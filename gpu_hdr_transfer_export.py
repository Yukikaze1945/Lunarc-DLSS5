"""Experimental HDR10 on the GPU-resident export chain: opt-in multiscale, no image readback.

Selected by ``job_settings["experimental_hdr_scrgb"] = True`` together with
``export_backend = "gpu_transfer"``. The flag defaults to off everywhere, the SDR
chain and its defaults are untouched by this file, and a request this chain cannot
serve is refused with every unmet condition listed rather than exported another way -
an HDR10 picture drawn by an 8-bit path would otherwise look like a successful HDR run.

Per frame, at a working scale below 1.0:

    nvdec_bridge        ``nvdec_bridge_open_hdr``: NVDEC decode into a borrowed P010
                        CUDA surface, with the frame's own colour enums, PTS and the
                        source's ST 2086 / CTA-861.3 metadata as the decoder read it
    gpu_hdr_scrgb_bridge  PQ -> linear scRGB FP16 at full size (decode), Lanczos-2
                        working master (downscale), display proxy + motion-only
                        RGBA8 guidance (proxy)
    nvof_bridge         flow read from that RGBA8 guidance plane, left on the device
    dlss_gpu_host       Feature 18 on an FP16 session: device colour in, borrowed
                        R16G16B16A16_FLOAT D3D12 texture out
    gpu_hdr_scrgb_bridge  Catmull-Rom residual against the pristine master
                        (reconstruct), then composited scRGB -> PQ-coded BT.2020
                        limited-range P010 (encode)
    nvenc_bridge        ``nvenc_p010_open`` + ``nvenc_encode_p010``: in-process
                        HEVC Main10 or AV1 Main/10-bit, with codec-specific
                        mastering-display and content-light SEIs / Metadata OBUs

Fixed parameters, chosen once per job and never derived from the content:

  * ``HDR_SUPPORT_CEILING_SCRGB`` (250.0) is the declared domain the decode validates
    against. It is a bound, not a measurement: Rec.2020 -> Rec.709 through this
    bridge's matrix has positive row sums whose largest is ~1.6602, so legal HDR10
    content peaks at 1.6602 x 125 = 207.53 scRGB, and 250 sits above that with margin.
    Content at the ceiling would already be out of domain, which is what keeps the
    above-ceiling count a statement about the source rather than a tautology.
  * ``HDR_CLIP_CEILING_SCRGB`` (125.0) is the encode's clamp top - ST 2084's 10000
    cd/m2 at this bridge's 80 cd/m2 reference white. It is never MaxCLL, and MaxCLL is
    never used as a ceiling anywhere in this chain.
  * ``hdr_paper_white_scrgb`` (8.0 = 640 cd/m2) is chosen once per job and handed to
    ``gpu_hdr_open``, which reports back the value it applies; a bridge that applies a
    different number refuses the open.

The chain moves no pixel across a host boundary. The host traffic it does have is
scalar: every executed pass of the colour bridge copies its counter run into a
CPU-visible readback buffer, because that is what the report structs are answered from.
The contract binds
image transfers rather than diagnostics, so this is permitted - and it is unmeasured,
which the transfer report records as ``hdr_scalar_report_readback`` with a byte count
of None instead of borrowing the image-readback counter to stand in for it.

What this chain deliberately does not do: it runs no multi-pass, resizes no output,
tonemaps nothing, carries no HLG, and re-encodes no picture. The encoder writes one
video stream into one Matroska file; a job that also
wants audio, subtitles, chapters, attachments, format metadata or the source video's
own title, language and disposition carried across gets the GPU-transfer chain's own
copy pass - ``gtx.needs_copy_pass`` to decide it and ``gtx.build_mux_command`` to
express it, with the encode as input 0, the source as input 1 and ``-c:v copy`` holding
the bitstream still. A request the shared pass cannot honour is refused at the gate
rather than silently delivered without it, and that includes every non-empty
``extra_args``: this stage carries no caller-written FFmpeg arguments at all, because
no denylist can tell a harmless addition from one that strips the mastering-display and
content-light SEIs out of the deliverable.

Performance is NOT_CHARACTERIZED, and nothing here claims formal HDR conformance:
FP16 quantisation loss is the Stage A record's, not this file's claim.
"""
from __future__ import annotations

import ctypes as C
import math
import os
import tempfile
import time
from fractions import Fraction

import bundle_paths
import gpu_transfer_export as gtx
import hdr_gpu_engine as hdr
import pipeline
from native_watchdog import phase as native_phase
from layer_settings import layer_parameters, resolve_layer_settings

NATIVE_DIR = bundle_paths.native_dir()

# The additive entries this chain needs. NVDEC HDR is an extra open on ABI 3 (a
# P010 surface is reported only to a bridge opened by nvdec_bridge_open_hdr); the
# NVENC HDR10 entry and its SEI writer are additive sections under their own
# version numbers, so each is checked against its own getter rather than assumed
# from the SDR ABI the shared library set was loaded on.
NVDEC_HDR_ABI = 3
NVENC_HDR_ABI = 1
NVENC_HDR_SEI_ABI = 1
# NvdecDeviceFrame.software_format: 1 = NV12, 2 = P010. Only a 2 is an HDR borrow.
NVDEC_SOFTWARE_FORMAT_P010 = 2

# --- the fixed policy -----------------------------------------------------------

HDR_SETTING = "experimental_hdr_scrgb"
HDR_PAPER_WHITE_SETTING = "hdr_paper_white_scrgb"

HDR_SUPPORT_CEILING_SCRGB = 250.0
HDR_CLIP_CEILING_SCRGB = 125.0
HDR_PAPER_WHITE_SCRGB = 8.0
# 10000 cd/m2 at this bridge's 80 cd/m2 reference white, stated as the arithmetic
# the two constants above are anchored to, so a reader can check 125 rather than
# take it on faith.
NITS_PER_SCRGB = 80.0
# The largest derived peak: a legal BT.2020 RGB sample that is at the PQ top on its
# most-excited channel converts to 1.66020 x 125 scRGB through this bridge's own
# to_scrgb row sums. The declared support ceiling has to sit above *that*, not above
# whatever one clip happens to measure.
DERIVED_LEGAL_PEAK_SCRGB = 207.53

HDR_CODEC = "hevc_nvenc"
HDR_PROFILE = "main10"
P010_PROFILES = {"hevc_nvenc": "main10", "av1_nvenc": "main"}
HDR_CONTAINER = ".mkv"
# The source shape the decode accepts: 10-bit 4:2:0, limited/full range, BT.2020
# primaries, ST 2084 transfer, BT.2020 non-constant-luminance matrix. ffprobe's own
# names, so the check is against what the probe reported rather than a second
# vocabulary. HLG is refused by name because it is the one thing that looks like it
# might be served by the same chain.
HDR_SOURCE_PIX_FMTS = ("yuv420p10le",)
HDR_SOURCE_RANGE = ("tv", "mpeg", "limited", "pc", "jpeg", "full")
HDR_SOURCE_PRIMARIES = ("bt2020",)
HDR_SOURCE_TRANSFER = ("smpte2084",)
HDR_SOURCE_SPACE = ("bt2020ncl", "bt2020nc")
HLG_TRANSFERS = ("arib-std-b67", "hlg")
# The encoder's HDR10 entry fixes these itself; passing one through extra_options
# refuses the open, so a setting that would produce one is refused here instead.
LOCKED_HDR_AV_OPTIONS = ("profile", "pix_fmt", "sw_pix_fmt", "rc", "colorspace",
                         "color_range", "alpha_bits")
# "preserve" is what the chain does: the source's own declaration, which the encoder
# entry re-states. "bt2020_pq" asks for exactly the declaration the entry insists on,
# so it agrees rather than adding a clause. hdr_to_sdr and every other preset is a
# tonemap or a metadata rewrite this chain will not perform.
HDR_ALLOWED_COLOR_PRESETS = ("preserve", "none", "", "None", "bt2020_pq")
HDR_ALLOWED_PIXEL_FORMATS = ("auto", "p010le", "yuv420p10le")
# The copy-pass request keys this chain expresses on top of ``gtx.MUX_ONLY_KEYS``.
# ``copy_video_stream_metadata`` is an override for the operator who already has a way to
# say what they want - ``copy_metadata`` - and it defaults to that: a job that did not
# name this key still gets the source video's own title, language and flags carried onto
# the deliverable, because declining the descriptive fields of one stream while copying
# the descriptive fields of the container would be a half-honoured promise. It stays out
# of the shared list on purpose: the SDR chain runs the same pass with no descriptive-video
# clause in its command, so a key it accepted and did not carry would be the silent
# loss this project refuses everywhere else. There it remains an unknown setting, which
# is a refusal; here it is honoured by ``gtx.build_mux_command``. Listing it is what
# keeps a job that *does* name it from being refused as an unsupported encoder-plan key.
HDR_MUX_ONLY_KEYS = frozenset({"copy_video_stream_metadata"})


class HdrExportError(RuntimeError):
    """A stage of the HDR chain refused; the text is that stage's own."""


# --- native struct mirrors for the additive HDR entries -------------------------
# Field for field from native/nvdec_bridge.h and native/nvenc_bridge.h, so a header
# drift is a struct_size the compiled bridge itself rejects rather than a silent
# misread. Sizes are never hardcoded: every struct is passed with C.sizeof of this
# file's own mirror, and an entry that answers rc != 0 on a struct_size mismatch is
# recorded as a bridge that does not match this checkout.

_U32, _I32, _U64, _I64 = (C.c_uint32, C.c_int32, C.c_uint64, C.c_int64)
_CP, _VP = C.c_char_p, C.c_void_p


class HdrMetadata(C.Structure):
    """Mirror of NvdecHdrMetadata, which is NvencHdrMetadata field for field.

    One mirror covers the handoff, which is the point: the decode query fills it and
    the encoder open takes it, so the source's own rationals reach the SEI without
    passing through a float. A zero in has_* means the source said nothing, and the
    encoder then attaches nothing rather than inventing a default.
    """
    _fields_ = [("struct_size", _U32), ("has_mastering", _I32),
                ("primary_x_num", _I32 * 3), ("primary_x_den", _I32 * 3),
                ("primary_y_num", _I32 * 3), ("primary_y_den", _I32 * 3),
                ("white_x_num", _I32), ("white_x_den", _I32),
                ("white_y_num", _I32), ("white_y_den", _I32),
                ("min_luminance_num", _I32), ("min_luminance_den", _I32),
                ("max_luminance_num", _I32), ("max_luminance_den", _I32),
                ("mastering_has_primaries", _I32),
                ("mastering_has_luminance", _I32), ("has_content_light", _I32),
                ("max_cll", _U32), ("max_fall", _U32)]


class HdrOptions(C.Structure):
    """Mirror of NvencHdrOptions."""
    _fields_ = [("struct_size", _U32), ("codec", _CP), ("profile", _CP),
                ("preset", _CP), ("tune", _CP), ("multipass", _CP),
                ("rate_control", _CP), ("constant_quality", _I32),
                ("bitrate", _I64), ("gop", _I32), ("extra_options", _CP),
                ("sw_pixel_format", _I32), ("color_range", _I32),
                ("color_primaries", _I32), ("color_trc", _I32),
                ("color_space", _I32)]


class HdrDefaults(C.Structure):
    """Mirror of NvencHdrDefaults: the contract the installed entry insists on."""
    _fields_ = [("struct_size", _U32), ("sw_pixel_format", _I32),
                ("color_range", _I32), ("color_primaries", _I32),
                ("color_trc", _I32), ("color_space", _I32),
                ("profile", C.c_char * 32)]


class HdrCapability(C.Structure):
    """Mirror of NvencHdrCapability, answered by the installed encoder."""
    _fields_ = [("struct_size", _U32), ("encoder_present", _I32),
                ("p010_supported", _I32), ("cuda_supported", _I32),
                ("main10_profile_supported", _I32),
                ("cuda_device_created", _I32), ("hdr_metadata_options", _I32),
                ("supported", _I32), ("encoder_name", C.c_char * 64),
                ("profile_name", C.c_char * 32)]


class P010Capability(C.Structure):
    _fields_ = [('profile_supported' if name == 'main10_profile_supported' else name, kind)
                for name, kind in HdrCapability._fields_]


class HdrContract(C.Structure):
    """Mirror of NvencHdrContract."""
    _fields_ = [("struct_size", _U32), ("codec", C.c_char * 64),
                ("container", C.c_char * 32), ("profile_name", C.c_char * 32),
                ("requested_pix_fmt", _I32), ("actual_pix_fmt", _I32),
                ("requested_sw_pix_fmt", _I32), ("actual_sw_pix_fmt", _I32),
                ("actual_sw_pix_fmt_name", C.c_char * 32),
                ("requested_profile", _I32), ("actual_profile", _I32),
                ("requested_bits_per_raw_sample", _I32),
                ("actual_bits_per_raw_sample", _I32), ("requested_color_range", _I32),
                ("requested_color_primaries", _I32), ("requested_color_trc", _I32),
                ("requested_color_space", _I32), ("actual_color_range", _I32),
                ("actual_color_primaries", _I32), ("actual_color_trc", _I32),
                ("actual_color_space", _I32), ("width", _U32), ("height", _U32),
                ("hw_frames_pool_size", _I32), ("hw_frames_sw_format", _I32),
                ("metadata_written_to", _I32), ("has_mastering", _I32),
                ("has_content_light", _I32), ("cuda_adapter_luid", C.c_uint8 * 8),
                ("cuda_adapter_node_mask", _U32)]


class HdrCounters(C.Structure):
    """Mirror of NvencHdrCounters."""
    _fields_ = [("struct_size", _U32), ("frames_submitted", _U64),
                ("packets_written", _U64), ("p010_device_copy_bytes", _U64),
                ("host_pixel_bytes", _U64), ("rgb_conversion_bytes", _U64),
                ("mastering_side_data", _U64), ("content_light_side_data", _U64)]


class SeiCapability(C.Structure):
    """Mirror of NvencHdrSeiCapability: properties of the writer, not of NVENC."""
    _fields_ = [("struct_size", _U32), ("writer_available", _I32),
                ("encoder_offers_metadata_option", _I32),
                ("supports_partial_mastering", _I32), ("primary_order", _I32),
                ("chroma_denominator", _I32), ("luma_denominator", _I32),
                ("payload_type_mastering", _I32),
                ("payload_type_content_light", _I32),
                ("prefix_sei_nal_type", _I32), ("mastering_payload_size", _U32),
                ("content_light_payload_size", _U32), ("max_nal_bytes", _U32),
                ("max_insert_bytes", _U32)]


class P010MetadataStatus(C.Structure):
    _fields_ = [("struct_size", _U32)] + [(name, _I32) for name in (
        "codec_kind", "color_mode", "writer_available", "metadata_requested",
        "mastering_serialized", "content_light_serialized", "error")] + [
        (name, _U64) for name in ("packets_seen", "mastering_inserted",
        "content_light_inserted", "mastering_already_present",
        "content_light_already_present", "bytes_inserted")] + [
        ("wire_values", _U32 * 12), ("max_chromaticity_error", C.c_double),
        ("max_luminance_error", C.c_double), ("detail", C.c_char * 256)]


class SeiStatus(C.Structure):
    """Mirror of NvencHdrSeiStatus: what the writer did to this session's packets."""
    _fields_ = [("struct_size", _U32), ("writer_available", _I32),
                ("metadata_requested", _I32), ("mastering_serialized", _I32),
                ("content_light_serialized", _I32), ("error", _I32),
                ("detail", C.c_char * 192), ("framing", _I32),
                ("packet_sample", C.c_char * 64), ("packets_seen", _U64),
                ("nals_inserted", _U64), ("bytes_inserted", _U64),
                ("mastering_inserted", _U64), ("content_light_inserted", _U64),
                ("mastering_already_present", _U64),
                ("content_light_already_present", _U64),
                ("packets_without_vcl", _U64), ("vcl_bytes", _U64),
                ("vcl_hash", _U64), ("existing_nal_bytes", _U64),
                ("existing_nal_hash", _U64)]


# Where the bridge wrote the metadata it was given; only the in-bitstream bit is a
# statement about the elementary stream, so the report names the bits set rather
# than printing a number a reader would have to decode.
METADATA_DESTINATIONS = {1: "frame_side_data", 2: "stream_coded_side_data",
                         4: "encoder_coded_side_data", 8: "container_colour_tags",
                         16: "in_bitstream_prefix_sei", 32: "in_bitstream_metadata_obu"}
SEI_ERRORS = {0: "ok", 1: "nothing_to_do", 2: "partial_mastering",
              3: "bad_rational", 4: "out_of_range", 5: "conflict",
              6: "framing_unsupported", 7: "no_vcl", 8: "malformed",
              9: "capacity"}


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if value is None else str(value)


def _fields(struct, skip=("struct_size",)):
    """Every declared field of a mirror as plain data, char arrays decoded."""
    out = {}
    for name, _ in struct._fields_:
        if name in skip:
            continue
        value = getattr(struct, name)
        if isinstance(value, bytes):
            out[name] = value.decode("utf-8", "replace")
        elif isinstance(value, C.Array):
            out[name] = list(value)
        elif isinstance(value, float):
            out[name] = value
        else:
            out[name] = int(value)
    return out


def mastering_wire_report(snapshot, codec=HDR_CODEC):
    """Audit the serializer's integer-grid conversion without changing source data.

    Matroska float metadata often becomes a rational with a power-of-two
    denominator. HEVC uses 1/50000 chromaticity and 1/10000 luminance units;
    AV1 uses 0.16 chromaticity, 24.8 maximum and 18.14 minimum luminance.
    Report each conversion and its exact error, including unchanged values.
    Native validation remains responsible for rejecting invalid declarations.
    """
    if not snapshot or not (snapshot.get("has_mastering") and
            snapshot.get("mastering_has_primaries") and
            snapshot.get("mastering_has_luminance")):
        return {"policy": "nearest_wire_unit_ties_up", "fields": [],
                "quantized_fields": 0}
    av1 = codec == 'av1_nvenc'
    chroma_scale = 65536 if av1 else 50000
    values = []
    for i, colour in enumerate(("red", "green", "blue")):
        for axis in ("x", "y"):
            values.append((f"{colour}_{axis}", snapshot[f"primary_{axis}_num"][i],
                           snapshot[f"primary_{axis}_den"][i], chroma_scale))
    for name, scale in (("white_x", chroma_scale), ("white_y", chroma_scale),
                        ("max_luminance", 256 if av1 else 10000),
                        ("min_luminance", 16384 if av1 else 10000)):
        values.append((name, snapshot[name + "_num"], snapshot[name + "_den"], scale))
    fields = []
    for name, num, den, scale in values:
        if den <= 0 or num < 0:
            fields.append({"field": name, "valid_rational": False})
            continue
        whole, remainder = divmod(num * scale, den)
        units = whole + int(2 * remainder >= den)
        source = Fraction(num, den)
        wire = Fraction(units, scale)
        fields.append({"field": name, "source": str(source), "wire": str(wire),
                       "wire_units": units, "wire_denominator": scale,
                       "quantized": wire != source,
                       "absolute_error": str(abs(wire - source))})
    return {"policy": "nearest_wire_unit_ties_up", "fields": fields,
            "quantized_fields": sum(bool(f.get("quantized")) for f in fields),
            "original_container_rationals_retained": True}


# --- the CUDA driver, narrowly --------------------------------------------------

class _Cuda:
    """cuInit / push / pop, and nothing else.

    The decoder's CUcontext is borrowed, and both the HDR bridge and the HDR encoder
    require it current on the calling thread. Making it current is the one driver
    operation this chain has to perform, so only the three entry points are resolved;
    there is no allocation, no memcpy and no stream here, which is part of what the
    zero-host-traffic claim rests on.
    """

    def __init__(self):
        bundle_paths.add_dll_directory(NATIVE_DIR)
        self._lib = C.WinDLL("nvcuda.dll")
        self._lib.cuInit.argtypes = [C.c_uint]
        self._lib.cuInit.restype = C.c_int
        self._lib.cuCtxPushCurrent_v2.argtypes = [_VP]
        self._lib.cuCtxPushCurrent_v2.restype = C.c_int
        self._lib.cuCtxPopCurrent_v2.argtypes = [C.POINTER(_VP)]
        self._lib.cuCtxPopCurrent_v2.restype = C.c_int

    def init(self):
        code = self._lib.cuInit(0)
        if code:
            raise HdrExportError(f"cuInit failed ({code})")

    def push(self, context):
        code = self._lib.cuCtxPushCurrent_v2(C.c_void_p(int(context or 0)))
        if code:
            raise HdrExportError(
                f"cuCtxPushCurrent_v2 refused the decoder's CUcontext "
                f"({hex(int(context or 0))}): rc {code}")
        return True

    def pop(self):
        previous = _VP()
        code = self._lib.cuCtxPopCurrent_v2(C.byref(previous))
        if code:
            raise HdrExportError(f"cuCtxPopCurrent_v2 failed ({code})")
        return int(previous.value or 0)


# --- the additive entries, bound on the library set the SDR backend loaded ------

_HDR_NVDEC = {
    "nvdec_bridge_open_hdr": (_I32, [_CP, C.POINTER(_VP)]),
    "nvdec_bridge_get_hdr_metadata": (_I32, [_VP, C.POINTER(HdrMetadata)]),
}
_HDR_NVENC = {
    "nvenc_hdr_abi_version": (_U32, []),
    "nvenc_hdr_get_defaults": (_I32, [C.POINTER(HdrDefaults)]),
    "nvenc_hdr_capability": (_I32, [_VP, _U32, _U32, C.POINTER(HdrCapability)]),
    "nvenc_hdr_open": (_I32, [_VP, _U32, _U32, _U32, _U32, _I32, _I32, _CP,
                              C.POINTER(HdrOptions), C.POINTER(HdrMetadata),
                              C.POINTER(_VP)]),
    "nvenc_encode_p010": (_I32, [_VP, _U64, C.c_size_t, _U64, C.c_size_t, _I64]),
    "nvenc_hdr_get_counters": (_I32, [_VP, C.POINTER(HdrCounters)]),
    "nvenc_hdr_get_contract": (_I32, [_VP, C.POINTER(HdrContract)]),
    "nvenc_hdr_sei_abi_version": (_U32, []),
    "nvenc_hdr_sei_capability": (_I32, [C.POINTER(SeiCapability)]),
    "nvenc_hdr_sei_get": (_I32, [_VP, C.POINTER(SeiStatus)]),
}


_P010_NVENC = {
    "nvenc_p010_abi_version": (_U32, []),
    "nvenc_p010_open": (_I32, [_VP, _U32, _U32, _U32, _U32, _I32, _I32, _CP,
        C.POINTER(HdrOptions), C.POINTER(HdrMetadata), _I32, C.POINTER(_VP)]),
    "nvenc_p010_capability": (_I32, [_VP, _U32, _U32, _CP, C.POINTER(P010Capability)]),
    "nvenc_p010_get_metadata_status": (_I32, [_VP, C.POINTER(P010MetadataStatus)]),
    "nvenc_p010_get_contract": (_I32, [_VP, C.POINTER(HdrContract)]),
}


def open_p010_encoder(library, sdr10, context, width, height, fps_num, fps_den,
                      tb_num, tb_den, path, options, metadata, handle):
    bound = gtx._bind_optional(library, _P010_NVENC)
    if 'nvenc_p010_open' in bound:
        if library.nvenc_p010_abi_version() != 1:
            raise gtx.BridgeError('Unsupported P010 encoder ABI')
        return library.nvenc_p010_open(context, width, height, fps_num, fps_den,
            tb_num, tb_den, path, options, metadata, 0 if sdr10 else 1, handle)
    if C.cast(options, C.POINTER(HdrOptions)).contents.codec != b'hevc_nvenc':
        raise gtx.BridgeError('AV1 10-bit requires the rebuilt P010 encoder bridge')
    entry = library.nvenc_sdr10_open if sdr10 else library.nvenc_hdr_open
    return entry(context, width, height, fps_num, fps_den, tb_num, tb_den,
                 path, options, metadata, handle)


def load_hdr_capabilities(bridges, width=None, height=None, codec=HDR_CODEC) -> dict:
    """Bind the additive HDR entries onto the loaded set and ask them what they are.

    Called on the export thread, after ``gtx.load_bridges`` and before any frame:
    presence, ABI number, struct size acceptance and the installed encoder's own
    HDR10 declaration are all facts negotiation needs, and none of them needs a
    device. ``nvenc_hdr_capability`` is asked with a NULL context, so it reports the
    static capacity and explicitly does not create a CUDA device object; the
    report says so rather than implying more than was asked.
    """
    out = {"nvdec": {}, "nvenc": {}, "colour_bridge": {}, "missing": []}
    nvdec = bridges.get("nvdec")
    nvenc = bridges.get("nvenc")
    if nvdec is None or nvenc is None:
        out["missing"].append("the loaded bridge set carries no decoder or encoder")
        return out
    abi = int(nvdec.nvdec_bridge_abi_version())
    out["nvdec"]["abi"] = abi
    if abi != NVDEC_HDR_ABI:
        out["missing"].append(
            f"nvdec_bridge.dll reports ABI {abi}; the P010 borrow is additive at "
            f"ABI {NVDEC_HDR_ABI}")
        return out
    bound = gtx._bind_optional(nvdec, _HDR_NVDEC)
    for symbol in _HDR_NVDEC:
        if symbol not in bound:
            out["missing"].append(
                f"nvdec_bridge.dll exports no {symbol}, so no P010 frame can be "
                "borrowed; rebuild it from native/nvdec_bridge.cpp")
    if out["missing"]:
        return out

    hdr_abi = int(nvenc.nvenc_hdr_abi_version())
    out["nvenc"]["hdr_abi"] = hdr_abi
    if hdr_abi != NVENC_HDR_ABI:
        out["missing"].append(
            f"nvenc_bridge.dll's HDR10 entry reports ABI {hdr_abi}, this chain "
            f"reads {NVENC_HDR_ABI}")
        return out
    sei_abi = int(nvenc.nvenc_hdr_sei_abi_version())
    out["nvenc"]["sei_abi"] = sei_abi
    if sei_abi != NVENC_HDR_SEI_ABI:
        out["missing"].append(
            f"the in-bitstream SEI writer reports ABI {sei_abi}, this chain reads "
            f"{NVENC_HDR_SEI_ABI}")
        return out
    bound = gtx._bind_optional(nvenc, _HDR_NVENC)
    for symbol in _HDR_NVENC:
        if symbol not in bound:
            out["missing"].append(
                f"nvenc_bridge.dll exports no {symbol}; the HDR10 entry is "
                "incomplete and must be rebuilt from native/nvenc_bridge.cpp")
    if out["missing"]:
        return out

    # struct_size gate: each entry rejects a size that does not match the compiled
    # struct, so an rc of 0 below is the bridge's own statement that this file's
    # mirror is its layout. A non-zero rc is a missing capability, not a retry.
    defaults = HdrDefaults(C.sizeof(HdrDefaults))
    rc = int(nvenc.nvenc_hdr_get_defaults(C.byref(defaults)))
    out["defaults_rc"] = rc
    out["struct_sizes_accepted"] = {"NvencHdrDefaults": rc == 0}
    if rc:
        out["missing"].append(
            f"nvenc_hdr_get_defaults failed ({rc}): "
            f"{_text(nvenc.nvenc_last_error())}")
        return out
    out["encoder_defaults"] = _fields(defaults)
    out["encoder_defaults"]["profile"] = P010_PROFILES.get(codec, "")
    p010_bound = gtx._bind_optional(nvenc, _P010_NVENC)
    out["p010_available"] = set(_P010_NVENC).issubset(p010_bound) and nvenc.nvenc_p010_abi_version() == 1
    out["encoder_codec"] = codec
    if codec not in P010_PROFILES or (codec == 'av1_nvenc' and not out['p010_available']):
        out['missing'].append('Requested codec needs the generic P010 bridge: ' + codec)
        return out

    sei_cap = SeiCapability(C.sizeof(SeiCapability))
    rc = int(nvenc.nvenc_hdr_sei_capability(C.byref(sei_cap)))
    out["struct_sizes_accepted"]["NvencHdrSeiCapability"] = rc == 0
    if rc:
        out["missing"].append(
            f"nvenc_hdr_sei_capability failed ({rc}): "
            f"{_text(nvenc.nvenc_last_error())}")
        return out
    out["sei_capability"] = _fields(sei_cap)

    cap_type = P010Capability if out['p010_available'] else HdrCapability
    cap = cap_type(C.sizeof(cap_type))
    if out['p010_available']:
        rc = int(nvenc.nvenc_p010_capability(None, int(width or 0), int(height or 0), codec.encode(), C.byref(cap)))
    else:
        rc = int(nvenc.nvenc_hdr_capability(None, int(width or 0), int(height or 0), C.byref(cap)))
    out["struct_sizes_accepted"]["NvencHdrCapability"] = rc == 0
    if rc:
        out["missing"].append(
            f"the installed FFmpeg has no usable {codec} for the P010 entry "
            f"({rc}): {_text(nvenc.nvenc_last_error())}")
        return out
    out["capability"] = _fields(cap)

    library = hdr.load()
    out["colour_bridge"] = {
        "abi": hdr.GPU_HDR_ABI_VERSION,
        "path": getattr(library, "_dll_path", None),
        "bound_exports": sorted(name for name, _, _ in hdr._BRIDGE),
        "paper_white_default": HDR_PAPER_WHITE_SCRGB,
        # The optional multiscale foundation, asked by symbol name. A capability
        # read costs a name lookup and nothing else - no device, no pixel - and the
        # gate needs the answer before it can say whether a second layer is
        # available at all, so it is taken here rather than left for the open.
        # Speaking ABI 1 says nothing about it.
        "multiscale": hdr.probe_multiscale(library),
    }
    # The structural half of the no-pixel-readback claim, recorded as a fact about
    # this process rather than asserted in prose: an entry point that was never
    # resolved cannot have moved a pixel. These are the image-plane readbacks. The
    # bridge's per-pass counter readback is a different class of traffic - scalars,
    # on every dispatch, and not counted by report_readback_bytes - and the transfer
    # report names it as the boundary that exists but has no measurement.
    out["colour_bridge"]["pixel_readback_exports_bound"] = sorted(
        name for name in out["colour_bridge"]["bound_exports"] if "readback" in name)
    out["colour_bridge"]["scalar_counter_readback_per_dispatch"] = {
        "crosses_host": True,
        "bytes_from_source": 256,
        "measured_by": None,
        "reading": "kCounterSlots * sizeof(uint32_t) of counters per executed pass, "
                   "copied to a CPU-visible readback buffer by run_pass. Permitted: "
                   "the contract binds image transfers, not scalar diagnostics. No "
                   "ABI counter measures it, so report_readback_bytes is not its "
                   "measurement and zero there is not a zero here."}
    out["nvdec"]["pixel_readback_entry_bound"] = bool(
        "nvdec_bridge_readback_last_frame" in vars(nvdec))
    # The typed HDR half of the fusion bridge, asked of the library the bridge set
    # already loaded. A set loaded for a single layer carries no fusion library at
    # all, and then no read is made and nothing is claimed: the gate refuses a
    # data-consistency job for want of the read rather than assuming the group is
    # there because the SDR entries are.
    fusion = bridges.get("fusion")
    if fusion is not None:
        import hdr_fusion_engine
        out["fusion_bridge"] = hdr_fusion_engine.probe_hdr_fusion(fusion)
        out["fusion_bridge"]["path"] = getattr(fusion, "_dll_path", None)
    return out


def load_sdr10_capabilities(bridges, width=None, height=None, codec=HDR_CODEC):
    """Shared FP16/P010 facilities, with separately verified SDR colour entries."""
    caps = load_hdr_capabilities(bridges, width, height, codec=codec)
    for key, name, original in (("nvdec", "nvdec_bridge_open_p010", "nvdec_bridge_open_hdr"),
                                ("nvenc", "nvenc_sdr10_open", "nvenc_hdr_open")):
        try:
            library = bridges[key]
            entry = getattr(library, name)
            prototype = getattr(library, original)
            entry.argtypes, entry.restype = prototype.argtypes, prototype.restype
        except (KeyError, AttributeError) as error:
            caps["missing"].append(f"missing SDR10 entry {name}: {error}")
    library = hdr.load()
    for name in ("gpu_sdr10_open", "gpu_sdr10_decode", "gpu_sdr10_contract"):
        if not hasattr(library, name):
            caps["missing"].append(f"missing SDR10 colour entry {name}")
    defaults = dict(caps.get("encoder_defaults") or {})
    defaults.update(color_range=1, color_primaries=1, color_trc=1, color_space=1)
    caps["encoder_defaults"] = defaults
    caps["colour_mode"] = "sdr10_bt709"
    return caps


def sdr10_source(source):
    """Only the SDR domain with implemented colour math; never infer HDR from depth."""
    supported_format = (source.get("pix_fmt") == "yuv420p10le" or
        (str(source.get("codec_name", "")).lower() == "prores" and
         source.get("pix_fmt") == "yuv422p10le"))
    return (supported_format
            and source.get("color_transfer") == "bt709"
            and source.get("color_primaries") == "bt709"
            and source.get("color_space") == "bt709"
            and source.get("color_range") in ("tv", "mpeg", "limited"))


def release_loader_cookies(bridges):
    """Give back the DLL-directory cookies a bridge set is still holding.

    ``gtx.load_bridges`` opens them so the FFmpeg-side imports of nvdec and nvenc
    resolve, and hands them to whoever takes the set. The export that runs a frame
    closes them in its own ``finally``; a job that never reached that point - refused by
    the gate, or raised on the way to it - is the one that still owns them, and a
    refusal that does not say so leaks a handle per refused job for the life of the
    process.

    Only the cookies close. The libraries they made resolvable belong to the process,
    and in a frozen build to whatever other session imported them first, so nothing here
    unloads or frees them. The list is emptied as it is released, which makes a second
    call - from the export's ``finally`` on a set that got that far - do nothing rather
    than close the same cookie twice.

    Returns how many cookies this call released.
    """
    if not bridges:
        return 0
    cookies = list(bridges.get("cookies") or [])
    for cookie in cookies:
        gtx._close_cookie(cookie)
    bridges["cookies"] = []
    return len(cookies)


# --- the request --------------------------------------------------------------

_TRUE_FORMS = (True, "1", "true", "True", "yes", "on")
_FALSE_FORMS = (False, None, "", "0", "false", "False", "no", "off")


def hdr_requested(job_settings) -> bool:
    """Whether this job explicitly turned the experimental HDR chain on.

    Explicit only: an absent key, a false value or any unreadable value is "not
    requested", which leaves the job on the SDR chain - where a PQ source is refused
    by that chain's own gate rather than exported wrongly. A caller routes on this and
    never on a guess about what the user meant.
    """
    value = (job_settings or {}).get(HDR_SETTING)
    if isinstance(value, str):
        return value.strip() in _TRUE_FORMS or value.strip().lower() in (
            "true", "yes", "on")
    return value in (True,) or (isinstance(value, int) and not isinstance(
        value, bool) and value == 1)


def hdr_flag_reason(job_settings):
    """A reason when the flag is neither clearly on nor clearly off, else None."""
    value = (job_settings or {}).get(HDR_SETTING)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return None
    text_value = str(value).strip()
    if text_value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off",
                              ""):
        return None
    return (f"{HDR_SETTING}={value!r} is neither an explicit on nor an explicit off; "
            "the experimental HDR chain is opted into with a boolean and is never "
            "guessed at")


def paper_white_reason(job_settings):
    """A reason when the job's paper white is not a usable fixed value, else None."""
    value = (job_settings or {}).get(HDR_PAPER_WHITE_SETTING, HDR_PAPER_WHITE_SCRGB)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (f"{HDR_PAPER_WHITE_SETTING}={value!r} is not a number; it is the "
                f"diffuse white the HDR bridge is opened with ({HDR_PAPER_WHITE_SCRGB} "
                "scRGB = 640 cd/m2) and is chosen per job, never per frame")
    if not math.isfinite(number) or number < 1.0:
        return (f"{HDR_PAPER_WHITE_SETTING}={number!r} is not finite and at least "
                "1.0 scRGB")
    return None


def plan_hdr_encoder(transcode_settings, cq: int, source: dict,
                     option_available=None, hdr_caps=None,
                     tune_available=None) -> dict:
    """Turn the job's encoding settings into the HDR10 entry's own request.

    The same option vocabulary ``gtx.plan_encoder`` uses, minus the names the HDR10
    entry fixes for itself: ``profile`` goes into the struct's own field, and
    ``pix_fmt``, ``sw_pix_fmt``, ``rc``, ``colorspace``, ``color_range`` and
    ``alpha_bits`` have no place in the pass-through at all, because an extra option
    that names one refuses the open. The five numeric declarations are sent as the
    installed entry reported them, so this file hardcodes no FFmpeg enumerator.
    """
    ts = dict(transcode_settings or {})
    if str(ts.get("codec", "")).lower() in ("prores_ks", "prores"):
        from prores_export import encoder_plan
        return encoder_plan(ts, source)
    unsupported = []
    codec = str(ts.get("codec", HDR_CODEC))
    profile = P010_PROFILES.get(codec, '')
    if not profile:
        unsupported.append(f'codec={codec}: P010 supports HEVC or AV1 only')
    if codec == 'av1_nvenc' and not (hdr_caps or {}).get('p010_available'):
        unsupported.append('AV1 10-bit requires generic P010 encoder capabilities')
    if (hdr_caps or {}).get('encoder_codec', codec) != codec:
        unsupported.append('Encoder capabilities were queried for a different codec')
    requested_profile = str(ts.get('profile', 'auto')).strip()
    accepted_profiles = ('auto', '', 'None', 'none', profile)
    if codec == 'hevc_nvenc': accepted_profiles += ('Main 10', 'main 10', 'Main10')
    if requested_profile not in accepted_profiles:
        unsupported.append(f'profile={requested_profile}: {codec} requires {profile}')

    pix_fmt = str(ts.get("pixel_format", "auto")).strip().lower()
    if pix_fmt not in HDR_ALLOWED_PIXEL_FORMATS:
        unsupported.append(
            f"pixel_format={pix_fmt}: the HDR chain submits P010 (10-bit "
            "4:2:0) planes and refuses an 8-bit request rather than encoding one")

    struct = {"codec": codec.encode(), "profile": profile.encode(),
              "preset": None, "tune": None, "multipass": None,
              "rate_control": None, "constant_quality": 0, "bitrate": 0, "gop": 0,
              # Zero means "the entry's own HDR10 value", which is what is filled in
              # below from nvenc_hdr_get_defaults; nothing here is an opinion about
              # what a FFmpeg pixel-format or colour-space enumerator equals.
              "sw_pixel_format": 0, "color_range": 0, "color_primaries": 0,
              "color_trc": 0, "color_space": 0}
    defaults = (hdr_caps or {}).get("encoder_defaults") or {}
    for field, key in (("sw_pixel_format", "sw_pixel_format"),
                       ("color_range", "color_range"),
                       ("color_primaries", "color_primaries"),
                       ("color_trc", "color_trc"),
                       ("color_space", "color_space")):
        if key in defaults:
            struct[field] = int(defaults[key])
    declared_profile = str(defaults.get("profile") or "").strip()
    if declared_profile and declared_profile != profile:
        unsupported.append(
            f"the installed HDR10 entry names its profile {declared_profile!r}, not "
            f"{profile!r}; the request is refused rather than sent to an "
            "encoder that would reject it after the first frame")

    av_options = []
    preset = str(ts.get("preset", "p7"))
    struct["preset"] = preset.encode()
    tune = requested_tune = "hq" if not ts else str(ts.get("tune", "auto"))
    # The entry passes tune straight into the encoder's option table, where an
    # enumerator that table does not list is an EINVAL that refuses the open after
    # the decode and the reconstruction have already run. Which codecs list uhq is a
    # property of the loaded libavcodec, so the probe is the caller's and the
    # value passes through when nobody answered.
    if tune == "uhq" and tune_available is not None and not tune_available(codec):
        tune = "hq"
    if tune and tune != "auto":
        struct["tune"] = tune.encode()
    multipass = str(ts.get("multipass", "auto"))
    if not ts:
        struct["multipass"] = b"qres"
    elif preset != "p7" and multipass != "auto":
        struct["multipass"] = multipass.encode()

    rate_mode = str(ts.get("rate_mode", "cq"))
    if rate_mode == "cq":
        struct["rate_control"] = b"vbr"
        struct["constant_quality"] = int(ts.get("cq", cq))
    elif rate_mode == "vbr":
        struct["rate_control"] = b"vbr"
        av_options.append(("b", str(ts.get("video_bitrate", "80M"))))
    elif rate_mode == "cbr":
        struct["rate_control"] = b"cbr"
        bitrate = str(ts.get("video_bitrate", "80M"))
        av_options += [("b", bitrate), ("maxrate", bitrate), ("bufsize", bitrate)]
    elif rate_mode == "constqp":
        struct["rate_control"] = b"constqp"
        av_options.append(("qp", str(int(ts.get("qp", 20)))))
    else:
        unsupported.append(f"rate_mode={rate_mode}")
    if not struct["rate_control"]:
        struct["rate_control"] = b"vbr"

    if rate_mode in ("cq", "vbr") and str(ts.get("peak_policy", "")) == "custom":
        av_options += [("maxrate", str(ts.get("maxrate", "380M"))),
                       ("bufsize", str(ts.get("bufsize", "1520M")))]

    fps = gtx.fps_rational(source) or (60, 1)
    if str(ts.get("gop_mode", "auto")) == "auto":
        struct["gop"] = max(1, int(round(fps[0] / fps[1] * 2.0)))
    else:
        struct["gop"] = max(1, int(ts.get("gop", 240)))

    for option, key, default in (
            ("bf", "b_frames", "auto"), ("b_ref_mode", "b_ref_mode", "auto"),
            ("rc-lookahead", "lookahead", "auto"),
            ("lookahead_level", "lookahead_level", "auto"),
            ("weighted_pred", "weighted_prediction", None),
            ("tf_level", "temporal_filter", "auto"),
            ("split_encode_mode", "split_encode", "auto"),
            ("level", "level", "auto"), ("tier", "tier", "auto"),
            ("gpu", "gpu", "-1")):
        text = default if key not in ts else str(ts.get(key))
        if text in (None, "", "auto", "None", "-1"):
            continue
        if key == "weighted_prediction":
            text = "1" if text in ("1", "true", "True", "on") else "0"
        if codec == 'av1_nvenc' and key == 'weighted_prediction' and text == '0':
            continue
        av_options.append((option, text))

    aq_mode = str(ts.get("aq_mode", "spatial"))
    if aq_mode == "spatial":
        av_options += [("spatial-aq", "1"), ("temporal-aq", "0"),
                       ("aq-strength", str(ts.get("aq_strength", 8)))]
    elif aq_mode == "temporal":
        av_options += [("spatial-aq", "0"), ("temporal-aq", "1")]
    elif aq_mode in ("off", "none"):
        av_options += [("spatial-aq", "0"), ("temporal-aq", "0")]
    else:
        unsupported.append(f"aq_mode={aq_mode}")

    for name, value in list(av_options):
        if name in LOCKED_HDR_AV_OPTIONS:
            av_options.remove((name, value))
            unsupported.append(
                f"encoder option {name}={value} is fixed by the HDR10 entry and "
                "cannot be passed to it")
        elif option_available is not None and not option_available(codec, name):
            av_options.remove((name, value))
            unsupported.append(f"encoder {codec} has no option {name}, so the "
                               "setting for it cannot be applied")

    value = int(struct['constant_quality'])
    if not 0 <= value <= (63 if codec == 'av1_nvenc' else 51):
        unsupported.append(f'CQ out of range for {codec}: {value}')
    if codec == 'av1_nvenc' and ts.get('weighted_prediction'):
        unsupported.append('AV1 does not support weighted prediction')
    if codec == 'av1_nvenc' and ts.get('b_ref_mode') == 'each':
        unsupported.append('AV1 does not support each B-frame as reference')
    ignored = {k for k in ts if k not in gtx.MUX_ONLY_KEYS
               and k not in HDR_MUX_ONLY_KEYS
               and k not in gtx.ENCODER_KEYS and k not in gtx.QUEUE_ONLY_KEYS}
    for key in sorted(ignored):
        unsupported.append(f"transcode_settings[{key!r}] has no equivalent this "
                           "encoder can be given")
    return {"struct": struct, "av_options": av_options, "codec": codec,
            "tune_adjustment": {"requested": requested_tune, "applied": tune,
                "reason": "loaded encoder does not support UHQ"} if tune != requested_tune else None,
            "profile": profile, "unsupported": unsupported}


# --- the gate -------------------------------------------------------------------

# The copy pass is not a general FFmpeg front for this chain. ``extra_args`` is appended
# after every clause ``gtx.build_mux_command`` writes, FFmpeg honours the later clause,
# and the clauses it would have to leave alone to deliver an HDR10 picture are the video
# codec, the stream mapping, the colour declaration, the bitstream filters and the
# overwrite decision - which is most of what an argument list is for. Telling those
# apart from a harmless addition needs a parser that reads FFmpeg's option grammar the
# way FFmpeg does, and the denylist this file used to carry did not: a caller could write
# ``-bsf:v filter_units=remove_types=39|40`` and have the mastering-display and
# content-light SEIs stripped from the deliverable while every gate still said yes. So
# the contract is the narrow honest one: an HDR10 job names its mapping with the form's
# own audio, subtitle, attachment, metadata and chapter choices, and carries no extra
# arguments at all. The SDR chain is untouched - its ``extra_args`` still reaches its
# command exactly as it always has.
def hdr_mux_refusals(transcode_settings) -> list:
    """What the copy pass cannot be asked to carry, before a single frame is encoded.

    The pass is ``gtx.build_mux_command`` with no HDR branch of its own, so which
    modes it expresses is that helper's vocabulary rather than a second list invented
    here. Refusing at the gate is what keeps a mode the pass cannot honour from
    arriving after the GPU work is already spent.
    """
    ts = dict(transcode_settings or {})
    reasons = []
    audio_mode = str(ts.get("audio_mode", "copy"))
    if audio_mode not in gtx.MUX_AUDIO_MODES:
        reasons.append(f"audio_mode={audio_mode}: the container copy pass expresses "
                       f"only {', '.join(gtx.MUX_AUDIO_MODES)}, and this chain runs "
                       "no audio handling of its own to serve another value")
    subtitle_mode = str(ts.get("subtitle_mode", "copy"))
    if subtitle_mode not in gtx.MUX_SUBTITLE_MODES:
        reasons.append(
            f"subtitle_mode={subtitle_mode}: the container copy pass expresses only "
            f"{', '.join(gtx.MUX_SUBTITLE_MODES)}. An unrecognised value would leave "
            "the muxer to pick a subtitle codec on its own, which is a silent "
            "conversion rather than the copy that was asked for")
    # A field the caller left out is not an argument list: only a value with
    # something in it is a request this chain has to refuse.
    named = gtx.extra_args_request(ts)
    if named:
        reasons.append(
            f"extra_args={named!r}: this stage of the HDR10 chain accepts no extra "
            "FFmpeg arguments at all. The clauses the copy pass already owns are the "
            "video codec, the stream mapping, the colour declaration, the bitstream "
            "filters and the overwrite decision, FFmpeg honours the later clause, and "
            "no list of option names can tell a harmless addition from one that "
            "strips the mastering-display and content-light SEIs out of the "
            "deliverable - so the mapping is named by audio_mode, subtitle_mode, "
            "copy_attachments, copy_metadata and copy_chapters instead, and a job "
            "that also names extra_args is refused rather than run")
    return reasons


def wants_video_descriptive_metadata(transcode_settings) -> bool:
    """Whether this job wants the source video's title, language and flags carried.

    Derived from the switch the operator already has: ``copy_metadata`` asks that the
    deliverable describe the material rather than only contain it, and a track name and a
    language are exactly that kind of description. So the default is to copy them, and
    ``copy_metadata=False`` - a job that declined descriptive metadata - declines the
    video track's own fields with it. ``copy_video_stream_metadata`` overrides in either
    direction for a caller that means the video stream specifically, which is why the
    key is accepted rather than refused; nobody has to set it, and it never reaches the
    encoder plan as an unknown setting.
    """
    ts = dict(transcode_settings or {})
    named = ts.get("copy_video_stream_metadata")
    if named is not None:
        return bool(named)
    return bool(ts.get("copy_metadata", True))


def descriptive_video_metadata_request(source, transcode_settings):
    """What the pass is handed for the video stream's own descriptive fields, or ``None``.

    ``None`` means this run writes no descriptive clause for video and promises nothing:
    a job that declined the metadata, a source that was never probed for those fields, or
    a track with nothing descriptive on it. Anything else is the narrowed field set
    ``gtx.build_mux_command`` puts on the *output* video stream - the encoder's own - so
    the source's technical tags (``ENCODER``, ``BPS``, ``DURATION``, frame counts) are
    never restated as facts about this run's encode, and the source's video packets are
    never mapped.
    """
    if not wants_video_descriptive_metadata(transcode_settings):
        return None
    return gtx.source_video_descriptive(source)


def hdr_multi_layer_refusals(multi, caps, live, flow_provider, uses_flow,
                             working) -> list:
    """Why a second HDR layer would run here, or exactly why it would not.

    Four separately-versioned things have to be true and each is asked on its own:
    the colour bridge's optional multiscale group, the Feature 18 host's ability to
    open an extra RGBA16F session on the anchor's core, a flow provider that can
    give each layer its own motion history, and the typed half of the fusion bridge
    when the job asks for a data-consistency solve. None of them is inferred from
    ``GPU_HDR_ABI_VERSION``, and a missing one is a refusal the operator can read -
    a second layer is not the one layer it would otherwise have run asked twice.
    """
    reasons = []
    multiscale = ((caps or {}).get("colour_bridge") or {}).get("multiscale")
    if multiscale is None:
        reasons.append("no HDR multiscale capability read was made on the loaded "
                       "colour bridge, so an extra layer's master import, "
                       "observation and composite are unverified")
    elif not multiscale.get("supported"):
        reasons.append(multiscale.get("reason") or
                       "the loaded colour bridge has no HDR multiscale foundation")
    if tuple(multi["dims"][0]) != tuple(working):
        reasons.append(
            f"layer 0 resolves to {multi['dims'][0][0]}x{multi['dims'][0][1]}, not "
            f"the {working[0]}x{working[1]} the anchor session and the primary "
            "bridge are opened at")
    capability = getattr(live, "layer_capability", None)
    if not callable(capability):
        reasons.append("the Feature 18 session reports no layer capability, so an "
                       "extra session on its core cannot be verified before the "
                       "open")
    else:
        answer = capability() or {}
        missing = [name for name in ("shared_layer_sessions",
                                     "format_aware_layer_open",
                                     "rgba16f_sessions") if not answer.get(name)]
        if missing:
            reasons.append(
                "the loaded GPU host opens no extra RGBA16F layer session (no "
                f"{', '.join(missing)}), so a dual or triple reconstruction has "
                "nothing to run the second and third Feature 18 passes on")
    if uses_flow and not (callable(getattr(flow_provider, "new_independent_layer",
                                           None)) and
                          callable(getattr(flow_provider,
                                           "close_independent_layer", None))):
        reasons.append("guidance_mode=1 over several layers needs an independent "
                       "NVOF layer each: this flow provider exports no "
                       "new_independent_layer and close_independent_layer, so the "
                       "layers would share one motion history and one reset")
    if multi["fusion"] == "data_consistency_experimental":
        fusion = (caps or {}).get("fusion_bridge")
        if fusion is None:
            reasons.append("no typed HDR fusion capability read was made on the "
                           "loaded bridge set, so gpu_fusion_open_hdr and "
                           "gpu_fusion_process_hdr are unverified and a "
                           "data-consistency solve cannot be promised")
        elif not fusion.get("supported"):
            reasons.append(fusion.get("reason") or
                           "the fusion bridge carries no typed linear-scRGB group")
    return reasons


def negotiate_hdr(dims: dict, source: dict, job_settings, transcode_settings,
                  cq: int, with_audio: bool, out_path: str, live,
                  hdr_caps=None, option_available=None, bridges_ok=True,
                  bridge_error=None, flow_provider=None,
                  extra_reasons=None, tune_available=None,
                  colour_mode="hdr10") -> dict:
    """Say whether this HDR10 export can run on the GPU, and exactly why not.

    The same shape and the same rule set as ``gtx.negotiate`` wherever the two chains
    share a requirement, and deliberately narrower where they do not: one pass, no
    resize, no tonemap, and a source that is actually in the BT.2100 PQ domain rather
    than merely tagged for it. The layer vector is the shared helper's own
    ``gtx.multi_scale_config``, so a dual or triple request is judged by the rules the
    SDR backend judges it by and the only HDR-specific question left is whether the
    four separately-versioned parts a second layer needs are actually loaded. A job
    that needs anything besides its video
    stream is accepted onto the GPU-transfer chain's own copy pass, because the rule
    that decides it and the command that carries it out are the same two shared helpers
    - ``gtx.needs_copy_pass`` and ``gtx.build_mux_command`` - and the only thing this
    gate adds is what an HDR10 bitstream cannot let a caller override. Nothing here
    touches an image, so the whole decision is testable without a display attached.
    """
    js = dict(job_settings or {})
    ts = dict(transcode_settings or {})
    sdr10 = colour_mode == "sdr10_bt709"
    if colour_mode not in ("hdr10", "sdr10_bt709"):
        raise ValueError(f"unknown high precision colour mode {colour_mode!r}")
    reasons = []
    if not bridges_ok:
        reasons.append("bridges: " + (bridge_error or "unavailable"))
    caps = hdr_caps or {}
    for gap in caps.get("missing") or []:
        reasons.append(gap)
    if not caps:
        reasons.append("no HDR capability read was made, so the additive decoder, "
                       "colour-bridge and encoder entries are unverified")

    flag = hdr_flag_reason(js) if not sdr10 else None
    if flag:
        reasons.append(flag)
    white_reason = paper_white_reason(js) if not sdr10 else None
    if white_reason:
        reasons.append(white_reason)
    # The value the chain would open its colour bridge with. A job that named a
    # paper white the check above refused is refused outright, so this only has to
    # be readable when the run is accepted; the fixed default keeps the policy dict
    # a description of what would run rather than a description of nothing.
    try:
        paper_white = float(js.get(HDR_PAPER_WHITE_SETTING, HDR_PAPER_WHITE_SCRGB))
        if not math.isfinite(paper_white) or paper_white < 1.0:
            paper_white = HDR_PAPER_WHITE_SCRGB
    except (TypeError, ValueError):
        paper_white = HDR_PAPER_WHITE_SCRGB
    if sdr10:
        paper_white = 1.0
        if hdr_requested(js):
            reasons.append("SDR10 must not be selected with experimental_hdr_scrgb")

    if str(js.get("export_backend")) != "gpu_transfer":
        reasons.append(
            f"export_backend={js.get('export_backend')!r}: the experimental HDR "
            "chain is a mode of the GPU-transfer backend and never falls back to "
            "the FFmpeg pipe, which carries 8-bit SDR pixels")
    if str(js.get("dlss_backend")) != "gpu_flow_d3d12":
        reasons.append("the HDR chain runs on dlss_backend=gpu_flow_d3d12, the "
                       "project GPU host")

    colour_format = getattr(live, "color_format", None)
    if colour_format != "rgba16f":
        reasons.append(
            f"the open Feature 18 session is {colour_format!r}, not 'rgba16f': the "
            "HDR chain hands the bridge's FP16 display proxy straight to inference, "
            "so an 8-bit session cannot be used for it and is refused rather than "
            "accepted as an approximation. Open the session with "
            "color_format='rgba16f'.")
    if not callable(getattr(live, "d3d12_device", None)):
        reasons.append("the Feature 18 session cannot report the D3D12 device the "
                       "HDR bridge is opened on")

    guidance_mode = int(js.get("guidance_mode", 0) or 0)
    import native_effects
    uses_flow = guidance_mode == 1 and not isinstance(live, native_effects.Passthrough)
    if guidance_mode not in (0, 1):
        reasons.append(f"guidance_mode={guidance_mode} is neither 0 (off) nor "
                       "1 (flow); this chain runs one or the other")
    if uses_flow:
        if str(js.get("flow_engine", "nvof")) not in ('nvof', 'fidelityfx', 'raft', 'raft_large'):
            reasons.append(f"flow_engine={js.get('flow_engine')} has no device path")
        if not callable(getattr(flow_provider, "get_flow_device_rgba", None)):
            reasons.append("the flow provider cannot be handed a CUDA surface, so "
                           "the motion-only guidance plane would have to be "
                           "downloaded to compute flow")
        elif not flow_provider.supports_device_rgba_flow():
            reasons.append("the NVOF bridge has no device RGBA input entry point")
    elif guidance_mode == 0:
        if not (callable(getattr(live,
                                 "process_device_color_zero_motion_resource", None))
                and bool(getattr(live, "supports_zero_motion", lambda: False)())):
            reasons.append("guidance_mode=0 needs a Feature 18 session that binds a "
                           "resident zero motion texture (dlss_gpu_host ABI 3) and "
                           "answers it for FP16 colour")

    if int(js.get("multi_pass", 1) or 1) != 1:
        reasons.append("multi_pass is not implemented by the production pipeline, "
                       "and the HDR chain does not add it")

    src_w, src_h = int(dims["src_w"]), int(dims["src_h"])
    import native_effects
    process_w, process_h = native_effects.processing_size(js, (src_w, src_h))
    live_w, live_h = int(dims["live_w"]), int(dims["live_h"])
    dst_w, dst_h = int(dims["dst_w"]), int(dims["dst_h"])
    scale = native_effects.core_working_scale(js)
    percent = gtx.working_percent(scale)
    if percent is None:
        reasons.append(
            f"working_scale={scale!r} is not a whole percentage between "
            f"{gtx.WORKING_PERCENT_MIN} and {gtx.WORKING_PERCENT_MAX}; the HDR chain "
            "resolves its working extent with the same resolve_working_dims rule the "
            "CPU plan and the SDR backend use, and adds no scale list of its own")
    elif (live_w, live_h) != pipeline.resolve_working_dims(
            process_w, process_h, percent / 100.0):
        reasons.append(
            f"{percent}% of {src_w}x{src_h} resolves to "
            f"{pipeline.resolve_working_dims(src_w, src_h, percent / 100.0)}, not "
            f"the {live_w}x{live_h} this job declared")
    elif live_w > process_w or live_h > process_h:
        reasons.append(
            f"working extent {live_w}x{live_h} exceeds the full extent "
            f"{src_w}x{src_h}; the HDR bridge downsamples and back up again")
    if (dst_w, dst_h) != native_effects.output_size(js, (src_w, src_h)):
        from export_resize import validate
        try:
            validate((dst_w, dst_h), ts.get('scale_algorithm', 'lanczos'))
        except ValueError as exc:
            reasons.append(str(exc))
    prores_source = (str(source.get("codec_name", "")).lower() == "prores" and
                     str(source.get("pix_fmt", "")).lower() == "yuv422p10le")
    # The requested output codec is independent of the input decoder codec.
    prores_output = str(ts.get("codec", "")).lower() in ("prores_ks", "prores")
    if src_w % 2 or src_h % 2:
        reasons.append("source dimensions must both be even (the working colour bridge requires even extents)")

    # The layer vector is validated by the same helper the SDR backend validates its
    # own with, so a duplicated percentage, an illegal weight or a tolerance the
    # solver could not stop on is refused here with the same words it is refused
    # there. The HDR chain adds no scale rule and drops none; a single-layer job
    # passes through the same call and gets the one layer, one scale and the
    # catmull baseline it already had, which is why nothing about a single job's
    # answer changed when this line did.
    multi = None
    try:
        multi = gtx.multi_scale_config(js, process_w, process_h)
    except (TypeError, ValueError) as error:
        reasons.append(f"reconstruction: {error}")
    if multi is not None and len(multi["scales"]) > 1:
        reasons += hdr_multi_layer_refusals(multi, caps, live, flow_provider,
                                            uses_flow, (live_w, live_h))

    if source.get("video_stream_count") != 1:
        reasons.append(f"{source.get('video_stream_count')} video streams: the "
                       "decoder bridge opens exactly one")
    if source.get("pix_fmt") not in HDR_SOURCE_PIX_FMTS and not prores_source:
        reasons.append(
            f"pix_fmt={source.get('pix_fmt')}: the HDR chain needs a supported 10-bit "
            "source (P010 4:2:0 or ProRes yuv422p10le 4:2:2)")
    transfer = str(source.get("color_transfer") or "")
    if sdr10:
        if prores_source:
            valid_prores_sdr = (source.get("color_range") in ("tv", "mpeg", "limited") and
                source.get("color_primaries") == "bt709" and
                source.get("color_transfer") == "bt709" and
                source.get("color_space") == "bt709")
            if not valid_prores_sdr:
                reasons.append("ProRes SDR10 requires explicit limited-range BT.709 range, primaries, transfer and matrix tags")
        elif not sdr10_source(source):
            reasons.append("SDR10 requires explicitly tagged limited-range BT.709 "
                           "10-bit 4:2:0 input; PQ/HLG or unspecified colour is refused")
    elif transfer in HLG_TRANSFERS:
        reasons.append(
            f"color_transfer={transfer}: HLG is a different reference and a "
            "different inverse transfer function; this chain carries ST 2084 (PQ) "
            "and refuses to guess a display size for HLG")
    elif transfer not in HDR_SOURCE_TRANSFER:
        reasons.append(
            f"color_transfer={transfer or 'unspecified'} is not smpte2084: the "
            "bridge's decode inverse transfer is PQ, and an SDR-tagged or "
            "linear source would be read as if it were not")
    range_name = str(source.get("color_range") or "")
    if not sdr10 and range_name not in HDR_SOURCE_RANGE:
        reasons.append(
            f"color_range={range_name or 'unspecified'} must explicitly declare "
            "limited or full range; the bridge normalizes the declared input "
            "range and writes limited-range HDR10 P010")
    primaries = str(source.get("color_primaries") or "")
    if not sdr10 and primaries not in HDR_SOURCE_PRIMARIES:
        reasons.append(
            f"color_primaries={primaries or 'unspecified'} is not bt2020: the "
            "bridge converts BT.2020 RGB and refuses another primaries set")
    space = str(source.get("color_space") or "")
    if not sdr10 and space not in HDR_SOURCE_SPACE:
        reasons.append(
            f"color_space={space or 'unspecified'} is not bt2020ncl: constant-"
            "luminance BT.2020 (CL) and an unspecified matrix are both refused by "
            "the native decode, which is the refusal this chain would forward")

    preset = str(ts.get("color_preset", "preserve"))
    if preset not in HDR_ALLOWED_COLOR_PRESETS:
        reasons.append(
            f"color_preset={preset}: this chain writes the source's own HDR10 "
            "declaration and no other. hdr_to_sdr and every tone-mapping preset are "
            "refused because the bridge holds a signed, unclamped working master "
            "precisely so that nothing is mapped twice")
    for key in ("custom_color_space", "custom_color_primaries",
                "custom_color_transfer", "custom_color_range"):
        value = str(ts.get(key, "") or "")
        if value and value not in ("auto", "preserve"):
            reasons.append(
                f"{key}={value}: a colour clause written by a muxer would be a "
                "second, contradicting declaration of what the encoder itself "
                "insists on")

    container = os.path.splitext(str(out_path).lower())[1]
    if prores_output:
        if container != ".mov":
            reasons.append(f"ProRes output container {container or '(none)'} must be .mov")
    elif container != HDR_CONTAINER:
        reasons.append(
            f"output container {container or '(none)'}: the HDR encode entry writes "
            f"one video stream into {HDR_CONTAINER}, which is also the intermediate "
            "the copy pass wraps, so a differently-named deliverable cannot be "
            "produced")
    # The rewrap is the project's existing FFmpeg map-and-copy pass, unchanged: the
    # encoder writes a video-only MKV, that file becomes input 0 of
    # ``gtx.build_mux_command``, and the source becomes input 1 for the audio,
    # subtitles, attachments, metadata, chapters and the video stream's own descriptive
    # fields the bridge cannot carry. ``-c:v copy`` is what makes this a rewrap rather
    # than a second encode, and it is the reason the encoded HDR declaration, its SEIs
    # and its bitstream survive it: this chain adds no muxer and no mapping rule of its
    # own.
    #
    # Decided from the probe before the pass is negotiated, so a source whose video track
    # is named, has a language or carries flags is rewrapped for it rather than having
    # the request dropped after the GPU work is already spent. A run reports afterwards
    # which fields it actually put on the deliverable: a source with no title, no
    # language and no flag set carries nothing, and that is reported as nothing rather
    # than as a promise kept.
    video_stream_metadata = descriptive_video_metadata_request(source, ts)
    copy_pass = gtx.needs_copy_pass(source, ts, with_audio, out_path,
                                    video_descriptive_metadata=video_stream_metadata)
    if copy_pass:
        reasons += hdr_mux_refusals(ts)

    fps = gtx.fps_rational(source) or gtx.fps_rational({}, dims.get("fps"))
    if not fps:
        reasons.append("frame rate is not a positive rational, so the encoder cannot "
                       "be given a frame rate at all")
    plan = plan_hdr_encoder(ts, cq, source, option_available, caps, tune_available)
    reasons += plan["unsupported"]
    reasons += list(extra_reasons or [])

    layers = len(multi["scales"]) if multi else 1
    policy = {"support_ceiling_scrgb": HDR_SUPPORT_CEILING_SCRGB,
              "clip_ceiling_scrgb": HDR_CLIP_CEILING_SCRGB,
              "paper_white_scrgb": paper_white,
              "pq_top_scrgb": HDR_CLIP_CEILING_SCRGB,
              "derived_legal_peak_scrgb": DERIVED_LEGAL_PEAK_SCRGB,
              "nits_per_scrgb": NITS_PER_SCRGB,
              "ceiling_source": "fixed per job: never derived from MaxCLL, never "
                                "adapted per frame",
              "single_layer": layers == 1, "layers": layers,
              "reconstruction_mode": (multi or {"mode": "single"})["mode"],
              "fusion_mode": (multi or {"fusion": "catmull_baseline"})["fusion"],
              "per_layer": "each layer owns its colour bridge, Feature 18 session, "
                           "guidance plane and observation; the pristine master, the "
                           "paper white and the proxy policy are the job's and are "
                           "identical for every layer",
              "multi_pass": 1, "output_resize": (dst_w, dst_h) != native_effects.output_size(js, (src_w, src_h)),
              "tonemap": False,
              "mux": "the existing gtx.build_mux_command copy pass, run only when the "
                     "request carries anything beyond video, with -c:v copy: never a "
                     "second encode and never a second muxer"}
    if sdr10:
        # Existing native structs retain their field names, but SDR values are
        # relative linear Rec.709, not physical scRGB luminance.
        policy.update(colour_mode=colour_mode, support_ceiling_scrgb=2.0,
                      clip_ceiling_scrgb=1.0, paper_white_scrgb=1.0,
                      pq_top_scrgb=None, derived_legal_peak_scrgb=None,
                      nits_per_scrgb=None, linear_units="relative Rec.709, white=1")
        policy.update(nr_input_domain='bt709_rgb_code_fp16',
            nr_reconstruction='exact_bt709_eotf_delta',
            preserve_source_detail=(layers != 1 or bool(js.get('residual_pyramid', True))))
        if source.get('colour_resolution'):
            policy['source_colour_resolution'] = source['colour_resolution']
    if prores_output:
        policy.update(encoder="PyAV/libavcodec ProRes 422 HQ",
                      static_metadata="not authored by the ProRes software encoder")
    if prores_source:
        plan["_source"] = source
    contract_struct = plan["struct"]
    return {"supported": not reasons, "reasons": reasons, "plan": plan,
            "fps": fps, "guidance_mode": guidance_mode, "uses_flow": uses_flow,
            "downscaled": (live_w, live_h) != (src_w, src_h),
            "plan_driven": (live_w, live_h) != (src_w, src_h),
            "working_scale": scale, "copy_pass": copy_pass,
            "multi_scale": multi, "layers": layers,
            "hdr_video_stream_metadata": video_stream_metadata,
            "hdr_policy": policy,
            "hdr_encoder_contract_declared": {
                "codec": plan["codec"], "profile": plan["profile"],
                "sw_pixel_format": contract_struct.get("sw_pixel_format"),
                "color_range": contract_struct.get("color_range"),
                "color_primaries": contract_struct.get("color_primaries"),
                "color_trc": contract_struct.get("color_trc"),
                "color_space": contract_struct.get("color_space")},
            "hdr_caps": {key: caps.get(key) for key in
                         ("nvdec", "nvenc", "defaults_rc", "encoder_defaults",
                          "sei_capability", "capability", "colour_bridge",
                          "fusion_bridge", "struct_sizes_accepted", "p010_available", "encoder_codec")
                         if key in caps}}


def negotiate_sdr10(*args, **kwargs):
    """Reuse lifecycle/fusion/remux admission with the genuine SDR colour domain."""
    return negotiate_hdr(*args, **kwargs, colour_mode="sdr10_bt709")


# --- the run --------------------------------------------------------------------

class HdrChain:
    """The HDR bridges for one export, opened on the first borrowed frame.

    Nothing can be opened before the decoder reports its CUDA context, extents and
    colour fields, and the Feature 18 session's D3D12 device is what the HDR bridge
    is opened on, so the first frame decides the shape of every frame after it. Each
    stage hands back borrowed device memory that the next call into the same bridge
    invalidates, which is why ``process`` is one straight line that cannot be
    reordered, overlapped or split across threads.
    """

    def __init__(self, bridges, hdr_caps, video, encoder_output, live, plan, fps,
                 policy, dims, flow_provider=None, residual_strength=1.0,
                 uses_flow=True, preview_consumer=None):
        self.b = bridges
        self.caps = hdr_caps
        self.video = video
        self.source = {}
        self.encoder_output = encoder_output
        self.preview_consumer = preview_consumer
        self.live = live
        self.plan = plan
        self.fps = fps
        self.policy = dict(policy or {})
        self.sdr10 = self.policy.get("colour_mode") == "sdr10_bt709"
        from prores_decode import is_prores
        self.source = dict(plan.get("_source", (hdr_caps or {}).get("_prores_source", {})) or {})
        self.prores_input = is_prores(self.source)
        self.prores_output_enabled = str(plan.get("codec", "")).lower() == "prores_ks"
        self.prores_decoder = None
        self.prores_output = None
        if self.prores_output_enabled and preview_consumer is None:
            from prores_export import ProResOutput
            self.prores_output = ProResOutput(self)
        self.bridge_type = hdr.GpuSdr10Bridge if self.sdr10 else hdr.GpuHdrBridge
        self.flow_provider = flow_provider
        self.uses_flow = bool(uses_flow)
        self.strength = float(residual_strength)
        self.src = (int(dims["src_w"]), int(dims["src_h"]))
        self.working = (int(dims["live_w"]), int(dims["live_h"]))
        self.decoder = None
        self.bridge = None
        self.encoder = None
        self.cuda = None
        self.context = None
        self.pushed = False
        self.first = True
        self.flow_calls = 0
        self.zero_motion_calls = 0
        self.source_metadata = None
        self.metadata_snapshots = []
        self.metadata_divergence = None
        self.frame_reports = []
        self.contract = None
        self.sei_status = None
        self.encoder_counters = None
        self.bridge_counters = None
        from post_control import GpuPostControl
        self.post_control = GpuPostControl(hdr=True, sdr10=self.sdr10)
        from temporal_control import GpuTemporalControl
        self.temporal_control = GpuTemporalControl(hdr=True)
        self._frame_reset = True
        self.input_effect = self.output_effect = None
        self.native_output_exchange = None
        self.processing_size = self.src
        self.output_size = (int(dims['dst_w']), int(dims['dst_h']))
        self.decode_bridge = self.encode_bridge = self.encode_exchange = None
        self.framegen = self.framegen_bridge = self.framegen_exchange = None

    def _consume_preview_output(self, frame, composited):
        """Synchronously pass the borrowed final FP16 plane to preview, if set."""
        if self.preview_consumer is None:
            return None
        descriptor = composited.as_descriptor()
        descriptor.update({
            "format": composited.format,
            "pixel_format": composited.pixel_format,
            "width": composited.width, "height": composited.height,
            "cuda_context": self.context,
            "d3d12_device": self.live.d3d12_device(),
            "dxgi_format": 10,
            "state": 0x800,
            "producer_completed": True,
        })
        if getattr(self,'truehdr_tail',None):
            if getattr(self, 'output_effect', None) and self.output_effect.output_hdr:
                descriptor['output_hdr'] = True
            descriptor = self.truehdr_tail.process(descriptor,frame)
        return self.preview_consumer(descriptor, frame)

    # -- lifetime -------------------------------------------------------------

    def open_decoder(self):
        if self.prores_input:
            from prores_decode import ProResDecoder
            self.prores_decoder = ProResDecoder(self.video, self.source)
            self.decoder = self.prores_decoder
            return
        handle = C.c_void_p()
        opener = ("nvdec_bridge_open_p010" if self.sdr10 else "nvdec_bridge_open_hdr")
        code = int(getattr(self.b["nvdec"], opener)(os.fsencode(self.video),
                                                  C.byref(handle)))
        if code:
            gtx._bridge_error(self.b["nvdec"], "nvdec_bridge_last_error", code)
        self.decoder = handle

    def next_frame(self, frame):
        if self.prores_input:
            return int(self.prores_decoder.next(frame))
        return int(self.b["nvdec"].nvdec_bridge_next(self.decoder, C.byref(frame)))

    def seek_decoder(self, relative_us):
        if self.prores_input:
            self.prores_decoder.seek_us(relative_us)
            return 0
        lib = self.b["nvdec"]
        lib.nvdec_bridge_seek_us.argtypes = [C.c_void_p, C.c_int64]
        lib.nvdec_bridge_seek_us.restype = C.c_int32
        return int(lib.nvdec_bridge_seek_us(self.decoder, int(relative_us)))

    def cancel_decoder(self):
        if self.prores_input:
            if self.prores_decoder:
                self.prores_decoder.cancel()
        elif self.decoder:
            self.b["nvdec"].nvdec_bridge_cancel(self.decoder)

    def close(self):
        """Reverse owner order: encoder, colour bridge, Feature 18, context, decoder.

        The HDR bridge imports the Feature 18 output texture, so it goes before the
        session that produced it; the decoder's CUcontext outlives every stage that
        borrowed memory from it and is popped only once they are gone; and the
        decoder is released last, because the bridge and the encoder were opened on
        its context. Every close here is idempotent or guarded, so a failed run
        tearing down after a partially-opened chain is the same path.
        """
        from gpu_cleanup import release_owned, finish_cleanup
        errors = []
        release_owned(self, 'post_control', errors)
        release_owned(self, 'temporal_control', errors)
        for name in ('truehdr_tail','prores_output','export_resize', 'export_resize_bridge', 'framegen_tail', 'ordered_framegen', 'framegen_input', 'output_effect', 'native_output_exchange', 'input_effect',
                     'framegen', 'framegen_exchange', 'framegen_bridge',
                     'encode_exchange', 'encode_bridge', 'decode_bridge'):
            release_owned(self, name, errors)
        release_owned(self, 'encoder', errors, lambda handle: self.b['nvenc'].nvenc_close(handle))
        release_owned(self, 'bridge', errors)
        if getattr(self.flow_provider, 'borrows_decoder_context', False):
            release_owned(self, 'flow_provider', errors, method='close_independent_layer')
        release_owned(self, 'live', errors)
        if self.pushed:
            try:
                self.cuda.pop()
            except Exception as error:
                errors.append(('decoder_context_pop', error))
            finally:
                self.pushed = False
        if self.prores_input:
            release_owned(self, 'prores_decoder', errors)
            self.decoder = None
        else:
            release_owned(self, 'decoder', errors, lambda handle: self.b['nvdec'].nvdec_bridge_close(handle))
        finish_cleanup(errors)

    def _fail(self, symbol, code):
        gtx._bridge_error(self.b["nvenc"], symbol, code)

    def _read_metadata(self):
        """The mastering and content-light the decoder parsed from the frame it is on.

        Only valid on the thread that called ``nvdec_bridge_next`` and while that
        frame is current, which is why every read happens at the top of ``process``
        rather than from another thread or after the decoder has advanced. The struct
        itself comes back, because the encoder open takes that exact shape: the
        source's rationals reach the SEI without passing through a float, and the
        snapshot a caller reads is the same struct's fields.
        """
        if self.prores_input:
            return self.prores_decoder.metadata()
        metadata = HdrMetadata(C.sizeof(HdrMetadata))
        code = int(self.b["nvdec"].nvdec_bridge_get_hdr_metadata(self.decoder,
                                                                 C.byref(metadata)))
        if code:
            gtx._bridge_error(self.b["nvdec"], "nvdec_bridge_last_error", code)
        return metadata

    @staticmethod
    def _half_present(snapshot):
        """True when the source declared a mastering display whose halves disagree.

        The writer refuses partial ST 2086 metadata, so the chain says so before it
        has an encoder rather than after: the source's own claim is carried whole or
        it is not carried at all, and nothing here invents the missing half.
        """
        return bool(snapshot["has_mastering"]) and bool(
            snapshot["mastering_has_primaries"]) != bool(
            snapshot["mastering_has_luminance"])

    # -- the frame ------------------------------------------------------------

    def _open_stages(self, frame):
        """The context, the metadata, the colour bridge and the encoder, in that order.

        All four need the first frame's own facts: its context, its extent, its time
        base and the metadata the decoder parsed from it. Nothing here is guessed
        from the file name or from what the container's stream header said - the
        colour fields the bridge is handed are the ones the decoder reported.
        """
        expected_format = 3 if self.prores_input else NVDEC_SOFTWARE_FORMAT_P010
        if int(frame.software_format) != expected_format:
            raise HdrExportError(
                f"the decoder reported software_format {frame.software_format}, not "
                f"{expected_format} ({'P210' if self.prores_input else 'P010'}): this chain "
                "refuses a surface with the wrong bit depth or chroma layout")
        if (int(frame.width), int(frame.height)) != self.src:
            raise HdrExportError(
                f"the decoder yielded {frame.width}x{frame.height} where this export "
                f"negotiated {self.src[0]}x{self.src[1]}")
        context = int(frame.cuda_context or 0)
        if not context:
            raise HdrExportError("the decoder reported no CUDA context to borrow")

        self.cuda = _Cuda()
        self.cuda.init()
        self.cuda.push(context)
        self.pushed = True
        self.context = context

        metadata = self._read_metadata()
        snapshot = _fields(metadata)
        self.metadata_snapshots.append(snapshot)
        self.source_metadata = snapshot
        if self.sdr10 and (snapshot["has_mastering"] or snapshot["has_content_light"]):
            raise HdrExportError("SDR10 source unexpectedly carries HDR metadata; refused")
        if self._half_present(snapshot) and not getattr(self,'truehdr_tail',None):
            raise HdrExportError(
                "the source declares a mastering display whose primaries and "
                f"luminance are half present (primaries={snapshot['mastering_has_primaries']}, "
                f"luminance={snapshot['mastering_has_luminance']}); the HDR10 writer "
                "refuses partial ST 2086 metadata and this chain does not invent the "
                "missing half")

        self.bridge = self.bridge_type(
            d3d12_device=self.live.d3d12_device(), cuda_context=context,
            full_width=self.processing_size[0], full_height=self.processing_size[1],
            working_width=self.working[0], working_height=self.working[1],
            paper_white_scrgb=float(self.policy["paper_white_scrgb"]))
        self.contract = self.bridge.contract
        if self.prores_input or self.processing_size != self.src or (self.preview_consumer is not None and self.input_effect is not None):
            self.decode_bridge = self.bridge_type(d3d12_device=self.live.d3d12_device(),
                cuda_context=context, full_width=self.src[0], full_height=self.src[1],
                working_width=self.src[0], working_height=self.src[1],
                paper_white_scrgb=float(self.policy['paper_white_scrgb']),
                purpose='decode422' if self.prores_input else 'decode')

        # Preview owns display/readback. Preserve the decoder, context and colour
        # bridge setup, but do not open an output encoder for a preview-only chain.
        if self.preview_consumer is not None or getattr(self,'truehdr_tail',None):
            return
        if self.prores_output is not None:
            return

        options = HdrOptions()
        options.struct_size = C.sizeof(HdrOptions)
        struct = self.plan["struct"]
        options.codec = struct["codec"]
        options.profile = struct["profile"]
        options.preset = struct["preset"]
        options.tune = struct["tune"]
        options.multipass = struct["multipass"]
        options.rate_control = struct["rate_control"]
        options.constant_quality = int(struct["constant_quality"])
        options.bitrate = int(struct["bitrate"])
        options.gop = int(struct["gop"])
        options.extra_options = gtx.extra_options_blob(self.plan["av_options"]) or None
        options.sw_pixel_format = int(struct["sw_pixel_format"])
        options.color_range = int(struct["color_range"])
        options.color_primaries = int(struct["color_primaries"])
        options.color_trc = int(struct["color_trc"])
        options.color_space = int(struct["color_space"])
        # Source absence stays absence: metadata the source never declared is passed
        # as NULL, which the entry reads as "do not attach anything".
        attach = None if not (snapshot["has_mastering"] or
                              snapshot["has_content_light"]) else metadata
        encoder = C.c_void_p()
        from vfx_framegen import encoder_timing
        rate, tb = encoder_timing(getattr(self, 'framegen_settings', {}), self.fps,
            (int(frame.time_base_num), int(frame.time_base_den)))
        code = int(open_p010_encoder(self.b["nvenc"], self.sdr10,
            C.c_void_p(context), self.output_size[0], self.output_size[1], *rate,
            C.c_int32(tb[0]), C.c_int32(tb[1]),
            os.fsencode(self.encoder_output), C.byref(options), attach,
            C.byref(encoder)))
        if code:
            self._fail("nvenc_last_error", code)
        self.encoder = encoder

    def _assert_metadata_stable(self):
        """Every later frame's own claim, compared against the frame that opened.

        The encoder was opened with the mastering and content-light numbers of the
        first frame, and its writer attaches those to every packet. A source whose
        metadata moves mid-job would therefore be described by the wrong numbers
        from that frame on, which is a claim this chain will not make silently: the
        divergence is recorded and the run refuses.
        """
        snapshot = _fields(self._read_metadata())
        self.metadata_snapshots.append(snapshot)
        if snapshot == self.source_metadata:
            return
        changed = sorted(name for name in snapshot
                         if snapshot[name] != self.source_metadata[name])
        self.metadata_divergence = {"frame_idx": len(self.metadata_snapshots) - 1,
                                    "changed_fields": changed,
                                    "first_frame": self.source_metadata,
                                    "this_frame": snapshot}
        raise HdrExportError(
            f"the source's HDR metadata changed at frame "
            f"{len(self.metadata_snapshots) - 1} ({', '.join(changed)}): this chain "
            "opens the encoder on the first frame's mastering display and "
            "content-light numbers and carries those to every packet, so dynamic "
            "or per-frame-varying metadata is refused rather than reported as "
            "preserved")

    def _primary_colour(self, frame, colour, stages, reset=False):
        """The decode, the working master and the proxy, timed as the chain reports them.

        These three calls are the whole of the frame's colour work before any
        inference, and every layer shape this file runs starts with them: one layer
        reconstructs from what they produce, several import the pristine master they
        produced into their own bridges. Sharing the sequence is what keeps the
        milliseconds, the ``source_report`` read and the order in which the borrowed
        planes succeed one another identical between the two, rather than a second
        description of the same three calls that could drift from the first.
        """
        mark = time.perf_counter()
        decoder_bridge = getattr(self, 'decode_bridge', None) or self.bridge
        supplied = getattr(self, '_ordered_color', None)
        if supplied is None:
            master = decoder_bridge.decode(
                p010_y_ptr=frame.y_device_ptr, p010_y_pitch=frame.y_pitch_bytes,
                p010_uv_ptr=frame.uv_device_ptr, p010_uv_pitch=frame.uv_pitch_bytes,
                support_ceiling_scrgb=float(self.policy["support_ceiling_scrgb"]),
                **colour)
            source_report = decoder_bridge.source_report()
            if self.prores_input and self.input_effect is None:
                master = self.bridge.import_master(master.as_descriptor())
        else:
            from hdr_gpu_engine import DevicePlane
            master = DevicePlane(fmt='rgba16f_cuda', pixel_format='linear GPU VFG output',
                bytes_per_pixel=8, **{k: supplied[k] for k in
                    ('width', 'height', 'cuda_context', 'device_ptr', 'pitch_bytes')})
            source_report = self.ordered_framegen.source_report
            if self.input_effect is None:
                master = self.bridge.import_master(supplied)
        self._preview_source = master.as_descriptor()
        stages["hdr_decode"] = (time.perf_counter() - mark) * 1000.0

        if self.input_effect is not None:
            self._prefix_discontinuous = bool(self._detect_scene(master, reset, False) or reset)
            mark = time.perf_counter()
            if getattr(self,'_preview_reuse_prefix',False):
                color = self._preview_prefix_color
            else:
                color = self.input_effect.color(master.as_descriptor(),frame,self._frame_reset)
                if self.preview_consumer is not None: self._preview_prefix_color = dict(color)
            master = self.bridge.import_master(color)
            stages['native_input_effect'] = (time.perf_counter() - mark) * 1000
            stages['native_input_reused'] = bool(getattr(self,'_preview_reuse_prefix',False))

        mark = time.perf_counter()
        working_master = self.bridge.downscale()
        stages["hdr_downscale"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        proxy, guidance = self.bridge.proxy()
        proxy_report = self.bridge.proxy_report()
        stages["hdr_proxy"] = (time.perf_counter() - mark) * 1000.0
        return (master, working_master, proxy, guidance, source_report,
                proxy_report)

    def _guidance_and_inference(self, live, flow_provider, proxy, guidance, frame,
                                index, reset, is_discontinuous, timings, counted=None):
        """Motion, then Feature 18, on the session and plane this layer was given.

        The motion source is the one the negotiation agreed to, not whichever object
        happens to be in the argument: a guided job whose NVOF call raises is a failed
        job, and silently drawing the next frame from resident zero motion would
        report an unguided result under a guided job's settings. ``counted`` is this
        layer's own motion tally - the dict with a ``flow`` and a ``zero_motion`` slot
        that says which of the two this call is about to make - and is ``None`` only
        for the single-layer chain, which has one layer and needs no breakdown of it.
        The whole-job counts are bumped for every layer, so on a multi-scale run
        ``flow_calls`` is the number of NVOF calls this chain actually made rather
        than the number of frames it decoded.
        """
        descriptor = proxy.as_descriptor()
        mark = time.perf_counter()
        flow = None
        if self.uses_flow:
            flow = flow_provider.get_flow_device_rgba(
                index, guidance.as_descriptor(), frame.cuda_context,
                is_discontinuous=is_discontinuous)
            flow_ms = (time.perf_counter() - mark) * 1000.0
            self.flow_calls += 1
            if counted is not None:
                counted["flow"] += 1
            dlss = live.process_device_color_flow_resource(descriptor, flow,
                                                           reset=reset)
        else:
            flow_ms = None
            if not getattr(live, 'native_passthrough', False):
                self.zero_motion_calls += 1
            if counted is not None:
                counted["zero_motion"] += 1
            dlss = live.process_device_color_zero_motion_resource(descriptor,
                                                                 reset=reset)
        inference_ms = (time.perf_counter() - mark) * 1000.0 - (flow_ms or 0.0)
        timings["nvof_flow"] = flow_ms
        timings["dlss_inference"] = inference_ms
        if counted is not None:
            timings["layer_nvof_flow_ms"] = flow_ms
            timings["layer_dlss_inference_ms"] = inference_ms
        return dlss, flow

    def _encode(self, frame, stages, composited=None):
        """Composited scRGB to P010 to the encoder - the shared tail of every shape.

        The encode stage is the only thing in this chain that clamps, and the encoder
        consumes the borrowed planes inside ``nvenc_encode_p010``, which copies and
        synchronises before it returns. Both halves are the same two calls in the same
        order for a one-layer and a many-layer frame, so a fused composite and a
        baseline composite are encoded by identical code.
        """
        if self.preview_consumer is not None:
            return None, None
        mark = time.perf_counter()
        encoder_bridge = self.encode_bridge or self.bridge
        if self.preview_consumer is None and not (getattr(self, 'framegen_settings', None) and not getattr(self, 'ordered_framegen', None)):
            encoder_bridge = self._resize_output(composited, stages) or encoder_bridge
        framegen_active = bool(getattr(self, 'framegen_settings', None) and not getattr(self, 'ordered_framegen', None))
        if self.preview_consumer is None and framegen_active:
            mark = time.perf_counter()
            if self.framegen is None:
                from vfx_framegen import Interpolator
                self.framegen = Interpolator(self.context, self.framegen_stage_size, self.framegen_settings,
                    self.fps, wide=True, hdr=not self.sdr10 or getattr(self, 'framegen_hdr', False), cancel=getattr(self, 'cancel', None))
            from native_effects import frame_at_size
            self._framegen_frame = frame_at_size(frame, (frame.width, frame.height))
            color = composited.as_descriptor()
            if getattr(self, 'framegen_hdr', False): color['output_hdr'] = True
            self.framegen.submit(color, frame, self._frame_reset, self._emit_generated)
            stages['vfg_and_encoder_submit'] = (time.perf_counter()-mark)*1000
            return None, None
        if getattr(self, 'truehdr_tail', None):
            color = composited.as_descriptor()
            if getattr(self, 'output_effect', None) and self.output_effect.output_hdr:
                color['output_hdr'] = True
            return self.truehdr_tail.encode(self, color, frame)
        if self.prores_output is not None:
            color = (getattr(self, '_resized_color', None)
                     if encoder_bridge is getattr(self, 'export_resize_bridge', None)
                     else composited.as_descriptor())
            _, encode_report = self.prores_output.encode(color, frame, int(frame.pts))
            stages['prores_submit'] = (time.perf_counter() - mark) * 1000.0
            return None, encode_report
        planes = encoder_bridge.encode(clip_ceiling_scrgb=float(self.policy["clip_ceiling_scrgb"]))
        encode_report = encoder_bridge.encode_report()
        stages["hdr_encode"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        code = int(self.b["nvenc"].nvenc_encode_p010(
            self.encoder, planes.y.device_ptr, planes.y.pitch_bytes,
            planes.uv.device_ptr, planes.uv.pitch_bytes, int(frame.pts)))
        if code:
            self._fail("nvenc_last_error", code)
        stages["nvenc_submit"] = (time.perf_counter() - mark) * 1000.0
        return planes, encode_report

    def _emit_generated(self, color, pts, kind):
        if self.framegen_bridge is None:
            from native_effects import Exchange
            render_size = getattr(self, 'render_output_size', self.output_size)
            self.framegen_exchange = Exchange(self.live.d3d12_device(), self.context, render_size, wide=True)
            self.framegen_bridge = self.bridge_type(d3d12_device=self.live.d3d12_device(), cuda_context=self.context,
                full_width=render_size[0], full_height=render_size[1],
                working_width=render_size[0], working_height=render_size[1],
                paper_white_scrgb=float(self.policy['paper_white_scrgb']))
        tail = getattr(self, 'framegen_tail', None)
        if tail:
            if getattr(self, 'framegen_input', None) is None:
                from native_effects import Exchange
                self.framegen_input = Exchange(self.live.d3d12_device(), self.context, self.framegen_stage_size, wide=True)
            texture = self.framegen_input.texture(color['device_ptr'], color['pitch_bytes'])
            if color.get('output_hdr'): texture['output_hdr'] = True
            texture = tail.process(texture, self._framegen_frame,
                self.framegen.emitted == 0 or (kind == 'original' and self._frame_reset))
            color = self.framegen_exchange.color(texture)
        else:
            texture = self.framegen_exchange.texture(color['device_ptr'], color['pitch_bytes'])
            if color.get('output_hdr'): texture['output_hdr'] = True
        if getattr(self, 'truehdr_tail', None):
            self.truehdr_tail.encode(self, texture, self._framegen_frame, pts)
            return
        if self.prores_output is not None:
            resized = self._resize_output(color, {})
            self.prores_output.encode(self._resized_color if resized is not None else color, self._framegen_frame, pts)
            return
        self.framegen_bridge.import_master(color)
        self.framegen_bridge.set_composite(texture)
        encoder_bridge = self._resize_output(color, {}) or self.framegen_bridge
        planes = encoder_bridge.encode(clip_ceiling_scrgb=float(self.policy['clip_ceiling_scrgb']))
        code = int(self.b['nvenc'].nvenc_encode_p010(self.encoder, planes.y.device_ptr, planes.y.pitch_bytes,
            planes.uv.device_ptr, planes.uv.pitch_bytes, pts))
        if code:
            self._fail('nvenc_last_error', code)

    def _resize_output(self, color, stages):
        if color is None:
            return None
        color = color.as_descriptor() if hasattr(color, 'as_descriptor') else color
        size = (color['width'], color['height'])
        if size == self.output_size:
            return None
        if getattr(self, 'export_resize', None) is None:
            from export_resize import Resize
            self.export_resize = Resize(self.live.d3d12_device(), self.context, size,
                self.output_size, self.export_scale_algorithm, wide=True)
            self.export_resize_bridge = self.bridge_type(d3d12_device=self.live.d3d12_device(), cuda_context=self.context,
                full_width=self.output_size[0], full_height=self.output_size[1],
                working_width=self.output_size[0], working_height=self.output_size[1],
                paper_white_scrgb=float(self.policy['paper_white_scrgb']))
        mark = time.perf_counter()
        resized = self.export_resize.color(color)
        self._resized_color = resized.as_descriptor()
        self.export_resize_bridge.import_master(resized)
        self.export_resize_bridge.set_composite(self.export_resize.as_texture(resized))
        stages['export_resize'] = (time.perf_counter()-mark)*1000
        return self.export_resize_bridge

    def process(self, frame, flow_provider, index, is_discontinuous):
        """One frame, device to device, in the order the borrows require.

        The encoder consumes the borrowed P010 planes inside ``nvenc_encode_p010``,
        which copies and synchronises before it returns, so the next
        ``nvdec_bridge_next`` - which recycles the surface this call read - is only
        ever reached after the frame has been fully used. That is the whole reason the
        encode is the last stage of the frame rather than a queued step.
        """
        is_first = self.first
        if is_first:
            self._open_stages(frame)
            self.first = False
        else:
            # The decoder is still on this frame and this is still the thread that
            # decoded it, which is the only moment its metadata can be read.
            self._assert_metadata_stable()

        colour = {name: int(getattr(frame, name)) for name in
                  ("color_range", "color_primaries", "color_transfer",
                   "color_space")}
        if self.sdr10:
            from source_colour import resolve_frame_colour
            colour = resolve_frame_colour(colour, self.policy.get('source_colour_resolution'))
        stages = {}
        (master, working_master, proxy, guidance, source_report,
         proxy_report) = self._primary_colour(frame, colour, stages, is_first or is_discontinuous)

        is_discontinuous = (self._prefix_discontinuous if self.input_effect is not None
                            else self._detect_scene(master, is_first, is_discontinuous))
        reset = is_first or is_discontinuous
        primary_dlaa = getattr(self.live, 'native_dlaa', False)
        dlss, flow = self._guidance_and_inference(
            self.live, flow_provider, working_master if primary_dlaa else proxy,
            guidance, frame, index, reset,
            is_discontinuous, stages)

        mark = time.perf_counter()
        if primary_dlaa:
            # Equal-extent DLAA operates on the linear picture directly. The
            # bounded proxy above is only used for optical-flow guidance.
            composited = self.bridge.set_composite(dlss)
            reconstruct_report = None
        else:
            composited = self.bridge.reconstruct(dlss, residual_strength=self.strength,
                **({'preserve_source_detail': self.policy.get('preserve_source_detail', True)} if self.sdr10 else {}))
            reconstruct_report = self.bridge.reconstruct_report()
        stages["hdr_reconstruct"] = (time.perf_counter() - mark) * 1000.0

        composited = self._post_process(master, composited, stages, frame)
        planes, encode_report = self._encode(frame, stages, composited)
        preview_output = self._consume_preview_output(frame, composited)

        self.frame_reports.append({
            "frame_idx": index, "pts": int(frame.pts),
            "time_base": [int(frame.time_base_num), int(frame.time_base_den)],
            "colour_fields_sent": colour, "reset": bool(reset),
            "motion_source": (self.flow_provider.timing().get('guidance_backend', 'nvof_device_rgba')
                              if self.uses_flow else "zero_motion"),
            "source_report": source_report, "proxy_report": proxy_report,
            "reconstruct_report": reconstruct_report,
            "encode_report": encode_report,
            "planes": {"master_full": master.as_dict(),
                       "working_master": working_master.as_dict(),
                       "proxy": proxy.as_dict(), "guidance": guidance.as_dict(),
                       "composited": composited.as_dict(),
                       "encoded_y": planes.y.as_dict() if planes else None,
                       "encoded_uv": planes.uv.as_dict() if planes else None},
            # The Feature 18 output as the bridge saw it: which DXGI format it
            # validated and which extent it checked it against.
            "feature18_output": {"width": dlss.get("width"),
                                 "height": dlss.get("height"),
                                 "dxgi_format": dlss.get("dxgi_format")},
        })
        outcome = {"stages": stages, "flow": flow,
                "nvof_timing": (flow_provider.timing()
                                if self.uses_flow and flow_provider is not None
                                and hasattr(flow_provider, "timing") else None)}
        if self.preview_consumer is not None:
            outcome["preview_output"] = preview_output
            outcome['_preview_master'] = self._preview_source
        return outcome

    def _post_process(self, master, composited, stages, frame=None):
        capture = getattr(self, '_preview_post_capture', None)
        if self.post_control.controls.neutral and self.temporal_control.controls.mode == "off":
            if capture:
                capture(device=self.live.d3d12_device(), context=self.context,
                    width=master.width, height=master.height, wide=True,
                    source_ptr=master.device_ptr, source_pitch=master.pitch_bytes,
                    candidate=composited.as_descriptor(), frame=frame)
            return self._native_post_process(composited, stages, frame)
        mark = time.perf_counter()
        stable = self.temporal_control.process(device=self.live.d3d12_device(),
            cuda_context=self.context, source_ptr=master.device_ptr,
            source_pitch=master.pitch_bytes, width=self.processing_size[0], height=self.processing_size[1],
            candidate={"device_ptr": composited.device_ptr,
                       "pitch_bytes": composited.pitch_bytes}, reset=self._frame_reset)
        if capture:
            capture(device=self.live.d3d12_device(), context=self.context,
                width=self.processing_size[0], height=self.processing_size[1], wide=True,
                source_ptr=master.device_ptr, source_pitch=master.pitch_bytes, candidate=stable, frame=frame)
        controlled = self.post_control.process(device=self.live.d3d12_device(),
            cuda_context=self.context, source_ptr=master.device_ptr,
            source_pitch=master.pitch_bytes, width=self.processing_size[0], height=self.processing_size[1],
            candidate=stable)
        output = self.bridge.set_composite(controlled)
        stages["post_control"] = (time.perf_counter() - mark) * 1000.0
        return self._native_post_process(output, stages, frame)

    def _native_post_process(self, composited, stages, frame):
        if getattr(self, 'output_effect', None) is None:
            return composited
        mark = time.perf_counter()
        if self.native_output_exchange is None:
            from native_effects import Exchange
            self.native_output_exchange = Exchange(self.live.d3d12_device(), self.context, self.processing_size, wide=True)
        texture = self.native_output_exchange.texture(composited.device_ptr, composited.pitch_bytes)
        result = self.output_effect.process(texture, frame, self._frame_reset)
        target_size = self.output_effect.target
        if target_size != self.processing_size:
            from native_effects import Exchange
            if self.encode_bridge is None:
                self.encode_bridge = self.bridge_type(d3d12_device=self.live.d3d12_device(),
                    cuda_context=self.context, full_width=target_size[0], full_height=target_size[1],
                    working_width=target_size[0], working_height=target_size[1],
                    paper_white_scrgb=float(self.policy['paper_white_scrgb']))
                self.encode_exchange = Exchange(self.live.d3d12_device(), self.context, target_size, wide=True)
            self.encode_bridge.import_master(self.encode_exchange.color(result))
            output = self.encode_bridge.set_composite(result)
        else:
            output = self.bridge.set_composite(result)
        stages['native_output_effect'] = (time.perf_counter()-mark)*1000
        return output

    def _detect_scene(self, master, is_first, is_discontinuous):
        scene = self.temporal_control.observe(device=self.live.d3d12_device(),
            cuda_context=self.context, source_ptr=master.device_ptr,
            source_pitch=master.pitch_bytes, width=master.width, height=master.height,
            reset=is_first or is_discontinuous) if self.temporal_control.controls.enabled else {"scene_cut": False}
        discontinuous = bool(is_discontinuous or scene["scene_cut"])
        self._frame_reset = bool(is_first or discontinuous)
        return discontinuous

    # -- what the bridges say about the run -----------------------------------

    def counters(self):
        """Every transfer counter this chain can read, or nothing for what it cannot.

        Read after ``finish`` and before the bridges go away. A stage that never
        opened contributes no key rather than a zero, because a zero here would be a
        claim that a boundary was crossed zero times and only a bridge may make it.
        """
        out = {}
        if getattr(self,'truehdr_tail',None): out['truehdr'] = self.truehdr_tail.report()
        if self.prores_output is not None: out['prores_encoder'] = self.prores_output.report()
        if self.prores_decoder is not None: out['prores_decoder'] = self.prores_decoder.counters()
        if getattr(self, 'framegen', None) is not None:
            out['frame_generation'] = self.framegen.report()
            if hasattr(self, 'framegen_stage_index'):
                from ordered_vfg import stage_report
                out['frame_generation']['pipeline'] = stage_report(self)
        if getattr(self, 'export_resize', None):
            out['export_resize'] = self.export_resize.report()
        for name in ('decode_bridge', 'encode_bridge', 'framegen_bridge', 'export_resize_bridge'):
            bridge = getattr(self, name, None)
            if bridge is not None:
                out[name] = bridge.transfer_counters()
        for name in ('input_effect', 'output_effect'):
            effect = getattr(self, name, None)
            if effect is not None:
                out['native_' + name] = effect.report()
        if self.decoder and not self.prores_input:
            block = gtx._NvdecCounters()
            block.struct_size = C.sizeof(gtx._NvdecCounters)
            if not self.b["nvdec"].nvdec_bridge_get_transfer_counters(
                    self.decoder, C.byref(block)):
                out["decoder"] = {name: getattr(block, name) for name, _
                                  in gtx._NvdecCounters._fields_[1:]}
        if self.bridge is not None and self.bridge.is_alive():
            out["hdr_bridge"] = self.bridge.transfer_counters()
            self.bridge_counters = out["hdr_bridge"]
            # The primary bridge's own dispatch count is what lets the scalar
            # counter-readback boundary be stated as a range instead of as nothing:
            # every completed pass copied 256 bytes of counters into a readback heap,
            # including this bridge's decode. Asked only when the group was verified
            # present, because on a build without it the honest answer is that no
            # counter exists to ask.
            group = ((self.caps.get("colour_bridge") or {}).get("multiscale") or {})
            if group.get("supported"):
                out["hdr_primary_multiscale"] = self.bridge.multiscale_counters()
        if self.encoder:
            block = HdrCounters(C.sizeof(HdrCounters))
            if not self.b["nvenc"].nvenc_hdr_get_counters(self.encoder,
                                                          C.byref(block)):
                out["encoder"] = _fields(block)
                self.encoder_counters = out["encoder"]
            contract = HdrContract(C.sizeof(HdrContract))
            if not self.b["nvenc"].nvenc_hdr_get_contract(self.encoder,
                                                          C.byref(contract)):
                out["encoder_contract"] = _fields(contract)
            sei = SeiStatus(C.sizeof(SeiStatus))
            if not self.b["nvenc"].nvenc_hdr_sei_get(self.encoder, C.byref(sei)):
                out["sei"] = _fields(sei)
                self.sei_status = out["sei"]
            if hasattr(self.b['nvenc'], 'nvenc_p010_get_metadata_status'):
                status = P010MetadataStatus(C.sizeof(P010MetadataStatus))
                if not self.b['nvenc'].nvenc_p010_get_metadata_status(self.encoder, C.byref(status)):
                    out['p010_metadata'] = _fields(status)
        if self.live is not None:
            try:
                out["dlss"] = self.live.transfer_counters()
            except Exception:
                pass
        out["nvof"] = (list(self.flow_provider.device_input_counters())
                       if self.uses_flow and self.flow_provider is not None
                       and hasattr(self.flow_provider, "device_input_counters")
                       else None)
        out["post_control"] = self.post_control.diagnostics()
        out["temporal_control"] = self.temporal_control.diagnostics()
        out["metadata_snapshots"] = self.metadata_snapshots
        out["metadata_divergence"] = self.metadata_divergence
        return out

    def finish(self):
        if getattr(self, 'framegen', None) is not None:
            self.framegen.finish(getattr(self, 'framegen_emit', self._emit_generated))
        if self.encoder:
            code = int(self.b["nvenc"].nvenc_finish(self.encoder))
            if code:
                self._fail("nvenc_last_error", code)
        if getattr(self,'truehdr_tail',None): self.truehdr_tail.finish()
        if self.prores_output is not None: self.prores_output.finish()


class HdrMultiScaleChain(HdrChain):
    """A dual or triple HDR reconstruction: one pristine master, one observation per layer.

    Layer 0 is exactly the chain a single-layer run is - the decoder's P010, the
    primary colour bridge's own downscale and proxy, the anchor Feature 18 session -
    so ``single`` keeps its own reconstruct path and is not re-expressed as a solver.
    Every layer above it owns a colour bridge of its own, opened on that layer's
    session's D3D12 device, on the decoder's CUDA context and at the job's one paper
    white, and takes its main colour by a device-to-device copy of the *pristine*
    full-resolution linear scRGB master that ``decode()`` produced: an extra layer
    never resamples, tone maps or 8-bit-approximates the picture, and its residual is
    measured against its own working master at its own resolved extent.

    The layers meet in exactly one place. ``data_consistency_experimental`` hands the
    master and one signed working residual per layer to the typed half-float solver
    and imports the full-resolution output it stores through the primary bridge's
    ``set_composite``, over the producer fence that says the store retired;
    ``catmull_baseline`` runs every layer's observation for the record and then
    composites layer 0's through the unchanged single-layer ``reconstruct``, which is
    the baseline itself and never a sum of residuals. Nothing re-encodes, re-scales or
    re-tonemaps in either mode, and the encode tail is shared verbatim.

    The borrowed-plane rule is why one frame is one straight line: an observation is
    the bridge's own residual plane and survives until the next call into that bridge,
    so every layer's ``observe`` happens before the solve and the solve copies all of
    them inside its one call. Feature 18 outputs are per-session, so a layer's texture
    is not disturbed by another layer's inference.
    """

    def __init__(self, bridges, hdr_caps, video, encoder_output, live, plan, fps,
                 policy, dims, flow_provider=None, residual_strength=1.0,
                 uses_flow=True, *, multi, extra_layers, fusion_plans,
                 preview_consumer=None):
        super().__init__(bridges, hdr_caps, video, encoder_output, live, plan, fps,
                         policy, dims, flow_provider=flow_provider,
                         residual_strength=residual_strength, uses_flow=uses_flow,
                         preview_consumer=preview_consumer)
        self.multi = dict(multi)
        self.extra_layers = list(extra_layers)
        self.fusion_plans = list(fusion_plans)
        self.fusion = None
        self.fusion_reports = []
        self.layer_calls = [0] * (1 + len(self.extra_layers))
        # One motion tally per layer, in the shape ``_guidance_and_inference`` bumps:
        # a dict, because the call that decides which kind of motion ran is the call
        # that records it, and reaching for a slot by index would be that decision
        # restated one function away from where it is made.
        self.layer_motion_calls = [{"flow": 0, "zero_motion": 0}
                                   for _ in self.layer_calls]
        self.layer_reset_flags = [[] for _ in self.layer_calls]
        self.layer_session_ids = None
        self.layer_papers = None

    # -- lifetime -------------------------------------------------------------

    def _require_multiscale_group(self):
        """Refuse to build a layer on a colour bridge that cannot take an import.

        The gate asked the same question of the same library before the job was
        accepted; this is the run's own copy of the answer, because by here a frame
        has been decoded and a bridge opened on a build without the group would fail
        in the middle of a layer instead of at the door.
        """
        group = (self.caps.get("colour_bridge") or {}).get("multiscale")
        if group is None:
            raise HdrExportError(
                "no HDR multiscale capability read was made on the loaded colour "
                "bridge, so this run will not import a master, observe a residual or "
                "take a fused composite on a group that was never verified")
        if not group.get("supported"):
            raise HdrExportError(group.get("reason") or
                                 "the loaded colour bridge has no HDR multiscale "
                                 "foundation for a second layer")

    def _open_stages(self, frame):
        """The single-layer stages, then one colour bridge per extra layer, then the solver.

        Each extra bridge is opened at the full extent it must be able to hold and the
        working extent its own scale resolved to, on the device that session's Feature
        18 output belongs to. The job's fixed paper white is handed to every one of
        them and each bridge reports back the value it applies, so a layer running at a
        different reference white is an open that fails, not a silent second policy.
        """
        super()._open_stages(frame)
        self._require_multiscale_group()
        session_ids = [self.live.session_id] + [
            layer["live"].session_id for layer in self.extra_layers]
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in session_ids)
                or len(set(session_ids)) != len(session_ids)):
            raise HdrExportError(
                "HDR multiscale layers require distinct, positive reportable session IDs")
        paper_white = float(self.policy["paper_white_scrgb"])
        try:
            for layer in self.extra_layers:
                width, height = layer["dims"]
                layer["bridge"] = self.bridge_type(
                    d3d12_device=layer["live"].d3d12_device(),
                    cuda_context=self.context,
                    full_width=self.processing_size[0], full_height=self.processing_size[1],
                    working_width=width, working_height=height,
                    paper_white_scrgb=paper_white)
            if self.multi["fusion"] == "data_consistency_experimental":
                self._open_fusion()
        except Exception as open_error:
            for cleanup_error in self._close_layers(suppress_errors=True):
                if hasattr(open_error, "add_note"):
                    open_error.add_note(
                        f"HDR multiscale rollback also failed: {cleanup_error}")
            raise
        self.layer_session_ids = session_ids
        self.layer_papers = [self.bridge.paper_white] + [
            layer["bridge"].paper_white for layer in self.extra_layers]

    def _open_fusion(self):
        """The typed solver, on the anchor's device and the pipeline's own tap plans.

        Importing the wrapper is deferred to here so a single-layer job, a
        ``catmull_baseline`` job and an import of this module load no fusion DLL at
        all. The layer descriptions are the shared ``TapPlans`` the negotiation's
        resolved extents produced - the same float32 tables the SDR chain marshals -
        and they are kept on the chain for the life of the open solver, whose options
        point into them.
        """
        import hdr_fusion_engine
        dims = self.multi["dims"]
        self.fusion = hdr_fusion_engine.HdrFusionBridge(
            d3d12_device=self.live.d3d12_device(),
            full_width=self.processing_size[0], full_height=self.processing_size[1],
            layers=[{"plan": plan, "dims": dims[index],
                     "weight": self.multi["weights"][index]}
                    for index, plan in enumerate(self.fusion_plans)],
            beta=self.multi["beta"], residual_strength=self.strength,
            relative_tolerance=self.multi["relative_tolerance"],
            max_iterations=self.multi["max_iterations"],
            library=self.b.get("fusion"))

    def _close_layers(self, *, suppress_errors=False):
        """Reverse ownership: solver, then each layer's bridge, session and flow layer.

        The solver's texture is what the primary bridge imported, so it goes first; a
        layer's bridge imports that layer's Feature 18 output, so it goes before the
        session that drew it; a layer's session goes before the independent motion
        layer it was fed from. The anchor's session, the primary bridge, the context
        and the decoder are the parent's to release in the parent's order, and the root
        flow provider was never this chain's to close - only the layers it handed out.
        Every step is guarded so a partially-opened chain tears down by the same code
        as a completed one, and no handle is released twice.
        """
        errors = []

        def release(owner, attr, method="close"):
            resource = owner.get(attr) if isinstance(owner, dict) else getattr(owner, attr)
            if resource is None:
                return
            if isinstance(owner, dict):
                owner[attr] = None
            else:
                setattr(owner, attr, None)
            try:
                getattr(resource, method)()
            except Exception as error:
                errors.append(error)

        release(self, "fusion")
        for layer in reversed(self.extra_layers):
            release(layer, "bridge")
            release(layer, "live")
            release(layer, "flow_provider", "close_independent_layer")
        if errors and not suppress_errors:
            error = HdrExportError(
                "one or more HDR multiscale resources failed to close: "
                + "; ".join(str(item) for item in errors))
            raise error from errors[0]
        return errors

    def close(self):
        errors = self._close_layers(suppress_errors=True)
        try:
            super().close()
        except Exception as error:
            errors.append(error)
        if errors:
            failure = HdrExportError(
                "HDR multiscale cleanup failed: "
                + "; ".join(str(item) for item in errors))
            raise failure from errors[0]

    # -- the frame ------------------------------------------------------------

    def process(self, frame, flow_provider, index, is_discontinuous):
        """Decode once, drive every layer over the one master, decide once, encode once.

        The frame is one line in the order the borrows require: the primary bridge's
        decode is the only measurement of the source and the only producer of the
        master, each extra bridge imports that master before anything else touches it,
        every observation is taken before the solve that consumes them, and the encode
        is the last stage. A layer whose NVOF call or Feature 18 pass raises fails the
        job: this chain never substitutes another layer's motion, another session's
        resident zero motion or another algorithm for the one that refused.
        """
        if self.multi['fusion'] == 'sequential':
            from sequential_nr import process_wide
            return process_wide(self, frame, flow_provider, index, is_discontinuous)
        is_first = self.first
        if is_first:
            self._open_stages(frame)
            self.first = False
        else:
            self._assert_metadata_stable()
        if (int(frame.width), int(frame.height)) != self.src:
            raise HdrExportError(
                f"the decoder yielded {frame.width}x{frame.height} at frame {index} "
                f"where every layer was opened on {self.src[0]}x{self.src[1]}: a "
                "multi-scale reconstruction rebuilds its masters, sessions and "
                "observation planes rather than reusing another extent's history")

        colour = {name: int(getattr(frame, name)) for name in
                  ("color_range", "color_primaries", "color_transfer",
                   "color_space")}
        if self.sdr10:
            from source_colour import resolve_frame_colour
            colour = resolve_frame_colour(colour, self.policy.get('source_colour_resolution'))
        stages = {}
        (master, working_master, proxy, guidance, source_report,
         proxy_report) = self._primary_colour(frame, colour, stages, is_first or is_discontinuous)
        is_discontinuous = (self._prefix_discontinuous if self.input_effect is not None
                            else self._detect_scene(master, is_first, is_discontinuous))
        reset = is_first or is_discontinuous

        layers = [{
            "number": 0, "scale": self.multi["scales"][0],
            "dims": list(self.working), "bridge": self.bridge,
            "live": self.live, "flow_provider": flow_provider,
            "weight": self.multi["weights"][0],
            "provenance": "decoded_pristine_master", "master": master,
            "proxy": proxy, "guidance": guidance, "proxy_report": proxy_report,
            "paper_white_scrgb": self.bridge.paper_white, "stages": {},
        }]
        for number, layer in enumerate(self.extra_layers, 1):
            mark = time.perf_counter()
            imported = layer["bridge"].import_master(master)
            layer["bridge"].downscale()
            layer_proxy, layer_guidance = layer["bridge"].proxy()
            stages["hdr_layers_colour"] = stages.get(
                "hdr_layers_colour", 0.0) + (time.perf_counter() - mark) * 1000.0
            layers.append({
                "number": number, "scale": layer["scale"],
                "dims": list(layer["dims"]), "bridge": layer["bridge"],
                "live": layer["live"], "flow_provider": layer["flow_provider"],
                "weight": self.multi["weights"][number],
                "provenance": "imported_pristine_master", "master": imported,
                "proxy": layer_proxy, "guidance": layer_guidance,
                "proxy_report": layer["bridge"].proxy_report(),
                "paper_white_scrgb": layer["bridge"].paper_white, "stages": {},
            })

        observations = []
        flow = None
        reused_layers = []
        for entry in layers:
            number = entry["number"]
            from preview_layer_cache import evaluate
            (dlss, entry_flow), reused = evaluate(self, number, lambda: self._guidance_and_inference(
                entry["live"], entry["flow_provider"], entry["proxy"],
                entry["guidance"], frame, index, reset, is_discontinuous,
                entry["stages"],
                counted=self.layer_motion_calls[number]))
            reused_layers.append(reused)
            if not reused:
                self.layer_calls[number] += 1
                self.layer_reset_flags[number].append(bool(reset))
            mark = time.perf_counter()
            observation = entry["bridge"].observe(dlss)
            entry["stages"]["hdr_observe"] = (
                (time.perf_counter() - mark) * 1000.0)
            entry["dlss"] = dlss
            entry["observation"] = observation
            observations.append(observation)
            if number == 0:
                flow = entry_flow
        stages["nvof_flow"] = layers[0]["stages"].get("nvof_flow")
        stages["dlss_inference"] = layers[0]["stages"].get("dlss_inference")
        self.layer_inference_ms = [float(entry["stages"].get("dlss_inference") or 0.0)
                                   for entry in layers]
        stages["nr_total_ms"] = sum(self.layer_inference_ms)
        for key in ("layer_nvof_flow_ms", "layer_dlss_inference_ms"):
            stages[key] = sum(entry["stages"].get(key) or 0.0
                              for entry in layers[1:])

        reconstruct_report = None
        fused = None
        if self.multi["fusion"] == "data_consistency_experimental":
            mark = time.perf_counter()
            if self.fusion is None:
                self._open_fusion()
            fused = self.fusion.process(cuda_context=self.context, master=master,
                                        observations=observations)
            stages["hdr_fusion"] = (time.perf_counter() - mark) * 1000.0
            self.fusion_reports.append(fused["report"])
            mark = time.perf_counter()
            composited = self.bridge.set_composite(fused)
            stages["hdr_composite_import"] = (time.perf_counter() - mark) * 1000.0
        else:
            # The baseline, unchanged: layer 0's observation and the reconstruct's own
            # residual pass are the same computation over the same planes, so the
            # composite is what a single-layer job would have written and no residual
            # from any other layer is added into it.
            mark = time.perf_counter()
            composited = self.bridge.reconstruct(layers[0]["dlss"],
                                                 residual_strength=self.strength)
            reconstruct_report = self.bridge.reconstruct_report()
            stages["hdr_reconstruct"] = (time.perf_counter() - mark) * 1000.0

        composited = self._post_process(master, composited, stages, frame)
        planes, encode_report = self._encode(frame, stages, composited)
        preview_output = self._consume_preview_output(frame, composited)

        report = {
            "frame_idx": index, "pts": int(frame.pts),
            "time_base": [int(frame.time_base_num), int(frame.time_base_den)],
            "colour_fields_sent": colour, "reset": bool(reset),
            "motion_source": (self.flow_provider.timing().get('guidance_backend', 'nvof_device_rgba')
                              if self.uses_flow else "zero_motion"),
            "source_report": source_report, "proxy_report": proxy_report,
            "encode_report": encode_report,
            "planes": {"master_full": master.as_dict(),
                       "working_master": working_master.as_dict(),
                       "proxy": proxy.as_dict(), "guidance": guidance.as_dict(),
                       "composited": composited.as_dict(),
                       "encoded_y": planes.y.as_dict() if planes else None,
                       "encoded_uv": planes.uv.as_dict() if planes else None},
            "feature18_output": {"width": layers[0]["dlss"].get("width"),
                                 "height": layers[0]["dlss"].get("height"),
                                 "dxgi_format": layers[0]["dlss"].get("dxgi_format")},
            "reconstruction_mode": self.multi["mode"],
            "layers": [self._layer_record(entry) for entry in layers],
        }
        if reconstruct_report is not None:
            # Only where the composite pass actually ran. An observation runs the same
            # residual pass, but the composite half of reconstruct_report describes a
            # pass this frame did not run, so a data-consistency frame reports the
            # solver instead of a reconstruct total.
            report["reconstruct_report"] = reconstruct_report
            report["fusion_mode_actual"] = "catmull_baseline_layer_0"
        if fused is not None:
            report["fusion_mode_actual"] = "typed_gpu_data_consistency_solve"
            report["solver"] = {
                "iterations": fused["iterations"],
                "converged": fused["converged"],
                "unrepresentable_components": fused["unrepresentable_components"],
                "relative_tolerance": self.fusion.relative_tolerance,
                "max_iterations": self.fusion.max_iterations,
                "output": {"width": fused["width"], "height": fused["height"],
                           "dxgi_format": fused["dxgi_format"],
                           "state": fused["state"],
                           "fence_value_present": bool(fused["fence_value"])},
                "report": fused["report"]}
        self.frame_reports.append(report)
        outcome = {"stages": stages, "flow": flow, "layer_inference_reused": reused_layers,
                "nvof_timing": (flow_provider.timing()
                                if self.uses_flow and flow_provider is not None
                                and hasattr(flow_provider, "timing") else None)}
        if self.preview_consumer is not None:
            outcome["preview_output"] = preview_output
            outcome['_preview_master'] = self._preview_source
        return outcome

    @staticmethod
    def _layer_record(entry):
        """One layer's frame, as its own bridges reported it.

        A layer's ``provenance`` says how its master got in and only layer 0 carries a
        ``source_report``: an imported plane was measured by the bridge that decoded
        it, and reading a source report off an importing bridge would be reading the
        decode counters of a pass that bridge never ran.
        """
        dlss = entry["dlss"]
        return {
            "layer": entry["number"], "scale": entry["scale"],
            "working_dims": entry["dims"], "weight": entry["weight"],
            "session_id": entry["live"].session_id,
            "master_provenance": entry["provenance"],
            "paper_white_scrgb": entry["paper_white_scrgb"],
            "bridge_extents": {"full": list(entry["bridge"].full_extent),
                               "working": list(entry["bridge"].working_extent)},
            "proxy_report": entry["proxy_report"],
            "planes": {"master": entry["master"].as_dict(),
                       "proxy": entry["proxy"].as_dict(),
                       "guidance": entry["guidance"].as_dict(),
                       "observation": entry["observation"].as_dict()},
            "feature18_output": {"width": dlss.get("width"),
                                 "height": dlss.get("height"),
                                 "dxgi_format": dlss.get("dxgi_format")},
            "stages": dict(entry["stages"]),
        }

    # -- what the bridges say about the run -----------------------------------

    def layer_diagnostics(self):
        """What the layer vector ran, as this chain recorded making it.

        The session ids are the host's own per-core counters, not addresses; the call
        counts are this chain's record of the calls it made; the byte figures are in
        ``counters``, out of the bridges that counted them.
        """
        return {
            "layer_count": len(self.layer_calls),
            "layer_inference_ms": list(getattr(self, "layer_inference_ms", [])),
            "reconstruction_mode": self.multi["mode"],
            "fusion_mode_requested": self.multi["fusion"],
            "layer_scales": list(self.multi["scales"]),
            "layer_working_dims": [list(dims) for dims in self.multi["dims"]],
            "layer_weights": list(self.multi["weights"]),
            "layer_parameters": [layer_parameters(getattr(owner, 'settings', {}))
                                 for owner in [self.live] +
                                 [layer['live'] for layer in self.extra_layers]],
            "layer_session_ids": list(self.layer_session_ids or []),
            "layer_sessions_distinct": (
                len(set(self.layer_session_ids)) == len(self.layer_session_ids)
                if self.layer_session_ids else None),
            "layer_paper_white_scrgb": list(self.layer_papers or []),
            "layer_calls": list(self.layer_calls),
            "layer_flow_calls": [entry["flow"]
                                 for entry in self.layer_motion_calls],
            "layer_zero_motion_calls": [entry["zero_motion"]
                                        for entry in self.layer_motion_calls],
            "layer_reset_flags": [list(flags) for flags in self.layer_reset_flags],
            "fusion_beta": self.multi["beta"],
            "fusion_relative_tolerance": self.multi["relative_tolerance"],
            "fusion_max_iterations": self.multi["max_iterations"],
            "solver_open": self.fusion is not None,
        }

    def counters(self):
        """The parent's blocks plus every extra bridge, session and the solver.

        Each layer's colour bridge answers for itself, so the same device traffic is
        attributed to the bridge that moved it rather than merged into one number the
        primary bridge never saw. ``pass_dispatches`` is the multiscale group's own
        count of the dispatches that each copied their counter run back to the host -
        the scalar boundary the image counters do not measure.
        """
        out = super().counters()
        out["hdr_layers"] = []
        for layer in self.extra_layers:
            bridge = layer.get("bridge")
            if bridge is None or not bridge.is_alive():
                continue
            entry = {"dims": list(layer["dims"]), "scale": layer["scale"],
                     "session_id": (layer["live"].session_id
                                    if layer.get("live") else None),
                     "bridge": bridge.transfer_counters(),
                     "multiscale": bridge.multiscale_counters()}
            live = layer.get("live")
            counters = getattr(live, "transfer_counters", None)
            entry["dlss"] = counters() if callable(counters) else None
            provider = layer.get("flow_provider")
            device_counters = getattr(provider, "device_input_counters", None)
            entry["nvof"] = (list(device_counters())
                             if callable(device_counters) else None)
            out["hdr_layers"].append(entry)
        if self.fusion is not None:
            out["fusion"] = self.fusion.counters()
        out["hdr_multi_scale"] = self.layer_diagnostics()
        return out


# --- what the per-frame reports add up to ---------------------------------------

# The count fields of each stage's report, named per stage so a reader can tell the
# proxy's negatives from the reconstruction's. Every one of these is a component count
# the compiled bridge itself recorded; nothing here is inferred, and a stage that never
# ran contributes no key rather than a zero.
_COUNT_FIELDS = {
    "source": ("nonfinite_samples", "negative_components",
               "above_ceiling_components"),
    "proxy": ("negative_components", "pinned_components", "dead_zone_components",
              "above_actionable_components", "above_saturation_components"),
    "reconstruct": ("applied_components", "gain_cap_components",
                    "rejected_pinned_components", "rejected_dead_components",
                    "rejected_negative_components", "nonfinite_components",
                    "composite_out_of_domain_components",
                    "composite_nonfinite_components"),
    "encode": ("clamped_high_components", "clamped_low_components",
               "out_of_gamut_components", "nonfinite_components"),
}
# The extremes, with how they combine across frames: a sum of two maxima is nothing,
# so the envelope is kept as the low and high of what the bridges reported.
_EXTREME_FIELDS = {
    "source": {"min_component": min, "max_component": max},
    "proxy": {"min_code": min, "max_code": max},
    "reconstruct": {"max_abs_stops": max, "max_abs_residual_scrgb": max,
                    "composite_min_component": min, "composite_max_component": max},
    "encode": {"max_linear_component": max, "max_code_y": max, "min_code_y": min,
               "max_code_uv": max, "min_code_uv": min},
}


def aggregate_hdr_reports(frame_reports) -> dict:
    """Total clamping, rejection, negative, non-finite and above-ceiling counts.

    The brief's requirement that every one of these counts be reported, not
    summarised away: the pristine master's negative and out-of-domain channels are
    preserved by the existing native math and survive to the encode as code 0, so the
    only honest description of the run is how many components each stage refused to
    clamp, pinned, rejected or clipped, and over how many pixels it decided that.
    """
    out = {"frames_reported": len(frame_reports)}
    for kind, names in _COUNT_FIELDS.items():
        reports = [item.get(kind + "_report") or {} for item in frame_reports]
        if not reports or any(name not in report for report in reports
                              for name in names):
            continue
        for name in names:
            out[f"{kind}_{name}"] = sum(int(report[name]) for report in reports)
        out[f"{kind}_pixels_total"] = sum(int(report.get("pixels", 0))
                                          for report in reports)
    for kind, names in _EXTREME_FIELDS.items():
        reports = [item.get(kind + "_report") or {} for item in frame_reports]
        if not reports or any(name not in report for report in reports
                              for name in names):
            continue
        for name, combine in names.items():
            out[f"{kind}_{name}_envelope"] = combine(
                float(report[name]) for report in reports)
    return out


def _hashable(snapshot):
    """A metadata dict as something a set can compare, lists included."""
    return tuple((name, tuple(value) if isinstance(value, list) else value)
                 for name, value in sorted((snapshot or {}).items()))


def hdr_transfer_report(counters, processed, src_w, src_h, working, policy,
                        aggregate, negotiation=None, chain=None) -> dict:
    """Every host boundary this chain could cross, as its bridge reported it.

    Same rule as ``gtx.transfer_report``: a ``_measured`` field is a bridge's own
    number or None when nothing answered, and a ``_logical`` field is the size of a
    buffer the code describes rather than a traffic figure. Adding a logical figure to
    a measured one would double-count the same bytes, so the two are never summed.
    A third kind of entry covers a boundary the source says exists and no counter
    measures: ``hdr_scalar_report_readback``, whose byte figure is None and whose
    derived per-dispatch size carries that provenance in its own name. None of these
    are physical PCIe measurements; a bridge counts the calls it makes.
    """
    live_w, live_h = working or (src_w // 2, src_h // 2)
    decoder = counters.get("decoder") or {}
    bridge = counters.get("hdr_bridge") or {}
    dlss = counters.get("dlss") or {}
    encoder = counters.get("encoder") or {}
    contract = counters.get("encoder_contract") or {}
    sei = counters.get("sei") or {}
    nvof = counters.get("nvof")
    report = {
        # The frames this chain actually drove, which is not what was asked for until
        # the run says so: a decode that stopped early reports the short count here and
        # the requested count beside it.
        "hdr_frames_processed_measured": processed,
        # Decoder. The two host fields are the counter that only the readback entry
        # point raises, and production never binds it; a zero here is that counter
        # answering, and None is a bridge that did not answer at all.
        "decode_host_copy_bytes_measured": decoder.get("host_frame_copy_bytes"),
        "decode_host_copy_count_measured": decoder.get("host_frame_copy_count"),
        "decode_frames_decoded_measured": decoder.get("frames_decoded"),
        "decode_device_frame_count_measured": decoder.get("device_frame_count"),
        "decode_bitstream_submitted_bytes_measured":
            decoder.get("bitstream_bytes_submitted"),
        # The open_hdr contract: the decoder scales nothing, so its scale field is a
        # measurement of a stage that the HDR open makes impossible.
        "decode_scale_device_output_bytes_measured":
            decoder.get("scale_device_output_bytes"),
        "decode_p010_bytes_logical": src_w * src_h * 3,
        # Colour bridge: device writes and the device-to-device copy of the borrowed
        # P010, plus the report reads, which are scalars and counted apart.
        "hdr_process_count_measured": bridge.get("process_count"),
        "hdr_source_device_copy_bytes_measured":
            bridge.get("source_device_copy_bytes"),
        "hdr_source_device_write_bytes_measured":
            bridge.get("source_device_write_bytes"),
        "hdr_working_device_write_bytes_measured":
            bridge.get("working_device_write_bytes"),
        "hdr_guidance_device_write_bytes_measured":
            bridge.get("guidance_device_write_bytes"),
        "hdr_composited_device_write_bytes_measured":
            bridge.get("composited_device_write_bytes"),
        "hdr_encode_device_write_bytes_measured":
            bridge.get("encode_device_write_bytes"),
        # The bridge's own counter for image planes copied out to the host. It moves
        # only when a ``gpu_hdr_readback_*`` entry runs, and this chain binds none of
        # them, so a production run reads that counter's zero and means it. It is not
        # a measurement of the per-pass counter readback, which is the next line.
        "hdr_diagnostic_image_readback_bytes_measured":
            bridge.get("report_readback_bytes"),
        # Scalar counters do come back to the host - every executed dispatch copies
        # its counter run into a CPU-visible readback buffer, and the report structs
        # above are answered out of that copy. No field of the ABI measures those
        # bytes, so the honest answer here is None: nothing was counted, which is not
        # the same as having counted zero. Reporting the diagnostic counter in this
        # slot would do exactly that.
        "hdr_scalar_report_readback_bytes_measured": None,
        "hdr_scalar_report_readback": {
            "crosses_host": True,
            "bytes_measured": None,
            "per_dispatch_bytes_from_source": 256,
            "measured_by_this_report": False,
            "evidence": "native/gpu_hdr_scrgb_bridge.cpp run_pass: CopyBufferRegion "
                        "from kPlaneCounters to kPlaneCounterReadback for "
                        "kCounterSlots * sizeof(uint32_t), on a buffer created in a "
                        "D3D12_HEAP_TYPE_READBACK heap, then Unmap/Map and memcpy "
                        "into the host snapshot every report getter answers from",
            "why_not_a_total": "256 is read off the source constants (64 slots of 4 "
                               "bytes), not counted by the bridge. Multiplying it by a "
                               "dispatch count would be an arithmetic estimate about "
                               "bytes this run moved, so no such product is reported "
                               "here as a measurement."},
        # The complete HDR picture-byte readback figure: no counter of any bridge in
        # this chain answers it, so it stays unanswered rather than becoming a zero.
        "hdr_pixel_readback_bytes_measured": None,
        # Feature 18 on the FP16 session.
        "dlss_host_color_stage_bytes_measured": dlss.get("host_color_stage_bytes"),
        "dlss_host_color_stage_count_measured": dlss.get("host_color_stage_count"),
        "dlss_device_color_copy_bytes_measured": dlss.get("device_color_copy_bytes"),
        "dlss_output_readback_bytes_measured": dlss.get("output_readback_bytes"),
        # NVENC HDR10 entry: host_pixel_bytes and rgb_conversion_bytes are fields the
        # bridge itself carries and insists are always zero on this entry, which is
        # why they are read rather than asserted.
        "encoder_frames_submitted_measured": encoder.get("frames_submitted"),
        "encoder_packets_written_measured": encoder.get("packets_written"),
        "encoder_p010_device_copy_bytes_measured":
            encoder.get("p010_device_copy_bytes"),
        "encoder_host_pixel_bytes_measured": encoder.get("host_pixel_bytes"),
        "encoder_rgb_conversion_bytes_measured": encoder.get("rgb_conversion_bytes"),
        "encoder_mastering_side_data_bytes_measured":
            encoder.get("mastering_side_data"),
        "encoder_content_light_side_data_bytes_measured":
            encoder.get("content_light_side_data"),
        # The planes, sized by the extents the code was given: 8 bytes per pixel of
        # FP16 colour for the picture path, 4 for the 8-bit motion guidance, and the
        # 10-bit 4:2:0 P010 pair the encoder receives.
        "hdr_master_full_fp16_bytes_logical": src_w * src_h * 8,
        "hdr_working_master_fp16_bytes_logical": live_w * live_h * 8,
        "hdr_proxy_fp16_bytes_logical": live_w * live_h * 8,
        "hdr_guidance_rgba8_bytes_logical": live_w * live_h * 4,
        "hdr_composited_fp16_bytes_logical": src_w * src_h * 8,
        "hdr_encoded_p010_bytes_logical": src_w * src_h * 3,
        "nvof_device_input_copy_count_total": (nvof or (None, None))[0],
        "nvof_device_input_bytes_total": (nvof or (None, None))[1],
    }
    report.update({f"total_{name}": value
                   for name, value in aggregate.items()})
    if chain is not None:
        report["hdr_dlss_flow_calls_measured"] = chain.flow_calls
        report["hdr_dlss_zero_motion_calls_measured"] = chain.zero_motion_calls
        report["hdr_metadata_snapshot_count_measured"] = len(
            chain.metadata_snapshots)
        report["hdr_metadata_stable_across_frames"] = (
            chain.metadata_divergence is None and len(set(
                _hashable(item) for item in chain.metadata_snapshots)) <= 1
            if chain.metadata_snapshots else None)
        report["hdr_metadata_divergence"] = chain.metadata_divergence
    multi = counters.get("hdr_multi_scale")
    if multi is not None:
        # The layer vector as this chain recorded running it. The session ids are the
        # Feature 18 host's own per-core counters, not addresses, and their distinctness
        # is the auditable form of "each layer drew its own frames": one session asked
        # twice would report one id.
        report["hdr_layers_measured"] = multi["layer_count"]
        report["hdr_reconstruction_mode"] = multi["reconstruction_mode"]
        report["hdr_fusion_mode_requested"] = multi["fusion_mode_requested"]
        report["hdr_layer_scales"] = multi["layer_scales"]
        report["hdr_layer_working_dims"] = multi["layer_working_dims"]
        report["hdr_layer_weights"] = multi["layer_weights"]
        report["hdr_layer_session_ids"] = multi["layer_session_ids"]
        report["hdr_layer_sessions_distinct"] = multi["layer_sessions_distinct"]
        report["hdr_layer_paper_white_scrgb"] = multi["layer_paper_white_scrgb"]
        report["hdr_layer_calls_measured"] = multi["layer_calls"]
        report["hdr_layer_flow_calls_measured"] = multi["layer_flow_calls"]
        report["hdr_layer_zero_motion_calls_measured"] = (
            multi["layer_zero_motion_calls"])
        report["hdr_layer_reset_flags"] = multi["layer_reset_flags"]
        report["hdr_layer_masters"] = {
            "layer_0": "decoded_pristine_master",
            "other_layers": "imported_pristine_master_by_device_to_device_copy",
            "source_report_of_an_imported_master": None,
            "why": "an import copies a plane another bridge already measured; only "
                   "the bridge that ran the decode pass has a source report, and no "
                   "importing bridge's decode counters were read as its measurement"}
    layer_blocks = counters.get("hdr_layers") or []
    if layer_blocks:
        # Per bridge, per session, per engine: what each one said it moved. Not summed
        # with the primary block into one figure, because the same master bytes do
        # cross twice - once into the primary's master by decode, once into each
        # layer's master by import - and only the attribution per bridge says which
        # crossing each number belongs to.
        report["hdr_layer_transfers_measured"] = []
        for number, block in enumerate(layer_blocks, 1):
            bridge = block.get("bridge") or {}
            scale = block.get("multiscale") or {}
            report["hdr_layer_transfers_measured"].append({
                "layer": number, "working_dims": list(block["dims"]),
                "session_id": block.get("session_id"),
                "master_provenance_measured": scale.get("master_provenance"),
                "process_count_measured": bridge.get("process_count"),
                "master_import_count_measured": scale.get("master_import_count"),
                "master_import_device_bytes_measured":
                    scale.get("master_import_device_bytes"),
                "observe_count_measured": scale.get("observe_count"),
                "observe_copy_bytes_measured": scale.get("observe_copy_bytes"),
                "observe_residual_bytes_measured":
                    scale.get("observe_residual_bytes"),
                "composite_import_count_measured":
                    scale.get("composite_import_count"),
                "composite_import_bytes_measured":
                    scale.get("composite_import_bytes"),
                "source_device_write_bytes_measured":
                    bridge.get("source_device_write_bytes"),
                "working_device_write_bytes_measured":
                    bridge.get("working_device_write_bytes"),
                "guidance_device_write_bytes_measured":
                    bridge.get("guidance_device_write_bytes"),
                "composited_device_write_bytes_measured":
                    bridge.get("composited_device_write_bytes"),
                "encode_device_write_bytes_measured":
                    bridge.get("encode_device_write_bytes"),
                "hdr_diagnostic_image_readback_bytes_measured":
                    bridge.get("report_readback_bytes"),
                "dlss_measured": block.get("dlss"),
                "nvof_device_input_counters_measured": block.get("nvof"),
                "pass_dispatches_measured": scale.get("pass_dispatches"),
            })
    solves = getattr(chain, "fusion_reports", None) if chain is not None else None
    if solves:
        report["hdr_solver_frames_measured"] = len(solves)
        report["hdr_solver_iterations_max_measured"] = max(
            int(item["iterations"]) for item in solves)
        report["hdr_solver_initial_residual_norm_max_measured"] = max(
            float(item["initial_residual_norm"]) for item in solves)
        report["hdr_solver_final_residual_norm_max_measured"] = max(
            float(item["final_residual_norm"]) for item in solves)
        report["hdr_solver_relative_residual_max_measured"] = max(
            float(item["relative_residual"]) for item in solves)
        report["hdr_solver_converged_all_frames"] = all(
            bool(item["converged"]) for item in solves)
        report["hdr_solver_unrepresentable_components_total_measured"] = sum(
            int(item["unrepresentable_components"]) for item in solves)
        report["hdr_solver_threshold"] = multi["fusion_relative_tolerance"] if (
            multi) else None
        report["hdr_solver_max_iterations"] = multi["fusion_max_iterations"] if (
            multi) else None
    if layer_blocks:
        # The layer's own working plane sizes, named per layer so a reader can tell
        # which extent a figure describes. These are sizes of buffers the code
        # describes, not traffic: the traffic of the same planes is the measured
        # block above.
        for entry, block in zip(report["hdr_layer_transfers_measured"], layer_blocks):
            width, height = block["dims"]
            entry["working_plane_fp16_bytes_logical"] = width * height * 8
            entry["guidance_plane_rgba8_bytes_logical"] = width * height * 4
    fusion_counters = counters.get("fusion")
    if fusion_counters is not None:
        report["hdr_fusion_counters_measured"] = fusion_counters
    dispatches = {}
    if counters.get("hdr_primary_multiscale") is not None:
        dispatches["primary"] = int(
            counters["hdr_primary_multiscale"].get("pass_dispatches") or 0)
    for number, block in enumerate(layer_blocks, 1):
        value = (block.get("multiscale") or {}).get("pass_dispatches")
        if value is not None:
            dispatches[f"layer_{number}"] = int(value)
    if dispatches:
        # Every completed pass of every colour bridge in this chain copied its counter
        # run into a CPU-visible readback buffer, and no field of the ABI measures those
        # bytes. The dispatch counts are the bridges' own, and 256 is read off the
        # source's constants, so their product is stated as what it is: a range derived
        # from a counted number, reported beside - never instead of, and never folded
        # into ``hdr_scalar_report_readback_bytes_measured``, which stays None.
        report["hdr_scalar_report_readback"].update({
            "pass_dispatches_measured": dispatches,
            "pass_dispatches_total_measured": sum(dispatches.values()),
            "range_bytes_from_completed_dispatch_count":
                sum(dispatches.values()) * 256,
            "fusion_scalar_readback_bytes_measured": (
                fusion_counters or {}).get("scalar_readback_bytes"),
            "provenance": "GpuHdrMultiscaleCounters.pass_dispatches per colour "
                          "bridge, incremented inside run_pass for each completed "
                          "dispatch; GpuFusionHdrCounters.scalar_readback_bytes from "
                          "the typed solver's own counters"})
    if policy:
        report["hdr_fixed_policy"] = dict(policy)
    if contract:
        written = int(contract.get("metadata_written_to") or 0)
        report["encoder_metadata_written_to_measured"] = written
        report["encoder_metadata_destinations"] = [
            name for bit, name in sorted(METADATA_DESTINATIONS.items())
            if written & bit]
        # The only bit that is a statement about the elementary stream the file
        # holds; the side-data and container bits describe other carriers.
        report["encoder_in_bitstream_sei_written"] = bool(written & 16)
        report["encoder_in_bitstream_obu_written"] = bool(written & 32)
        report["encoder_contract"] = contract
    if counters.get('p010_metadata'):
        report['p010_metadata_writer'] = counters['p010_metadata']
        report['p010_metadata_writer']['format'] = 'av1_metadata_obu' if counters['p010_metadata']['codec_kind'] == 2 else 'hevc_prefix_sei'
        if counters['p010_metadata']['codec_kind'] == 2:
            sei = None
    if sei:
        report["sei_writer"] = {
            "writer_available": sei.get("writer_available"),
            "metadata_requested": sei.get("metadata_requested"),
            "mastering_serialized": sei.get("mastering_serialized"),
            "content_light_serialized": sei.get("content_light_serialized"),
            "error": SEI_ERRORS.get(int(sei.get("error") or 0),
                                    sei.get("error")),
            "detail": sei.get("detail"),
            "packets_seen": sei.get("packets_seen"),
            "nals_inserted": sei.get("nals_inserted"),
            "bytes_inserted": sei.get("bytes_inserted"),
            "mastering_inserted": sei.get("mastering_inserted"),
            "content_light_inserted": sei.get("content_light_inserted"),
            "mastering_already_present": sei.get("mastering_already_present"),
            "content_light_already_present":
                sei.get("content_light_already_present"),
        }
    if negotiation:
        report["hdr_negotiation"] = {
            key: negotiation.get(key) for key in
            ("guidance_mode", "uses_flow", "downscaled", "working_scale",
             "hdr_encoder_contract_declared", "copy_pass",
             "hdr_video_stream_metadata")}
    return report


# --- the run --------------------------------------------------------------------

def _reserve_intermediate(negotiation):
    """The one path this run could ever own, or ``None`` when the encode is the file.

    A job that needs the copy pass is encoded into a file of its own, unique to this
    run, so a remux that fails cannot leave - or produce - anything under a name the
    caller asked for. The pre-delete is the existing rule: mkstemp hands out a name it
    has reserved, and the encoder bridge wants to create the file itself.
    """
    if not negotiation["copy_pass"]:
        return None
    suffix = (".mov" if negotiation.get("plan", {}).get("codec") == "prores_ks"
              else gtx.INTERMEDIATE_CONTAINER)
    handle, path = tempfile.mkstemp(suffix=suffix, prefix="dlss_gpu_hdr_")
    os.close(handle)
    os.unlink(path)
    return path


def _remove_intermediate(path):
    """Remove the file this run created under a name nobody else could have chosen.

    Called from the guard around the whole run, so it answers every exit - success, a
    failed frame, a cancel, a teardown that raised, a refused verification - and not
    only the paths that reach the remux. A partial HDR10 master left behind is a file
    that looks like a deliverable, which is why a cleanup that could not finish is
    reported and never promoted to the run's own failure: an unlink that fails hides no
    information a caller needs, while the exception it would swallow does. The source
    and the caller's ``out_path`` are never this path.
    """
    if not path:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError as cleanup_error:
        print(f"[gpu_hdr_transfer] intermediate cleanup failed: {cleanup_error}",
              flush=True)


def export_gpu_hdr_transfer(video, live, flow_provider, out_path, dims, frames,
                            cq, with_audio, transcode_settings, job_settings,
                            negotiation, progress=None, cancel=None,
                            on_proc_started=None, on_proc_exited=None,
                            diagnostics_out=None, profiler=None,
                            transfer_collector=None, bridges=None,
                            hdr_caps=None, verify=None, lock=None):
    """Run the experimental HDR10 export and return ``(out_path, report)``.

    The loop is the GPU-transfer loop, and deliberately so: same lock per frame, same
    watchdog, same cancellation between frames, same shortfall rule and the same output
    verification. What differs is the chain it drives. The second half is the same one
    the SDR chain runs: when the negotiation says the request carries anything beyond
    video, the encoder writes a uniquely named video-only MKV, that file is closed and
    released before ``gtx.run_copy_pass`` maps it as input 0 beside the source as input
    1, and the deliverable is the remux's output - never the encoder's, and never a
    second encode of the picture.

    This entry point owns that intermediate for the entire run: the name is reserved
    here and removed in the ``finally`` here, so the guard covers construction of the
    chain, the decode, every frame, ``finish``, teardown, the report, the remux and the
    verification instead of only the block that runs FFmpeg. Single threaded by
    necessity, like the SDR chain: every bridge here is synchronous and bound to the
    decoder's CUDA context and this thread, and each hand-off is a borrowed pointer the
    next call invalidates.
    """
    # The fuse, before the decoder opens the source and before the encoder's first
    # write: an HDR run that overwrote its own master would destroy the reference the
    # whole chain is measured against.
    pipeline.assert_output_not_input(video, out_path)
    import native_effects
    if native_effects.has_stage(job_settings or {},'vsr'):
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime((job_settings or {}).get('vsr_runtime_path'))
    intermediate = _reserve_intermediate(negotiation)
    try:
        return _export_gpu_hdr_transfer(
            video, live, flow_provider, out_path, dims, frames, cq, with_audio,
            transcode_settings, job_settings, negotiation, progress=progress,
            cancel=cancel, on_proc_started=on_proc_started,
            on_proc_exited=on_proc_exited, diagnostics_out=diagnostics_out,
            profiler=profiler, transfer_collector=transfer_collector,
            bridges=bridges, hdr_caps=hdr_caps, verify=verify, lock=lock,
            intermediate=intermediate)
    finally:
        _remove_intermediate(intermediate)


def _open_extra_layers(multi, live, flow_provider, uses_flow, src_w, src_h, settings=None):
    """One independent Feature 18 session and motion history per layer above the first.

    Built before the first frame and before the chain exists, because an extra layer
    that cannot be opened is a job that never starts rather than a run that discovers
    it mid-picture. Each goes through ``open_layer_ex`` on the anchor's core: the NGX
    infrastructure may be shared, the Feature is not, and a second layer drawn from
    the same session would be one Feature called twice wearing a layer's name. The
    colour bridges those sessions feed are opened later, on the CUDA context only the
    decoder can report.

    A failure part-way through is unwound in the order the objects were made, so a
    layer that opened hands its session and its independent NVOF layer back and the
    caller still sees the original error.
    """
    import dlss_gpu_engine
    layers = []
    try:
        for scale, (width, height) in zip(multi["scales"][1:], multi["dims"][1:]):
            layer = {"dims": (width, height), "scale": scale, "live": None,
                     "bridge": None,
                     "flow_provider": (flow_provider.new_independent_layer()
                                       if uses_flow else None)}
            layers.append(layer)
            from layer_settings import resolve_layer_settings
            layer["live"] = dlss_gpu_engine.Live(
                width, height, resolve_layer_settings(
                    dict(live.settings, **(settings or {})), len(layers)), shared_with=live,
                color_format="rgba16f")
    except Exception as open_error:
        cleanup_errors = []
        for layer in reversed(layers):
            if layer["live"]:
                try:
                    layer["live"].close()
                except Exception as error:
                    cleanup_errors.append(error)
            if layer["flow_provider"]:
                try:
                    layer["flow_provider"].close_independent_layer()
                except Exception as error:
                    cleanup_errors.append(error)
        for cleanup_error in cleanup_errors:
            if hasattr(open_error, "add_note"):
                open_error.add_note(
                    f"HDR layer-open rollback also failed: {cleanup_error}")
        raise
    return layers


def _export_gpu_hdr_transfer(video, live, flow_provider, out_path, dims, frames,
                             cq, with_audio, transcode_settings, job_settings,
                             negotiation, progress=None, cancel=None,
                             on_proc_started=None, on_proc_exited=None,
                             diagnostics_out=None, profiler=None,
                             transfer_collector=None, bridges=None,
                             hdr_caps=None, verify=None, lock=None,
                             intermediate=None):
    """The HDR run itself, on an intermediate whose removal its caller guards.

    ``intermediate`` is the encoder's destination when the request carries anything
    beyond video, and ``None`` when it does not - in which case the encode writes the
    deliverable itself and there is nothing here to clean up.
    """
    js = dict(job_settings or {})
    src_w, src_h = int(dims["src_w"]), int(dims["src_h"])
    import native_effects
    process_w, process_h = native_effects.processing_size(js, (src_w, src_h))
    sdr10 = (negotiation.get("hdr_policy") or {}).get("colour_mode") == "sdr10_bt709"
    transfer_mode = "gpu_sdr10_transfer" if sdr10 else "gpu_hdr_transfer"
    from post_control import GpuPostControl, controls_from_settings
    post_controls = controls_from_settings(js)
    from temporal_control import GpuTemporalControl, temporal_from_settings
    temporal_controls = temporal_from_settings(js)
    live_w, live_h = int(dims["live_w"]), int(dims["live_h"])
    uses_flow = bool(negotiation["uses_flow"])
    bridges = bridges or gtx.load_bridges(requires_flow=uses_flow)
    cap_codec = (HDR_CODEC if str((transcode_settings or {}).get('codec', '')).lower() in
                 ('prores_ks', 'prores') else (transcode_settings or {}).get('codec', HDR_CODEC))
    hdr_caps = hdr_caps or (load_sdr10_capabilities if sdr10 else load_hdr_capabilities)(
        bridges, src_w, src_h, codec=cap_codec)
    encoder_output = intermediate or out_path
    multi = negotiation.get("multi_scale")
    extra_layers = []
    if multi and len(multi["scales"]) > 1:
        extra_layers = _open_extra_layers(multi, live, flow_provider, uses_flow,
                                          process_w, process_h, settings=js)
    strength = float(js.get("residual_strength", 1.0))
    if extra_layers:
        try:
            fusion_plans = [gtx.TapPlans(process_w, process_h, width, height)
                            for width, height in multi["dims"]]
            chain = HdrMultiScaleChain(
                bridges, hdr_caps, video, encoder_output, live,
                negotiation["plan"], negotiation["fps"],
                negotiation["hdr_policy"], dims, flow_provider=flow_provider,
                residual_strength=strength, uses_flow=uses_flow, multi=multi,
                extra_layers=extra_layers, fusion_plans=fusion_plans)
        except Exception as construction_error:
            # These independently opened sessions/providers become chain-owned only
            # after construction succeeds. Until then, unwind them here without
            # touching the caller's anchor session or root provider.
            for layer in reversed(extra_layers):
                for key, method in (("live", "close"),
                                    ("flow_provider", "close_independent_layer")):
                    resource = layer.get(key)
                    if resource is None:
                        continue
                    layer[key] = None
                    try:
                        getattr(resource, method)()
                    except Exception as cleanup_error:
                        if hasattr(construction_error, "add_note"):
                            construction_error.add_note(
                                "HDR multiscale construction rollback also failed: "
                                f"{cleanup_error}")
            raise
    else:
        chain = HdrChain(bridges, hdr_caps, video, encoder_output, live,
                         negotiation["plan"], negotiation["fps"],
                         negotiation["hdr_policy"], dims,
                         flow_provider=flow_provider, residual_strength=strength,
                         uses_flow=uses_flow)
    chain.post_control = GpuPostControl(post_controls, hdr=True, sdr10=sdr10)
    chain.temporal_control = GpuTemporalControl(temporal_controls, hdr=True)
    import native_effects
    native_effects.install(chain, live, js, (src_w, src_h), flow_provider)
    chain.render_output_size = native_effects.output_size(js, (src_w, src_h))
    chain.export_scale_algorithm = (transcode_settings or {}).get('scale_algorithm', 'lanczos')
    chain.cancel = cancel
    if sdr10:
        chain.post_control._report["domain"] = "relative_linear_rec709"
    collector = (gtx.transfer_profiler.TransferProfiler()
                 if transfer_collector is None else transfer_collector)
    profiler = profiler or pipeline.StageProfiler()
    processed = 0
    previous_timestamp = None
    started = time.time()
    last_frame_t = time.time()
    counters = {}
    task_error = None
    cleanup_error = None
    profiler.start()
    try:
        native_phase('打开 HDR/10-bit 解码器', 300)
        chain.open_decoder()
        frame = gtx._DeviceFrame()
        frame.struct_size = C.sizeof(gtx._DeviceFrame)
        from export_range import SelectedDecoder, validate_range
        selected_decoder = SelectedDecoder(chain, validate_range(js.get("_frame_range")), negotiation["fps"], cancel)
        while processed < frames:
            if cancel and cancel():
                raise pipeline.JobCancelled("用户已取消任务")
            if time.time() - last_frame_t > gtx.WATCHDOG_TIMEOUT_SEC:
                raise RuntimeError(
                    f"HDR GPU 转码管线在第 {processed} 帧严重停滞（超过 "
                    f"{gtx.WATCHDOG_TIMEOUT_SEC} 秒无有效帧），NVDEC / Feature18 / "
                    "NVENC 可能已死锁")
            t_decode = time.perf_counter()
            native_phase(f'HDR/10-bit 解码第 {processed + 1} 帧', 300)
            state = selected_decoder.next(frame)
            decode_ms = (time.perf_counter() - t_decode) * 1000.0
            if state == 0:
                break
            if state < 0:
                gtx._bridge_error(bridges["nvdec"], "nvdec_bridge_last_error", state)
            before = (flow_provider.device_input_counters()
                      if uses_flow and hasattr(flow_provider,
                                               "device_input_counters") else None)
            t_frame = time.perf_counter()
            discontinuous = processed == 0 or bool(
                previous_timestamp and gtx._frame_timestamp_discontinuity(
                    previous_timestamp, frame, negotiation["fps"]))
            with gtx._session(lock):
                native_phase(f'HDR/10-bit 处理第 {processed + 1} 帧', 300)
                outcome = chain.process(frame, flow_provider, processed,
                                       is_discontinuous=discontinuous)
            previous_timestamp = (int(frame.pts), int(frame.time_base_num),
                                  int(frame.time_base_den))
            frame_ms = (time.perf_counter() - t_frame) * 1000.0
            from gpu_memory import capture
            capture(chain, live, phase='export')
            after = (flow_provider.device_input_counters()
                     if uses_flow and hasattr(flow_provider,
                                              "device_input_counters") else None)
            report_item = {
                "frame_idx": processed,
                "decode": decode_ms,
                "frame_total": frame_ms,
                "decode_requested": "cpu" if chain.prores_input else "cuda",
                "decode_actual": "libavcodec_prores_p210" if chain.prores_input else "cuda_nvdec_bridge_p010",
                "export_backend_requested": str(js.get("export_backend", "auto")),
                "export_backend_actual": "gpu_transfer",
                "transfer_mode_requested": transfer_mode,
                "transfer_mode_actual": transfer_mode,
                "experimental_hdr_scrgb": not sdr10,
                "dlss_backend_actual": "gpu_flow_d3d12",
                "hdr_color_format": getattr(live, "color_format", None),
                "hdr_motion_source": (flow_provider.timing().get('guidance_backend', 'nvof_device_rgba') if uses_flow
                                      else "zero_motion"),
                "guidance_backend": (outcome.get("nvof_timing") or {}).get(
                    "guidance_backend"),
                "flow_pts": int(frame.pts),
                **{name: ms for name, ms in outcome["stages"].items()},
            }
            if before and after and all(value is not None for value in before + after):
                # The root provider's own counters, which is layer 0's motion input.
                # The extra layers' NVOF traffic is in the per-layer blocks of the
                # transfer report, read off the layers' own providers at the end.
                report_item["nvof_device_input_copy_count"] = after[0] - before[0]
                report_item["nvof_device_input_bytes"] = after[1] - before[1]
            if multi and len(multi["scales"]) > 1:
                # The frame's own record of what the layers did, so a run that drew
                # one layer where two were asked for is visible in the per-frame
                # ledger and not only in the settings that requested them.
                record = chain.frame_reports[-1]
                solver = record.get("solver") or {}
                report_item["hdr_layer_count"] = len(record["layers"])
                report_item["hdr_fusion_mode_actual"] = record["fusion_mode_actual"]
                report_item["hdr_solver_iterations"] = solver.get("iterations")
                report_item["hdr_solver_converged"] = solver.get("converged")
                report_item["hdr_solver_unrepresentable"] = solver.get(
                    "unrepresentable_components")
            if sdr10:
                report_item = sdr10_report(report_item)
            collector.record(report_item)
            profiler.record({k: v for k, v in report_item.items()
                             if k != "flow_pts"})
            processed += 1
            last_frame_t = time.time()
            if progress:
                elapsed = max(0.001, time.time() - started)
                progress(processed, frames,
                         f"[GPU 显存直传 {'SDR 10-bit' if sdr10 else 'HDR10'} 导出] {processed}/{frames} "
                         f"({processed / frames * 100:.1f}%) | "
                         f"{processed / elapsed:.1f} fps")
        native_phase('排空 HDR/10-bit 编码器', 180)
        chain.finish()
    except BaseException as error:
        task_error = error
        raise
    finally:
        # A hung call is observed independently; no foreign thread closes it.
        native_phase('关闭 HDR/10-bit 导出链路', 90)
        try:
            chain.cancel_decoder()
        except Exception as cancel_error:
            print(f"[gpu_hdr_transfer] decoder cancel failed: {cancel_error}",
                  flush=True)
        try:
            counters = chain.counters()
        except Exception as counter_error:
            print(f"[gpu_hdr_transfer] transfer counters unavailable: "
                  f"{counter_error}", flush=True)
        profiler.stop()
        try:
            chain.close()
        except Exception as error:
            if task_error is None:
                cleanup_error = error
            elif hasattr(task_error, "add_note"):
                task_error.add_note(f"HDR export cleanup also failed: {error}")
        finally:
            # The set this run was handed, released through the one path that says what
            # "released" means: a job whose cookies the entry point already gave back
            # finds an empty list here instead of closing a foreign handle.
            release_loader_cookies(bridges)
        profiler.print_summary(f"GPU HDR TRANSFER EXPORT ({processed} FRAMES)")
        if cleanup_error is not None:
            raise cleanup_error

    native_phase('导出报告及容器封装', None)
    aggregate = aggregate_hdr_reports(chain.frame_reports)
    report = hdr_transfer_report(counters, processed, src_w, src_h,
                                 (live_w, live_h), negotiation["hdr_policy"],
                                 aggregate, negotiation=negotiation, chain=chain)
    report["hdr_frames_requested"] = frames
    report["post_control"] = counters.get("post_control")
    report["temporal_control"] = counters.get("temporal_control")
    report["render_pipeline"] = native_effects.stages(js)
    report["native_input_effect"] = counters.get("native_input_effect")
    report["native_output_effect"] = counters.get("native_output_effect")
    report['frame_generation'] = counters.get('frame_generation')
    report['export_resize'] = counters.get('export_resize')
    report['truehdr'] = counters.get('truehdr')
    if js.get('truehdr_enabled'):
        report['transfer_mode_actual'] = 'gpu_sdr10_truehdr_transfer' if sdr10 else 'gpu_hdr10_sdr_truehdr_transfer'
        report['output_color_contract'] = 'BT.2020 / PQ / limited range / P010 / ' + negotiation['plan']['codec'] + ' / ' + negotiation['plan']['profile']
        report.setdefault('sdr10_not_claimed',{})['formal_hdr'] = 'TrueHDR HDR10 conversion; no formal conformance certification'
    report['extra_colour_bridges'] = {k: counters[k] for k in ('decode_bridge', 'encode_bridge', 'framegen_bridge', 'export_resize_bridge') if k in counters}
    report["primary_vendor_feature"] = (counters.get("dlss") or {}).get(
        "vendor_feature", "NGX Feature18 / DLSSNR")
    report["export_backend_requested"] = str(js.get("export_backend", "auto"))
    report["export_backend_actual"] = "gpu_transfer"
    report["export_backend_fallback_reason"] = None
    report["transfer_mode_requested"] = transfer_mode
    report["transfer_mode_actual"] = (('gpu_sdr10_truehdr_transfer' if sdr10 else 'gpu_hdr10_sdr_truehdr_transfer')
                                     if js.get('truehdr_enabled') else transfer_mode)
    report["experimental_hdr_scrgb"] = not sdr10
    report["dlss_backend_actual"] = "gpu_flow_d3d12"
    report["hdr_color_format"] = getattr(live, "color_format", None)
    report["hdr_full_dims"] = [src_w, src_h]
    report["hdr_working_dims"] = [live_w, live_h]
    report["hdr_working_scale"] = negotiation["working_scale"]
    report["hdr_layers"] = (negotiation.get("layers") or 1)
    if multi and len(multi["scales"]) > 1:
        # Which decision the frames actually got, read off the per-frame records rather
        # than restated from the settings: a run that solved is told apart from a run
        # that drew the baseline, and a run that did both is visible as both.
        report["hdr_fusion_modes_actual"] = sorted(
            {item["fusion_mode_actual"] for item in chain.frame_reports})
    report["hdr_output_dims"] = [int(dims["dst_w"]), int(dims["dst_h"])]
    from vfx_framegen import output_fps
    report["hdr_frame_rate"] = "%d/%d" % output_fps(js, negotiation['fps'])
    report["hdr_guidance_mode"] = negotiation["guidance_mode"]
    report["hdr_uses_flow"] = uses_flow
    report["hdr_feature_resets_measured"] = sum(
        1 for item in chain.frame_reports if item.get("reset"))
    report["hdr_encoder_codec"] = negotiation["plan"]["codec"]
    report["hdr_encoder_profile"] = negotiation["plan"]["profile"]
    report["hdr_encoder_av_options"] = negotiation["plan"]["av_options"]
    report["hdr_encoder_tune_adjustment"] = negotiation["plan"].get("tune_adjustment")
    # Which file the caller's name belongs to. The encoder always writes one video
    # stream into one container; when the request carries anything else, the pass below
    # is what produces the deliverable, and a run has to be able to say which of the
    # two shapes it ran instead of leaving it to be guessed from the frame count.
    report["hdr_copy_pass"] = bool(negotiation["copy_pass"])
    report["hdr_deliverable_written_by"] = (
        "ffmpeg_map_and_copy_pass" if negotiation["copy_pass"]
        else "prores_software_encoder" if chain.prores_output_enabled
        else "nvenc_hdr_bridge")
    # The video stream's own descriptive fields this run asked the pass to put on the
    # deliverable: title and language as the source wrote them, and the flag list its
    # disposition carried. None is a request that was never made, or a source that has
    # nothing descriptive - it is not a claim that a title was preserved.
    report["hdr_video_stream_metadata"] = negotiation.get(
        "hdr_video_stream_metadata")
    report["hdr_bridge_dll"] = (hdr_caps.get("colour_bridge") or {}).get("path")
    report["hdr_source_metadata"] = chain.source_metadata
    report["hdr_mastering_wire_conversion"] = (None if counters.get('truehdr') or chain.prores_output_enabled else
        mastering_wire_report(chain.source_metadata, negotiation['plan']['codec']))
    if chain.prores_output_enabled:
        report['hdr_static_metadata'] = 'not authored by the ProRes software encoder'
        report['hdr_software_encoder'] = counters.get('prores_encoder')
        report['export_backend_actual'] = 'gpu_render_cpu_prores'
        report['encoder_host_pixel_bytes_measured'] = counters['prores_encoder']['frame_host_readback_bytes']
        report['encoder_frames_submitted_measured'] = counters['prores_encoder']['frames_submitted']
        report['zero_copy'] = False
        report['hdr_encoded_p010_bytes_logical'] = None
        report['encoded_p210_bytes_logical'] = int(dims['dst_w']) * int(dims['dst_h']) * 4
    if counters.get('prores_decoder'):
        report['software_decoder'] = dict(codec='prores', pixel_format='yuv422p10le', **counters['prores_decoder'])
        report['decode_p010_bytes_logical'] = None
        report['decode_p210_bytes_logical'] = src_w * src_h * 4
    report["hdr_pixel_readback_exports_bound"] = (
        (hdr_caps.get("colour_bridge") or {}).get("pixel_readback_exports_bound"))
    report["decoder_pixel_readback_entry_bound"] = bool(
        (hdr_caps.get("nvdec") or {}).get("pixel_readback_entry_bound"))
    report["hdr_not_claimed"] = {
        "performance": "NOT_CHARACTERIZED: no throughput or latency claim is made "
                       "by this run, and the per-stage milliseconds are timings of "
                       "one process, not a benchmark",
        "formal_hdr": "no HDR conformance claim: the picture is a chain that carries "
                      "an HDR10 declaration, not a certification of one",
        "lossless": "no losslessness claim; FP16 quantisation is the Stage A record's",
        "quality": "no quality-improvement claim",
        "mux": "the copy pass rewrites a container and never the picture: the encoded "
               "video is packet-copied, and every other stream is whatever FFmpeg's own "
               "copy or convert made from the source. No claim is made for a container "
               "other than Matroska. A caller's extra_args are not supported at this "
               "stage in any form - the request is refused whatever the argument list "
               "names - and the source video's title, language and disposition reach "
               "the deliverable when the job's copy_metadata left them on (the default) "
               "or named them directly, and the source actually carried them; the "
               "source's own technical stream tags "
               "(ENCODER, BPS, DURATION, frame counts) are never restated as facts "
               "about this run's encode",
        "domains": "HDR10 limited/full input range is normalized during GPU decode; "
                   "encoded output is limited-range. HLG, unspecified range, "
                   "non-BT.2020 primaries, constant-luminance "
                   "matrix and 8-bit are refused; final output resizing uses the GPU export resampler",
        "layers": "one, two or three reconstruction layers are supported; multi_pass "
                  "greater than 1 is not supported. Not every source and scale "
                  "combination has been verified: this run's short fixture evidence "
                  "does not establish general picture correctness",
        "host_traffic": "no claim that nothing crossed a host boundary: no picture "
                        "byte did, and the colour bridge's per-pass counter readback "
                        "did on every dispatch. The count of those scalar bytes is "
                        "unmeasured, so it is reported as None rather than as a zero. "
                        "When the request needed a copy pass, the finished compressed "
                        "bitstream and the source's other streams crossed to the host "
                        "through FFmpeg's own buffers, which is mux traffic and not "
                        "picture traffic"}
    if chain.prores_output_enabled or chain.prores_input:
        report['hdr_not_claimed']['host_traffic'] = 'ProRes uses CPU codec frames and explicit GPU upload/readback; these are reported separately from scalar readback and compressed remux traffic.'
        report['hdr_not_claimed']['mux'] = 'MOV ProRes output uses FFmpeg video packet copy when remuxing; audio/subtitle compatibility follows the selected MOV stream settings.'
    if diagnostics_out is not None:
        if sdr10:
            report = sdr10_report(report)
        diagnostics_out.update(report)
        prefix = "sdr10" if sdr10 else "hdr"
        diagnostics_out[f"{prefix}_frame_reports"] = (sdr10_report(chain.frame_reports)
                                                     if sdr10 else chain.frame_reports)
        diagnostics_out[f"{prefix}_capabilities"] = hdr_caps
        diagnostics_out[f"{prefix}_negotiation"] = (sdr10_report(negotiation)
                                                   if sdr10 else negotiation)
    mux_command = None
    if cancel and cancel():
        raise pipeline.JobCancelled("用户已取消任务")

    if js.get("_frame_range") and processed != frames:
        raise RuntimeError(f"选区导出帧数不完整：预期 {frames}，实际 {processed}")
    if diagnostics_out is not None and js.get("_frame_range"):
        diagnostics_out["frame_range"] = dict(js["_frame_range"], source_start_seconds=selected_decoder.source_first_seconds, emitted_input_frames=processed)
    if processed < frames:
        shortfall = frames - processed
        tolerance = max(5, int(round(frames * 0.01)))
        if shortfall > tolerance:
            raise RuntimeError(f"HDR 解码提前终止！预期读取 {frames} 帧，实际解码 "
                               f"{processed} 帧（输入文件可能损坏或解码中断）")

    if intermediate:
        # The encoder is closed by the block above, so what is on disk is a
        # finished file and the copy pass is reading a complete bitstream.
        transcode_settings = selected_decoder.mux_settings(transcode_settings, pipeline._bundle_ffprobe())
        if diagnostics_out is not None and js.get('_frame_range'):
            diagnostics_out['frame_range']['mux_source_start_seconds'] = transcode_settings['_source_range_start_seconds']
        if not os.path.isfile(intermediate):
            raise HdrExportError("the encoder produced no intermediate to wrap")
        ffmpeg = pipeline._bundle_ffmpeg()
        if not ffmpeg:
            raise HdrExportError("ffmpeg not found for the HDR container copy "
                                 "pass")
        mux_command = gtx.run_copy_pass(
            ffmpeg, intermediate, video, transcode_settings, out_path,
            with_audio, on_proc_started=on_proc_started,
            on_proc_exited=on_proc_exited,
            video_descriptive_metadata=negotiation.get(
                "hdr_video_stream_metadata"),
            video_authoritative=True,
            ffprobe=pipeline._bundle_ffprobe())
        if cancel and cancel():
            # The remux has written the deliverable and the run has not yet been called
            # a success, so a cancel that landed while FFmpeg was running is honoured
            # here instead of being turned into a report that says the job finished. The
            # queue's own process hooks are what reach the child; this is the last
            # checkpoint before the file is read back and returned.
            raise pipeline.JobCancelled("用户已取消任务")
    if mux_command is not None:
        report["sdr10_mux_command" if sdr10 else "hdr_mux_command"] = mux_command
        if diagnostics_out is not None:
            diagnostics_out["sdr10_mux_command" if sdr10 else "hdr_mux_command"] = mux_command

    if processed:
        # The count has to come from the bytes this run just wrote, and from the file
        # the caller was promised: with a copy pass that is the remux's output, so this
        # check runs after it and never reports the intermediate as the deliverable. A
        # path probed earlier in the process would answer the question from memory.
        from vfx_framegen import output_frame_count
        pipeline.verify_export_output(out_path, output_frame_count(js, negotiation['fps'], frames),
                                      probe=verify or pipeline.video_info_uncached)
    return out_path, sdr10_report(report) if sdr10 else report


def sdr10_report(value):
    """Label reused ABI counters with their SDR domain rather than asserting HDR.

    ABI field names are historical. Public SDR diagnostics use relative linear
    units, and retain the distinction between measured image and scalar traffic.
    """
    if isinstance(value, list):
        return [sdr10_report(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        name = ("sdr10_" + key[4:]) if key.startswith("hdr_") else key
        if key != "experimental_hdr_scrgb":
            name = name.replace("_scrgb", "_relative_linear")
        result[name] = sdr10_report(item)
    if "sdr10_not_claimed" in result:
        converted = bool(result.get('truehdr'))
        result["sdr10_not_claimed"].update(
            formal_hdr="TrueHDR HDR10 conversion; no formal conformance certification" if converted else "not HDR: BT.709 SDR, no PQ or HDR SEI",
            domains="SDR10 input → TrueHDR → linear scRGB → P010 / PQ / BT.2020 / HEVC or AV1" if converted else "limited-range 10-bit BT.709 SDR, HEVC or AV1 10-bit MKV; "
                    "other SDR domains are not supported; final output resizing uses the GPU export resampler",
            lossless="FP16 and lossy video encoding are not lossless")
    if "sdr10_policy" in result:
        result["sdr10_policy"]["linear_units"] = "relative Rec.709, white=1"
    return result
