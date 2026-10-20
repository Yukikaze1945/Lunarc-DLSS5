"""Reusable production binding for the HDR10 scRGB GPU bridge (ABI 1).

This is the Python owner for one ``GpuHdrBridge`` from
``native/gpu_hdr_scrgb_bridge.dll``. It mirrors the current header's structs and
entry points, loads the bundled DLL lazily through ``bundle_paths``, and drives
the colour chain the way ``native/gpu_hdr_scrgb_bridge.h`` describes it: P010 in,
linear scRGB FP16 held on the device end to end, P010 out.

What this is deliberately not:

  * it is not an export pipeline - there is no decoder, NVOF, Feature 18 or NVENC
    orchestration here, and no product wiring. Those are staged separately;
  * it moves no pixel to or from the host. Every stage hands back the bridge's
    own device plane as a ``(pointer, pitch)`` descriptor, and the readback entry
    points are not bound at all. Scalars do cross: the bridge brings each pass's
    counter run back to host memory so the report getters can answer from it, and
    ``GpuHdrTransferCounters`` has no field that measures those bytes. See
    ``native/gpu_hdr_scrgb_bridge.h``;
  * it adds no resampler and no second rounding rule. The tap plans are the CPU
    reference's own - ``pipeline._build_lanczos2_axis_plan`` down and
    ``pipeline._build_catmull_rom_axis_plan`` up - serialized into the header's
    ``GpuHdrTapTable`` so the device resamplers are the reference in half
    precision rather than a new opinion of it.

The physical units are the bridge's, fixed by the header and reported by
``contract``: linear scRGB 1.0 is 80 cd/m2, and the ST 2084 top of 10000 cd/m2 is
125.0 scRGB. ``paper_white_scrgb`` is chosen once at open and never adapted per
frame; ``support_ceiling_scrgb`` (decode) and ``clip_ceiling_scrgb`` (encode) are
declared per call by the caller and are never derived from MaxCLL here.

The multiscale foundation - ``import_master``, ``observe``, ``set_composite`` and
``multiscale_counters`` - is an optional capability group with its own version. It
is probed by symbol name, because speaking ABI 1 says nothing about those entries
existing: a bridge built without them still loads and runs the single layer exactly
as before, and a call into the group on such a bridge is refused with the reason
rather than attempted.

Performance is NOT_CHARACTERIZED by this layer.
"""
import ctypes
import math
import threading

import numpy as np

import bundle_paths


# The one ABI this wrapper speaks, mirroring GPU_HDR_ABI_VERSION in the header.
GPU_HDR_ABI_VERSION = 1
# The optional multiscale foundation, versioned on its own and probed by symbol
# name. Speaking ABI 1 says nothing about these entries being present, so a bridge
# built without them still loads and runs the single layer, and a multiscale call
# against it is refused with the reason rather than guessed at.
GPU_HDR_MULTISCALE_ABI_VERSION = 1
# GPU_HDR_MAX_TAPS: the widest tap row a plan may carry; a wider plan is refused,
# never truncated.
GPU_HDR_MAX_TAPS = 24
# The header's own unit anchor, named here so a caller reads it once: 1.0 scRGB is
# 80 cd/m2 and the ST 2084 top is 125.0 scRGB.
SCRGB_PER_RELATIVE = 125.0

# DXGI_FORMAT_R16G16B16A16_FLOAT, the only format the reconstruction accepts for a
# Feature 18 output texture.
DXGI_R16G16B16A16_FLOAT = 10

# The AVColor* enums the source must declare for decode to accept it: limited
# range, BT.2020 primaries, ST 2084 transfer, BT.2020 non-constant-luminance
# space. Handed to the bridge as int32 and refused by it on any other combination;
# named so a caller does not have to guess the numbers.
AV_COLOR_RANGE_LIMITED = 1
AV_COLOR_RANGE_FULL = 2
AV_COLOR_PRIMARIES_BT2020 = 9
AV_COLOR_TRANSFER_ST2084 = 16
# AVCOL_SPC_BT2020_NCL in libavutil/pixfmt.h; 10 is AVCOL_SPC_BT2020_CL.
AV_COLOR_SPACE_BT2020_NC = 9

# The bridge's plane numbering, kept for callers that name a plane; the readback
# entries that would use them are intentionally not bound here.
PLANE_SOURCE_Y, PLANE_SOURCE_UV = 0, 1
PLANE_MASTER_FULL, PLANE_WORKING_MASTER = 2, 3
PLANE_PROXY, PLANE_GUIDANCE = 4, 5
PLANE_WORKING_RESIDUAL, PLANE_COMPOSITED = 6, 7
PLANE_ENCODED_Y, PLANE_ENCODED_UV = 8, 9

# Descriptor format strings, matched to what the downstream consumers already
# check: dlss_gpu_engine.Live takes "rgba16f_cuda" for an FP16 colour and NVOF
# takes "rgba8_cuda" for the motion-only guidance surface. The FP16 scRGB planes
# and the P010 encode planes are the bridge's own; the encode pair has no dict
# consumer in this layer, only a typed shape.
FORMAT_RGBA16F_CUDA = "rgba16f_cuda"
FORMAT_RGBA8_CUDA = "rgba8_cuda"
FORMAT_P010_CUDA = "p010_cuda"


class HdrBridgeError(RuntimeError):
    """A bridge refusal, carrying the native last-error text when there is one."""


# ---------------------------------------------------------------------------
# ctypes mirrors of native/gpu_hdr_scrgb_bridge.h - field for field, so a header
# drift is a wrong struct_size the compiled bridge itself rejects.
# ---------------------------------------------------------------------------

class GpuHdrTapTable(ctypes.Structure):
    """Mirror of GpuHdrTapTable: one row stride shared by both axes, [dst][taps]
    row-major, clamp-to-edge already resolved in the indices."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("taps", ctypes.c_uint32),
        ("vertical_src_len", ctypes.c_uint32),
        ("vertical_dst_len", ctypes.c_uint32),
        ("vertical_indices", ctypes.POINTER(ctypes.c_int32)),
        ("vertical_weights", ctypes.POINTER(ctypes.c_float)),
        ("horizontal_src_len", ctypes.c_uint32),
        ("horizontal_dst_len", ctypes.c_uint32),
        ("horizontal_indices", ctypes.POINTER(ctypes.c_int32)),
        ("horizontal_weights", ctypes.POINTER(ctypes.c_float)),
    ]


class GpuHdrContract(ctypes.Structure):
    """Mirror of GpuHdrContract: the numbers this build applies, reported rather
    than copied from the prose."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("scrgb_per_nit", ctypes.c_double),
        ("nits_per_scrgb", ctypes.c_double),
        ("to_scrgb", ctypes.c_double * 9),
        ("to_bt2020", ctypes.c_double * 9),
        ("pq_m1", ctypes.c_double),
        ("pq_m2", ctypes.c_double),
        ("pq_c1", ctypes.c_double),
        ("pq_c2", ctypes.c_double),
        ("pq_c3", ctypes.c_double),
        ("kr", ctypes.c_double),
        ("kg", ctypes.c_double),
        ("kb", ctypes.c_double),
        ("proxy_knee", ctypes.c_double),
        ("proxy_shoulder_span", ctypes.c_double),
        ("proxy_shoulder_k", ctypes.c_double),
        ("proxy_saturation_relative", ctypes.c_double),
        ("proxy_actionable_relative", ctypes.c_double),
        ("max_stops", ctypes.c_double),
        ("min_slope_per_stop", ctypes.c_double),
    ]

    def as_dict(self):
        """Serializable scalars; the two 9-tuples are exposed separately so a
        report stays flat and small."""
        scalars = {name: getattr(self, name) for name, _ in self._fields_
                   if name not in ("struct_size", "to_scrgb", "to_bt2020")}
        scalars["to_scrgb"] = list(self.to_scrgb)
        scalars["to_bt2020"] = list(self.to_bt2020)
        return scalars


class GpuHdrSourceReport(ctypes.Structure):
    """Mirror of GpuHdrSourceReport."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("support_ceiling_scrgb", ctypes.c_double),
        ("paper_white_scrgb", ctypes.c_double),
        ("nonfinite_samples", ctypes.c_uint32),
        ("negative_components", ctypes.c_uint32),
        ("above_ceiling_components", ctypes.c_uint32),
        ("min_component", ctypes.c_double),
        ("max_component", ctypes.c_double),
        ("pixels", ctypes.c_uint32),
    ]


class GpuHdrProxyReport(ctypes.Structure):
    """Mirror of GpuHdrProxyReport."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("negative_components", ctypes.c_uint32),
        ("pinned_components", ctypes.c_uint32),
        ("dead_zone_components", ctypes.c_uint32),
        ("above_actionable_components", ctypes.c_uint32),
        ("above_saturation_components", ctypes.c_uint32),
        ("min_code", ctypes.c_double),
        ("max_code", ctypes.c_double),
        ("pixels", ctypes.c_uint32),
    ]


class GpuHdrReconstructReport(ctypes.Structure):
    """Mirror of GpuHdrReconstructReport."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("applied_components", ctypes.c_uint32),
        ("gain_cap_components", ctypes.c_uint32),
        ("rejected_pinned_components", ctypes.c_uint32),
        ("rejected_dead_components", ctypes.c_uint32),
        ("rejected_negative_components", ctypes.c_uint32),
        ("nonfinite_components", ctypes.c_uint32),
        ("max_abs_stops", ctypes.c_double),
        ("max_abs_residual_scrgb", ctypes.c_double),
        ("composite_out_of_domain_components", ctypes.c_uint32),
        ("composite_nonfinite_components", ctypes.c_uint32),
        ("composite_min_component", ctypes.c_double),
        ("composite_max_component", ctypes.c_double),
        ("pixels", ctypes.c_uint32),
    ]


class GpuHdrEncodeReport(ctypes.Structure):
    """Mirror of GpuHdrEncodeReport."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("clip_ceiling_scrgb", ctypes.c_double),
        ("clamped_high_components", ctypes.c_uint32),
        ("clamped_low_components", ctypes.c_uint32),
        ("out_of_gamut_components", ctypes.c_uint32),
        ("nonfinite_components", ctypes.c_uint32),
        ("max_linear_component", ctypes.c_double),
        ("max_code_y", ctypes.c_double),
        ("min_code_y", ctypes.c_double),
        ("max_code_uv", ctypes.c_double),
        ("min_code_uv", ctypes.c_double),
    ]


class GpuHdrTransferCounters(ctypes.Structure):
    """Mirror of GpuHdrTransferCounters: device-to-device traffic plus the
    diagnostic image-plane readback counter, so a working bridge has the first six
    rising together. The last field is not a measure of the per-pass counter
    readback - that is scalar host traffic which no field here counts, and a report
    must leave its byte count unanswered."""
    _fields_ = [("struct_size", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "process_count",
            "source_device_copy_bytes",
            "source_device_write_bytes",
            "working_device_write_bytes",
            "guidance_device_write_bytes",
            "composited_device_write_bytes",
            "encode_device_write_bytes",
            "report_readback_bytes",
        )]


class GpuHdrMultiscaleCounters(ctypes.Structure):
    """Mirror of GpuHdrMultiscaleCounters: what the multiscale entries alone did,
    in a struct of its own because GpuHdrTransferCounters' layout is what every
    existing consumer reads and does not change. The same device traffic is in the
    fields above too; this one attributes it per entry and says what last wrote the
    master, so a source report is never read as a measurement of an imported
    plane. Like the other struct it does not measure the per-pass counter
    readback; ``pass_dispatches`` is what lets a caller state it separately."""
    _fields_ = [("struct_size", ctypes.c_uint32),
                ("master_provenance", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "master_import_count",
            "master_import_device_bytes",
            "observe_count",
            "observe_copy_bytes",
            "observe_residual_bytes",
            "composite_import_count",
            "composite_import_bytes",
            "pass_dispatches",
        )]


_REPORT_STRUCTS = {
    "source": (GpuHdrSourceReport, "gpu_hdr_get_source_report"),
    "proxy": (GpuHdrProxyReport, "gpu_hdr_get_proxy_report"),
    "reconstruct": (GpuHdrReconstructReport, "gpu_hdr_get_reconstruct_report"),
    "encode": (GpuHdrEncodeReport, "gpu_hdr_get_encode_report"),
}


# ---------------------------------------------------------------------------
# The exports this layer binds. Pixel readbacks and plane sizing are excluded on
# purpose: they are correctness instrumentation, not the chain, and binding them
# would invite a host-side image path this module must never offer.
# ---------------------------------------------------------------------------

_REQUIRED_EXPORTS = (
    "gpu_hdr_abi_version", "gpu_hdr_fill_contract", "gpu_hdr_open",
    "gpu_hdr_paper_white", "gpu_hdr_decode", "gpu_hdr_downscale",
    "gpu_hdr_proxy", "gpu_hdr_reconstruct", "gpu_hdr_encode",
    "gpu_hdr_get_source_report", "gpu_hdr_get_proxy_report",
    "gpu_hdr_get_reconstruct_report", "gpu_hdr_get_encode_report",
    "gpu_hdr_get_transfer_counters", "gpu_hdr_close", "gpu_hdr_last_error",
)

# name -> (argtypes, restype). void* is c_void_p, out double* / uint* / uint64* are
# pointers to the ctypes scalar, const char* last_error is c_char_p.
_u64p = ctypes.POINTER(ctypes.c_uint64)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_BRIDGE = [
    ("gpu_hdr_abi_version", [], ctypes.c_uint32),
    ("gpu_hdr_fill_contract", [ctypes.POINTER(GpuHdrContract)], None),
    ("gpu_hdr_open", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(GpuHdrTapTable), ctypes.POINTER(GpuHdrTapTable),
        ctypes.c_double, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int32),
    ("gpu_hdr_paper_white", [ctypes.c_void_p], ctypes.c_double),
    ("gpu_hdr_decode", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_double, _u64p, _u32p],
        ctypes.c_int32),
    ("gpu_hdr_downscale", [ctypes.c_void_p, _u64p, _u32p], ctypes.c_int32),
    ("gpu_hdr_proxy", [ctypes.c_void_p, _u64p, _u32p, _u64p, _u32p],
        ctypes.c_int32),
    ("gpu_hdr_reconstruct", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.c_double, _u64p, _u32p], ctypes.c_int32),
    ("gpu_hdr_encode", [ctypes.c_void_p, ctypes.c_double, _u64p, _u32p, _u64p,
        _u32p], ctypes.c_int32),
    ("gpu_hdr_get_source_report", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrSourceReport)], ctypes.c_int32),
    ("gpu_hdr_get_proxy_report", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrProxyReport)], ctypes.c_int32),
    ("gpu_hdr_get_reconstruct_report", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrReconstructReport)], ctypes.c_int32),
    ("gpu_hdr_get_encode_report", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrEncodeReport)], ctypes.c_int32),
    ("gpu_hdr_get_transfer_counters", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrTransferCounters)], ctypes.c_int32),
    ("gpu_hdr_close", [ctypes.c_void_p], None),
    ("gpu_hdr_last_error", [], ctypes.c_char_p),
]

# GpuHdrMultiscaleCounters.master_provenance: what last wrote the bridge's
# full-resolution master. An imported master has no source report - the decode pass
# is the only thing that measures one - so a consumer reads this before treating a
# source report as a statement about the plane the bridge holds now.
MASTER_FROM_NOTHING, MASTER_FROM_DECODE, MASTER_FROM_IMPORT = 0, 1, 2

# The optional multiscale foundation, bound on demand and never as part of the ABI
# gate: an ABI 1 bridge that predates these entries must still open and run the
# single layer, so their absence is a capability answer, not a load failure.
_MULTISCALE_BRIDGE = [
    ("gpu_hdr_multiscale_abi_version", [], ctypes.c_uint32),
    ("gpu_hdr_import_master", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, _u64p, _u32p], ctypes.c_int32),
    ("gpu_hdr_observe", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64, _u64p, _u32p], ctypes.c_int32),
    ("gpu_hdr_set_composite", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64, _u64p, _u32p], ctypes.c_int32),
    ("gpu_hdr_get_multiscale_counters", [ctypes.c_void_p,
        ctypes.POINTER(GpuHdrMultiscaleCounters)], ctypes.c_int32),
]

MULTISCALE_SYMBOLS = tuple(name for name, _, _ in _MULTISCALE_BRIDGE)


def probe_multiscale(library):
    """Ask the bridge, by symbol name, whether it carries the multiscale
    foundation.

    Returns ``{"supported": bool, "version": int, "missing": [...], "reason": str}``
    and never raises, because a caller that only wants the single layer has to be
    able to ask this before deciding what to bind. Speaking ABI 1 implies nothing
    about these entries, and a partial group is reported as missing rather than
    driven half-way.
    """
    missing = []
    for name, argtypes, restype in _MULTISCALE_BRIDGE:
        try:
            entry = getattr(library, name)
        except AttributeError:
            missing.append(name)
            continue
        entry.argtypes = argtypes
        entry.restype = restype
    if missing:
        return {"supported": False, "version": 0, "missing": missing,
                "reason": "gpu_hdr_scrgb_bridge.dll does not export "
                          f"{', '.join(missing)}: this build has no HDR multiscale "
                          "foundation, so an extra layer cannot take a master, an "
                          "observation or a composite"}
    try:
        version = int(library.gpu_hdr_multiscale_abi_version())
    except (TypeError, ValueError):
        version = -1
    if version != GPU_HDR_MULTISCALE_ABI_VERSION:
        return {"supported": False, "version": version, "missing": [],
                "reason": f"the HDR bridge's multiscale foundation reports version "
                          f"{version}, this wrapper speaks "
                          f"{GPU_HDR_MULTISCALE_ABI_VERSION}"}
    return {"supported": True, "version": version, "missing": [], "reason": ""}


# The one loaded, ABI-checked, prototype-bound library (lazy; module import does
# not touch the GPU or any native code).
_library = None


def last_error_text(library):
    """The bridge's own last message, decoded, or an empty string."""
    getter = getattr(library, "gpu_hdr_last_error", None)
    detail = getter() if callable(getter) else None
    if isinstance(detail, bytes):
        return detail.decode("utf-8", "replace")
    return detail or ""


def bind_prototypes(library):
    """Give every bound entry its argtypes/restype, or raise naming what is
    missing. This is the export-presence check the ABI gate depends on."""
    for name, argtypes, restype in _BRIDGE:
        try:
            entry = getattr(library, name)
        except AttributeError:
            raise HdrBridgeError(
                f"gpu_hdr_scrgb_bridge.dll does not export {name!r}; this wrapper "
                f"speaks ABI {GPU_HDR_ABI_VERSION} and cannot run without it")
        entry.argtypes = argtypes
        entry.restype = restype
    return library


def validate_library(library):
    """Reject a bridge that is not this ABI, after binding the prototypes it uses.
    A missing export or an ABI mismatch is a clear error, never a silent fall
    back to a partial chain."""
    try:
        abi = int(library.gpu_hdr_abi_version())
    except AttributeError:
        raise HdrBridgeError(
            "gpu_hdr_scrgb_bridge.dll does not export gpu_hdr_abi_version; cannot "
            "check the ABI before use")
    bind_prototypes(library)
    if abi != GPU_HDR_ABI_VERSION:
        raise HdrBridgeError(
            f"gpu_hdr_scrgb_bridge.dll ABI mismatch: it reports {abi}, this wrapper "
            f"speaks {GPU_HDR_ABI_VERSION}")
    return library


def load():
    """The bundled bridge, loaded once and validated. Import of this module never
    calls here."""
    global _library
    if _library is not None:
        return _library
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    path = bundle_paths.require_native(
        "gpu_hdr_scrgb_bridge.dll", "run the HDR10 scRGB GPU colour bridge")
    library = ctypes.CDLL(path)
    library._dll_path = path
    _library = validate_library(library)
    return _library


def reset_loaded_library():
    """Drop the cached library handle. For tests that inject a fake and must not
    leak it into a later real load."""
    global _library
    _library = None


# ---------------------------------------------------------------------------
# Typed device descriptors. A descriptor is a small, serializable statement of
# what a bridge plane is; ``as_descriptor`` emits the exact field names the
# downstream consumers check (dlss_gpu_engine.Live, NVOF), and nothing here ever
# reinterprets one format as another.
# ---------------------------------------------------------------------------

class DevicePlane:
    """One device-resident plane the bridge owns: a pointer, a byte pitch, and
    the truth about its format."""

    __slots__ = ("format", "pixel_format", "width", "height", "cuda_context",
                 "device_ptr", "pitch_bytes", "bytes_per_pixel", "plane")

    def __init__(self, *, fmt, pixel_format, width, height, cuda_context,
                 device_ptr, pitch_bytes, bytes_per_pixel, plane=None):
        self.format = fmt
        self.pixel_format = pixel_format
        self.width = int(width)
        self.height = int(height)
        self.cuda_context = cuda_context
        self.device_ptr = int(device_ptr)
        self.pitch_bytes = int(pitch_bytes)
        self.bytes_per_pixel = int(bytes_per_pixel)
        self.plane = plane

    def as_descriptor(self):
        """The dict shape ``process_device_*`` and NVOF accept."""
        return {
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "cuda_context": self.cuda_context,
            "device_ptr": self.device_ptr,
            "pitch_bytes": self.pitch_bytes,
        }

    def as_dict(self):
        return {
            "format": self.format,
            "pixel_format": self.pixel_format,
            "width": self.width,
            "height": self.height,
            "pitch_bytes": self.pitch_bytes,
            "bytes_per_pixel": self.bytes_per_pixel,
            "plane": self.plane,
        }

    def __repr__(self):
        return (f"DevicePlane({self.pixel_format}, {self.width}x{self.height}, "
                f"pitch={self.pitch_bytes}, bpp={self.bytes_per_pixel}, "
                f"ptr=0x{self.device_ptr:x})")


class P010Output:
    """The two semi-planar P010 planes the encode hands to the encoder."""

    __slots__ = ("y", "uv")

    def __init__(self, y, uv):
        self.y = y
        self.uv = uv

    def as_dict(self):
        return {"format": FORMAT_P010_CUDA, "y": self.y.as_dict(),
                "uv": self.uv.as_dict()}

    def __repr__(self):
        return f"P010Output(y={self.y!r}, uv={self.uv!r})"


# ---------------------------------------------------------------------------
# CPU tap plans -> GpuHdrTapTable. Reuses the pipeline reference plans and the
# header's shared-stride serialization; introduces no rounding of its own.
# ---------------------------------------------------------------------------

def build_tap_table(vertical_plan, horizontal_plan):
    """Widen two pipeline.AxisPlans into one GpuHdrTapTable.

    The bridge reads exactly ``taps`` columns per row with one stride for both
    axes, so the narrower axis is padded with weight 0.0 at a valid (already
    clamp-to-edge) index rather than dropped. A plan wider than GPU_HDR_MAX_TAPS
    is refused here, never truncated: a cropped plan is a different filter than
    the CPU reference.

    Returns ``(table, keepalive)``. The table holds raw pointers into the arrays,
    so the caller must keep ``keepalive`` alive for as long as the bridge can read
    from the plan - which is its whole open life.
    """
    taps = max(int(vertical_plan.max_taps), int(horizontal_plan.max_taps))
    if taps < 1:
        raise HdrBridgeError("a tap plan must carry at least one tap column")
    if taps > GPU_HDR_MAX_TAPS:
        raise HdrBridgeError(
            f"the {vertical_plan.kernel}/{horizontal_plan.kernel} plan needs "
            f"{taps} tap columns; the bridge accepts at most {GPU_HDR_MAX_TAPS}")

    def axis(plan):
        if int(plan.max_taps) > taps:
            raise HdrBridgeError(
                f"{plan.kernel} {plan.src_len}->{plan.dst_len} carries "
                f"{plan.max_taps} columns and the table stride is {taps}; refusing "
                f"to drop taps")
        if int(plan.dst_len) < 1:
            raise HdrBridgeError(f"{plan.kernel} plan has a zero-length destination")
        indices = np.zeros((plan.dst_len, taps), np.int32)
        weights = np.zeros((plan.dst_len, taps), np.float32)
        indices[:, :plan.max_taps] = plan.indices
        weights[:, :plan.max_taps] = plan.weights
        if int(plan.src_len) < 1:
            raise HdrBridgeError(f"{plan.kernel} plan has a zero-length source")
        if indices.size and (int(indices.min()) < 0
                             or int(indices.max()) >= int(plan.src_len)):
            raise HdrBridgeError(
                f"{plan.kernel} plan index leaves [0, {plan.src_len})")
        return indices, weights

    vertical_indices, vertical_weights = axis(vertical_plan)
    horizontal_indices, horizontal_weights = axis(horizontal_plan)

    table = GpuHdrTapTable()
    table.struct_size = ctypes.sizeof(GpuHdrTapTable)
    table.taps = taps
    table.vertical_src_len = int(vertical_plan.src_len)
    table.vertical_dst_len = int(vertical_plan.dst_len)
    table.vertical_indices = vertical_indices.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))
    table.vertical_weights = vertical_weights.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    table.horizontal_src_len = int(horizontal_plan.src_len)
    table.horizontal_dst_len = int(horizontal_plan.dst_len)
    table.horizontal_indices = horizontal_indices.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))
    table.horizontal_weights = horizontal_weights.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    keepalive = (vertical_indices, vertical_weights, horizontal_indices,
                 horizontal_weights)
    return table, keepalive


def make_hdr_plans(full_width, full_height, working_width, working_height):
    """The two production plans for one extent pair, as ``build_tap_table``
    expects them: colour down (Lanczos-2, full->working) and residual up
    (Catmull-Rom, working->full). Imported lazily so importing this module pulls
    in no GPU code."""
    import pipeline
    down = (pipeline._build_lanczos2_axis_plan(full_height, working_height),
            pipeline._build_lanczos2_axis_plan(full_width, working_width))
    up = (pipeline._build_catmull_rom_axis_plan(working_height, full_height),
          pipeline._build_catmull_rom_axis_plan(working_width, full_width))
    return down, up


# ---------------------------------------------------------------------------
# Validation helpers. These check the caller's declarations; they never re-derive
# a ceiling from metadata and never touch paper white after open.
# ---------------------------------------------------------------------------

def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HdrBridgeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number) or number <= 0.0:
        raise HdrBridgeError(f"{name} must be finite and positive, got {number!r}")
    return number


def _paper_white_valid(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HdrBridgeError(f"paper_white_scrgb must be a number, got {value!r}")
    if not math.isfinite(number) or number < 1.0:
        raise HdrBridgeError(
            "paper_white_scrgb must be finite and at least 1.0 (it is the proxy's "
            f"diffuse white in linear scRGB); got {number!r}")
    return number


def _int_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise HdrBridgeError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise HdrBridgeError(f"{name} must be at least 1, got {value}")
    return value


def _as_int32(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise HdrBridgeError(f"{name} must be an int32 enum value, got {value!r}")
    if not (-0x80000000 <= value <= 0x7FFFFFFF):
        raise HdrBridgeError(f"{name} is outside int32: {value}")
    return value


def _to_int_ptr(value):
    """A borrowed handle as a plain int, accepting an int, a ctypes c_void_p, a
    c_uint64 device pointer, or None for null."""
    if value is None:
        return 0
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    return int(getattr(value, "value", value))


def _plane_field(source, name):
    """One field of a plane, from a DevicePlane or the dict shape a producer
    hands over. Absent is None, and each caller says what a missing field means."""
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _require_scrgb16f_plane(source, *, extent, cuda_context, label):
    """Refuse a borrowed full-resolution linear scRGB FP16 plane before the bridge
    ever sees it, and return the three numbers the call needs.

    A bare CUDA address carries no format, extent or context, so the caller
    declares them and this is where a declaration that cannot be a pristine HDR
    master is rejected: an RGBA8 or P010 plane, another extent, a pitch that cannot
    hold a row, a pitch a pitched copy cannot walk, or a plane that lives in some
    other CUDA context. Nothing here reinterprets a format - an imported plane is
    copied, so if it were not already FP16 scRGB the bridge would be storing 8-bit
    samples as if they were HDR colour, which is the one thing a second layer must
    never do.
    """
    if source is None:
        raise HdrBridgeError(f"{label} is required")
    device_ptr = _to_int_ptr(_plane_field(source, "device_ptr"))
    if not device_ptr:
        raise HdrBridgeError(f"{label} has no device pointer")
    fmt = _plane_field(source, "format")
    if fmt != FORMAT_RGBA16F_CUDA:
        raise HdrBridgeError(
            f"{label} must be {FORMAT_RGBA16F_CUDA}, got {fmt!r}: only a linear "
            f"scRGB FP16 plane is an HDR master, and an {FORMAT_RGBA8_CUDA} or "
            f"{FORMAT_P010_CUDA} plane would be copied in as if it were one")
    pixel_format = _plane_field(source, "pixel_format")
    if pixel_format and ("R8G8B8A8" in str(pixel_format)
                         or "P010" in str(pixel_format)):
        raise HdrBridgeError(
            f"{label} declares {pixel_format}, which is not 16-bit float per "
            f"channel; the format string and the pixel format may not disagree")
    bytes_per_pixel = _plane_field(source, "bytes_per_pixel")
    if bytes_per_pixel is not None and int(bytes_per_pixel) != 8:
        raise HdrBridgeError(
            f"{label} carries {bytes_per_pixel} bytes per sample; R16G16B16A16_"
            f"FLOAT is 8")
    width = _plane_field(source, "width")
    height = _plane_field(source, "height")
    if width is None or height is None:
        raise HdrBridgeError(
            f"{label} must declare its extent; the bridge copies a whole plane and "
            f"cannot size one from an address")
    if (int(width), int(height)) != tuple(extent):
        raise HdrBridgeError(
            f"{label} extent [{width}, {height}] does not match the full extent "
            f"{list(extent)}")
    pitch = _plane_field(source, "pitch_bytes")
    if pitch is None:
        raise HdrBridgeError(f"{label} must declare its row pitch in bytes")
    pitch = int(pitch)
    row_bytes = int(width) * 8
    if pitch < row_bytes:
        raise HdrBridgeError(
            f"{label} pitch {pitch} is under the {row_bytes} bytes an FP16 RGBA row "
            f"of {width} samples carries")
    if pitch % 16:
        raise HdrBridgeError(
            f"{label} pitch {pitch} is not a multiple of 16, which is what a "
            f"pitched device-to-device copy requires")
    source_context = _to_int_ptr(_plane_field(source, "cuda_context"))
    if source_context != int(cuda_context):
        raise HdrBridgeError(
            f"{label} lives in CUDA context 0x{source_context:x}, not the bridge's "
            f"0x{int(cuda_context):x}; copying across contexts would read memory "
            f"this bridge has no claim on")
    return {"device_ptr": device_ptr, "width": int(width),
            "height": int(height), "pitch_bytes": pitch}


class GpuHdrBridge:
    """One open GpuHdrBridge, borrowed device and CUDA context.

    The device and context are borrowed: the caller owns them and this object
    never destroys them. It owns the bridge handle instead - closing it exactly
    once, idempotently, and refusing every stage call once closed. It is an
    explicit-lifetime object: no destructor touches native code on an arbitrary
    thread, so cleanup is either ``close()`` or a ``with`` block.

    Like the rest of the chain it is a synchronous, single-thread owner. The
    thread that opened the bridge is recorded and every stage must be called from
    that thread; the caller is responsible for pushing the bridge's CUDA context
    current on that thread before calling, exactly as the reference chain does.
    """

    _open_symbol = "gpu_hdr_open"
    _decode_symbol = "gpu_hdr_decode"
    _contract_symbol = "gpu_hdr_fill_contract"
    _linear_label = "scRGB"

    def __init__(self, *, d3d12_device, cuda_context, full_width, full_height,
                 working_width, working_height, paper_white_scrgb, library=None, purpose="full"):
        purposes = {"full": 0, "guidance": 1, "decode": 2, "encode": 3,
                    "decode422": 4, "encode422": 5}
        if purpose not in purposes:
            raise HdrBridgeError(f"unknown HDR bridge purpose {purpose!r}")
        self.purpose = purpose
        # Validate whatever library we are about to drive, injected or bundled:
        # a bridge that is not ABI 1, or is missing an entry this wrapper binds,
        # is refused here rather than mid-frame.
        self._lib = validate_library(
            library if library is not None else load())

        full_width = _int_positive(full_width, "full_width")
        full_height = _int_positive(full_height, "full_height")
        working_width = _int_positive(working_width, "working_width")
        working_height = _int_positive(working_height, "working_height")
        # The downscale plan runs full->working and the reconstruct plan the other
        # way, so a working extent wider than the full one has no plan that fits.
        # This is a plan-shape requirement, not a scale-policy judgement.
        if working_width > full_width or working_height > full_height:
            raise HdrBridgeError(
                f"working extent {working_width}x{working_height} must not exceed "
                f"the full extent {full_width}x{full_height}; the tap plans down "
                f"and back up")
        self._full = (full_width, full_height)
        self._working = (working_width, working_height)
        self._configured_paper_white = _paper_white_valid(paper_white_scrgb)

        down_axes, up_axes = make_hdr_plans(full_width, full_height,
                                            working_width, working_height)
        self._downscale_plan, down_keep = build_tap_table(*down_axes)
        self._reconstruct_plan, up_keep = build_tap_table(*up_axes)
        # The only backing store the tables point into; held for the bridge's life.
        self._plan_keepalive = (down_keep, up_keep)

        self._handle = ctypes.c_void_p()
        self._cuda_context = _to_int_ptr(cuda_context)
        self._bind_colour_mode()
        open_entry = getattr(self._lib, self._open_symbol)
        purpose_args = ()
        if purpose != "full":
            symbol = self._open_symbol + "_purpose"
            try:
                open_entry = getattr(self._lib, symbol)
            except AttributeError as error:
                raise HdrBridgeError(f"missing {symbol}; rebuild the colour bridge for resource purposes") from error
            signature = next(args for name, args, _ in _BRIDGE if name == "gpu_hdr_open")
            open_entry.argtypes = signature[:-1] + [ctypes.c_uint32, signature[-1]]
            open_entry.restype = ctypes.c_int32
            purpose_args = (purposes[purpose],)
        code = open_entry(
            ctypes.c_void_p(_to_int_ptr(d3d12_device)),
            ctypes.c_void_p(self._cuda_context),
            full_width, full_height, working_width, working_height,
            ctypes.byref(self._downscale_plan),
            ctypes.byref(self._reconstruct_plan),
            self._configured_paper_white, *purpose_args, ctypes.byref(self._handle))
        if code:
            detail = last_error_text(self._lib)
            # open reported failure; free whatever it may still have set, then say why.
            self._release_handle()
            raise HdrBridgeError(f"gpu_hdr_open failed ({code}): {detail}")

        self._owner_thread = threading.get_ident()
        self._contract = None
        # The capability probe, done once on first use: an ABI 1 bridge may or may
        # not carry the multiscale group and the answer is a symbol lookup.
        self._multiscale = None
        if not self._handle.value:
            self._plan_keepalive = None
            raise HdrBridgeError(
                f"gpu_hdr_open reported success ({code}) without a bridge handle")

        # The bridge reports the paper white it actually applies. A non-finite or
        # different value means every downstream number would be wrong while the
        # chain looked healthy, so the handle is given back and the open refused;
        # the requested value is never rewritten to match.
        try:
            effective = float(self._lib.gpu_hdr_paper_white(self._handle))
        except (TypeError, ValueError):
            self._release_handle()
            raise HdrBridgeError(
                "gpu_hdr_paper_white did not report a number for the open bridge")
        if not math.isfinite(effective) or effective != self._configured_paper_white:
            self._release_handle()
            raise HdrBridgeError(
                f"the bridge applies paper white {effective!r}, not the requested "
                f"{self._configured_paper_white!r}: {last_error_text(self._lib)}")
        self._effective_paper_white = effective

    def _bind_colour_mode(self):
        """HDR entries are bound by validate_library; additive modes bind separately."""

    # -- lifetime -----------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def is_alive(self):
        return bool(self._handle.value)

    def _release_handle(self):
        """Give the native handle back and drop the plan backing store. The only
        place that calls gpu_hdr_close."""
        if self._handle.value:
            self._lib.gpu_hdr_close(self._handle)
        self._handle = ctypes.c_void_p()
        self._plan_keepalive = None

    def close(self):
        """Close once and forget the handle; safe to call again.

        The wrong thread is refused before the native call: the bridge's CUDA
        context is current on the thread that opened it, so closing elsewhere
        would destroy the handle through a context that does not own it. An
        already-closed bridge stays a no-op, including from another thread.
        """
        if not self._handle.value:
            return
        if threading.get_ident() != self._owner_thread:
            raise HdrBridgeError(
                "the HDR bridge must be closed from the thread that opened it")
        self._release_handle()

    def _require_open(self):
        if not self._handle.value:
            raise HdrBridgeError("the HDR bridge is closed")
        if threading.get_ident() != self._owner_thread:
            raise HdrBridgeError(
                "the HDR bridge must be used from the thread that opened it")

    def _fail(self, stage, code):
        return HdrBridgeError(
            f"{stage} failed ({code}): {last_error_text(self._lib)}")

    # -- fixed configuration, read-only -------------------------------------

    @property
    def full_extent(self):
        return self._full

    @property
    def working_extent(self):
        return self._working

    @property
    def configured_paper_white(self):
        """The paper white this bridge was opened with."""
        return self._configured_paper_white

    @property
    def paper_white(self):
        """The value the bridge reports it applies, read from the open bridge."""
        return self._effective_paper_white

    @property
    def contract(self):
        """The numbers this build applies, filled once and cached as a dict."""
        if self._contract is None:
            struct = GpuHdrContract(ctypes.sizeof(GpuHdrContract))
            getattr(self._lib, self._contract_symbol)(ctypes.byref(struct))
            self._contract = struct.as_dict()
        return self._contract

    # -- the chain ----------------------------------------------------------

    def decode(self, *, p010_y_ptr, p010_y_pitch, p010_uv_ptr, p010_uv_pitch,
               color_range, color_primaries, color_transfer, color_space,
               support_ceiling_scrgb):
        """P010 -> full-resolution linear scRGB FP16. The AVColor* enums are the
        source's own, passed through for the bridge to refuse if they are not
        limited or full-range BT.2020 NC with ST 2084 (SDR10 remains limited
        BT.709). ``support_ceiling_scrgb`` is the
        caller's declared domain ceiling; it is validated as finite and positive
        and never derived from anything here."""
        self._require_open()
        ceiling = _positive_finite(support_ceiling_scrgb, "support_ceiling_scrgb")
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = getattr(self._lib, self._decode_symbol)(
            self._handle, ctypes.c_uint64(_to_int_ptr(p010_y_ptr)),
            ctypes.c_size_t(_to_int_ptr(p010_y_pitch)),
            ctypes.c_uint64(_to_int_ptr(p010_uv_ptr)),
            ctypes.c_size_t(_to_int_ptr(p010_uv_pitch)),
            _as_int32(color_range, "color_range"),
            _as_int32(color_primaries, "color_primaries"),
            _as_int32(color_transfer, "color_transfer"),
            _as_int32(color_space, "color_space"),
            ceiling, ctypes.byref(out_ptr), ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_decode", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA, pixel_format=f"R16G16B16A16_FLOAT ({self._linear_label})",
            width=self._full[0], height=self._full[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_MASTER_FULL)

    def downscale(self):
        """The pristine working master: Lanczos-2 of the full-resolution scRGB,
        signed and unclamped. Feature 18 never reads this plane."""
        self._require_open()
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = self._lib.gpu_hdr_downscale(self._handle, ctypes.byref(out_ptr),
                                           ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_downscale", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA, pixel_format=f"R16G16B16A16_FLOAT ({self._linear_label})",
            width=self._working[0], height=self._working[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_WORKING_MASTER)

    def proxy(self):
        """The bounded display proxy and the motion-only guidance surface, from
        one pass. Two separate descriptors, never conflated: the proxy is FP16 and
        is what Feature 18 sees; the guidance is RGBA8 and only NVOF reads it.
        Handing the guidance to a colour output or treating it as FP16 is a caller
        error this return shape makes hard to commit."""
        self._require_open()
        proxy_ptr, proxy_pitch = ctypes.c_uint64(), ctypes.c_uint32()
        guidance_ptr, guidance_pitch = ctypes.c_uint64(), ctypes.c_uint32()
        code = self._lib.gpu_hdr_proxy(
            self._handle, ctypes.byref(proxy_ptr), ctypes.byref(proxy_pitch),
            ctypes.byref(guidance_ptr), ctypes.byref(guidance_pitch))
        if code:
            raise self._fail("gpu_hdr_proxy", code)
        proxy = DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA,
            pixel_format="R16G16B16A16_FLOAT (display proxy)",
            width=self._working[0], height=self._working[1],
            cuda_context=self._cuda_context, device_ptr=proxy_ptr.value,
            pitch_bytes=proxy_pitch.value, bytes_per_pixel=8, plane=PLANE_PROXY)
        guidance = DevicePlane(
            fmt=FORMAT_RGBA8_CUDA, pixel_format="R8G8B8A8_UNORM (motion-only)",
            width=self._working[0], height=self._working[1],
            cuda_context=self._cuda_context, device_ptr=guidance_ptr.value,
            pitch_bytes=guidance_pitch.value, bytes_per_pixel=4,
            plane=PLANE_GUIDANCE)
        return proxy, guidance

    def reconstruct(self, dlss_output, fence=None, fence_value=None,
                    residual_strength=1.0):
        """Feature 18 RGBA16F output -> composited full-resolution scRGB.

        ``dlss_output`` is either the dict ``process_device_color_flow_resource``
        returned (validated for format and extent when it carries those fields) or
        a raw texture pointer, in which case ``fence`` and ``fence_value`` are
        required. The result is the bridge's full-size FP16 composite plane; there
        is no fallback and no re-derived shader here.
        """
        self._require_open()
        texture, fence_ptr, value = self._producer_texture(
            dlss_output, fence, fence_value, extent=self._working,
            extent_name="working", label="Feature 18 output",
            role="reconstruction")
        strength = float(residual_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise HdrBridgeError(
                f"residual_strength must be finite and non-negative, got "
                f"{residual_strength!r}")
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = self._lib.gpu_hdr_reconstruct(
            self._handle, ctypes.c_void_p(_to_int_ptr(texture)),
            ctypes.c_void_p(_to_int_ptr(fence_ptr)),
            ctypes.c_uint64(_to_int_ptr(value)),
            strength, ctypes.byref(out_ptr), ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_reconstruct", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA,
            pixel_format=f"R16G16B16A16_FLOAT (composited {self._linear_label})",
            width=self._full[0], height=self._full[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_COMPOSITED)

    def encode(self, *, clip_ceiling_scrgb):
        """Composited full-resolution scRGB -> PQ-coded BT.2020 limited-range P010
        at full size, returned as two CUDA planes for the encoder. ``clip_ceiling``
        is the top of the clamp interval in linear scRGB (125.0 is the ST 2084
        top); it is validated finite and positive and is never taken from MaxCLL."""
        self._require_open()
        ceiling = _positive_finite(clip_ceiling_scrgb, "clip_ceiling_scrgb")
        y_ptr, y_pitch = ctypes.c_uint64(), ctypes.c_uint32()
        uv_ptr, uv_pitch = ctypes.c_uint64(), ctypes.c_uint32()
        code = self._lib.gpu_hdr_encode(self._handle, ceiling,
                                        ctypes.byref(y_ptr),
                                        ctypes.byref(y_pitch),
                                        ctypes.byref(uv_ptr),
                                        ctypes.byref(uv_pitch))
        if code:
            raise self._fail("gpu_hdr_encode", code)
        is_p210 = self.purpose == "encode422"
        y = DevicePlane(fmt=FORMAT_P010_CUDA, pixel_format=("P210LE" if is_p210 else "P010LE") + " luma",
                        width=self._full[0], height=self._full[1],
                        cuda_context=self._cuda_context, device_ptr=y_ptr.value,
                        pitch_bytes=y_pitch.value, bytes_per_pixel=2,
                        plane=PLANE_ENCODED_Y)
        # Both formats store one interleaved Cb,Cr 10-bit pair per two pixels.
        # P210 retains one pair on every row; P010 has one per 2x2 cell.
        uv = DevicePlane(fmt=FORMAT_P010_CUDA,
                         pixel_format=("P210LE" if is_p210 else "P010LE") +
                            (" chroma pairs (Cb/Cr interleaved, 4:2:2)" if is_p210 else
                             " chroma pairs (Cb/Cr interleaved, 4:2:0)"),
                         width=self._full[0] // 2,
                         height=self._full[1] if is_p210 else self._full[1] // 2,
                         cuda_context=self._cuda_context,
                         device_ptr=uv_ptr.value, pitch_bytes=uv_pitch.value,
                         bytes_per_pixel=4, plane=PLANE_ENCODED_UV)
        return P010Output(y, uv)

    # -- the multiscale foundation (optional capability group) --------------
    #
    # Three GPU-only abilities, each of which keeps this bridge's own layer
    # resources independent of every other layer's:
    #
    #   * ``import_master`` puts a borrowed pristine full-resolution linear scRGB
    #     FP16 plane into this bridge's master, by device-to-device copy and with
    #     no colour decision of its own, so an extra layer starts from a master the
    #     job already decoded instead of decoding the source again;
    #   * ``observe`` returns the signed working residual the existing reconstruction
    #     measures - the same kPassResidual helper gpu_hdr_reconstruct runs, with no
    #     upscale, no strength and no composite;
    #   * ``set_composite`` puts a fused full-resolution FP16 texture into the
    #     composited plane so the existing encode stage writes PQ from it.
    #
    # Units throughout are the bridge's: linear scRGB, 1.0 == 80 cd/m2, and the
    # residual is the same signed scRGB difference the single-layer composite would
    # have upsampled. Borrowed planes and textures are read or copied inside the
    # call and never retained.

    @property
    def multiscale_capability(self):
        """What the bridge said, by symbol name, about the multiscale group:
        ``{"supported", "version", "missing", "reason"}``. Probed once per bridge,
        and never inferred from ABI 1."""
        if self._multiscale is None:
            self._multiscale = probe_multiscale(self._lib)
        return self._multiscale

    @property
    def has_multiscale(self):
        return bool(self.multiscale_capability["supported"])

    def _require_multiscale(self):
        capability = self.multiscale_capability
        if not capability["supported"]:
            raise HdrBridgeError(capability["reason"])
        return capability

    def import_master(self, source):
        """Take this layer's main colour from a borrowed pristine master.

        ``source`` is a full-resolution linear scRGB FP16 plane - a ``DevicePlane``
        as another bridge's ``decode()`` returned it, or the dict shape a producer
        describes it with. The bridge copies it whole into its own master, so the
        caller keeps ownership and may reuse or free the source as soon as this
        returns.

        The completion rule is the caller's to keep and the bridge cannot check it:
        the plane must already be fully written and device-visible. A plane handed
        back by ``decode()`` is - the decode pass waits its own fence before that
        call returns - and so is any producer that blocks until its GPU work has
        retired. There is no fence argument because a CUDA device pointer has no
        fence to name, and this entry will not invent one.

        Nothing is done to the samples. The layer then runs ``downscale``, ``proxy``
        and ``observe`` on its own working resources, as a single layer does.
        """
        self._require_open()
        self._require_multiscale()
        plane = _require_scrgb16f_plane(source, extent=self._full,
                                        cuda_context=self._cuda_context,
                                        label="imported master")
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = self._lib.gpu_hdr_import_master(
            self._handle, ctypes.c_uint64(plane["device_ptr"]), plane["width"],
            plane["height"], plane["pitch_bytes"], ctypes.byref(out_ptr),
            ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_import_master", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA, pixel_format=f"R16G16B16A16_FLOAT ({self._linear_label})",
            width=self._full[0], height=self._full[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_MASTER_FULL)

    def observe(self, dlss_output, fence=None, fence_value=None):
        """This layer's Feature 18 output -> the signed working residual.

        Returns the bridge's own working-resolution FP16 plane: RGB in linear scRGB
        units holding reconstruct(working master) - working master, so a component
        is legitimately negative and nothing is normalised into [0, 1]. Alpha is
        0.0 in every sample - padding, not coverage.

        No Catmull-Rom upscale, no ``residual_strength``, no composite and no
        encode: that is the fused decision, made once over all layers. The residual
        is measured against the pristine working master this bridge derived from the
        master it holds now, which is why the native entry refuses an observation
        whose downscale belongs to an older master, or whose proxy read a working
        plane a later downscale replaced.

        ``dlss_output`` is the dict a Feature 18 host returned, or a raw texture
        pointer with ``fence`` and ``fence_value``, exactly as ``reconstruct`` takes
        them.
        """
        self._require_open()
        self._require_multiscale()
        texture, fence_ptr, value = self._producer_texture(
            dlss_output, fence, fence_value, extent=self._working,
            extent_name="working", label="Feature 18 output",
            role="observation")
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = self._lib.gpu_hdr_observe(
            self._handle, ctypes.c_void_p(_to_int_ptr(texture)),
            ctypes.c_void_p(_to_int_ptr(fence_ptr)),
            ctypes.c_uint64(_to_int_ptr(value)), ctypes.byref(out_ptr),
            ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_observe", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA,
            pixel_format="R16G16B16A16_FLOAT (signed working residual, linear "
                         "scRGB; alpha 0.0 padding)",
            width=self._working[0], height=self._working[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_WORKING_RESIDUAL)

    def set_composite(self, composite_output, fence=None, fence_value=None):
        """The fused full-resolution composite -> this bridge's composited plane.

        ``composite_output`` is a full-extent R16G16B16A16_FLOAT texture made by the
        same D3D12 device, with the producer's ``fence`` and ``fence_value``: the
        solver writes it asynchronously on another queue, so the fence is required
        and a missing one is refused here and by the bridge, not assumed complete.
        The copy is the same placed-footprint device-to-device mechanism the Feature
        18 output comes in by, one extent up, and no pixel crosses to the host.

        The values are stored as they arrive; ``encode`` stays the only stage that
        clamps, and this adds no tone map, PQ maths, colour claim or encoder.
        """
        self._require_open()
        self._require_multiscale()
        texture, fence_ptr, value = self._producer_texture(
            composite_output, fence, fence_value, extent=self._full,
            extent_name="full", label="fused composite", role="composite import")
        if fence_ptr is None or value is None:
            raise HdrBridgeError(
                "the fused composite needs its producer fence and value: the solver "
                "is still writing it on another queue")
        out_ptr = ctypes.c_uint64()
        out_pitch = ctypes.c_uint32()
        code = self._lib.gpu_hdr_set_composite(
            self._handle, ctypes.c_void_p(_to_int_ptr(texture)),
            ctypes.c_void_p(_to_int_ptr(fence_ptr)),
            ctypes.c_uint64(_to_int_ptr(value)), ctypes.byref(out_ptr),
            ctypes.byref(out_pitch))
        if code:
            raise self._fail("gpu_hdr_set_composite", code)
        return DevicePlane(
            fmt=FORMAT_RGBA16F_CUDA,
            pixel_format=f"R16G16B16A16_FLOAT (composited {self._linear_label})",
            width=self._full[0], height=self._full[1],
            cuda_context=self._cuda_context, device_ptr=out_ptr.value,
            pitch_bytes=out_pitch.value, bytes_per_pixel=8,
            plane=PLANE_COMPOSITED)

    def multiscale_counters(self):
        """The multiscale foundation's own attribution, as the ABI exposes it.

        The same device traffic also lands in ``transfer_counters`` - an imported
        master is a full-resolution FP16 write, an observation's residual is working
        traffic, an imported composite is a full-resolution FP16 write - so the two
        reports add up instead of being two versions of one number.
        ``pass_dispatches`` is how many shader dispatches these entries ran, which is
        how a caller states the scalar counter readback each dispatch crosses
        separately rather than calling it zero.
        """
        self._require_open()
        self._require_multiscale()
        counters = GpuHdrMultiscaleCounters(
            ctypes.sizeof(GpuHdrMultiscaleCounters))
        code = self._lib.gpu_hdr_get_multiscale_counters(self._handle,
                                                         ctypes.byref(counters))
        if code:
            raise self._fail("gpu_hdr_get_multiscale_counters", code)
        return {name: getattr(counters, name) for name, _ in
                GpuHdrMultiscaleCounters._fields_ if name != "struct_size"}

    def _producer_texture(self, output, fence, fence_value, *, extent,
                          extent_name, label, role):
        """Pull (texture, fence, fence_value) out of a borrowed D3D12 producer
        output, checking the descriptor when it declares its format and extent.

        One rule for every texture the bridge copies in - the Feature 18 output and
        the fused composite: R16G16B16A16_FLOAT, this device, the extent the call
        is defined over. ``require_fence`` is the bridge's own business and is not
        guessed at here: a descriptor without one still reaches the entry that
        refuses it.
        """
        if isinstance(output, dict):
            if ("d3d12_resource" not in output or "d3d12_fence" not in output
                    or "fence_value" not in output):
                raise HdrBridgeError(
                    f"{label} descriptor must carry d3d12_resource, d3d12_fence "
                    f"and fence_value")
            fmt = output.get("dxgi_format")
            if fmt is not None and int(fmt) != DXGI_R16G16B16A16_FLOAT:
                raise HdrBridgeError(
                    f"{label} must be R16G16B16A16_FLOAT (dxgi {DXGI_R16G16B16A16_FLOAT}), "
                    f"got {fmt}; the {role} never reads an 8-bit colour")
            width, height = output.get("width"), output.get("height")
            if width is not None and height is not None and \
                    (int(width), int(height)) != tuple(extent):
                raise HdrBridgeError(
                    f"{label} extent [{width}, {height}] does not match "
                    f"the {extent_name} extent {list(extent)}")
            return (output["d3d12_resource"], output["d3d12_fence"],
                    output["fence_value"])
        if output is None:
            raise HdrBridgeError(f"a {label} texture is required")
        if fence is None or fence_value is None:
            raise HdrBridgeError(
                "a raw texture needs its fence and fence_value")
        return output, fence, fence_value

    # -- reports (small structs only, never pixels) -------------------------
    # Each of these is answered on the host out of the snapshot the pass itself
    # took when it copied its counter run into the CPU-visible readback buffer, so
    # a report arriving at all is evidence that those 256 bytes-per-dispatch of
    # scalars crossed. Nothing here re-reads a plane to produce them.

    def _read_report(self, kind):
        self._require_open()
        struct_cls, getter_name = _REPORT_STRUCTS[kind]
        struct = struct_cls(ctypes.sizeof(struct_cls))
        code = getattr(self._lib, getter_name)(self._handle, ctypes.byref(struct))
        if code:
            raise self._fail(getter_name, code)
        return {name: getattr(struct, name) for name, _ in struct_cls._fields_
                if name != "struct_size"}

    def source_report(self):
        return self._read_report("source")

    def proxy_report(self):
        return self._read_report("proxy")

    def reconstruct_report(self):
        return self._read_report("reconstruct")

    def encode_report(self):
        return self._read_report("encode")

    def allocated_device_bytes(self):
        """Actual DEFAULT heap allocation sizes, excluding CPU counter heaps."""
        self._require_open()
        entry = getattr(self._lib, "gpu_hdr_allocated_device_bytes", None)
        if entry is None:
            return None
        entry.argtypes, entry.restype = [ctypes.c_void_p], ctypes.c_uint64
        return int(entry(self._handle))

    def transfer_counters(self):
        """The device-traffic counters as a dict. Every field is what the ABI
        actually exposes; nothing here is a defaulted zero standing in for a
        counter the bridge does not carry. The dict is not a complete account of
        what crossed to the host: the per-pass counter readback is in none of these
        fields, and the caller who needs that boundary must report it as unmeasured
        rather than as a zero read out of here."""
        self._require_open()
        counters = GpuHdrTransferCounters(ctypes.sizeof(GpuHdrTransferCounters))
        code = self._lib.gpu_hdr_get_transfer_counters(self._handle,
                                                       ctypes.byref(counters))
        if code:
            raise self._fail("gpu_hdr_get_transfer_counters", code)
        return {name: getattr(counters, name) for name, _ in
                GpuHdrTransferCounters._fields_ if name != "struct_size"}


class GpuSdr10Bridge(GpuHdrBridge):
    """P010 BT.709 SDR → relative linear Rec.709 FP16, using shared GPU stages.

    The reference white is fixed at 1.0 for the entire job. These numbers are
    relative SDR values, not absolute HDR luminance or PQ codes. The native
    additive entries refuse a different transfer, matrix, range or primaries.
    """
    _open_symbol = "gpu_sdr10_open"
    _decode_symbol = "gpu_sdr10_decode"
    _contract_symbol = "gpu_sdr10_contract"
    _linear_label = "relative linear Rec.709 SDR"

    def __init__(self, **kwargs):
        if float(kwargs.get("paper_white_scrgb", 1.0)) != 1.0:
            raise HdrBridgeError("SDR10 uses fixed relative reference white 1.0")
        kwargs["paper_white_scrgb"] = 1.0
        super().__init__(**kwargs)

    def _bind_colour_mode(self):
        signatures = {name: (args, result) for name, args, result in _BRIDGE}
        for symbol, original in ((self._open_symbol, "gpu_hdr_open"),
                                 (self._decode_symbol, "gpu_hdr_decode"),
                                 (self._contract_symbol, "gpu_hdr_fill_contract")):
            try:
                entry = getattr(self._lib, symbol)
            except AttributeError as error:
                raise HdrBridgeError(f"missing SDR10 native entry {symbol}") from error
            entry.argtypes, entry.restype = signatures[original]
        if not hasattr(self._lib, 'gpu_sdr10_nr_color_abi_version') or self._lib.gpu_sdr10_nr_color_abi_version() != 1:
            raise HdrBridgeError('SDR10 NR requires the updated BT.709 colour bridge')
        self._lib.gpu_sdr10_reconstruct.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
        self._lib.gpu_sdr10_reconstruct.restype = ctypes.c_int32

    def reconstruct(self, dlss_output, fence=None, fence_value=None,
                    residual_strength=1.0, *, preserve_source_detail=True):
        self._require_open()
        texture, fence_ptr, value = self._producer_texture(dlss_output, fence, fence_value,
            extent=self._working, extent_name='working', label='Feature 18 output', role='reconstruction')
        strength = float(residual_strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1.5:
            raise HdrBridgeError('SDR10 residual strength must be finite in 0..1.5')
        pointer, pitch = ctypes.c_uint64(), ctypes.c_uint32()
        code = self._lib.gpu_sdr10_reconstruct(self._handle,
            ctypes.c_void_p(_to_int_ptr(texture)), ctypes.c_void_p(_to_int_ptr(fence_ptr)),
            ctypes.c_uint64(_to_int_ptr(value)), strength, int(preserve_source_detail),
            ctypes.byref(pointer), ctypes.byref(pitch))
        if code:
            raise self._fail('gpu_sdr10_reconstruct', code)
        return DevicePlane(fmt=FORMAT_RGBA16F_CUDA, pixel_format='linear Rec.709 SDR NR output',
            width=self._full[0], height=self._full[1], cuda_context=self._cuda_context,
            device_ptr=pointer.value, pitch_bytes=pitch.value, bytes_per_pixel=8, plane=PLANE_COMPOSITED)
