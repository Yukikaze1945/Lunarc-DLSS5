"""Opt-in production export backend that keeps every frame on the GPU.

Selected with ``job_settings["export_backend"] = "gpu_transfer"``; the default
stays the FFmpeg stdin pipe in :func:`pipeline.export_dlss_stream`. Nothing here
runs unless it is asked for, and a request that cannot be met is refused with
the unmet conditions listed instead of quietly exported the old way - the rule
``transfer_mode=reduced`` and ``residual_backend=native`` already follow.

Per frame the chain is, at a ``working_scale`` below 1.0:

    nvdec_bridge   NVDEC decode into a borrowed NV12 CUDA surface, PTS and
                   colour metadata attached
    nv12_rgba      NV12 -> RGBA8 in CUDA with the source's BT.601 or BT.709
                   limited-range matrix
    gpu_lanczos    Lanczos-2 working colour, byte-identical to the CPU plan
    nvof_bridge    flow read from that device surface, result left on device.
                   guidance_mode=1 only: an unguided frame takes its motion
                   from the texture the DLSS host zeros at open, and then this
                   bridge is not loaded and the engine is not probed
    dlss_gpu_host  Feature 18, device colour in, borrowed D3D12 RGBA texture out
    gpu_residual   Catmull-Rom residual, full-size D3D12 texture out
    nvenc_bridge   RGB->NV12 in CUDA with the mate of the decoder's matrix,
                   submitted to an in-process encoder

Any whole percentage between 25 and 100 is a scale this backend serves, and the
two shapes a request resolves to are decided by arithmetic rather than by which
ratios happen to have been measured: 50% on a source whose axes are exactly
halvable runs the exact-2x kernels, whose taps are constants, and every other
percentage opens the same two stages by the caller's axis plans, through the
plan-driven exports the bridges added under those same ABI numbers, so the device
filter is the CPU plan's filter rather than a second implementation of it. Each
plan is built for the 8-aligned size ``pipeline.resolve_working_dims`` actually
returns for its percentage, which is what negotiation matches the request against,
and the window that size needs is checked against what the kernels walk before the
job starts.

At ``working_scale=1.0``, the 100% end of the range, the chain is the CPU plan's
own shape for that scale:
the full RGBA is the working colour (no Lanczos pass, because resampling a
plane to itself is not a stage), and there is no residual stage - the CPU
plan feeds the DLSS output straight to the encoder at 1.0, so the GPU chain
hands the DLSS host's output texture to NVENC as it stands, through the
producer-fence parameters the encoder already takes. A scale that is not a whole
percentage, one outside 25..100, or one whose resolved extents no chain shape
describes is refused rather than approximated, as is a plan-driven scale on a
build whose bridges have no plan-driven export to open.

This is not zero copy and does not claim to be. Each bridge reports its own
device-side and host-side counters and :func:`transfer_report` passes them
through unchanged, so an uninstrumented boundary reads ``None`` rather than
borrowing someone else's zero. Sizes of buffers that exist in code are labelled
logical, never measured. The decoder counts the frames it handed out and every
byte it copied to host memory, which is what lets the decode boundary report a
measured figure here instead of an absent one. The conversion and working-color
bridges count the device writes their kernels and copies make per successful
call; neither can see cache-line or PCIe traffic, so no figure here is a
physical bus measurement.

Performance is NOT_CHARACTERIZED.
"""
from __future__ import annotations

import ctypes as C
import contextlib
import glob
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

import bundle_paths
import pipeline
from native_watchdog import phase as native_phase
from layer_settings import layer_parameters, resolve_layer_settings
import transfer_profiler

# native/ as the running layout put it. A frozen build's contents directory is not the
# directory this file was read from, so __file__ here resolved a bundle that PyInstaller
# had already collected into a different tree.
NATIVE_DIR = bundle_paths.native_dir()

# The decoder and encoder use the same matrix selected from source metadata.
# Unspecified source matrix retains the original BT.601 behaviour.
COLOR_MATRIX_BT601 = 1
COLOR_MATRIX_BT709 = 0  # nvenc_bridge.h
NV12_COLOR_MATRIX_BT709 = 2  # nv12_rgba_bridge.h

# Settings build_transcode_command routes to the muxer rather than the encoder.
# They are not refusals; the copy pass applies them. The resolution family is here
# because a mode that reads one of them changes the resolved output size, which
# negotiate refuses on its own account rather than on the key's name.
MUX_ONLY_KEYS = frozenset({
    "container", "skip_existing", "overwrite_output", "decode_mode",
    "audio_mode", "subtitle_mode", "copy_attachments", "copy_metadata",
    "copy_chapters", "resolution_mode", "custom_width", "custom_height",
    "custom_factor", "keep_aspect", "scale_algorithm", "color_preset",
    "custom_color_space", "custom_color_primaries", "custom_color_transfer",
    "custom_color_range", "extra_args", "audio_bitrate",
})
# Two the delivery panel puts in every job and no chain is ever handed: the queue
# acts on finish_action after the run and suffix names the file before it. They are
# not encoding choices, so refusing them made the panel's own default payload a job
# this backend would not take.
QUEUE_ONLY_KEYS = frozenset({"suffix", "finish_action", "_source_range_start_seconds", "_source_range_duration_seconds"})
# Every transcode_settings key plan_encoder is known to turn into an encoder
# setting. Anything outside this and outside MUX_ONLY_KEYS is refused rather
# than dropped, because a silently ignored encoding choice is worse than a job
# that says it will not run.
ENCODER_KEYS = frozenset({
    "codec", "preset", "tune", "multipass", "rate_mode", "cq", "qp",
    "video_bitrate", "gop_mode", "gop", "peak_policy", "maxrate", "bufsize",
    "b_frames", "b_ref_mode", "lookahead", "lookahead_level", "aq_mode",
    "aq_strength", "weighted_prediction", "temporal_filter", "split_encode",
    "gpu", "profile", "level", "tier", "pixel_format", "rate_mode",
})
SUPPORTED_CODECS = ("av1_nvenc", "hevc_nvenc", "h264_nvenc")
# The encoder bridge writes one video stream, so this is the only container it
# can produce on its own; everything else arrives through the copy pass.
INTERMEDIATE_CONTAINER = ".mkv"
NVDEC_ABI, NV12_ABI, LANCZOS_ABI, RESIDUAL_ABI, NVENC_ABI = 3, 3, 2, 1, 2
# GPU_LANCZOS_MAX_TAPS and GPU_RESIDUAL_MAX_TAPS: the widest padded tap window
# each kernel walks. A plan with a shorter window is served by the bridge's own
# zero-weight padding; a plan wider than this has no kernel that can read it.
#
# The residual figure is the Catmull-Rom plan's own width: ``CATMULL_ROM_TAPS``
# rows are four taps wide for every upscale, because the window is the four
# samples around where the destination lands and does not depend on the ratio.
# The Lanczos figure is a bound on the downsample window, which does:
# ``_lanczos2_axis_windows`` gives a row ``ceil(position - 2/scale) ..
# floor(position + 2/scale)``, so no row is wider than ``floor(4 / scale) + 1``
# taps, and ``scale`` is the ratio 8-alignment resolved rather than the one that
# was asked for. Call the source axis S and the resolved working axis D.
# ``resolve_working_dims`` floors an axis at 16, and the floor can only be
# reached below S = 48, where D/S is over a third. Past that D = 8*round(S*r/8)
# with r >= 1/4, and Python's round-half-to-even costs at most four pixels, so
# D >= S/4 - 4. That bound is loose on its own - it is only the halfway case
# where the cost is actually paid: S = 32k + 16 rounds down to D = 8k, giving
# D/S = k/(4k+2), whose smallest value over the k the floor does not swallow
# (k = 2, the 80-pixel axis) is exactly 1/5, and which grows with k. So D/S is
# never below 0.2 for an even S at least 16 and at most the 16384 the bridge caps
# at, no plan is wider than floor(4/0.2) + 1 = 21 taps, and 21 is what an 80-pixel
# axis at 25% really asks for. 24 is that bound with the headroom a rounding
# change would want; a plan wider than it is still refused rather than truncated,
# and ``test_no_resolved_size_at_any_whole_percentage_outgrows_the_kernel_window``
# walks the axes and percentages to hold the claim to its arithmetic.
# RESIDUAL_MAX_TAPS needs no such argument: 4 is the plan width itself.
#
# This is a mirror of the header this source is compiled against, so it is what
# the plan serializer pads to. It is not what a request is measured against: that
# is the loaded DLL's own ``gpu_lanczos_max_taps``, because a mirror describes
# this checkout and says nothing about an older binary.
LANCZOS_MAX_TAPS, RESIDUAL_MAX_TAPS = 24, 4
# What to measure a request against when the loaded bridge answers nothing.
# ``gpu_lanczos_bridge`` published the plan-driven open under ABI 2 with a 16-wide
# window, then widened that window to 24 and added ``gpu_lanczos_max_taps`` in the
# same change, so the build an operator can still have installed is exactly the
# one that has the plan-driven open, reports no window, and walks 16. The figure is
# read out of that binary, not out of this file: the packaged
# ``dist/DLSS5Tool/_internal/native/gpu_lanczos_bridge.dll`` compiles
# ``#define TAPS 16u`` into its scaled shader and exports no getter, while the ABI-2
# build in ``native/`` compiles ``#define TAPS 24u`` and does export it. Answering a
# question the DLL never got would hand a 17..21 tap plan to a 16-wide kernel, which
# is a refusal at the open after the job has already decoded frames; the conservative
# bound refuses it before the first frame, and a build narrower still is refused by
# its own check rather than served.
LEGACY_LANCZOS_MAX_TAPS = 16
# The working scales this backend serves, as whole percentages of the source:
# every integer from MIN to MAX inclusive. The range is a cap on what the CPU
# plan's own filter is defined for (``resolve_working_dims`` clips its ratio to
# 0.25..1.0) and on what a downsample stage has to downsample, not a list of
# ratios that happen to have been measured - every percentage in it resolves to
# 8-aligned extents, and those extents decide the chain shape and the tap window.
WORKING_PERCENT_MIN, WORKING_PERCENT_MAX = 25, 100
BT601_FAMILY = (None, "unknown", "unspecified", "smpte170m", "bt470bg")


class BridgeError(RuntimeError):
    """A bridge could not be loaded, opened or driven; the text is its own."""


# --- loading ------------------------------------------------------------------

def ffmpeg_dll_directories() -> list:
    """Directories holding the FFmpeg runtime the native bridges were linked to.

    The bundled command line tool is a different installation from the
    development tree the bridges link against, and picking up the wrong
    ``avcodec-*.dll`` would fail inside this process with no useful message. The
    order matches the smoke scripts, so export and the correctness gates resolve
    the same libraries.

    Which installation that is belongs to the machine, not to this file: a literal
    from whoever's tree it was developed in resolves on nothing else. So the answer
    is the knob the bridges were linked with (``FFMPEG_SDK_DIR``, the one
    native/build_nvdec_bridge.bat uses), then a bundle's own contents directory, then
    the FFmpeg this process would run, then the one PATH would have handed the loader.
    """
    candidates = []
    sdk = os.environ.get("FFMPEG_SDK_DIR")
    if sdk:
        candidates.append(os.path.join(sdk, "bin"))
    candidates.append(os.path.join(os.environ.get('LUNARC_RUNTIME_DIR') or
        os.path.join(bundle_paths.app_dir(), 'runtime'), 'tools'))
    if getattr(sys, "frozen", False):
        candidates.append(bundle_paths.contents_dir())
    bundled = pipeline._bundle_ffmpeg()
    if bundled:
        candidates.append(os.path.dirname(bundled))
    candidates += _ffmpeg_shared_dirs_on_path()
    # A directory only counts if the libraries are in it: load_bridges registers the
    # first of these and nothing else, so a candidate that is merely an existing folder
    # (a shim directory, an install of the tool without the shared libraries beside it)
    # would be the one answer given and the one that fails.
    return [path for path in dict.fromkeys(candidates) if _holds_ffmpeg_libraries(path)]


def _holds_ffmpeg_libraries(directory):
    return bool(directory) and os.path.isdir(directory) and bool(
        glob.glob(os.path.join(directory, "avcodec-*.dll")))


def encoder_cli():
    """The command line tool beside the libraries this process loads its encoder from.

    The gate asks which option values an encoder accepts, and the encoder is a library
    in this process -- not whichever ``ffmpeg.exe`` PATH happens to hand over. One
    machine can hold two installations whose option lists differ (a value the CLI on
    PATH knows and the ``avcodec-*.dll`` the bridges bound to does not), and then the
    question has to be put to the installation that will do the encoding, or a job is
    approved that the bridge refuses once it has decoded a frame.
    """
    for directory in ffmpeg_dll_directories():
        cli = os.path.join(directory, "ffmpeg.exe")
        if os.path.isfile(cli):
            return cli
    return pipeline._bundle_ffmpeg()


def _ffmpeg_shared_dirs_on_path():
    """Every PATH directory holding an FFmpeg shared library, in PATH order.

    nvdec_bridge and nvenc_bridge import avcodec/avformat/avutil at load time, so when
    nothing better was registered this is what the loader settles for. Finding it here
    rather than hardcoding an installation is what lets a failed load name the directory
    the libraries actually came from - and keeps the source portable to a machine where
    that install does not exist.
    """
    return [os.path.abspath(directory) for directory in
            os.environ.get("PATH", "").split(os.pathsep)
            if _holds_ffmpeg_libraries(directory)]


def _bind(library, prototypes):
    for symbol, (returns, args) in prototypes.items():
        function = getattr(library, symbol, None)
        if function is None:
            raise BridgeError(f"{os.path.basename(library._name)} does not "
                              f"export {symbol}")
        function.restype = returns
        function.argtypes = args
    return library


def _require_abi(actual, expected, name):
    if int(actual) != expected:
        raise BridgeError(f"{name} reports ABI {int(actual)}, this module reads "
                          f"ABI {expected}; rebuild it from native/{name}.cpp")


def _bind_optional(library, prototypes):
    """Bind the exports a newer build added under an unchanged ABI number.

    The entry points every caller of this ABI already used keep their signature
    and behaviour, so a DLL that has never heard of them is still a valid ABI
    partner for the scales that do not need them. Presence is therefore recorded
    as a capability and negotiation refuses the scale that asks for it, rather
    than refusing to load the library set. Returns what it bound.
    """
    bound = {}
    for symbol, (returns, args) in prototypes.items():
        function = getattr(library, symbol, None)
        if function is None:
            continue
        function.restype = returns
        function.argtypes = args
        bound[symbol] = function
    return bound


def guidance_requires_flow(job_settings) -> bool:
    """Whether this guidance mode takes its motion from NVOF.

    One place decides, because the bridge set and the gate have to agree about
    what the run needs. Anything that is not clearly ``1`` is read as unguided,
    which is the same rule the gate applies; a mode it cannot read at all is
    answered as if it were guided, so an unreadable setting loses the optical
    flow bridge neither way and is refused for the mode itself.
    """
    try:
        return int(dict(job_settings or {}).get("guidance_mode", 0)) == 1
    except (TypeError, ValueError):
        return True


def native_tune_probe(library, *, value_probe=True):
    """Ask the loaded libavcodec, never a possibly different FFmpeg executable."""
    answers = {}
    def supports_uhq(codec):
        if codec not in answers:
            if value_probe:
                rc = library.nvenc_option_value_available(codec.encode(), b'tune', b'uhq')
            else:
                raise BridgeError('UHQ capability requires the rebuilt nvenc_bridge.dll; restart Lunarc to apply it')
            if rc < 0:
                _bridge_error(library, 'nvenc_last_error', rc)
            answers[codec] = rc == 1
        return answers[codec]
    return supports_uhq


def load_bridges(requires_flow: bool = True, requires_multiscale: bool = False):
    """Load and version-check the whole library set, or raise with the reason.

    Runs on the export thread. The NVOF and DLSS sessions bind to the thread and
    CUDA context that create them, so a probe from elsewhere could advertise a
    set of libraries the export then cannot actually drive.

    ``requires_flow`` is whether the guidance mode this set is being loaded for
    takes its motion from NVOF, which is guidance_mode=1. Then an optical flow
    bridge that cannot produce a device flow plane is refused here, because the
    frame cannot be guided without one. For guidance_mode=0 the motion is the
    DLSS host's own resident zero texture, so the module is not imported and the
    engine is not probed at all: a machine with no optical flow engine installed
    loads the same five bridges and runs unguided. The distinction is the
    caller's to state because the mode is what the caller negotiated on.
    """
    cookies = [os.add_dll_directory(path) for path in ffmpeg_dll_directories()[:1]]
    capabilities = {}
    try:
        nvdec = _bind(_load("nvdec_bridge"), {
            "nvdec_bridge_abi_version": (C.c_uint32, []),
            "nvdec_bridge_open": (C.c_int32, [C.c_char_p, C.POINTER(C.c_void_p)]),
            "nvdec_bridge_next": (C.c_int32, [C.c_void_p, C.POINTER(_DeviceFrame)]),
            "nvdec_bridge_get_transfer_counters": (C.c_int32, [C.c_void_p,
                C.POINTER(_NvdecCounters)]),
            "nvdec_bridge_cancel": (None, [C.c_void_p]),
            "nvdec_bridge_close": (None, [C.c_void_p]),
            "nvdec_bridge_last_error": (C.c_char_p, [])})
        _require_abi(nvdec.nvdec_bridge_abi_version(), NVDEC_ABI, "nvdec_bridge")

        nv12 = _bind(_load("nv12_rgba_bridge"), {
            "nv12_rgba_abi_version": (C.c_uint32, []),
            "nv12_rgba_open_matrix": (C.c_int32, [C.c_void_p, C.c_uint32, C.c_uint32,
                C.c_int32,
                C.POINTER(C.c_void_p)]),
            "nv12_rgba_convert": (C.c_int32, [C.c_void_p, C.c_uint64, C.c_uint32,
                C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64),
                C.POINTER(C.c_uint32)]),
            "nv12_rgba_get_transfer_counters": (C.c_int32, [C.c_void_p,
                C.POINTER(_Nv12Counters)]),
            "nv12_rgba_close": (None, [C.c_void_p]),
            "nv12_rgba_last_error": (C.c_char_p, [])})
        _require_abi(nv12.nv12_rgba_abi_version(), NV12_ABI, "nv12_rgba_bridge")

        lanczos = _bind(_load("gpu_lanczos_bridge"), {
            "gpu_lanczos_abi_version": (C.c_uint32, []),
            "gpu_lanczos_open": (C.c_int32, [C.c_void_p, C.c_uint32, C.c_uint32,
                C.POINTER(C.c_void_p)]),
            "gpu_lanczos_process": (C.c_int32, [C.c_void_p, C.c_uint64,
                C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]),
            "gpu_lanczos_get_transfer_counters": (C.c_int32, [C.c_void_p,
                C.POINTER(_LanczosCounters)]),
            "gpu_lanczos_close": (None, [C.c_void_p]),
            "gpu_lanczos_last_error": (C.c_char_p, [])})
        _require_abi(lanczos.gpu_lanczos_abi_version(), LANCZOS_ABI,
                     "gpu_lanczos_bridge")
        scaled = _bind_optional(lanczos, {
            "gpu_lanczos_open_scaled": (C.c_int32, [C.c_void_p, C.c_uint32,
                C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(_TapTables),
                C.POINTER(C.c_void_p)]),
            # The window this build's plan kernel walks, asked of the binary
            # rather than of the header this file was written against: a stale DLL
            # is a narrower kernel, and the honest answer is a refusal before the
            # job decodes a frame rather than an error at the open after it.
            "gpu_lanczos_max_taps": (C.c_uint32, [])})
        capabilities["lanczos_scaled"] = bool(scaled.get("gpu_lanczos_open_scaled"))
        if scaled.get("gpu_lanczos_max_taps") is not None:
            capabilities["lanczos_max_taps"] = int(scaled["gpu_lanczos_max_taps"]())

        residual = _bind(_load("gpu_residual_bridge"), {
            "gpu_residual_abi_version": (C.c_uint32, []),
            "gpu_residual_open": (C.c_int32, [C.c_void_p, C.c_uint32, C.c_uint32,
                C.POINTER(C.c_void_p)]),
            "gpu_residual_process": (C.c_int32, [C.c_void_p, C.c_void_p,
                C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64, C.c_size_t,
                C.c_void_p, C.c_uint64, C.c_size_t, C.c_float,
                C.POINTER(C.c_void_p)]),
            "gpu_residual_get_transfer_counters": (C.c_int32, [C.c_void_p,
                C.POINTER(_ResidualCounters)]),
            "gpu_residual_close": (None, [C.c_void_p]),
            "gpu_residual_last_error": (C.c_char_p, [])})
        _require_abi(residual.gpu_residual_abi_version(), RESIDUAL_ABI,
                     "gpu_residual_bridge")
        scaled = _bind_optional(residual, {
            "gpu_residual_open_scaled": (C.c_int32, [C.c_void_p, C.c_uint32,
                C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(_TapTables),
                C.POINTER(C.c_void_p)])})
        capabilities["residual_scaled"] = bool(scaled)

        low_residual = None
        if requires_multiscale:
            low_residual = _bind(_load("gpu_low_residual_bridge"), {
                "gpu_low_residual_abi_version": (C.c_uint32, []),
                "gpu_low_residual_open": (C.c_int32, [C.c_void_p,
                    C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]),
                "gpu_low_residual_process": (C.c_int32, [C.c_void_p,
                    C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p,
                    C.c_uint64, C.c_size_t, C.POINTER(C.c_void_p),
                    C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]),
                "gpu_low_residual_close": (None, [C.c_void_p]),
                "gpu_low_residual_get_transfer_counters": (C.c_int32,
                    [C.c_void_p, C.POINTER(_LowResidualCounters)]),
                "gpu_low_residual_last_error": (C.c_char_p, [])})
            _require_abi(low_residual.gpu_low_residual_abi_version(), 1,
                         "gpu_low_residual_bridge")
        capabilities["low_residual"] = low_residual is not None

        fusion = None
        if requires_multiscale:
            fusion = _bind(_load("gpu_multiscale_fusion_bridge"), {
                "gpu_fusion_abi_version": (C.c_uint32, []),
                "gpu_fusion_open": (C.c_int32, [C.c_void_p,
                    C.POINTER(_FusionOptions), C.POINTER(C.c_void_p)]),
                "gpu_fusion_process": (C.c_int32, [C.c_void_p,
                    C.c_void_p, C.c_uint64, C.c_size_t,
                    C.POINTER(_FusionObservation), C.c_uint32,
                    C.POINTER(C.c_void_p), C.POINTER(_FusionReport)]),
                "gpu_fusion_close": (None, [C.c_void_p]),
                "gpu_fusion_get_transfer_counters": (C.c_int32,
                    [C.c_void_p, C.POINTER(_FusionCounters)]),
                "gpu_fusion_last_error": (C.c_char_p, [])})
            _require_abi(fusion.gpu_fusion_abi_version(), 1,
                         "gpu_multiscale_fusion_bridge")
        capabilities["fusion"] = fusion is not None

        nvenc = _bind(_load("nvenc_bridge"), {
            "nvenc_abi_version": (C.c_uint32, []),
            "nvenc_option_available": (C.c_int32, [C.c_char_p, C.c_char_p]),
            "nvenc_open": (C.c_int32, [C.c_void_p, C.c_void_p, C.c_uint32,
                C.c_uint32, C.c_uint32, C.c_uint32, C.c_int32, C.c_int32,
                C.c_char_p, C.POINTER(_NvencOptions), C.POINTER(C.c_void_p)]),
            "nvenc_encode_texture": (C.c_int32, [C.c_void_p, C.c_void_p,
                C.c_void_p, C.c_uint64, C.c_int64]),
            "nvenc_get_transfer_counters": (C.c_int32, [C.c_void_p,
                C.POINTER(_NvencCounters)]),
            "nvenc_finish": (C.c_int32, [C.c_void_p]),
            "nvenc_close": (None, [C.c_void_p]),
            "nvenc_last_error": (C.c_char_p, [])})
        _require_abi(nvenc.nvenc_abi_version(), NVENC_ABI, "nvenc_bridge")
        value_probe = _bind_optional(nvenc, {
            "nvenc_option_value_available": (C.c_int32,
                [C.c_char_p, C.c_char_p, C.c_char_p])})

        import dlss_gpu_engine
        nvof_probe = None
        flow_module = None
        if requires_flow:
            try:
                import optical_flow
            except ImportError as missing:
                raise BridgeError(
                    "guidance_mode=1 takes its motion from NVOF, and the module "
                    f"that drives it cannot even be imported: {missing}")
            probe = optical_flow.probe_nvof_capabilities()
            if not probe.get("available"):
                raise BridgeError(f"NVOF is unavailable: {probe.get('error')}")
            if not probe.get("device_output"):
                raise BridgeError("nvof_bridge.dll has no device flow output export")
            if not probe.get("device_rgba_input"):
                raise BridgeError("nvof_bridge.dll has no device RGBA input export")
            nvof_probe = probe
            flow_module = optical_flow

        def option_available(codec, name, _probe=nvenc.nvenc_option_available):
            """Whether the named encoder has the named option, as the codec says.

            Wrapped because the bridge takes bytes and the settings are text; a
            caller must be able to ask in the same terms it writes a setting in.
            """
            return _probe(str(codec).encode("utf-8"),
                          str(name).encode("utf-8")) == 1

        tune_available = native_tune_probe(nvenc,
            value_probe="nvenc_option_value_available" in value_probe)

        return {"nvdec": nvdec, "nv12": nv12, "lanczos": lanczos,
                "residual": residual, "low_residual": low_residual,
                "fusion": fusion,
                "nvenc": nvenc, "cookies": cookies,
                "option_available": option_available,
                "tune_available": tune_available,
                "optical_flow": flow_module,
                # What this set does and does not have behind it. ``nvof_probe``
                # is None on an unguided set because nothing was asked of the
                # optical flow engine, not because it answered and failed; the
                # export reports the distinction so a mode 0 run can be told
                # apart from a guided run that lost its flow module.
                "nvof_probe": nvof_probe,
                "capabilities": capabilities,
                "ffmpeg_dll_directory": ffmpeg_dll_directories()[:1]}
    except Exception:
        for cookie in cookies:
            _close_cookie(cookie)
        raise


def _load(name):
    """One bridge, out of the native directory this layout actually has.

    ctypes reports a DLL that cannot load as a bare OSError, which is the same message
    for two different faults that need two different fixes: the file was never built or
    collected, or it is there and one of the libraries it imports at load time is not -
    the FFmpeg shared libraries for nvdec and nvenc, native/nvof_d3d12_interop.dll for
    three of the others. Each branch therefore says what to go and do.
    """
    bundle_paths.add_dll_directory(NATIVE_DIR)
    path = os.path.join(NATIVE_DIR, name + ".dll")
    if not os.path.isfile(path):
        raise BridgeError(str(bundle_paths.missing_binary(
            path, f"run the {name} stage of the GPU transfer export")))
    try:
        return C.CDLL(path)
    except OSError as error:
        found = ffmpeg_dll_directories()
        resolved = found[0] if found else "no directory holding avcodec-*.dll"
        raise BridgeError(
            f"{name}.dll at {path} loaded none of its imports: {error}. It needs the "
            f"FFmpeg shared libraries and native/nvof_d3d12_interop.dll; this process "
            f"resolved FFmpeg to {resolved}. Set FFMPEG_SDK_DIR to the installation "
            f"{name} was built against, or rebuild it with native/build_{name}.bat."
        ) from error


def _close_cookie(cookie):
    try:
        cookie.close()
    except Exception:
        pass


# --- native struct mirrors ----------------------------------------------------

class _DeviceFrame(C.Structure):
    """Mirror of NvdecDeviceFrame in native/nvdec_bridge.h."""
    _fields_ = [("struct_size", C.c_uint32), ("width", C.c_uint32),
        ("height", C.c_uint32), ("software_format", C.c_uint32),
        ("y_device_ptr", C.c_uint64), ("uv_device_ptr", C.c_uint64),
        ("y_pitch_bytes", C.c_int32), ("uv_pitch_bytes", C.c_int32),
        ("cuda_context", C.c_void_p), ("pts", C.c_int64),
        ("time_base_num", C.c_int32), ("time_base_den", C.c_int32),
        ("color_range", C.c_int32), ("color_primaries", C.c_int32),
        ("color_transfer", C.c_int32), ("color_space", C.c_int32)]


class _NvencOptions(C.Structure):
    """Mirror of NvencBridgeOptions, ABI 2 including the AVOption string."""
    _fields_ = [("struct_size", C.c_uint32), ("codec", C.c_char_p),
        ("preset", C.c_char_p), ("tune", C.c_char_p), ("multipass", C.c_char_p),
        ("rate_control", C.c_char_p), ("constant_quality", C.c_int32),
        ("bitrate", C.c_int64), ("gop", C.c_int32), ("color_matrix", C.c_int32),
        ("extra_options", C.c_char_p)]


def _counter_struct(*names):
    return type("Counters", (C.Structure,), {"_fields_":
        [("struct_size", C.c_uint32)] + [(n, C.c_uint64) for n in names]})


_NvencCounters = _counter_struct("frames_submitted", "packets_written",
    "staging_gpu_copy_bytes", "conversion_output_bytes", "host_pixel_bytes")
_ResidualCounters = _counter_struct("full_device_copy_bytes",
    "working_device_copy_bytes", "output_readback_bytes")
_LowResidualCounters = _counter_struct("process_count",
    "working_device_copy_bytes", "output_readback_bytes")
_FusionCounters = _counter_struct("process_count", "full_device_copy_bytes",
    "scalar_readback_bytes", "output_readback_bytes")
# Mirror of Nv12RgbaTransferCounters in native/nv12_rgba_bridge.h. The convert
# call makes no host copy, so the conversion boundary has no host field to read
# as zero: the two fields are the calls and the device bytes they wrote.
_Nv12Counters = _counter_struct("convert_count", "output_device_write_bytes")
# Mirror of GpuLanczosTransferCounters in native/gpu_lanczos_bridge.h. The first
# field is a copy the bridge itself performs device-to-device; the other two are
# the model passes' device writes. None of them is a host transfer.
_LanczosCounters = _counter_struct("process_count", "source_device_copy_bytes",
    "scratch_device_write_bytes", "output_device_write_bytes")
# Mirror of NvdecTransferCounters in native/nvdec_bridge.h. The readback entry
# point that raises the two host counters is diagnostic and stays unbound here:
# production hands frames out as borrowed device surfaces.
_NvdecCounters = _counter_struct("frames_decoded", "device_frame_count",
    "bitstream_bytes_submitted", "host_frame_copy_count", "host_frame_copy_bytes",
    "scale_device_output_bytes")


class _TapTables(C.Structure):
    """Mirror of GpuLanczosTapTables in native/gpu_lanczos_bridge.h.

    GpuResidualTapTable in native/gpu_residual_bridge.h repeats this layout field
    for field on purpose: one mirror then covers both bridges, and the header test
    fails if either header drifts from it. The vertical axis is listed first
    because that is the order both kernels run.
    """
    _fields_ = [("struct_size", C.c_uint32), ("taps", C.c_uint32),
        ("vertical_src_len", C.c_uint32), ("vertical_dst_len", C.c_uint32),
        ("vertical_indices", C.POINTER(C.c_int32)),
        ("vertical_weights", C.POINTER(C.c_float)),
        ("horizontal_src_len", C.c_uint32), ("horizontal_dst_len", C.c_uint32),
        ("horizontal_indices", C.POINTER(C.c_int32)),
        ("horizontal_weights", C.POINTER(C.c_float))]


class _FusionAxis(C.Structure):
    _fields_ = [("src_len", C.c_uint32), ("dst_len", C.c_uint32),
                ("taps", C.c_uint32),
                ("indices", C.POINTER(C.c_int32)),
                ("weights", C.POINTER(C.c_float))]


class _FusionLayer(C.Structure):
    _fields_ = [("width", C.c_uint32), ("height", C.c_uint32),
                ("horizontal", _FusionAxis), ("vertical", _FusionAxis),
                ("weight", C.c_float)]


class _FusionOptions(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
                ("full_width", C.c_uint32), ("full_height", C.c_uint32),
                ("layer_count", C.c_uint32), ("layers", _FusionLayer * 3),
                ("baseline_horizontal", _FusionAxis),
                ("baseline_vertical", _FusionAxis),
                ("beta", C.c_float), ("residual_strength", C.c_float),
                ("relative_tolerance", C.c_float),
                ("max_iterations", C.c_uint32)]


class _FusionObservation(C.Structure):
    _fields_ = [("signed_texture", C.c_void_p),
                ("producer_fence", C.c_void_p),
                ("producer_fence_value", C.c_uint64)]


class _FusionReport(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("iterations", C.c_uint32),
                ("initial_residual_norm", C.c_float),
                ("final_residual_norm", C.c_float),
                ("relative_residual", C.c_float), ("converged", C.c_uint32)]


def _fusion_axis(table, horizontal):
    prefix = "horizontal" if horizontal else "vertical"
    return _FusionAxis(getattr(table, prefix + "_src_len"),
                       getattr(table, prefix + "_dst_len"), table.taps,
                       getattr(table, prefix + "_indices"),
                       getattr(table, prefix + "_weights"))


def tap_tables(vertical, horizontal, max_taps):
    """Two pipeline.AxisPlans as the scaled bridges read them.

    Both axes are widened to one common tap count, because the bridge's plan
    struct carries a single ``taps`` row stride. The extra columns are weight 0.0
    at index 0, which is the padding ``_apply_axis_plan`` already sums over, so
    the widened plan is the same filter rather than an approximation of it. A
    window wider than the kernel can walk is refused here rather than truncated:
    silently dropping taps would build a different filter than the CPU plan.

    Returns ``(view, keepalive)``. The struct holds raw pointers into the arrays,
    so the caller has to keep the second return value alive for as long as the
    bridge can still read from it, which for a bridge is its whole open session.
    """
    def window(plan):
        if plan is None:
            raise BridgeError("a tap plan axis is missing")
        if plan.indices.shape[0] != plan.dst_len or plan.weights.shape[0] != plan.dst_len:
            raise BridgeError(f"{plan.kernel} plan claims {plan.dst_len} rows but "
                              f"carries {plan.indices.shape[0]}")
        return int(plan.max_taps)

    columns = max(window(vertical), window(horizontal))
    if columns > max_taps:
        raise BridgeError(f"the {columns}-tap window this scale needs is wider than "
                          f"the {max_taps} taps the bridge kernel walks")

    def columns_of(plan):
        indices = np.zeros((plan.dst_len, columns), np.int32)
        weights = np.zeros((plan.dst_len, columns), np.float32)
        indices[:, :plan.max_taps] = plan.indices
        weights[:, :plan.max_taps] = plan.weights
        return indices, weights

    vi, vw = columns_of(vertical)
    hi, hw = columns_of(horizontal)
    view = _TapTables()
    view.struct_size = C.sizeof(_TapTables)
    view.taps = columns
    view.vertical_src_len = vertical.src_len
    view.vertical_dst_len = vertical.dst_len
    view.vertical_indices = vi.ctypes.data_as(C.POINTER(C.c_int32))
    view.vertical_weights = vw.ctypes.data_as(C.POINTER(C.c_float))
    view.horizontal_src_len = horizontal.src_len
    view.horizontal_dst_len = horizontal.dst_len
    view.horizontal_indices = hi.ctypes.data_as(C.POINTER(C.c_int32))
    view.horizontal_weights = hw.ctypes.data_as(C.POINTER(C.c_float))
    return view, (vi, vw, hi, hw)


def working_color_plan(src_w, src_h, live_w, live_h):
    """The GPU Lanczos stage's plan: full extents in, working extents out."""
    return tap_tables(pipeline._build_lanczos2_axis_plan(src_h, live_h),
                      pipeline._build_lanczos2_axis_plan(src_w, live_w),
                      LANCZOS_MAX_TAPS)


def residual_plan(src_w, src_h, live_w, live_h):
    """The GPU residual stage's plan: working extents in, full extents out.

    The same Catmull-Rom axis plans ``reconstruct_residual`` builds, so the
    device upscale is the CPU reference's upscale rather than a second opinion
    of it.
    """
    return tap_tables(pipeline._build_catmull_rom_axis_plan(live_h, src_h),
                      pipeline._build_catmull_rom_axis_plan(live_w, src_w),
                      RESIDUAL_MAX_TAPS)


def working_percent(scale):
    """The requested working scale as a whole percentage, or None.

    One rule, shared by the gate and the delivery page: a request is a percentage
    of the source, so 0.67 is 67 and 33.333 is nothing this backend resolves. The
    tolerance is a float's, not a caller's - ``0.67 * 100`` is 67.00000000000001 -
    and it is tight enough that a ratio which is merely near a whole percentage
    (a third, 0.3333) is refused instead of being rounded into a job the operator
    did not ask for.
    """
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    percent = int(round(value * 100.0))
    if abs(value * 100.0 - percent) > 1e-6:
        return None
    return percent if WORKING_PERCENT_MIN <= percent <= WORKING_PERCENT_MAX else None


def plan_driven_scale(src_w, src_h, live_w, live_h, scale):
    """Which plan-driven percentage these extents are, or None for no chain.

    Matched on the extents and not on a list of ratios: any whole percentage in
    the served range below 100 is plan-driven once 8-alignment has produced a
    working size smaller than the source on both axes, because that is the one
    shape a tap plan describes. The extent test is the same one the
    working-color bridge makes of its own accord - a downsample stage has
    something to downsample - applied here, where a job can still be refused
    before it starts, and a caller that computed its own working size gets the
    refusal rather than a bridge that would sample a plane narrower than the one
    it was handed.
    """
    percent = working_percent(scale)
    if percent is None or percent == WORKING_PERCENT_MAX:
        return None
    if live_w >= src_w or live_h >= src_h:
        return None
    if (live_w, live_h) != pipeline.resolve_working_dims(src_w, src_h,
                                                         percent / 100.0):
        return None
    return percent / 100.0


def plan_tap_widths(src_w, src_h, live_w, live_h):
    """Taps the two CPU plans need for these extents, as (working color, upscale).

    Asked of the plan builders' own arithmetic - ``pipeline.lanczos2_axis_taps``
    and ``pipeline.CATMULL_ROM_TAPS`` - rather than of a second copy of it, so
    the gate cannot drift from the filter it is measuring. It builds no plan, so
    ``filter_plan_build_count`` still counts the four tables the export builds
    and nothing a negotiation looked at on the way to deciding.
    """
    working_color = max(pipeline.lanczos2_axis_taps(src_h, live_h),
                        pipeline.lanczos2_axis_taps(src_w, live_w))
    return working_color, pipeline.CATMULL_ROM_TAPS


class TapPlans:
    """The two plan-driven tables one chain needs, for one pair of extents.

    Built once per export from the resolved working size, then read by both
    resampling stages when they open. A bridge copies the table to the device as
    it opens, so the arrays outlive the upload by construction here: the views
    and what they point into are both attributes of this object, and nothing
    hands out one without the other.
    """

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int):
        self.working_color, self._color_buffers = working_color_plan(
            src_w, src_h, live_w, live_h)
        self.upscale, self._upscale_buffers = residual_plan(
            src_w, src_h, live_w, live_h)


def _bridge_error(library, symbol, code):
    """Raise with the bridge's own last message, which is the only detail that
    identifies what actually failed inside FFmpeg or CUDA."""
    getter = getattr(library, symbol, None)
    detail = getter() if callable(getter) else None
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", "replace")
    raise BridgeError(f"{symbol} reported {code}: {detail or 'no detail'}")


# --- what the source actually is ----------------------------------------------

def probe_source(video: str, ffprobe=None, include_chapters: bool = False,
                 include_video_metadata: bool = False) -> dict:
    """Everything negotiation needs about the input, from one ffprobe call.

    Two facts are opt-in because each costs something the call that has always been
    made does not:

    * ``include_chapters`` adds ``-show_chapters`` and counts the ``chapters`` list it
      returns. ``format.nb_chapters`` is not that count: a Matroska file is under no
      obligation to write the field at all, so the default answer for a source with two
      real chapters is ``0`` - and an admission test built on it would skip the copy
      pass and deliver the file without them.
    * ``include_video_metadata`` reports the first video stream's own ``tags`` and the
      set flags of its ``disposition``. No other caller reads them, and the SDR chain's
      probe and its answers are unchanged by them either way.
    """
    from image_source import is_image, probe
    if is_image(video):
        return probe(video)
    ffprobe = ffprobe or pipeline._bundle_ffprobe()
    if not ffprobe:
        raise BridgeError("ffprobe not found, so the source cannot be checked "
                          "before a GPU session is opened")
    arguments = ["-v", "error", "-print_format", "json", "-show_streams",
                 "-show_format"]
    if include_chapters:
        arguments.append("-show_chapters")
    dump = subprocess.check_output([ffprobe] + arguments + [video], timeout=120)
    data = json.loads(dump.decode("utf-8", "replace"))
    streams = data.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    first = video_streams[0] if video_streams else {}
    chapters = data.get("chapters") if include_chapters else None
    probed = {
        "video_stream_count": len(video_streams),
        "codec_name": first.get("codec_name"),
        "profile": first.get("profile"),
        "pix_fmt": first.get("pix_fmt"),
        "width": int(first.get("width") or 0),
        "height": int(first.get("height") or 0),
        "color_range": first.get("color_range"),
        "color_space": first.get("color_space"),
        "color_transfer": first.get("color_transfer"),
        "color_primaries": first.get("color_primaries"),
        "r_frame_rate": first.get("r_frame_rate"),
        "audio_streams": sum(1 for s in streams
                             if s.get("codec_type") == "audio"),
        "subtitle_streams": sum(1 for s in streams
                                if s.get("codec_type") == "subtitle"),
        "nb_chapters": (len(chapters) if chapters is not None
                        else int((data.get("format") or {}).get("nb_chapters") or 0)),
        "metadata": bool((data.get("format") or {}).get("tags")),
    }
    if include_video_metadata:
        probed["video_tags"] = dict(first.get("tags") or {})
        probed["video_disposition"] = {
            name: int(value) for name, value in
            (first.get("disposition") or {}).items() if value}
    from source_colour import resolve_source_colour
    probed['side_data_list'] = list(first.get('side_data_list') or [])
    return resolve_source_colour(probed)


def fps_rational(source: dict, override=None):
    """The frame rate as the encoder's rational, from the probed one."""
    text = str(override if override else source.get("r_frame_rate") or "")
    if not text or text in ("0/0", "0/1", "0"):
        return None
    try:
        value = (Fraction(text) if "/" in text or "." not in text
                 else Fraction(float(text))).limit_denominator(100000)
    except (ValueError, ZeroDivisionError):
        return None
    if value.numerator <= 0 or value.denominator <= 0:
        return None
    return (int(value.numerator), int(value.denominator))


# --- the encoder plan ---------------------------------------------------------

def plan_encoder(transcode_settings, cq: int, source: dict,
                 option_available=None, tune_available=None,
                 color_matrix=COLOR_MATRIX_BT601) -> dict:
    """Turn the project's encoding settings into what the bridge can apply.

    Struct fields carry what ``NvencBridgeOptions`` models; the rest become
    same-named AVOptions, because the name the command line used is the name
    the encoder's option table uses. ``option_available`` is the encoder itself
    answering whether it has one, so an unsupported setting is refused up front
    rather than dropped on the way to NVENC.
    """
    ts = dict(transcode_settings or {})
    if ts.get('codec') == 'prores_ks':
        from prores_export import encoder_plan
        plan = encoder_plan(ts, source)
        plan['struct']['color_matrix'] = color_matrix
        return plan
    unsupported = []
    codec = str(ts.get("codec", "av1_nvenc"))
    if codec not in SUPPORTED_CODECS:
        unsupported.append(f"codec={codec}: the in-process encoder only drives "
                           f"{', '.join(SUPPORTED_CODECS)}")
    preset = str(ts.get("preset", "p7"))

    struct = {"codec": codec.encode(), "preset": preset.encode(), "tune": None,
              "multipass": None, "rate_control": None, "constant_quality": 0,
              "bitrate": 0, "gop": 0, "color_matrix": color_matrix}
    av_options = []

    pix_fmt = str(ts.get("pixel_format", "auto"))
    if pix_fmt not in ("auto", "yuv420p"):
        unsupported.append(f"pixel_format={pix_fmt}: the bridge submits NV12 "
                           "8-bit 4:2:0 and nothing else")

    # Without transcode_settings the legacy command picks av1_nvenc/p7/hq/qres/
    # vbr/cq/b:v 0/g=fps*2/yuv420p, so this plan starts from the same place.
    tune = requested_tune = "hq" if not ts else str(ts.get("tune", "auto"))
    # The legacy command downgrades uhq to hq only where the encoder cannot have
    # it; answering for the codec here would silently change a setting the caller
    # asked for, so the probe is the caller's and the value passes through when
    # nobody answered.
    if tune == "uhq" and (codec == "h264_nvenc" or
                          (tune_available is not None and not tune_available(codec))):
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

    fps = fps_rational(source) or (60, 1)
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
            ("profile", "profile", "auto"), ("level", "level", "auto"),
            ("tier", "tier", "auto"), ("gpu", "gpu", "-1")):
        text = default if key not in ts else str(ts.get(key))
        if text in (None, "", "auto", "None", "-1"):
            continue
        if key == "weighted_prediction":
            text = "1" if text in ("1", "true", "True", "on") else "0"
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

    if option_available is not None:
        for name, value in list(av_options):
            if not option_available(codec, name):
                av_options.remove((name, value))
                unsupported.append(f"encoder {codec} has no option {name}, so "
                                   f"the setting for it cannot be applied")

    ignored = {k for k in ts
               if k not in MUX_ONLY_KEYS and k not in ENCODER_KEYS
               and k not in QUEUE_ONLY_KEYS}
    for key in sorted(ignored):
        unsupported.append(f"transcode_settings[{key!r}] has no equivalent this "
                           "encoder can be given")
    return {"struct": struct, "av_options": av_options, "codec": codec,
            "tune_adjustment": {"requested": requested_tune, "applied": tune,
                "reason": "loaded encoder does not support UHQ"} if tune != requested_tune else None,
            "unsupported": unsupported}


def _encoder_option(value):
    """One resolved encoder option as text, for a report and not for a bridge.

    The struct holds bytes because the bridge takes bytes; a run's own record has to
    be readable, and ``None`` has to stay ``None`` -- an option the plan did not set
    is a different claim from one it set to an empty string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def extra_options_blob(av_options) -> bytes:
    """Newline-joined name=value pairs for the bridge's pass-through field."""
    if not av_options:
        return b""
    for name, value in av_options:
        if "=" in name or "\n" in str(value) or "\n" in name:
            raise BridgeError(f"encoder option {name}={value} cannot be passed "
                              "to the bridge safely")
    return "\n".join(f"{name}={value}" for name, value in av_options).encode()


# --- container, audio and mapping ---------------------------------------------

# How long the copy pass may take before the run is called a failure. A remux moves
# compressed packets, so this is a generous wall-clock bound rather than a budget:
# it exists to stop a job waiting on an FFmpeg process that has stopped making
# progress, and it is the timeout the pass has always been given.
MUX_TIMEOUT_SEC = 1800.0
# The audio and subtitle modes ``build_mux_command`` expresses a clause for. The
# pass itself has always been permissive - an unknown ``subtitle_mode`` simply adds
# no ``-c:s`` clause and lets the container choose - so this vocabulary exists for
# gates that would rather refuse a mode than leave it to the muxer's default. An
# HDR10 bitstream is the case that cares: a stream the caller asked for and the pass
# silently re-encoded is a delivery that no longer matches the request.
MUX_AUDIO_MODES = ("copy", "aac", "opus", "flac", "pcm", "remove")
MUX_SUBTITLE_MODES = ("copy", "convert", "drop")
# The video stream's own descriptive fields a rewrap can carry across: what the track
# is called and what language it is in. They are the whole vocabulary on purpose.
# ``ENCODER``, ``BPS``, ``DURATION``, ``NUMBER_OF_FRAMES`` and their like are answers
# about the file the source *was*, and putting them on a stream this chain just encoded
# would state a measurement as if it had been made here.
VIDEO_DESCRIPTIVE_TAGS = ("title", "language")
# The flag a container assumes for a track nobody declared. An all-zero source video
# states "no flags", and the only way the deliverable can say that is to have the pass
# clear the one the muxer would otherwise leave on a stream it is about to write.
VIDEO_DESCRIPTIVE_DEFAULT_FLAG = "default"


def extra_args_request(transcode_settings) -> str:
    """The caller's extra FFmpeg arguments, or ``""`` when nothing was named.

    ``extra_args`` absent and ``extra_args`` present-but-empty are the same request -
    none - and so is a field the caller wrote as ``None``: JSON ``null`` is how a form
    that was never filled in arrives, not a decision to run the literal token ``None``
    as an argument. Treating it as an argument was a real defect, not a hypothetical
    one: it made ``needs_copy_pass`` rewrap a file for nothing and
    ``build_mux_command`` append ``shlex.split("None")`` to the FFmpeg line.
    """
    value = dict(transcode_settings or {}).get("extra_args")
    return "" if value is None else str(value).strip()


def source_video_descriptive(source):
    """The source video's descriptive fields, narrowed, or ``None`` if it has none.

    Returns ``{"title": str, "language": str, "disposition": [...]}`` with only the keys
    the source actually answers, and only when answering them changes something: an
    empty title is not carried, and a track with no title, no language and no flag set
    is not a reason to rewrap a file. A probe that never reported ``video_tags`` or
    ``video_disposition`` - every probe the SDR chain has always run - yields ``None``,
    which is what keeps a request for the fields from forcing a remux of a file that
    cannot satisfy it. Flags are sorted so that the command, the report and a test's
    expectation all say the same thing in the same order.

    When there is something to carry and the probe *did* read a disposition, an all-zero
    track is reported as ``disposition: []`` rather than as no answer: the source says no
    flags, and the pass has to clear the muxer's own default instead of delivering it as
    though the source had set it.
    """
    found = {}
    for name in VIDEO_DESCRIPTIVE_TAGS:
        value = (source.get("video_tags") or {}).get(name)
        if value is not None and str(value).strip():
            found[name] = str(value)
    disposition = source.get("video_disposition")
    stated = isinstance(disposition, dict)
    if stated:
        flags = sorted(name for name, value in disposition.items() if value)
    else:
        flags = sorted(set(disposition or ()))
    if not found and not flags:
        return None
    if stated or flags:
        found["disposition"] = flags
    return found


def video_descriptive_clauses(metadata) -> list:
    """The FFmpeg clauses that put one source video's descriptive fields on the output.

    ``-metadata:s:v:0`` writes to the first *output* video stream, which is the
    encoder's, and is never a ``-map`` of the source's video; ``-disposition:v:0``
    states the whole flag list the source carried instead of patching one flag onto
    whatever the copy happened to leave there. An empty list the source actually stated
    is the mirror image, and is written ``-default`` because FFmpeg's disposition option
    takes ``+flag`` and ``-flag`` and rejects the ``default=0`` spelling outright: the
    deliverable then says "no flags", which is what the source said, instead of keeping
    whatever flag the muxer assumed for the stream the encoder bridge wrote. Every value
    reaches FFmpeg as its own argument, so a title containing a space, an equals sign or
    a quote is one argv element and nothing more - no shell parses this command.
    """
    clauses = []
    for name in VIDEO_DESCRIPTIVE_TAGS:
        value = (metadata or {}).get(name)
        if value:
            clauses += ["-metadata:s:v:0", f"{name}={value}"]
    if "disposition" not in (metadata or {}):
        return clauses
    flags = metadata["disposition"] or ()
    if flags:
        clauses += ["-disposition:v:0",
                    "".join(f"+{name}" for name in sorted(flags))]
    else:
        clauses += ["-disposition:v:0",
                    "".join(f"-{name}" for name in (VIDEO_DESCRIPTIVE_DEFAULT_FLAG,))]
    return clauses


def needs_copy_pass(source: dict, transcode_settings, with_audio: bool,
                    out_path: str, video_descriptive_metadata=None) -> bool:
    """Whether the encoded video has to be rewrapped to honour the request.

    The bridge writes one video stream into one container. Anything the request
    adds - audio, subtitles, attachments, metadata, chapters, an MP4 tag or
    ``+faststart`` - has to come from the source, and the copy pass is what
    brings it across without a second encode.

    ``video_descriptive_metadata`` is the narrowed ``source_video_descriptive`` answer of
    a caller that intends to put a source video's own title, language and flags on the
    deliverable, because the encode is a stream the bridge wrote and inherits none of
    them. It is a parameter rather than a key read out of ``transcode_settings`` on
    purpose: the pass that would carry it has to be the same pass that was asked to, and
    a shared gate that rewrapped a file for a clause its builder never writes would be
    the silent loss this project refuses everywhere else. The SDR chain passes nothing and
    decides exactly what it always decided.
    """
    ts = dict(transcode_settings or {})
    if video_descriptive_metadata:
        # Checked ahead of the empty-plan short-circuit below: a caller that named the
        # fields and got a non-empty answer from the probe has made a request, whatever
        # the rest of its plan says.
        return True
    lowered = out_path.lower()
    if not lowered.endswith(INTERMEDIATE_CONTAINER):
        return True
    if not ts:
        return bool(with_audio and source.get("audio_streams"))
    # with_audio is the job's answer to "keep the sound?", and an encoder plan that says
    # nothing about audio leaves it at its default. A request with no audio has nothing
    # to carry from the source on that account, so the flag wins over the default.
    if (with_audio and str(ts.get("audio_mode", "copy")) != "remove"
            and source.get("audio_streams")):
        return True
    if str(ts.get("subtitle_mode", "copy")) != "drop" and source.get("subtitle_streams"):
        return True
    if ts.get("copy_attachments", True):
        return True
    if ts.get("copy_metadata", True) and source.get("metadata"):
        return True
    if ts.get("copy_chapters", True) and source.get("nb_chapters"):
        return True
    return bool(extra_args_request(ts))


def encoded_video_span(ffprobe: str, encoded_video: str) -> str:
    """Return the encoded video packet span as an FFmpeg duration string."""
    try:
        raw = subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=time_base:packet=pts,duration",
            "-show_packets", "-of", "json", encoded_video],
            stderr=subprocess.DEVNULL, timeout=30)
        data = json.loads(raw.decode("utf-8"))
        streams, packets = data.get("streams") or [], data.get("packets") or []
        if not streams or not packets:
            raise ValueError("video stream has no packets")
        num, den = (int(part) for part in streams[0]["time_base"].split("/"))
        if num <= 0 or den <= 0:
            raise ValueError("invalid video time base")
        pts = [int(packet["pts"]) for packet in packets if "pts" in packet]
        if len(pts) != len(packets) or any(
                int(packet.get("duration", 0)) <= 0 for packet in packets):
            raise ValueError("video packet timestamps or durations are missing")
        ends = [int(packet["pts"]) + int(packet["duration"])
                for packet in packets]
        span = Fraction(max(ends) - min(pts)) * Fraction(num, den)
        if span <= 0:
            raise ValueError("video packet span is not positive")
        # Round upward at sub-picosecond precision so decimal conversion cannot
        # cut the final packet boundary down.
        scale = 10**12
        ticks = (span.numerator * scale + span.denominator - 1) // span.denominator
        whole, fractional = divmod(ticks, scale)
        return f"{whole}.{fractional:012d}".rstrip("0").rstrip(".")
    except Exception as exc:
        raise BridgeError(f"cannot determine encoded video packet span: {exc}") from exc


def build_mux_command(ffmpeg: str, encoded_video: str, source: str,
                      transcode_settings, out_path: str,
                      with_audio: bool = True,
                      video_descriptive_metadata=None,
                      video_authoritative: bool = False,
                      video_duration: str | None = None) -> list:
    """Assemble the copy-and-map pass around the already encoded video.

    Input 0 is the encode and input 1 is the original, which is the same shape
    ``build_transcode_command`` gives its own mapping clauses, so audio,
    subtitles, attachments, metadata and chapters land the way they would have.
    ``-c:v copy`` is what keeps this from being a re-encode.

    ``video_descriptive_metadata`` is the narrowed answer of
    ``source_video_descriptive`` above - a source video's title, language and set
    disposition flags - and it is applied to the *output* video stream, which is the
    encoder's stream. That distinction is the whole clause: the source's video is never
    mapped, its bitstream is never substituted, and nothing the source's own encoder
    recorded about bitrate, duration or frame count is written onto a stream this run
    produced. Left out, which is how every job that never asked for it arrives, the
    command is exactly what it has always been.
    """
    ts = dict(transcode_settings or {})
    # The source is input 1 of this command and out_path is its -y destination, so
    # one file on both sides would be truncated while still being read for the
    # streams the bridge cannot carry.
    pipeline.assert_output_not_input(source, out_path)
    cmd = [ffmpeg, "-y", "-loglevel", "warning", "-i", encoded_video]
    if "_source_range_start_seconds" in ts:
        start, duration = float(ts["_source_range_start_seconds"]), float(ts["_source_range_duration_seconds"])
        if not math.isfinite(start) or start < 0 or not math.isfinite(duration) or duration <= 0:
            raise BridgeError("invalid source selection timeline")
        cmd += ["-ss", f"{start:.9f}", "-t", f"{duration:.9f}"]
    cmd += ["-i", source, "-map", "0:v:0", "-c:v", "copy"]
    # The same precedence the encoder plan is held to above: a job that asked for no
    # audio gets no audio, whatever the mode field defaults to.
    audio_mode = ("remove" if not with_audio else
                  (str(ts.get("audio_mode", "copy")) if ts else "copy"))
    if audio_mode != "remove":
        cmd += ["-map", "1:a?"]
    subtitle_mode = str(ts.get("subtitle_mode", "copy")) if ts else "copy"
    if subtitle_mode != "drop":
        cmd += ["-map", "1:s?"]
    attachments = bool(ts.get("copy_attachments", True)) if ts else False
    if attachments and out_path.lower().endswith(".mkv"):
        cmd += ["-map", "1:t?"]
    if ts:
        cmd += ["-map_metadata", "1" if ts.get("copy_metadata", True) else "-1",
                "-map_chapters", "1" if ts.get("copy_chapters", True) else "-1"]
    else:
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]
    cmd += video_descriptive_clauses(video_descriptive_metadata)

    bitrate = str(ts.get("audio_bitrate", "320k"))
    shortest = [] if video_authoritative else ["-shortest"]
    if audio_mode == "copy":
        cmd += ["-c:a", "copy"] + shortest
        if "_source_range_start_seconds" in ts:
            # A demux seek starts at a preceding video keyframe. Audio packets
            # before the selected time must not shift the encoded video forward.
            cmd += ["-copypriorss:a", "0"]
    elif audio_mode == "aac":
        cmd += ["-c:a", "aac", "-b:a", bitrate] + shortest
    elif audio_mode == "opus":
        cmd += ["-c:a", "libopus", "-b:a", bitrate] + shortest
    elif audio_mode == "flac":
        cmd += ["-c:a", "flac"] + shortest
    elif audio_mode == "pcm":
        cmd += ["-c:a", "pcm_s24le"] + shortest
    elif audio_mode == "remove":
        cmd += ["-an"]
    else:
        raise BridgeError(f"audio_mode={audio_mode} is not a mode the pipe path "
                          "offers either")

    if subtitle_mode == "copy":
        cmd += ["-c:s", "copy"]
    elif subtitle_mode == "convert":
        cmd += ["-c:s", "webvtt" if out_path.lower().endswith(".webm") else
                "srt" if out_path.lower().endswith(".mkv") else "mov_text"]
    elif subtitle_mode == "drop":
        cmd += ["-sn"]
    if subtitle_mode != 'drop' and "_source_range_start_seconds" in ts:
        duration = float(ts['_source_range_duration_seconds'])
        # Clip packet display intervals while preserving copied subtitle bytes.
        cmd += ['-bsf:s', f"setts=pts='max(PTS,0)':duration='max(0,min(PTS+DURATION,{duration:.9f}/TB)-max(PTS,0))'"]
    if attachments and out_path.lower().endswith(".mkv"):
        cmd += ["-c:t", "copy"]

    if out_path.lower().endswith((".mp4", ".mov")):
        tag = {"av1_nvenc": "av01", "hevc_nvenc": "hvc1"}.get(
            str(ts.get("codec", "av1_nvenc")), "")
        if tag:
            cmd += ["-tag:v", tag]
        cmd += ["-movflags", "+faststart"]
    extra = extra_args_request(ts)
    if extra:
        extra_tokens = shlex.split(extra)
        if video_authoritative and any(t == "-shortest" or t.startswith("-shortest=")
                                       for t in extra_tokens):
            raise BridgeError("-shortest conflicts with the encoded-video timeline")
        cmd += extra_tokens
    if video_authoritative:
        if video_duration is None:
            raise BridgeError("video-authoritative remux requires encoded video duration")
        try:
            duration = float(video_duration)
        except (TypeError, ValueError) as exc:
            raise BridgeError("encoded video duration must be a positive number") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise BridgeError("encoded video duration must be a positive number")
        cmd += ["-t", str(video_duration)]
    if "_source_range_start_seconds" in ts:
        cmd += ["-avoid_negative_ts", "disabled"]
    return cmd + [out_path]


def run_copy_pass(ffmpeg: str, encoded_video: str, source: str,
                  transcode_settings, out_path: str, with_audio: bool = True,
                  on_proc_started=None, on_proc_exited=None,
                  timeout: float = MUX_TIMEOUT_SEC,
                  video_descriptive_metadata=None,
                  video_authoritative: bool = False,
                  ffprobe: str | None = None) -> list:
    """Run ``build_mux_command``'s command over an already encoded video.

    This is the execution half of the copy pass, kept with its command builder so a
    caller that has to rewrap an encode gets both from one place. The process handle
    is handed to the queue's hooks exactly as it always was, which is what lets a
    cancellation terminate the remux rather than leave it writing.

    ``video_descriptive_metadata`` is passed to the builder untouched; a caller that
    probed the source for it (``source_video_descriptive``) hands over the narrowed
    fields, and every caller that did not asks for the command it has always got.

    In video-authoritative mode, ``-t`` bounds nonvideo packet starts to the encoded
    packet span while keeping every encoded video packet. Copy-mode audio padding and
    subtitle cue durations can extend beyond that cutoff, so container duration is not
    guaranteed to equal video duration.

    A non-zero exit is a ``BridgeError`` naming FFmpeg's own last words: a run that
    could not rewrap its deliverable has no file to report, and falling back to the
    encode alone would hand the caller a file missing the streams it asked for.
    Returns the command that ran, so the job's record can name the clauses.
    """
    command = build_mux_command(ffmpeg, encoded_video, source, transcode_settings,
                                out_path, with_audio,
                                video_descriptive_metadata=video_descriptive_metadata,
                                video_authoritative=video_authoritative,
                                video_duration=(encoded_video_span(ffprobe, encoded_video)
                                                if video_authoritative and ffprobe else None))
    mux = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
    if callable(on_proc_started):
        try:
            on_proc_started(mux)
        except Exception:
            pass
    try:
        try:
            _, stderr_data = mux.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The deliverable is half-written and the process still owns it. Killing
            # it is the only way the failure the job reports is the end of the work,
            # rather than a background writer that outlives the run.
            mux.kill()
            mux.communicate()
            raise BridgeError(f"容器封装复制超时（超过 {timeout:g} 秒），已终止 FFmpeg")
    finally:
        if callable(on_proc_exited):
            try:
                on_proc_exited(mux)
            except Exception:
                pass
    if mux.returncode != 0:
        detail = (stderr_data or b"").decode("utf-8", "replace")
        raise BridgeError(f"容器封装复制失败 (rc={mux.returncode}): "
                          f"{detail[-600:]}")
    return command


# --- the gate -----------------------------------------------------------------

def multi_scale_config(settings, full_width, full_height):
    """Validate an opt-in layer vector against the sealed 1% scale contract."""
    settings = dict(settings or {})
    from native_effects import nr_as_effect
    if not settings.get('_inline_hdr_nr') and nr_as_effect(settings):
        settings.update(reconstruction_mode='single', working_scale=1.0)
    mode = str(settings.get("reconstruction_mode", "single"))
    if mode not in ("single", "dual", "triple"):
        raise ValueError("reconstruction_mode must be single, dual, or triple")
    count = {"single": 1, "dual": 2, "triple": 3}[mode]
    # Single-layer jobs may carry saved experimental values from an earlier UI
    # selection. They must retain the sealed working_scale-only contract.
    scales = ([settings.get("working_scale", 1.0)] if count == 1
              else settings.get("layer_scales"))
    if scales is None:
        raise ValueError("dual/triple requires explicit layer_scales")
    if not isinstance(scales, (list, tuple)) or len(scales) != count:
        raise ValueError(f"{mode} requires exactly {count} layer_scales")
    percents = [working_percent(float(value)) for value in scales]
    if any(value is None for value in percents):
        raise ValueError("each layer scale must be an integer percent 25..100")
    if len(set(percents)) != count and settings.get('fusion_mode') != 'sequential':
        raise ValueError("layer scales must be distinct")
    if percents[0] != working_percent(float(settings.get("working_scale", 1.0))):
        raise ValueError("first layer scale must match working_scale")
    weights = ([1.0] if count == 1 else
               settings.get("layer_weights", [1.0] * count))
    if not isinstance(weights, (list, tuple)) or len(weights) != count:
        raise ValueError(f"{mode} requires exactly {count} layer_weights")
    weights = [float(value) for value in weights]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("layer weights must be finite and positive")
    fusion = ("catmull_baseline" if count == 1 else
              str(settings.get("fusion_mode", "catmull_baseline")))
    if fusion not in ("catmull_baseline", "data_consistency_experimental", "sequential"):
        raise ValueError("unknown fusion_mode")
    beta = float(settings.get("fusion_beta", 0.1)) if count > 1 else 0.1
    tolerance = (float(settings.get("fusion_relative_tolerance", 0.001))
                 if count > 1 else 0.001)
    iterations = int(settings.get("fusion_max_iterations", 128)) if count > 1 else 128
    if (not math.isfinite(beta) or beta <= 0 or
            not math.isfinite(tolerance) or not 0 < tolerance < 1 or
            not 1 <= iterations <= 1000):
        raise ValueError("fusion requires beta>0, 0<tolerance<1, and "
                         "1..1000 maximum iterations")
    dims = [pipeline.resolve_working_dims(full_width, full_height,
                                          percent / 100.0)
            for percent in percents]
    if any(w > full_width or h > full_height for w, h in dims):
        raise ValueError("a layer resolves beyond the full-resolution source")
    return {"mode": mode, "scales": [value / 100.0 for value in percents],
            "weights": weights, "dims": dims, "fusion": fusion,
            "beta": beta, "relative_tolerance": tolerance,
            "max_iterations": iterations}


def _frame_timestamp_discontinuity(previous, frame, fps):
    """Detect a missing/backward frame using decoder PTS in its own time base."""
    current = (int(frame.pts), int(frame.time_base_num),
               int(frame.time_base_den))
    if previous is None:
        return True
    old_pts, old_num, old_den = previous
    pts, num, den = current
    if num <= 0 or den <= 0 or (num, den) != (old_num, old_den):
        return True
    delta = pts - old_pts
    if delta <= 0:
        return True
    expected = Fraction(fps[1] * den, fps[0] * num)
    return Fraction(delta) > expected * Fraction(3, 2)

def negotiate(dims: dict, source: dict, job_settings, transcode_settings,
              cq: int, with_audio: bool, out_path: str, live,
              option_available=None, bridges_ok=True, bridge_error=None,
              flow_provider=None, extra_reasons=None, tune_available=None,
              capabilities=None) -> dict:
    """Say whether this export can run on the GPU, and exactly why not.

    Every input is a fact the caller already has: the probed source, the
    requested settings, the resolved dimensions, and what the loaded libraries
    advertise. Nothing here touches an image, which is what lets the whole gate
    be tested without a display attached.
    """
    js = dict(job_settings or {})
    ts = dict(transcode_settings or {})
    reasons = []
    import native_effects
    try:
        native_effects.validate(js, source)
    except ValueError as exc:
        reasons.append(str(exc))
    if not bridges_ok:
        reasons.append("bridges: " + (bridge_error or "unavailable"))
    if str(js.get("dlss_backend")) != "gpu_flow_d3d12":
        reasons.append("export_backend=gpu_transfer is the DLSS GPU-flow "
                       "backend; dlss_backend must be gpu_flow_d3d12 too")
    guidance_mode = int(js.get("guidance_mode", 0))
    uses_flow = guidance_mode == 1 and not isinstance(live, native_effects.Passthrough)
    if guidance_mode not in (0, 1):
        reasons.append(f"guidance_mode={guidance_mode} is neither 0 (off) nor "
                       "1 (flow); this chain runs one or the other")
    if uses_flow:
        if str(js.get("flow_engine", "nvof")) not in ('nvof', 'fidelityfx', 'raft', 'raft_large'):
            reasons.append(f"flow_engine={js.get('flow_engine')} has no device path")
        if not callable(getattr(flow_provider, "get_flow_device_rgba", None)):
            reasons.append("the flow provider cannot be handed a CUDA surface, so the "
                           "working frame would have to be downloaded to compute flow")
        elif not flow_provider.supports_device_rgba_flow():
            reasons.append("the NVOF bridge has no device RGBA input entry point")
    elif guidance_mode == 0:
        # Motion vectors come from a texture the DLSS host zeroed once at open and
        # nobody writes again, so an unguided frame needs no flow engine at all -
        # and must not quietly be given one, which is why flow_engine is not even
        # looked at here. What it does need is a session that binds that texture;
        # an ABI-2 host has no such entry point and cannot be talked into one.
        if not (callable(getattr(live,
                                 "process_device_color_zero_motion_resource", None))
                and bool(getattr(live, "supports_zero_motion", lambda: False)())):
            reasons.append("guidance_mode=0 needs a DLSS GPU host session that binds "
                           "a resident zero motion texture (dlss_gpu_host ABI 3); the "
                           "loaded session cannot answer for it")
    # The output side is the same question in both modes: whichever DLSS entry the
    # run calls, it has to hand back a texture, not a copy of one.
    dlss_entry = ("process_device_color_flow_resource" if uses_flow
                  else "process_device_color_zero_motion_resource")
    if not hasattr(live, dlss_entry):
        reasons.append(f"the DLSS session exposes no {dlss_entry}, so it cannot hand "
                       "back a D3D12 output texture and the stage that consumes it - "
                       "the residual at 0.5, the encoder at 1.0 - would have to read "
                       "it into CPU memory")
    if int(js.get("multi_pass", 1) or 1) != 1:
        reasons.append("multi_pass is not implemented by the production pipeline")

    src_w, src_h = int(dims['src_w']), int(dims['src_h'])
    try:
        src_w, src_h = native_effects.processing_size(js, (src_w, src_h))
    except ValueError:
        pass  # Native validation above already refused this configuration.
    live_w, live_h = int(dims["live_w"]), int(dims["live_h"])
    dst_w, dst_h = int(dims["dst_w"]), int(dims["dst_h"])
    multi = None
    try:
        multi = multi_scale_config(js, src_w, src_h)
    except (TypeError, ValueError) as error:
        reasons.append(f"multi-scale configuration: {error}")
    if multi is not None and len(multi["scales"]) > 1:
        if not (capabilities or {}).get("low_residual"):
            reasons.append("independent signed GPU residual bridge is unavailable")
        if multi["dims"][0] != (live_w, live_h):
            reasons.append("first layer extent differs from working dimensions")
        if not getattr(getattr(live, "_lib", None), "multi_session_supported", False):
            reasons.append("GPU host has no independent Feature 18 layer export")
        if multi["fusion"] == "data_consistency_experimental" and not (
                capabilities or {}).get("fusion"):
            reasons.append("GPU data-consistency fusion bridge is unavailable")
        if uses_flow and not callable(getattr(flow_provider,
                                           "new_independent_layer", None)):
            reasons.append("flow provider cannot create independent NVOF layers")
        for layer_scale, (layer_w, layer_h) in zip(multi["scales"][1:],
                                                    multi["dims"][1:]):
            is_exact = (layer_scale == 0.5 and
                        (layer_w, layer_h) == (src_w // 2, src_h // 2))
            is_identity = (layer_w, layer_h) == (src_w, src_h)
            if not (is_exact or is_identity) and not ((capabilities or {}).get("lanczos_scaled") and
                                     (capabilities or {}).get("residual_scaled")):
                reasons.append(f"layer {layer_scale:g} needs scaled bridge exports")
            width, _ = plan_tap_widths(src_w, src_h, layer_w, layer_h)
            reported = (capabilities or {}).get("lanczos_max_taps")
            capacity = (LEGACY_LANCZOS_MAX_TAPS if reported is None
                        else int(reported))
            if width > capacity:
                reasons.append(f"layer {layer_scale:g} needs {width} Lanczos "
                               f"taps but loaded bridge has {capacity}")
    scale = native_effects.core_working_scale(js)
    # Three chain shapes over one served range. 100% is the identity chain: the
    # full RGBA is the working colour and the DLSS output is the encoder input,
    # which is what the CPU plan does there - no working downsample and no
    # residual stage exist. 50% on a source whose axes it can halve exactly is
    # the pair of exact-2x stages, whose taps are constants in the kernels. Every
    # other whole percentage is that same pair of stages driven by the caller's
    # tap plan, because no constant table describes a resample that is not a
    # double. Nothing is served on the strength of having been measured before:
    # the extents the percentage resolves to decide the shape, and the two
    # capability questions a plan can still fail - does this build export a
    # plan-driven open, and does the kernel walk this window - are asked below.
    percent = working_percent(scale)
    identity = (percent == WORKING_PERCENT_MAX
                and (live_w, live_h) == (src_w, src_h))
    exact_two_x = (percent == 50 and (live_w, live_h) == (src_w // 2, src_h // 2))
    # The constant-tap kernels are the cheaper pair for the size they describe,
    # so 50% on a source it halves exactly is not a plan-driven request even
    # though a plan would serve it; every other percentage below 100 is.
    plan_scale = None if exact_two_x else plan_driven_scale(
        src_w, src_h, live_w, live_h, scale)
    plan_driven = plan_scale is not None
    downscaled = exact_two_x or plan_driven
    if not (identity or downscaled):
        if percent is None:
            detail = (f"working_scale={scale!r} is not a whole percentage "
                      f"between {WORKING_PERCENT_MIN} and {WORKING_PERCENT_MAX}, "
                      "and a fraction of the source that is not one of those has "
                      "no size this chain resolves")
        elif percent == WORKING_PERCENT_MAX:
            detail = (f"100% of {src_w}x{src_h} is the source size, and this job "
                      "declared a different working size")
        elif live_w >= src_w or live_h >= src_h:
            detail = (f"{percent}% of {src_w}x{src_h} leaves no axis below the "
                      "source once 8-alignment is applied, so a working downsample "
                      "has nothing to downsample")
        else:
            detail = (f"{percent}% of {src_w}x{src_h} resolves to "
                      "{}x{}, not the size this job declared".format(
                          *pipeline.resolve_working_dims(src_w, src_h,
                                                         percent / 100.0)))
        reasons.append(
            f"working_scale={scale} resolves to {live_w}x{live_h} "
            f"from {src_w}x{src_h}: {detail}. This backend serves every whole "
            "percentage from 25% to 100%: 1.0 is the identity chain (the full RGBA "
            "is the working colour and there is no residual stage), 0.5 on a source "
            "it halves exactly is a GPU Lanczos downsample and a GPU residual "
            "upscale on constant taps, and every other percentage is those same two "
            "stages opened with the CPU plan's own tap tables at the resolved size")
    if plan_driven:
        missing = [name for name, key in (("gpu_lanczos_bridge", "lanczos_scaled"),
                                          ("gpu_residual_bridge", "residual_scaled"))
                   if not (capabilities or {}).get(key)]
        if missing:
            reasons.append(" and ".join(missing) +
                           f" exports no plan-driven open, so this build cannot "
                           f"serve working_scale={plan_scale:g}; rebuild it from "
                           "native/. The exact-2x and identity chains are "
                           "unaffected.")
        # The other thing a plan can need that a build does not have is a wider
        # window than the kernel walks. Both limits are arithmetic the CPU plan
        # already implies for these extents, so a source whose plan outgrows the
        # kernel is refused here, before a single frame is decoded, instead of
        # failing at the open or - the alternative that would look like success -
        # truncating the taps and resampling with a different filter.
        working_color_taps, upscale_taps = plan_tap_widths(src_w, src_h,
                                                           live_w, live_h)
        # The window a request has to fit is the one the loaded binary walks, so
        # it is asked of the binary. A build that was never asked answers with the
        # width its own kernel was compiled with, not with this checkout's mirror:
        # see LEGACY_LANCZOS_MAX_TAPS for what that covers.
        reported = (capabilities or {}).get("lanczos_max_taps")
        window = LEGACY_LANCZOS_MAX_TAPS if reported is None else int(reported)
        unreported = "" if reported is not None else (
            " - this build exports no gpu_lanczos_max_taps, so the figure is the "
            "width its own kernel was compiled with")
        if working_color_taps > window:
            reasons.append(f"the {plan_scale:g} Lanczos plan is {working_color_taps} "
                           f"taps wide, past the {window} "
                           f"gpu_lanczos_bridge walks{unreported}; a truncated plan is a "
                           "different filter, so this is refused rather than "
                           "narrowed")
        if upscale_taps > RESIDUAL_MAX_TAPS:
            reasons.append(f"the {plan_scale:g} Catmull-Rom plan is {upscale_taps} "
                           f"taps wide, past the {RESIDUAL_MAX_TAPS} "
                           "gpu_residual_bridge walks; a truncated plan is a "
                           "different filter, so this is refused rather than "
                           "narrowed")
    if downscaled and not bool(js.get("residual_pyramid", True)) and (
            multi is None or multi["mode"] == "single"):
        # Pyramid off matters only where a residual stage exists to bypass it;
        # at 1.0 the CPU plan has none, so the setting is inert there.
        reasons.append("residual_pyramid off leaves a CPU resize in the path")
    expected_output = (int(dims['src_w']), int(dims['src_h']))
    try:
        expected_output = native_effects.output_size(js, expected_output)
    except ValueError as exc:
        reasons.append(str(exc))
    if (dst_w, dst_h) != expected_output:
        from export_resize import validate
        try:
            validate((dst_w, dst_h), ts.get('scale_algorithm', 'lanczos'))
        except ValueError as exc:
            reasons.append(str(exc))
    if src_w % 2 or src_h % 2:
        reasons.append("source dimensions must both be even")

    if source.get("video_stream_count") != 1:
        reasons.append(f"{source.get('video_stream_count')} video streams: the "
                       "decoder bridge opens exactly one")
    image_input = source.get('media_type') == 'image' and source.get('pix_fmt') == 'rgba'
    if not image_input and source.get("pix_fmt") not in ("yuv420p", "nv12"):
        reasons.append(f"pix_fmt={source.get('pix_fmt')} is not 8-bit 4:2:0, "
                       "which is the only thing NVDEC yields here as NV12")
    if str(source.get("color_transfer") or "") in ("smpte2084", "arib-std-b67"):
        reasons.append("HDR transfer: the chain carries 8-bit SDR pixels")
    if not image_input and source.get("color_range") in ("pc", "jpeg"):
        reasons.append("full-range NV12: the GPU converter only handles limited range")
    source_matrix = source.get("color_space")
    if not image_input and source_matrix not in BT601_FAMILY and source_matrix != "bt709":
        reasons.append(f"color_space={source.get('color_space')} is not the "
                       "supported BT.601 or BT.709 matrix")
    color_matrix = (COLOR_MATRIX_BT709 if source_matrix == "bt709"
                    else COLOR_MATRIX_BT601)
    preset = str(ts.get("color_preset", "preserve"))
    if preset not in ("preserve", "none", "", "None"):
        reasons.append(f"color_preset={preset} writes colour metadata that has "
                       "to come from the conversion this backend performs")

    # The probe's own rational wins: a frame rate that arrived here as a float has
    # already lost the 1001, and 29.97 of 100 is not 30000 of 1001.
    fps = fps_rational(source) or fps_rational({}, dims.get("fps"))
    if not fps:
        reasons.append("frame rate is not a positive rational, so the encoder "
                       "cannot be told a time base")
    plan = plan_encoder(dict(ts,pixel_format='auto',profile='auto') if js.get('truehdr_enabled') else ts,
                        cq, source, option_available, tune_available,
                        color_matrix=color_matrix)
    plan["decoder_color_matrix"] = (NV12_COLOR_MATRIX_BT709
                                    if source_matrix == "bt709" else COLOR_MATRIX_BT601)
    reasons += plan["unsupported"]
    reasons += list(extra_reasons or ())

    return {"supported": not reasons, "reasons": reasons, "plan": plan,
            "multi_scale": multi,
            "fps": fps,
            # Which motion source the run was accepted on. The export reads this
            # rather than re-reading job_settings, so a frame cannot be guided by
            # a decision the gate never made.
            "guidance_mode": guidance_mode,
            "uses_flow": uses_flow,
            # Which chain shape a supported request runs: the exact-2x stages at
            # 0.5, the same two stages driven by a tap plan at every other whole
            # percentage below 100, or the identity chain at 1.0 where the CPU
            # plan has no residual stage. The export reads this instead of
            # reclassifying, and works out the ratio it classified from
            # working_scale, which is the same number.
            "downscaled": downscaled,
            "plan_driven": plan_driven,
            "working_scale": scale,
            "declared_color_space": ("bt709" if color_matrix == COLOR_MATRIX_BT709
                                     else "smpte170m"),
            "copy_pass": needs_copy_pass(source, ts, with_audio, out_path)}


# --- the run ------------------------------------------------------------------

class Chain:
    def settings_native_mode(self):
        # Only the standalone VSR resource owner has no NGX inference feature.
        return 'vsr' if getattr(self.live, 'session_id', None) == 0 and getattr(
            self.live, 'host_dll', None) == 'gpu_rgba_exchange.dll' else 'nr'

    """The bridges for one export, opened on the first decoded frame.

    Nothing can be opened before the decoder reports its CUDA context and
    extents, and the DLSS host's D3D12 device is what the residual stage and
    the encoder are both validated against, so the first frame decides the shape
    of every one after it.

    ``downscaled`` is the chain shape the negotiation classified: True runs the
    two resampling stages (Lanczos down, residual up); False is the 1.0 identity
    chain, where the full RGBA is the working colour and the DLSS output is the
    encoder input - the same no-residual shape the CPU plan runs at 1.0.

    ``tap_plans`` is what those two stages resample with. None means the ratio is
    the exact double, whose taps are constants inside the kernels. A pair of
    plan views means the caller asked for a scale no constant table describes,
    and each stage is opened by its plan-driven export instead - which is a
    different entry point, not a different argument to the same one, so a bridge
    that cannot take a plan cannot be talked into running one.

    ``uses_flow`` is the motion source the negotiation accepted. True asks the
    flow provider for a device plane every frame and passes it to DLSS; False is
    guidance_mode=0, where the DLSS host binds its own resident zero motion
    texture and this chain never touches a flow engine - no NVOF session, no
    motion plane, no per-frame motion upload.

    ``preview_consumer`` is an optional synchronous sink for the final processed
    D3D12 texture. Its descriptor is borrowed and is valid only until the next
    operation on this chain; returning its value exposes it as ``preview_output``.
    With no consumer, the existing NVENC path is used unchanged.
    """

    def __init__(self, bridges, video, encoder_output, live, plan, fps, strength,
                 working, flow_provider=None, downscaled=True, tap_plans=None,
                 uses_flow=True, defer_encode=False, preview_consumer=None):
        self.b = bridges
        self.video = video
        self.encoder_output = encoder_output
        self.live = live
        self.plan = plan
        from prores_export import ProResOutput
        self.prores_output = ProResOutput(self) if plan.get('codec') == 'prores_ks' and preview_consumer is None else None
        self.fps = fps
        self.strength = strength
        self.working = (int(working[0]), int(working[1]))
        self.flow_provider = flow_provider
        self.downscaled = bool(downscaled)
        self.uses_flow = bool(uses_flow)
        self.tap_plans = tap_plans
        self.defer_encode = bool(defer_encode)
        self.preview_consumer = preview_consumer
        self._defer_preview_consumer = False
        self.flow_calls = 0
        self.zero_motion_calls = 0
        self.decoder = None
        self.image_decoder = None
        self._color_stages_opened = False
        self.nv12 = None
        self.lanczos = None
        self.residual = None
        self.encoder = None
        self.first = True
        from post_control import GpuPostControl
        self.post_control = GpuPostControl()
        from temporal_control import GpuTemporalControl
        self.temporal_control = GpuTemporalControl()
        self._frame_reset = True
        self.output_effect = None
        self.input_effect = None
        self.scene_observer = None
        self.framegen = self.framegen_input = self.framegen_output = None

    def _processing_frame(self, frame):
        import native_effects
        size = getattr(self, 'processing_size', (frame.width, frame.height))
        return native_effects.frame_at_size(frame, size) if size != (frame.width, frame.height) else frame

    def open_decoder(self):
        from image_source import is_image, DeviceImage
        if is_image(self.video):
            self.image_decoder = DeviceImage(self.video)
            return
        handle = C.c_void_p()
        code = self.b["nvdec"].nvdec_bridge_open(os.fsencode(self.video),
                                                 C.byref(handle))
        if code:
            _bridge_error(self.b["nvdec"], "nvdec_bridge_last_error", code)
        self.decoder = handle

    def next_frame(self, frame):
        if self.image_decoder is not None:
            return self.image_decoder.frame(frame)
        return int(self.b["nvdec"].nvdec_bridge_next(self.decoder,
                                                     C.byref(frame)))

    def cancel_decoder(self):
        if self.decoder:
            self.b["nvdec"].nvdec_bridge_cancel(self.decoder)

    def _open_color_stages(self, frame):
        if self._color_stages_opened:
            return
        self._color_stages_opened = True
        nv12 = self.b["nv12"]
        handle = C.c_void_p()
        if not hasattr(frame, 'image_rgba'):
            code = nv12.nv12_rgba_open_matrix(frame.cuda_context, frame.width,
                frame.height, self.plan["decoder_color_matrix"], C.byref(handle))
            if code:
                _bridge_error(nv12, "nv12_rgba_last_error", code)
            self.nv12 = handle
        frame = self._processing_frame(frame)
        if not self.downscaled:
            # The 1.0 chain has no working-color stage to open: the full RGBA
            # the converter just produced is the working colour itself.
            return
        lanczos = self.b["lanczos"]
        handle = C.c_void_p()
        if self.tap_plans is None:
            code = lanczos.gpu_lanczos_open(frame.cuda_context, frame.width,
                                            frame.height, C.byref(handle))
        else:
            # The extents the decoder reports are what the plan was built for,
            # and the bridge checks the axes against them before it uploads, so
            # a resolved size that does not match its own table fails here with
            # the row and tap that disagree rather than resampling wrongly.
            code = lanczos.gpu_lanczos_open_scaled(frame.cuda_context,
                frame.width, frame.height, self.working[0], self.working[1],
                C.byref(self.tap_plans.working_color), C.byref(handle))
        if code:
            _bridge_error(lanczos, "gpu_lanczos_last_error", code)
        self.lanczos = handle

    def _open_residual_and_encoder(self, dlss, frame):
        if self.downscaled and not self.defer_encode:
            residual = self.b["residual"]
            handle = C.c_void_p()
            if self.tap_plans is None:
                code = residual.gpu_residual_open(dlss["d3d12_device"],
                    self.working[0], self.working[1], C.byref(handle))
            else:
                code = residual.gpu_residual_open_scaled(dlss["d3d12_device"],
                    frame.width, frame.height, self.working[0],
                    self.working[1], C.byref(self.tap_plans.upscale),
                    C.byref(handle))
            if code:
                _bridge_error(residual, "gpu_residual_last_error", code)
            self.residual = handle

        # Preview consumers own the final texture themselves. Keep the residual
        # stage above, but never open an encoder for a preview-only chain.
        if self.preview_consumer is not None or self.output_effect is not None or getattr(self, 'truehdr_tail',None) or getattr(self, 'framegen_tail', None) or getattr(self, '_defer_layer_output', False):
            return
        self._open_encoder(dlss, frame, getattr(self, 'export_size', (frame.width, frame.height)))

    def _open_encoder(self, dlss, frame, size):
        if self.prores_output is not None:
            return
        struct = self.plan["struct"]
        blob = extra_options_blob(self.plan["av_options"])
        options = _NvencOptions(C.sizeof(_NvencOptions), struct["codec"],
            struct["preset"], struct["tune"], struct["multipass"],
            struct["rate_control"], struct["constant_quality"], struct["bitrate"],
            struct["gop"], struct["color_matrix"], blob or None)
        encoder = C.c_void_p()
        from vfx_framegen import encoder_timing
        rate, tb = encoder_timing({} if getattr(self, 'ordered_framegen', None) else getattr(self, 'framegen_settings', {}), getattr(self, '_encoder_fps', self.fps),
            (frame.time_base_num, frame.time_base_den))
        code = self.b["nvenc"].nvenc_open(dlss["d3d12_device"],
            frame.cuda_context, *size, *rate, *tb,
            os.fsencode(self.encoder_output), C.byref(options), C.byref(encoder))
        if code:
            _bridge_error(self.b["nvenc"], "nvenc_last_error", code)
        self.encoder = encoder

    def _submit_output(self, frame, candidate, stages, *, started=None):
        """Send the final texture to either a preview consumer or the encoder."""
        if self.preview_consumer is not None and self._defer_preview_consumer:
            return None
        if self.output_effect is not None:
            mark = time.perf_counter()
            candidate = self.output_effect.process(candidate, frame, self._frame_reset)
            stages['native_output_effect'] = (time.perf_counter() - mark) * 1000
        if self.preview_consumer is not None:
            if self._defer_preview_consumer:
                return None
            if getattr(self,'truehdr_tail',None):
                return self.preview_consumer(self.truehdr_tail.process(candidate,frame),frame)
            descriptor = dict(candidate)
            descriptor.update({
                "format": "rgba8_unorm",
                "width": int(candidate.get('width', frame.width)),
                "height": int(candidate.get('height', frame.height)),
                "cuda_context": int(frame.cuda_context),
                "dxgi_format": int(descriptor.get("dxgi_format", 28)),
                "state": int(descriptor.get("state", 0x800)),
            })
            return self.preview_consumer(descriptor, frame)

        if getattr(self, 'framegen_settings', None) and not getattr(self, 'ordered_framegen', None):
            from native_effects import Exchange, frame_at_size
            if self.framegen is None:
                from vfx_framegen import Interpolator
                size = (candidate.get('width', frame.width), candidate.get('height', frame.height))
                wide = candidate.get('dxgi_format') == 10
                self.framegen_input = Exchange(candidate['d3d12_device'], frame.cuda_context, size, wide=wide)
                self.framegen_output = Exchange(candidate['d3d12_device'], frame.cuda_context, size, wide=wide)
                self.framegen = Interpolator(frame.cuda_context, size, self.framegen_settings,
                    self.fps, wide=wide, hdr=bool(candidate.get('output_hdr')), cancel=getattr(self, 'cancel', None))
            mark = time.perf_counter()
            color = self.framegen_input.color(candidate)
            if candidate.get('output_hdr'): color['output_hdr'] = True
            self._framegen_frame = frame_at_size(frame, (frame.width, frame.height))
            self.framegen.submit(color, frame, self._frame_reset, self._emit_generated)
            stages['vfg_and_nvenc_submit'] = (time.perf_counter()-mark)*1000
            return None
        if getattr(self,'truehdr_tail',None):
            self.truehdr_tail.encode(self,candidate,frame)
            return None
        candidate = self._resize_output(candidate, frame, stages)
        if self.prores_output is not None:
            mark = time.perf_counter()
            self.prores_output.encode(candidate, frame)
            stages['prores_cpu_encode'] = (time.perf_counter()-mark)*1000
            return None
        if self.encoder is None and (self.output_effect is not None or getattr(self, 'export_resize', None) is not None or getattr(self,'_defer_layer_output',False)):
            self._open_encoder(candidate, frame, (candidate.get('width', frame.width), candidate.get('height', frame.height)))
        mark = time.perf_counter() if started is None else started
        code = self.b["nvenc"].nvenc_encode_texture(
            self.encoder, candidate["d3d12_resource"],
            candidate.get("d3d12_fence"), candidate.get("fence_value", 0),
            frame.pts)
        if code:
            _bridge_error(self.b["nvenc"], "nvenc_last_error", code)
        stages["nvenc_submit"] = (time.perf_counter() - mark) * 1000.0
        return None

    def process(self, frame, flow_provider, index, is_discontinuous):
        """One frame, in the order the borrowed pointers require.

        Each stage's output is valid only until the next call into its bridge,
        which is why nothing here can be reordered or overlapped. The returned
        durations are wall clock around each bridge call, which is as far as the
        measurement goes: a bridge that returns before the GPU has finished
        makes its own number a submission cost, not a completion cost.
        """
        nv12, lanczos = self.b["nv12"], self.b["lanczos"]
        stages = {}
        mark = time.perf_counter()
        full_ptr, full_pitch = C.c_uint64(), C.c_uint32()
        supplied = getattr(self, '_ordered_color', None)
        if supplied is None and hasattr(frame, 'image_rgba'):
            full_ptr.value, full_pitch.value = frame.image_rgba
        elif supplied is None:
            code = nv12.nv12_rgba_convert(self.nv12, frame.y_device_ptr,
                frame.y_pitch_bytes, frame.uv_device_ptr, frame.uv_pitch_bytes,
                C.byref(full_ptr), C.byref(full_pitch))
            if code:
                _bridge_error(nv12, "nv12_rgba_last_error", code)
        else:
            full_ptr.value, full_pitch.value = supplied['device_ptr'], supplied['pitch_bytes']
        stages["nv12_to_rgba"] = (time.perf_counter() - mark) * 1000.0
        scene = {"scene_cut": False, "classification": "disabled"}
        if self.temporal_control.controls.enabled:
            observer = self.temporal_control
            if self.input_effect is not None:
                # Detect source cuts before model history is consumed. The post-NR
                # temporal owner observes the processed extent independently below.
                if self.scene_observer is None:
                    from temporal_control import GpuTemporalControl
                    self.scene_observer = GpuTemporalControl(self.temporal_control.controls)
                observer = self.scene_observer
            scene = observer.observe(device=self.live.d3d12_device(),
                cuda_context=frame.cuda_context, source_ptr=full_ptr.value,
                source_pitch=full_pitch.value, width=supplied['width'] if supplied else frame.width,
                height=supplied['height'] if supplied else frame.height,
                reset=self.first or is_discontinuous)
        self._preview_original_color = dict(format='rgba8_cuda',width=frame.width,height=frame.height,
            cuda_context=int(frame.cuda_context),device_ptr=full_ptr.value,pitch_bytes=full_pitch.value)
        is_discontinuous = bool(is_discontinuous or scene["scene_cut"])
        self._frame_reset = self.first or is_discontinuous

        if self.input_effect is not None:
            mark = time.perf_counter()
            if getattr(self,'_preview_reuse_prefix',False):
                color = self._preview_prefix_color
            else:
                color = self.input_effect.color(supplied or dict(width=frame.width,height=frame.height,
                    device_ptr=full_ptr.value,pitch_bytes=full_pitch.value,cuda_context=frame.cuda_context),frame,self._frame_reset)
                if self.preview_consumer is not None: self._preview_prefix_color = dict(color)
            full_ptr, full_pitch = C.c_uint64(color['device_ptr']), C.c_uint32(color['pitch_bytes'])
            stages['native_input_effect'] = (time.perf_counter() - mark) * 1000
            stages['native_input_reused'] = bool(getattr(self,'_preview_reuse_prefix',False))
        frame = self._processing_frame(frame)
        if self.input_effect is not None and self.temporal_control.controls.enabled:
            self.temporal_control.observe(device=self.live.d3d12_device(),
                cuda_context=frame.cuda_context, source_ptr=full_ptr.value,
                source_pitch=full_pitch.value, width=frame.width, height=frame.height,
                reset=self._frame_reset)

        if self.downscaled:
            mark = time.perf_counter()
            work_ptr, work_pitch = C.c_uint64(), C.c_uint32()
            code = lanczos.gpu_lanczos_process(self.lanczos, full_ptr.value,
                C.c_size_t(full_pitch.value), C.byref(work_ptr),
                C.byref(work_pitch))
            if code:
                _bridge_error(lanczos, "gpu_lanczos_last_error", code)
            stages["working_color"] = (time.perf_counter() - mark) * 1000.0
            working = {"format": "rgba8_cuda", "width": self.working[0],
                       "height": self.working[1],
                       "cuda_context": frame.cuda_context,
                       "device_ptr": work_ptr.value,
                       "pitch_bytes": work_pitch.value}
        elif not self.downscaled:
            # working_scale 1.0: the working colour is the full RGBA the
            # converter just wrote. Borrowing its pointer is the whole point of
            # the scale - a resample to the same size would be a copy, and the
            # CPU plan does not make one either.
            working = {"format": "rgba8_cuda", "width": self.working[0],
                       "height": self.working[1],
                       "cuda_context": frame.cuda_context,
                       "device_ptr": full_ptr.value,
                       "pitch_bytes": full_pitch.value}

        mark = time.perf_counter()
        def infer_primary():
            nonlocal mark
            if self.uses_flow:
                flow = flow_provider.get_flow_device_rgba(index, working,
                    frame.cuda_context, is_discontinuous=is_discontinuous)
                stages["nvof_flow"] = (time.perf_counter() - mark) * 1000.0
                self.flow_calls += 1
                mark = time.perf_counter()
                dlss = self.live.process_device_color_flow_resource(working, flow,
                    reset=self.first or is_discontinuous)
            else:
                # Nothing to fetch and nothing to upload: DLSSNR.MVec is bound to the
                # texture the host zeroed at open, so the frame that arrives here is
                # the only device resource this stage reads besides the colour.
                flow = None
                if self.settings_native_mode() != 'vsr':
                    self.zero_motion_calls += 1
                dlss = self.live.process_device_color_zero_motion_resource(
                    working, reset=self.first or is_discontinuous)
            return dlss, flow
        from preview_layer_cache import evaluate
        (dlss, flow), dlss_reused = evaluate(self, 0, infer_primary)
        stage = 'vsr_input_device_copy' if self.settings_native_mode() == 'vsr' else 'dlss_inference'
        stages[stage] = 0.0 if dlss_reused else (time.perf_counter() - mark) * 1000.0
        if self.first:
            self._open_residual_and_encoder(dlss, frame)
            self.first = False

        if self.downscaled and not self.defer_encode:
            mark = time.perf_counter()
            texture = C.c_void_p()
            code = self.b["residual"].gpu_residual_process(self.residual,
                dlss["d3d12_resource"], dlss["d3d12_fence"], dlss["fence_value"],
                frame.cuda_context, full_ptr.value, C.c_size_t(full_pitch.value),
                frame.cuda_context, work_ptr.value, C.c_size_t(work_pitch.value),
                C.c_float(self.strength), C.byref(texture))
            if code:
                _bridge_error(self.b["residual"], "gpu_residual_last_error", code)
            stages["residual"] = (time.perf_counter() - mark) * 1000.0
            encode_texture, encode_fence, encode_value = texture.value, None, 0
        elif not self.downscaled:
            # No residual stage exists at working_scale 1.0 - the CPU plan feeds
            # the DLSS output to the encoder as it is. The DLSS host leaves its
            # texture COPY_SOURCE and has already waited on its own queue, and
            # the encoder's stage waits on the producer fence before touching
            # it, so the borrowed texture crosses to NVENC without a copy.
            encode_texture = dlss["d3d12_resource"]
            encode_fence = dlss["d3d12_fence"]
            encode_value = dlss["fence_value"]

        output_candidate = None
        preview_output = None
        if not self.defer_encode:
            candidate = {"d3d12_resource": encode_texture, "d3d12_fence": encode_fence,
                 "fence_value": encode_value, "d3d12_device": dlss["d3d12_device"],
                 "producer_completed": self.downscaled}
            controlled = (candidate if getattr(self, '_defer_layer_output', False) else
                          self._post_process(frame, full_ptr.value, full_pitch.value, candidate))
            output_candidate = controlled
            if not getattr(self, '_defer_layer_output', False):
                preview_output = self._submit_output(frame, controlled, stages)
        stages["decode_pts"] = frame.pts
        outcome = {"stages": stages, "flow": flow,
                "_processing_frame": frame,
                "_is_discontinuous": is_discontinuous, "_scene": scene,
                "_full_color": (full_ptr.value, full_pitch.value),
                "_working_color": working, "_dlss_resource": dlss, "_dlss_reused": dlss_reused,
                "nvof_timing": (flow_provider.timing()
                                if self.uses_flow and flow_provider is not None
                                else None)}
        if output_candidate is not None:
            outcome["_output_candidate"] = output_candidate
        if self.preview_consumer is not None and not self._defer_preview_consumer:
            outcome["preview_output"] = preview_output
        return outcome

    def _post_process(self, frame, source_ptr, source_pitch, candidate):
        # Stabilize first so final zero-strength and protected regions cannot
        # receive a residual reintroduced by temporal history.
        stable = self.temporal_control.process(device=candidate["d3d12_device"],
            cuda_context=frame.cuda_context, source_ptr=source_ptr,
            source_pitch=source_pitch, width=frame.width, height=frame.height,
            candidate=candidate, reset=self._frame_reset)
        capture = getattr(self, '_preview_post_capture', None)
        if capture:
            capture(device=self.live.d3d12_device(), context=frame.cuda_context,
                width=frame.width, height=frame.height, wide=False,
                source_ptr=source_ptr, source_pitch=source_pitch, candidate=stable, frame=frame)
        return self.post_control.process(device=candidate["d3d12_device"],
            cuda_context=frame.cuda_context, source_ptr=source_ptr,
            source_pitch=source_pitch, width=frame.width, height=frame.height,
            candidate=stable)

    def _emit_generated(self, color, pts, kind):
        texture = self.framegen_output.texture(color['device_ptr'], color['pitch_bytes'])
        if color.get('output_hdr'): texture['output_hdr'] = True
        tail = getattr(self, 'framegen_tail', None)
        frame = self._framegen_frame
        if tail:
            texture = tail.process(texture, frame, self.framegen.emitted == 0 or (kind == 'original' and self._frame_reset))
        if getattr(self,'truehdr_tail',None):
            self.truehdr_tail.encode(self,texture,frame,pts)
            return
        texture = self._resize_output(texture, frame, {})
        if self.prores_output is not None:
            self.prores_output.encode(texture, frame, pts)
            return
        if self.encoder is None:
            self._open_encoder(texture, frame, (texture['width'], texture['height']))
        code = self.b['nvenc'].nvenc_encode_texture(self.encoder, texture['d3d12_resource'],
            texture['d3d12_fence'], texture['fence_value'], pts)
        if code:
            _bridge_error(self.b['nvenc'], 'nvenc_last_error', code)

    def _resize_output(self, texture, frame, stages):
        size = (texture.get('width', frame.width), texture.get('height', frame.height))
        target = getattr(self, 'export_size', size)
        if size == target:
            return texture
        if getattr(self, 'export_resize', None) is None:
            from export_resize import Resize
            self.export_resize = Resize(texture['d3d12_device'], frame.cuda_context,
                size, target, self.export_scale_algorithm)
        start = time.perf_counter()
        result = self.export_resize.texture(texture)
        stages['export_resize'] = (time.perf_counter()-start)*1000
        return result

    def counters(self):
        out = {}
        if self.prores_output is not None:
            out['prores_output'] = self.prores_output.report()
        if getattr(self,'truehdr_tail',None): out['truehdr'] = self.truehdr_tail.report()
        if getattr(self, 'export_resize', None):
            out['export_resize'] = self.export_resize.report()
        if getattr(self, 'framegen', None) is not None:
            out['frame_generation'] = self.framegen.report()
            if hasattr(self, 'framegen_stage_index'):
                from ordered_vfg import stage_report
                out['frame_generation']['pipeline'] = stage_report(self)
        out["post_control"] = self.post_control.diagnostics()
        out["temporal_control"] = self.temporal_control.diagnostics()
        if self.scene_observer is not None:
            out['source_scene_observer'] = self.scene_observer.diagnostics()
        if self.decoder:
            block = _NvdecCounters()
            block.struct_size = C.sizeof(_NvdecCounters)
            if not self.b["nvdec"].nvdec_bridge_get_transfer_counters(
                    self.decoder, C.byref(block)):
                out["decoder"] = {name: getattr(block, name)
                                  for name, _ in _NvdecCounters._fields_[1:]}
        if self.nv12:
            block = _Nv12Counters()
            block.struct_size = C.sizeof(_Nv12Counters)
            if not self.b["nv12"].nv12_rgba_get_transfer_counters(self.nv12,
                    C.byref(block)):
                out["nv12"] = {name: getattr(block, name)
                               for name, _ in _Nv12Counters._fields_[1:]}
        if self.lanczos:
            block = _LanczosCounters()
            block.struct_size = C.sizeof(_LanczosCounters)
            if not self.b["lanczos"].gpu_lanczos_get_transfer_counters(
                    self.lanczos, C.byref(block)):
                out["lanczos"] = {name: getattr(block, name)
                                  for name, _ in _LanczosCounters._fields_[1:]}
        if self.encoder:
            block = _NvencCounters()
            block.struct_size = C.sizeof(_NvencCounters)
            if not self.b["nvenc"].nvenc_get_transfer_counters(self.encoder,
                    C.byref(block)):
                out["encoder"] = {name: getattr(block, name)
                                  for name, _ in _NvencCounters._fields_[1:]}
        if self.residual:
            block = _ResidualCounters()
            block.struct_size = C.sizeof(_ResidualCounters)
            if not self.b["residual"].gpu_residual_get_transfer_counters(
                    self.residual, C.byref(block)):
                out["residual"] = {name: getattr(block, name)
                                   for name, _ in _ResidualCounters._fields_[1:]}
        out["dlss"] = self.live.transfer_counters()
        if self.output_effect is not None:
            out['native_output_effect'] = self.output_effect.report()
        if self.input_effect is not None:
            out['native_input_effect'] = self.input_effect.report()
        out['render_pipeline'] = getattr(self, 'render_order', None)
        # The NVOF figures belong to the engine that ran, not to the DLSS host
        # that consumed its output, so they are read from the flow provider.
        out["nvof"] = (self.flow_provider.device_input_counters()
                       if self.flow_provider is not None else (None, None))
        return out

    def finish(self):
        if getattr(self, 'framegen', None) is not None:
            self.framegen.finish(getattr(self, 'framegen_emit', self._emit_generated))
        if self.encoder:
            code = self.b["nvenc"].nvenc_finish(self.encoder)
            if code:
                _bridge_error(self.b["nvenc"], "nvenc_last_error", code)
        if getattr(self,'truehdr_tail',None): self.truehdr_tail.finish()
        if self.prores_output is not None: self.prores_output.finish()

    def close(self):
        # Reverse creation order: each bridge borrows the device or context of
        # the one before it, so the decoder's CUDA context outlives all of them.
        # The DLSS session closes in its own slot, between the stages that import
        # its output texture and the memory its input was imported from; closing
        # it after the decoder had released that memory faults inside NGX, which
        # is the order the verified in-process smoke tears down in. Live.close()
        # is idempotent, so the session owner may still call it afterwards.
        from gpu_cleanup import release_owned, finish_cleanup
        errors = []
        release_owned(self, 'prores_output', errors)
        release_owned(self, 'post_control', errors)
        release_owned(self, 'temporal_control', errors)
        for name in ('truehdr_tail','export_resize', 'framegen_tail', 'ordered_framegen', 'framegen', 'framegen_output', 'framegen_input',
                     'scene_observer', 'output_effect', 'input_effect'):
            release_owned(self, name, errors)
        release_owned(self, 'encoder', errors, lambda handle: self.b['nvenc'].nvenc_close(handle))
        release_owned(self, 'residual', errors, lambda handle: self.b['residual'].gpu_residual_close(handle))
        if getattr(self.flow_provider, 'borrows_decoder_context', False):
            release_owned(self, 'flow_provider', errors, method='close_independent_layer')
        release_owned(self, 'live', errors)
        release_owned(self, 'lanczos', errors, lambda handle: self.b['lanczos'].gpu_lanczos_close(handle))
        release_owned(self, 'nv12', errors, lambda handle: self.b['nv12'].nv12_rgba_close(handle))
        release_owned(self, 'decoder', errors, lambda handle: self.b['nvdec'].nvdec_bridge_close(handle))
        release_owned(self, 'image_decoder', errors)
        finish_cleanup(errors)


class MultiScaleObservationChain(Chain):
    """Independent experimental Feature 18 observations for each scale.

    Each additional layer owns a Lanczos bridge, NVOF service, Feature 18
    session and signed low-resolution residual bridge. Observation-only mode
    encodes layer zero's Catmull baseline; experimental fusion instead solves
    and encodes a full-resolution residual on the GPU.
    """

    def __init__(self, *args, extra_layers, config, fusion_plans, **kwargs):
        super().__init__(*args,
            defer_encode=config["fusion"] == "data_consistency_experimental",
            **kwargs)
        self.extra_layers = extra_layers
        self._defer_layer_output = config['fusion'] == 'sequential'
        # The base call still builds layer 0's final candidate. A preview must
        # consume only after every independent layer and optional solve has run.
        self._defer_preview_consumer = self.preview_consumer is not None
        self.config = config
        self.fusion_plans = fusion_plans
        self.fusion_handle = None
        self.fusion_reports = []
        self.base_low_residual = None
        self.layer_calls = [0] * (1 + len(extra_layers))
        self.layer_reset_flags = [[] for _ in self.layer_calls]

    def _observe(self, handle, dlss, working, frame):
        library = self.b["low_residual"]
        if handle is None:
            handle = C.c_void_p()
            code = library.gpu_low_residual_open(dlss["d3d12_device"],
                working["width"], working["height"], C.byref(handle))
            if code:
                _bridge_error(library, "gpu_low_residual_last_error", code)
        texture, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        code = library.gpu_low_residual_process(handle,
            dlss["d3d12_resource"], dlss["d3d12_fence"],
            dlss["fence_value"], frame.cuda_context,
            working["device_ptr"], C.c_size_t(working["pitch_bytes"]),
            C.byref(texture), C.byref(fence), C.byref(value))
        if code:
            _bridge_error(library, "gpu_low_residual_last_error", code)
        return handle, (texture.value, fence.value, value.value)

    def _open_fusion(self, device, frame):
        options = _FusionOptions()
        options.struct_size = C.sizeof(_FusionOptions)
        options.full_width, options.full_height = frame.width, frame.height
        options.layer_count = len(self.fusion_plans)
        for index, (plan, dims, weight) in enumerate(zip(
                self.fusion_plans, self.config["dims"],
                self.config["weights"])):
            options.layers[index] = _FusionLayer(
                dims[0], dims[1],
                _fusion_axis(plan.working_color, True),
                _fusion_axis(plan.working_color, False), weight)
        options.baseline_horizontal = _fusion_axis(
            self.fusion_plans[0].upscale, True)
        options.baseline_vertical = _fusion_axis(
            self.fusion_plans[0].upscale, False)
        options.beta = self.config["beta"]
        options.residual_strength = self.strength
        options.relative_tolerance = self.config["relative_tolerance"]
        options.max_iterations = self.config["max_iterations"]
        handle = C.c_void_p()
        code = self.b["fusion"].gpu_fusion_open(device, C.byref(options),
                                                C.byref(handle))
        if code:
            _bridge_error(self.b["fusion"], "gpu_fusion_last_error", code)
        self.fusion_handle = handle

    def _open_color_stages(self, frame):
        super()._open_color_stages(frame)
        frame = self._processing_frame(frame)
        for layer in self.extra_layers:
            if layer["dims"] == (frame.width, frame.height):
                continue
            handle = C.c_void_p()
            lanczos = self.b["lanczos"]
            width, height = layer["dims"]
            if layer["tap_plans"] is None:
                code = lanczos.gpu_lanczos_open(frame.cuda_context,
                    frame.width, frame.height, C.byref(handle))
            else:
                code = lanczos.gpu_lanczos_open_scaled(frame.cuda_context,
                    frame.width, frame.height, width, height,
                    C.byref(layer["tap_plans"].working_color), C.byref(handle))
            if code:
                _bridge_error(lanczos, "gpu_lanczos_last_error", code)
            layer["lanczos"] = handle

    def process(self, frame, flow_provider, index, is_discontinuous):
        if self.config['fusion'] == 'sequential':
            from sequential_nr import process_sdr
            return process_sdr(self, frame, flow_provider, index, is_discontinuous)
        outcome = super().process(frame, flow_provider, index, is_discontinuous)
        frame = outcome.get('_processing_frame', frame)
        is_discontinuous = outcome.get("_is_discontinuous", is_discontinuous)
        reset = index == 0 or is_discontinuous
        reused_layers = [outcome["_dlss_reused"]]
        if not reused_layers[0]:
            self.layer_calls[0] += 1
            self.layer_reset_flags[0].append(bool(reset))
        self.base_low_residual, first_observation = self._observe(
            self.base_low_residual, outcome["_dlss_resource"],
            outcome["_working_color"], frame)
        observations = [first_observation]
        nr_times = [float(outcome["stages"].get("dlss_inference", 0.0))]
        full_ptr, full_pitch = outcome["_full_color"]
        for number, layer in enumerate(self.extra_layers, 1):
            width, height = layer["dims"]
            pointer, pitch = C.c_uint64(full_ptr), C.c_uint32(full_pitch)
            if layer["lanczos"]:
                code = self.b["lanczos"].gpu_lanczos_process(
                    layer["lanczos"], full_ptr, C.c_size_t(full_pitch),
                    C.byref(pointer), C.byref(pitch))
                if code:
                    _bridge_error(self.b["lanczos"], "gpu_lanczos_last_error", code)
            working = {"format": "rgba8_cuda", "width": width,
                       "height": height, "cuda_context": frame.cuda_context,
                       "device_ptr": pointer.value, "pitch_bytes": pitch.value}
            def infer_extra():
                if self.uses_flow:
                    flow = layer["flow_provider"].get_flow_device_rgba(
                        index, working, frame.cuda_context,
                        is_discontinuous=is_discontinuous)
                    inference_start = time.perf_counter()
                    dlss = layer["live"].process_device_color_flow_resource(
                        working, flow, reset=reset)
                else:
                    inference_start = time.perf_counter()
                    dlss = layer["live"].process_device_color_zero_motion_resource(
                        working, reset=reset)
                return dlss, flow if self.uses_flow else None, (time.perf_counter()-inference_start)*1000.0
            from preview_layer_cache import evaluate
            (dlss, _, inference_ms), reused = evaluate(self, number, infer_extra)
            reused_layers.append(reused)
            nr_times.append(0.0 if reused else inference_ms)
            layer["low_residual"], observed = self._observe(
                layer["low_residual"], dlss, working, frame)
            observations.append(observed)
            if not reused:
                self.layer_calls[number] += 1
                self.layer_reset_flags[number].append(bool(reset))
        outcome["_observations"] = observations
        outcome["layer_inference_reused"] = reused_layers
        self.layer_inference_ms = nr_times
        outcome["stages"]["nr_total_ms"] = sum(nr_times)
        if self.preview_consumer is not None:
            self._defer_preview_consumer = False
        if self.defer_encode:
            if self.fusion_handle is None:
                self._open_fusion(outcome["_dlss_resource"]["d3d12_device"],
                                  frame)
            descriptors = (_FusionObservation * len(observations))(*[
                _FusionObservation(*entry) for entry in observations])
            output = C.c_void_p()
            report = _FusionReport()
            report.struct_size = C.sizeof(_FusionReport)
            mark = time.perf_counter()
            code = self.b["fusion"].gpu_fusion_process(
                self.fusion_handle, frame.cuda_context, full_ptr,
                C.c_size_t(full_pitch), descriptors, len(observations),
                C.byref(output), C.byref(report))
            if code:
                _bridge_error(self.b["fusion"], "gpu_fusion_last_error", code)
            outcome["stages"]["fusion"] = (time.perf_counter() - mark) * 1000.0
            self.fusion_reports.append({
                "iterations": report.iterations,
                "initial_residual_norm": report.initial_residual_norm,
                "final_residual_norm": report.final_residual_norm,
                "relative_residual": report.relative_residual,
                "converged": bool(report.converged)})
            from fusion_diagnostics import read_solver_diagnostics
            diagnostics = read_solver_diagnostics(self.b["fusion"], self.fusion_handle)
            if diagnostics is not None:
                self.fusion_reports[-1]["diagnostics"] = diagnostics
            mark = time.perf_counter()
            controlled = self._post_process(frame, full_ptr, full_pitch,
                {"d3d12_resource": output.value, "d3d12_fence": None,
                 "fence_value": 0, "d3d12_device": outcome["_dlss_resource"]["d3d12_device"],
                 "producer_completed": True})
            preview_output = self._submit_output(frame, controlled,
                                                  outcome["stages"], started=mark)
            if self.preview_consumer is not None:
                outcome["preview_output"] = preview_output
        elif self.preview_consumer is not None:
            preview_output = self._submit_output(frame, outcome["_output_candidate"],
                                                  outcome["stages"])
            outcome["preview_output"] = preview_output
        return outcome

    def layer_diagnostics(self):
        return {"layer_count": len(self.layer_calls),
                "layer_parameters": [layer_parameters(getattr(owner, 'settings', {}))
                                     for owner in [self.live] +
                                     [layer['live'] for layer in self.extra_layers]],
                "layer_inference_ms": list(getattr(self, "layer_inference_ms", [])),
                "layer_working_dims": [list(self.working)] +
                                      [list(layer["dims"]) for layer in self.extra_layers],
                "layer_session_ids": [self.live.session_id] +
                                     [layer["live"].session_id for layer in self.extra_layers],
                "layer_reset_flags": self.layer_reset_flags,
                "layer_dlss_calls": self.layer_calls,
                "fusion_solver_frames": list(self.fusion_reports)}

    def counters(self):
        counters = super().counters()
        counters["extra_layers"] = []
        counters["low_residual_layers"] = []
        for handle in [self.base_low_residual] + [
                layer["low_residual"] for layer in self.extra_layers]:
            block = _LowResidualCounters()
            block.struct_size = C.sizeof(block)
            if handle and not self.b["low_residual"].gpu_low_residual_get_transfer_counters(
                    handle, C.byref(block)):
                counters["low_residual_layers"].append({
                    name: getattr(block, name)
                    for name, _ in _LowResidualCounters._fields_[1:]})
            else:
                counters["low_residual_layers"].append(None)
        if self.fusion_handle:
            block = _FusionCounters()
            block.struct_size = C.sizeof(block)
            if not self.b["fusion"].gpu_fusion_get_transfer_counters(
                    self.fusion_handle, C.byref(block)):
                counters["fusion"] = {
                    name: getattr(block, name)
                    for name, _ in _FusionCounters._fields_[1:]}
        for layer in self.extra_layers:
            item = {"dlss": layer["live"].transfer_counters(),
                    "nvof": (layer["flow_provider"].device_input_counters()
                             if layer["flow_provider"] else (None, None))}
            counters["extra_layers"].append(item)
        return counters

    def close(self):
        from gpu_cleanup import release_owned, finish_cleanup
        errors = []
        release_owned(self, 'fusion_handle', errors,
                      lambda handle: self.b['fusion'].gpu_fusion_close(handle))
        for layer in reversed(self.extra_layers):
            release_owned(layer, 'sequential_residual', errors,
                          lambda handle: self.b['residual'].gpu_residual_close(handle))
            release_owned(layer, 'low_residual', errors,
                          lambda handle: self.b['low_residual'].gpu_low_residual_close(handle))
            release_owned(layer, 'live', errors)
            release_owned(layer, 'lanczos', errors,
                          lambda handle: self.b['lanczos'].gpu_lanczos_close(handle))
            release_owned(layer, 'flow_provider', errors, method='close_independent_layer')
        release_owned(self, 'base_low_residual', errors,
                      lambda handle: self.b['low_residual'].gpu_low_residual_close(handle))
        release_owned(self, 'sequential_exchange', errors)
        try:
            super().close()
        except Exception as error:
            errors.append(('base_chain', error))
        finish_cleanup(errors)


def transfer_report(counters: dict, frames: int, src_w: int, src_h: int,
                    downscaled: bool = True, working=None) -> dict:
    """Per-boundary figures: a bridge's own number, or None for uninstrumented.

    None means nothing measured, and a zero in these fields is a claim that a
    boundary was crossed zero times - only a counter may make it. Sizes of
    buffers the code allocates are labelled logical instead, and adding a
    logical figure to a measured one would double-count the same bytes. The
    conversion and working-color fields are device writes the bridges count per
    successful call; neither bridge observes cache-line or PCIe traffic, so no
    field here is a physical bus measurement.

    ``downscaled`` says which chain the counters came from: the stages of a
    scale below 1.0, where the Lanczos and residual bridges answer for
    themselves, or the 1.0 identity chain, where neither bridge is ever opened
    and their fields stay None because nothing ran to count. ``working`` is that
    chain's resolved working size; None is the exact half.
    """
    live_w, live_h = working or (src_w // 2, src_h // 2)
    encoder = counters.get("encoder") or {}
    residual = counters.get("residual") or {}
    fusion = counters.get("fusion") or {}
    low_layers = counters.get("low_residual_layers") or []
    if fusion and low_layers and all(item is not None for item in low_layers):
        residual = {
            "full_device_copy_bytes": fusion["full_device_copy_bytes"],
            "working_device_copy_bytes": sum(
                item["working_device_copy_bytes"] for item in low_layers),
            "output_readback_bytes": fusion["output_readback_bytes"] + sum(
                item["output_readback_bytes"] for item in low_layers)}
    dlss = counters.get("dlss") or {}
    extra_dlss = [item.get("dlss") or {} for item in
                  counters.get("extra_layers") or []]
    if extra_dlss and all("host_color_stage_bytes" in item and
                          "output_readback_bytes" in item for item in extra_dlss):
        dlss = dict(dlss)
        for name in ("host_color_stage_bytes", "host_color_stage_count",
                     "device_color_copy_bytes", "output_readback_bytes"):
            if name in dlss:
                dlss[name] += sum(item[name] for item in extra_dlss)
    decoder = counters.get("decoder") or {}
    nv12 = counters.get("nv12") or {}
    lanczos = counters.get("lanczos") or {}
    return {
        "gpu_transfer_frames_requested": frames,
        # The decode bridge counts every frame it handed out and every byte it
        # moved to host memory. Production never calls its readback entry point,
        # so a zero in the host fields is that counter saying no frame crossed
        # to host memory - and None says the bridge did not answer at all.
        "decode_host_copy_bytes_measured": decoder.get("host_frame_copy_bytes"),
        "decode_host_copy_count_measured": decoder.get("host_frame_copy_count"),
        "decode_frames_decoded_measured": decoder.get("frames_decoded"),
        "decode_device_frame_count_measured": decoder.get("device_frame_count"),
        # Compressed bytes handed to the decoder, which uploads them itself:
        # packet traffic, not a frame transfer and not a PCIe figure.
        "decode_bitstream_submitted_bytes_measured": decoder.get("bitstream_bytes_submitted"),
        "decode_scale_device_output_bytes_measured": decoder.get("scale_device_output_bytes"),
        "decode_nv12_bytes_logical": src_w * src_h * 3 // 2,
        # The conversion kernel writes the whole RGBA8 output once per call, and
        # the working-color chain copies the full frame device-to-device and then
        # writes both of its passes. Those are the bytes the bridges' own code
        # determines; the reads around them are cache-served and their physical
        # cost is not observable, so no read figure is claimed here.
        "nv12_to_rgba_device_bytes_measured": nv12.get("output_device_write_bytes"),
        "nv12_to_rgba_convert_count_measured": nv12.get("convert_count"),
        "nv12_to_rgba_bytes_logical": src_w * src_h * 4,
        "working_color_device_bytes_measured": lanczos.get("output_device_write_bytes"),
        "working_color_process_count_measured": lanczos.get("process_count"),
        "working_color_source_device_copy_bytes_measured": lanczos.get("source_device_copy_bytes"),
        "working_color_scratch_device_write_bytes_measured": lanczos.get("scratch_device_write_bytes"),
        # At a scale below 1.0 this is the working plane the Lanczos pass owns,
        # at whatever size the plan resolved it to. At 1.0 the working colour is
        # the converter's full RGBA under its second name, so the figure labels
        # the same buffer the nv12_to_rgba line above labelled - it is not an
        # additional allocation, and no measured field claims a copy that the
        # bridge never made.
        "working_color_bytes_logical": ((live_w * live_h) if downscaled
                                        else src_w * src_h) * 4,
        "dlss_host_color_stage_bytes_measured": dlss.get("host_color_stage_bytes"),
        "dlss_host_color_stage_count_measured": dlss.get("host_color_stage_count"),
        "dlss_device_color_copy_bytes_measured": dlss.get("device_color_copy_bytes"),
        "dlss_output_readback_bytes_measured": dlss.get("output_readback_bytes"),
        "residual_full_device_copy_bytes_measured": residual.get("full_device_copy_bytes"),
        "residual_working_device_copy_bytes_measured": residual.get("working_device_copy_bytes"),
        "residual_output_readback_bytes_measured": residual.get("output_readback_bytes"),
        "fusion_scalar_readback_bytes_measured": fusion.get("scalar_readback_bytes"),
        "encoder_staging_device_copy_bytes_measured": encoder.get("staging_gpu_copy_bytes"),
        "encoder_conversion_device_write_bytes_measured": encoder.get("conversion_output_bytes"),
        "encoder_host_pixel_bytes_measured": encoder.get("host_pixel_bytes"),
        "encoder_frames_submitted_measured": encoder.get("frames_submitted"),
        "encoder_packets_written_measured": encoder.get("packets_written"),
    }


WATCHDOG_TIMEOUT_SEC = 30.0
# Same budget the legacy path gives the DLSS session lock: a wait longer than
# this is a stuck session, and the caller would rather have the frame fail than
# have the job hang.
LOCK_TIMEOUT_SEC = 15.0


@contextlib.contextmanager
def _session(lock):
    """Hold the shared DLSS lock for one frame of the chain, if there is one.

    The whole frame is inside the lock, not just the inference call: the decoder
    surface, the flow descriptor and the encoder's input texture are all borrowed
    from each other, so a preview that walked in mid-frame would be reading a
    stage that the next call has already recycled.
    """
    if lock is None:
        yield
        return
    if not lock.acquire(timeout=LOCK_TIMEOUT_SEC):
        raise RuntimeError(
            f"DLSS 会话锁 (_live_lock) 获取超时 {LOCK_TIMEOUT_SEC}s，GPU 显存直传"
            "管线无法进入推理：预览线程可能持有锁或上一会话未正确释放")
    try:
        yield
    finally:
        try:
            lock.release()
        except Exception:
            pass


def export_gpu_transfer(video, live, flow_provider, out_path, dims, frames,
                        cq, with_audio, transcode_settings, job_settings,
                        negotiation, progress=None, cancel=None,
                        on_proc_started=None, on_proc_exited=None,
                        diagnostics_out=None, profiler=None,
                        transfer_collector=None, bridges=None,
                        verify=None, lock=None):
    """Own the temporary encoded file across every export exit path."""
    pipeline.assert_output_not_input(video, out_path)
    intermediate = None
    if negotiation["copy_pass"]:
        handle, intermediate = tempfile.mkstemp(suffix='.mov' if (transcode_settings or {}).get('codec') == 'prores_ks' else INTERMEDIATE_CONTAINER,
                                                prefix="dlss_gpu_transfer_")
        os.close(handle)
        os.unlink(intermediate)
    try:
        return _export_gpu_transfer(video, live, flow_provider, out_path, dims,
            frames, cq, with_audio, transcode_settings, job_settings, negotiation,
            progress=progress, cancel=cancel, on_proc_started=on_proc_started,
            on_proc_exited=on_proc_exited, diagnostics_out=diagnostics_out,
            profiler=profiler, transfer_collector=transfer_collector,
            bridges=bridges, verify=verify, lock=lock, intermediate=intermediate)
    finally:
        if intermediate:
            try:
                os.unlink(intermediate)
            except FileNotFoundError:
                pass


def _export_gpu_transfer(video, live, flow_provider, out_path, dims, frames,
                        cq, with_audio, transcode_settings, job_settings,
                        negotiation, progress=None, cancel=None,
                        on_proc_started=None, on_proc_exited=None,
                        diagnostics_out=None, profiler=None,
                        transfer_collector=None, bridges=None,
                        verify=None, lock=None, intermediate=None):
    """Run the GPU-resident export and return ``out_path``.

    Single threaded by necessity: every bridge is synchronous and bound to the
    CUDA context and thread that opened it, and the hand-offs are borrowed
    pointers the next call invalidates. A producer thread would have to copy
    frames to outlive the decoder's memory, which is the round trip this
    backend exists to remove.
    """
    # The gate above is a negotiation about capability; this is a refusal about the
    # caller's own paths, and a direct caller of this entry has to meet it before the
    # decoder bridge opens the source.
    pipeline.assert_output_not_input(video, out_path)
    js = dict(job_settings or {})
    src_w, src_h = int(dims["src_w"]), int(dims["src_h"])
    import native_effects
    process_w, process_h = native_effects.processing_size(js, (src_w, src_h))
    from post_control import GpuPostControl, controls_from_settings
    post_controls = controls_from_settings(js)
    from temporal_control import GpuTemporalControl, temporal_from_settings
    temporal_controls = temporal_from_settings(js)
    live_w, live_h = int(dims["live_w"]), int(dims["live_h"])
    strength = float(js.get("residual_strength", 1.0))
    plan = negotiation["plan"]
    fps = negotiation["fps"]
    downscaled = bool(negotiation["downscaled"])
    uses_flow = bool(negotiation["uses_flow"])
    # A set the caller already loaded is taken as it is, because reloading it
    # here would drop the session its decoder context belongs to. A set loaded
    # for this run is loaded for the mode the negotiation accepted, so an
    # unguided export never asks an optical flow engine to introduce itself.
    bridges = bridges or load_bridges(
        requires_flow=uses_flow,
        requires_multiscale=bool(negotiation.get("multi_scale") and
            len(negotiation["multi_scale"]["scales"]) > 1))
    # A scale the constant tap tables cannot describe is opened by its plan, and
    # the plan is built from the extents this export resolved rather than from
    # anything the chain would have to recompute - so the two resampling stages
    # and the CPU reference are filtered by one set of coefficients.
    tap_plans = (TapPlans(process_w, process_h, live_w, live_h)
                 if negotiation["plan_driven"] else None)
    # The record names what actually ran, the way the CPU plan's own record
    # does: the residual stage's backend at a scale below 1.0, "none" where no
    # stage exists.
    residual_backend = ("gpu_multiscale_fusion_bridge_d3d12" if
        negotiation.get("multi_scale") and
        negotiation["multi_scale"]["fusion"] ==
        "data_consistency_experimental" else
        "gpu_residual_bridge_d3d12" if downscaled else "none")

    if negotiation["copy_pass"]:
        if intermediate is None:
            raise BridgeError("copy pass requires an owned intermediate path")
        encoder_output = intermediate
    else:
        intermediate = None
        encoder_output = out_path

    multi = negotiation.get("multi_scale")
    extra_layers = []
    if multi and len(multi["scales"]) > 1:
        import dlss_gpu_engine
        fusion_plans = [TapPlans(process_w, process_h, width, height)
                        for width, height in multi["dims"]]
        try:
            for scale, (width, height) in zip(multi["scales"][1:],
                                               multi["dims"][1:]):
                provider = (flow_provider.new_independent_layer()
                            if uses_flow else None)
                layer = {"dims": (width, height), "scale": scale,
                         "flow_provider": provider, "live": None,
                         "lanczos": None, "low_residual": None,
                         "tap_plans": (None if scale == 0.5 and
                                       (width, height) == (process_w // 2, process_h // 2)
                                       else TapPlans(process_w, process_h, width, height))}
                extra_layers.append(layer)
                from layer_settings import resolve_layer_settings
                layer["live"] = dlss_gpu_engine.Live(
                    width, height, resolve_layer_settings(dict(live.settings, **js), len(extra_layers)), shared_with=live)
        except Exception:
            for layer in reversed(extra_layers):
                if layer["live"]:
                    layer["live"].close()
                if layer["flow_provider"]:
                    layer["flow_provider"].close_independent_layer()
            raise
        chain = MultiScaleObservationChain(
            bridges, video, encoder_output, live, plan, fps, strength,
            (live_w, live_h), flow_provider, downscaled=downscaled,
            tap_plans=tap_plans, uses_flow=uses_flow,
            extra_layers=extra_layers, config=multi,
            fusion_plans=fusion_plans)
    else:
        chain = Chain(bridges, video, encoder_output, live, plan, fps, strength,
                      (live_w, live_h), flow_provider, downscaled=downscaled,
                      tap_plans=tap_plans, uses_flow=uses_flow)
    chain.post_control = GpuPostControl(post_controls)
    chain.temporal_control = GpuTemporalControl(temporal_controls)
    import native_effects
    native_effects.install(chain, live, js, (src_w, src_h), flow_provider)
    chain.export_size = (int(dims['dst_w']), int(dims['dst_h']))
    chain.export_scale_algorithm = (transcode_settings or {}).get('scale_algorithm', 'lanczos')
    chain.cancel = cancel
    # ``is None``, not ``or``: TransferProfiler defines __len__, so a caller's
    # collector that has recorded nothing yet is falsy, and ``or`` would replace
    # it with a private one and leave the caller reading an empty ledger.
    collector = (transfer_profiler.TransferProfiler() if transfer_collector is None
                 else transfer_collector)
    profiler = profiler or pipeline.StageProfiler()
    processed = 0
    previous_timestamp = None
    failed = False
    started = time.time()
    last_frame_t = time.time()
    counters = {}
    profiler.start()
    try:
        native_phase('打开 GPU 解码器', 300)
        chain.open_decoder()
        frame = _DeviceFrame()
        frame.struct_size = C.sizeof(_DeviceFrame)
        from export_range import SelectedDecoder, validate_range
        selected_decoder = SelectedDecoder(chain, validate_range(js.get("_frame_range")), fps, cancel)
        while processed < frames:
            if cancel and cancel():
                raise pipeline.JobCancelled("用户已取消任务")
            if time.time() - last_frame_t > WATCHDOG_TIMEOUT_SEC:
                raise RuntimeError(
                    f"GPU 转码管线在第 {processed} 帧严重停滞（超过 {WATCHDOG_TIMEOUT_SEC} "
                    f"秒无有效帧），NVDEC / Feature18 / NVENC 可能已死锁")
            t_decode = time.perf_counter()
            native_phase(f'GPU 解码第 {processed + 1} 帧', 300)
            state = selected_decoder.next(frame)
            decode_ms = (time.perf_counter() - t_decode) * 1000.0
            if state == 0:
                break
            if state < 0:
                _bridge_error(bridges["nvdec"], "nvdec_bridge_last_error", state)
            if isinstance(chain, MultiScaleObservationChain) and (
                    frame.width, frame.height) != (src_w, src_h):
                raise RuntimeError("multi-scale source extent changed during export; "
                                   "start a new job to rebuild and reset all layers")
            if chain.nv12 is None:
                chain._open_color_stages(frame)
            before = (flow_provider.device_input_counters()
                      if uses_flow and hasattr(flow_provider, "device_input_counters")
                      else None)
            t_frame = time.perf_counter()
            discontinuous = (processed == 0 or (
                isinstance(chain, MultiScaleObservationChain) and
                _frame_timestamp_discontinuity(previous_timestamp, frame, fps)))
            with _session(lock):
                native_phase(f'GPU 处理第 {processed + 1} 帧', 300)
                outcome = chain.process(frame, flow_provider, processed,
                                        is_discontinuous=discontinuous)
            previous_timestamp = (int(frame.pts), int(frame.time_base_num),
                                  int(frame.time_base_den))
            frame_ms = (time.perf_counter() - t_frame) * 1000.0
            from gpu_memory import capture
            capture(chain, live, phase='export')
            after = (flow_provider.device_input_counters()
                     if uses_flow and hasattr(flow_provider, "device_input_counters")
                     else None)
            record = {
                "frame_idx": processed,
                "decode": decode_ms,
                "frame_total": frame_ms,
                "decode_requested": "cuda",
                "decode_actual": "cuda_nvdec_bridge",
                "export_backend_requested": str(js.get("export_backend", "auto")),
                "export_backend_actual": "gpu_transfer",
                "transfer_mode_requested": "gpu_transfer",
                "transfer_mode_actual": "gpu_transfer",
                "dlss_backend_actual": "gpu_flow_d3d12",
                "residual_backend": residual_backend,
                "guidance_backend": (outcome.get("nvof_timing") or {}).get("guidance_backend"),
                "nvof_input_format": (outcome.get("nvof_timing") or {}).get("nvof_input_format"),
                "flow_pts": outcome["stages"]["decode_pts"],
                **{name: ms for name, ms in outcome["stages"].items()
                   if name != "decode_pts"},
            }
            # A delta of the engine's own cumulative counters, so the per-frame
            # figure is the copy that frame caused rather than a share of the
            # session total. Either end unreadable leaves the field absent.
            if before and after and all(v is not None for v in before + after):
                record["nvof_device_input_copy_count"] = after[0] - before[0]
                record["nvof_device_input_bytes"] = after[1] - before[1]
            collector.record(record)
            # flow_pts is a timestamp, not a duration, and the profiler prints every
            # field of its record under "Mean (ms)".
            profiler.record({k: v for k, v in record.items() if k != "flow_pts"})
            processed += 1
            last_frame_t = time.time()
            if progress:
                elapsed = max(0.001, time.time() - started)
                progress(processed, frames,
                         f"[GPU 显存直传导出] {processed}/{frames} "
                         f"({processed / frames * 100:.1f}%) | "
                         f"{processed / elapsed:.1f} fps")
        native_phase('排空 GPU 编码器', 180)
        chain.finish()
    except Exception:
        failed = True
        raise
    finally:
        # The independent watchdog observes hangs inside these calls. It only
        # requests process recovery; this owner thread performs normal cleanup.
        native_phase('关闭 GPU 导出链路', 90)
        try:
            chain.cancel_decoder()
        except Exception as cancel_error:
            print(f"[gpu_transfer] decoder cancel failed: {cancel_error}", flush=True)
        try:
            counters = chain.counters()
        except Exception as counter_error:
            print(f"[gpu_transfer] transfer counters unavailable: {counter_error}",
                  flush=True)
        profiler.stop()
        try:
            chain.close()
        finally:
            for cookie in bridges.get("cookies") or []:
                _close_cookie(cookie)
        if failed and intermediate and os.path.exists(intermediate):
            # A run that did not finish has no deliverable. The encoder writes a
            # trailer so the partial file stays probeable, and leaving it behind
            # would let a caller point at frames nobody asked to keep.
            try:
                os.unlink(intermediate)
            except OSError:
                pass
        profiler.print_summary(f"GPU TRANSFER EXPORT ({processed} FRAMES)")

    native_phase('导出报告及容器封装', None)

    report = transfer_report(counters, processed, src_w, src_h,
                             downscaled=downscaled, working=(live_w, live_h))
    report["post_control"] = counters.get("post_control")
    report['native_output_effect'] = counters.get('native_output_effect')
    report['native_input_effect'] = counters.get('native_input_effect')
    report['render_pipeline'] = native_effects.stages(js)
    report['nr_input_size'] = [process_w, process_h]
    report['frame_generation'] = counters.get('frame_generation')
    report['export_resize'] = counters.get('export_resize')
    report['truehdr'] = counters.get('truehdr')
    report['neural_mode'] = job_settings.get('neural_mode', 'nr')
    report["temporal_control"] = counters.get("temporal_control")
    report["export_backend_requested"] = str(js.get("export_backend", "auto"))
    report["export_backend_actual"] = "gpu_transfer"
    if counters.get('prores_output'):
        report['export_backend_actual'] = 'gpu_render_cpu_prores'
        report['prores_output'] = counters['prores_output']
        report['zero_copy'] = False
        report['encoder_host_pixel_bytes_measured'] = counters['prores_output']['frame_host_readback_bytes']
        report['encoder_frames_submitted_measured'] = counters['prores_output']['frames_submitted']
    report["export_backend_fallback_reason"] = None
    report["gpu_transfer_color_matrix"] = negotiation["declared_color_space"]
    report["gpu_transfer_encoder_codec"] = plan["codec"]
    report["gpu_transfer_encoder_av_options"] = plan["av_options"]
    report["gpu_transfer_encoder_tune_adjustment"] = plan.get("tune_adjustment")
    # The struct the bridge was actually handed, in the terms the plan wrote it in.
    # What a caller asked for and what the encoder received are two different facts --
    # a tune the loaded build does not have is downgraded here, and a constant quality
    # arrives through the payload rather than through the caller's default -- so a run
    # has to be able to name the second one without anybody re-deriving the first.
    report["gpu_transfer_encoder_options"] = {
        name: _encoder_option(plan["struct"].get(name))
        for name in ("preset", "tune", "multipass", "rate_control",
                     "constant_quality", "bitrate", "gop", "color_matrix")}
    report["gpu_transfer_copy_pass"] = bool(negotiation["copy_pass"])
    from vfx_framegen import output_fps
    report["gpu_transfer_frame_rate"] = "%d/%d" % output_fps(js, fps)
    # The chain shape the negotiation classified, so a finished run says which
    # of the scales it ran rather than leaving it to be inferred from which
    # bridge counters happen to be absent. The working size goes with it: for a
    # plan-driven ratio it is what resolve_working_dims returned for that ratio
    # (2880x1616 at 0.75 on a 4K source), which is what the tap plans were built
    # for and what the plan-driven stages were opened at.
    report["gpu_transfer_working_scale"] = negotiation["working_scale"]
    report["gpu_transfer_working_dims"] = [live_w, live_h]
    report["gpu_transfer_resampling"] = ("plan_driven" if tap_plans
                                        else "exact_two_x" if downscaled
                                        else "identity")
    report["gpu_transfer_residual_backend"] = residual_backend
    report["gpu_transfer_bridge_dll_directory"] = bridges.get("ffmpeg_dll_directory")
    # Whether the bridge set behind this run ever loaded and probed an optical
    # flow engine. The frame counters below say no frame asked it for motion;
    # this says the library was not even in the process, which is the stronger
    # claim an unguided run has to be able to make. Absent from a caller's stub
    # set, which loaded nothing.
    report["gpu_transfer_nvof_bridge_loaded"] = (
        bool(bridges.get("nvof_probe")) if "nvof_probe" in bridges else None)
    # Which motion source actually drove the DLSS stage. The counts are the
    # chain's own, so a guidance_mode=0 run has to show every frame going through
    # the host's resident zero motion texture and no frame ever reaching a flow
    # engine; the mode alone would only say what was asked for.
    report["gpu_transfer_guidance_mode"] = negotiation["guidance_mode"]
    report["gpu_transfer_uses_flow"] = uses_flow
    report["gpu_transfer_dlss_flow_calls"] = chain.flow_calls
    report["gpu_transfer_dlss_zero_motion_calls"] = chain.zero_motion_calls
    if isinstance(chain, MultiScaleObservationChain):
        report.update(chain.layer_diagnostics())
        report["layer_scales"] = multi["scales"]
        report["layer_weights"] = multi["weights"]
        report["fusion_mode"] = multi["fusion"]
        report["fusion_beta"] = multi["beta"]
        report["fusion_max_iterations"] = multi["max_iterations"]
        report["fusion_relative_tolerance"] = multi["relative_tolerance"]
        report["layer_transfer_counters"] = counters.get("extra_layers")
    report["nvof_device_input_copy_count_total"] = (counters.get("nvof") or
                                                    (None, None))[0]
    report["nvof_device_input_bytes_total"] = (counters.get("nvof") or
                                               (None, None))[1]
    if diagnostics_out is not None:
        diagnostics_out.update(report)
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
            raise RuntimeError(f"GPU 解码提前终止！预期读取 {frames} 帧，实际解码 "
                               f"{processed} 帧（输入文件可能损坏或解码中断）")

    if intermediate:
        transcode_settings = selected_decoder.mux_settings(transcode_settings, pipeline._bundle_ffprobe())
        if diagnostics_out is not None and js.get('_frame_range'):
            diagnostics_out['frame_range']['mux_source_start_seconds'] = transcode_settings['_source_range_start_seconds']
        if not os.path.isfile(intermediate):
            raise BridgeError("the encoder produced no intermediate to wrap")
        ffmpeg = pipeline._bundle_ffmpeg()
        if not ffmpeg:
            raise BridgeError("ffmpeg not found for the container copy pass")
        try:
            run_copy_pass(ffmpeg, intermediate, video, transcode_settings, out_path,
                          with_audio, on_proc_started=on_proc_started,
                          on_proc_exited=on_proc_exited,
                          video_authoritative=True,
                          ffprobe=pipeline._bundle_ffprobe())
        finally:
            try:
                os.unlink(intermediate)
            except OSError:
                pass

    if processed:
        # The count has to come from the bytes this run just wrote. pipeline.video_info
        # remembers a path for the life of the process, and a re-export over a name that
        # was probed earlier - the source of a previous job, an aborted attempt at this
        # one - would answer from that memory instead.
        from vfx_framegen import output_frame_count
        pipeline.verify_export_output(out_path, output_frame_count(js, fps, frames),
                                      probe=verify or pipeline.video_info_uncached)
    return out_path, report
