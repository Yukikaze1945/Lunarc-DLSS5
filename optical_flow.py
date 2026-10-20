#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_flow.py — Multi-Engine Optical Flow Provider for DLSS5 Standalone.

Supported Engines:
1. "nvof" (NVIDIA Optical Flow Hardware Backend):
   - Dedicated hardware optical flow accelerator on NVIDIA RTX GPUs.
   - Converts OpenCV BGR24 input explicitly to official ABGR8 format.
   - Grid options: 1x1 full-resolution output grid, 2x2, 4x4.
   - Presets: FAST, MEDIUM, SLOW.
   - Spatial bilinear interpolation for 2x2/4x4 without scaling displacement vectors.
   - Explicit CUcontext tracking, thread serialization, and temporal hint reset support.
   - Strictly NO silent fallback to DIS on initialization or execution failure.
2. "dis" (OpenCV DIS CPU):
   - CPU dense inverse search motion vectors.
3. "raft" (RAFT-Large Deep Neural Network):
   - PyTorch Raft-Large model in GPU VRAM or CPU.
4. Legacy "amd":
   - Automatically migrated to "dis" with a clear warning.
"""
import os
import sys
import time
import ctypes
from ctypes import Structure, c_int32, c_uint32, c_char, byref, c_void_p, POINTER, c_uint8, c_float, c_char_p, c_double, c_uint64, c_size_t
import numpy as np
import cv2

import bundle_paths
from torch_cuda_context import torch_primary_context

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import transfer_profiler
except ImportError:  # pragma: no cover - direct execution from another cwd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import transfer_profiler


class NvOFBridgeProbeInfo(Structure):
    _fields_ = [
        ('is_available', c_int32),
        ('gpu_name', c_char * 128),
        ('driver_version', c_uint32),
        ('max_api_version', c_uint32),
        ('min_width', c_uint32),
        ('min_height', c_uint32),
        ('max_width', c_uint32),
        ('max_height', c_uint32),
        ('supported_grid_mask', c_uint32),
        ('error_message', c_char * 256),
    ]


class NvOFBridgeTiming(Structure):
    """Mirror of NvOFBridgeTiming in native/nvof_bridge.h.

    One field per measured span. A span the bridge does not time arrives as
    NVOF_TIMING_UNMEASURED and is reported as "unavailable" rather than as a zero.
    """
    _fields_ = [
        ('upload_ms', c_double),
        ('submit_cpu_ms', c_double),
        ('dependency_wait_ms', c_double),
        ('download_ms', c_double),
        ('postprocess_wait_inclusive_ms', c_double),
        ('total_native_ms', c_double),
        # ABI 3, appended: the byte-map launches that fill the ABGR8 surface from uploaded
        # BGR24 staging, plus the synchronize that makes that surface safe to read. Measured
        # on the BGR input path only; unmeasured everywhere else, which shows as unavailable.
        ('input_convert_wait_ms', c_double),
    ]


NVOF_TIMING_UNMEASURED = -1.0
NVOF_TIMING_UNAVAILABLE = "unavailable"


def _timing_ms(value):
    """A measured duration, or the string 'unavailable' for a span that was never timed."""
    if value is None or value < 0.0:
        return NVOF_TIMING_UNAVAILABLE
    return float(value)


def _bgr24_host_view(arr, name: str) -> np.ndarray:
    """Check `arr` is a packed BGR24 host buffer the bridge can copy row by row.

    The BGR entry point hands the native side a pointer plus a row stride and the driver
    copies from that description without looking at it, so the layout has to be verified on
    the way in: three uint8 channels packed with stride 1, and rows at their natural 3-byte
    pitch. A row that is further apart than that is still fine - the stride is passed
    through - but a buffer whose channels are not adjacent is a different image.
    """
    if (arr is None or arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3
            or arr.strides[1] != 3 or arr.strides[2] != 1):
        raise ValueError(
            f"{name} must be a packed uint8 BGR24 host array; got dtype="
            f"{getattr(arr, 'dtype', None)} shape={getattr(arr, 'shape', None)} "
            f"strides={getattr(arr, 'strides', None)}")
    return arr


class NvOFBridgeCounters(Structure):
    """Mirror of NvOFBridgeCounters in native/nvof_bridge.h, ABI 3.

    The last three fields are appended, like the timing field above, and the version check in
    _load_native_bridge is what makes reading them safe: this struct is written by the library
    at its own sizeof, so a mirror that is shorter than the library's version is a buffer
    overrun rather than a harmless misread.
    """
    _fields_ = [
        ('execute_count', c_uint64),
        ('h2d_copy_count', c_uint64),
        ('h2d_bytes', c_uint64),
        ('d2h_copy_count', c_uint64),
        ('d2h_bytes', c_uint64),
        ('reset_count', c_uint64),
        ('bgr_staging_alloc_count', c_uint64),
        ('bgr_input_call_count', c_uint64),
        ('input_convert_launch_count', c_uint64),
    ]


_GLOBAL_NVOF_EXECUTE_COUNT = 0

def get_nvof_execute_count() -> int:
    global _GLOBAL_NVOF_EXECUTE_COUNT
    return _GLOBAL_NVOF_EXECUTE_COUNT

def reset_nvof_execute_count():
    global _GLOBAL_NVOF_EXECUTE_COUNT
    _GLOBAL_NVOF_EXECUTE_COUNT = 0


_NATIVE_BRIDGE = None
_NATIVE_PROBE_CACHE = None
_NATIVE_BRIDGE_LOAD_ERROR = None

# The timing struct's layout is not self-describing: reading an older or newer library with
# these field offsets yields numbers that look plausible and are wrong, so the version has
# to be checked before the struct is trusted. ABI 3 appended NvOFBridgeTiming.
# input_convert_wait_ms and three NvOFBridgeCounters fields, and added the
# nvof_bridge_execute_bgr_sequential export.
_EXPECTED_NVOF_BRIDGE_ABI = 3

# Set when the loaded library both reports ABI 3 and exports the BGR entry point. A
# capability of the binary, not of this module: a caller asks for it instead of re-deriving
# it from a version number, and refuses the reduced transfer path when it is false.
_NVOF_BGR_INPUT_EXPORT = False
_NVOF_DEVICE_OUTPUT_EXPORT = False
_NVOF_DEVICE_RGBA_INPUT_EXPORT = False


def _bridge_load_error():
    return _NATIVE_BRIDGE_LOAD_ERROR


def _load_native_bridge():
    global _NATIVE_BRIDGE, _NATIVE_BRIDGE_LOAD_ERROR
    global _NVOF_BGR_INPUT_EXPORT, _NVOF_DEVICE_OUTPUT_EXPORT
    global _NVOF_DEVICE_RGBA_INPUT_EXPORT
    if _NATIVE_BRIDGE is not None:
        return _NATIVE_BRIDGE

    native_dir = bundle_paths.native_dir()
    candidates = [
        bundle_paths.native_dll("nvof_bridge.dll"),
        os.path.join(native_dir, "build", "Release", "nvof_bridge.dll"),
        bundle_paths.collected("nvof_bridge.dll"),
    ]

    dll_path = None
    for c in candidates:
        if os.path.exists(c):
            dll_path = c
            break

    if not dll_path:
        _NATIVE_BRIDGE_LOAD_ERROR = (
            "nvof_bridge.dll not found; searched "
            + ", ".join(os.path.dirname(c) for c in candidates)
            + (" - rebuild it from native/, or build the bundle from the current "
               "spec, which collects it into native/"
               if getattr(sys, "frozen", False) else
               " - rebuild it with native/build.bat or native/CMakeLists.txt"))
        return None

    try:
        bridge = ctypes.CDLL(dll_path)
        bridge.nvof_bridge_get_abi_version.restype = c_uint32
        bridge.nvof_bridge_get_abi_version.argtypes = []
        try:
            abi = int(bridge.nvof_bridge_get_abi_version())
        except AttributeError:
            # A library that predates the version export cannot describe its struct.
            abi = 1
        if abi != _EXPECTED_NVOF_BRIDGE_ABI:
            _NATIVE_BRIDGE_LOAD_ERROR = (
                f"{dll_path} exports nvof_bridge ABI {abi}, this module reads ABI "
                f"{_EXPECTED_NVOF_BRIDGE_ABI}; rebuild it from native/nvof_bridge.cpp "
                "instead of running with a stale library")
            print(f"[flow] ERROR: {_NATIVE_BRIDGE_LOAD_ERROR}", flush=True)
            return None
        _NATIVE_BRIDGE_LOAD_ERROR = None

        bridge.nvof_bridge_probe.argtypes = [POINTER(NvOFBridgeProbeInfo)]
        bridge.nvof_bridge_probe.restype = c_int32

        bridge.nvof_bridge_create.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, POINTER(c_void_p)]
        bridge.nvof_bridge_create.restype = c_int32

        bridge.nvof_bridge_execute.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_float), c_uint32, c_int32]
        bridge.nvof_bridge_execute.restype = c_int32

        bridge.nvof_bridge_execute_timed.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_float), c_uint32, c_int32, POINTER(NvOFBridgeTiming)]
        bridge.nvof_bridge_execute_timed.restype = c_int32

        bridge.nvof_bridge_execute_sequential.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_float), c_uint32, c_int32, POINTER(NvOFBridgeTiming)]
        bridge.nvof_bridge_execute_sequential.restype = c_int32

        # Conditionally, so a library that predates ABI 3 still loads and runs the RGBA path
        # for every caller that does not ask for BGR input. Setting argtypes on a symbol that
        # is not there would raise here and take the whole engine down.
        _NVOF_BGR_INPUT_EXPORT = hasattr(bridge, "nvof_bridge_execute_bgr_sequential")
        if _NVOF_BGR_INPUT_EXPORT:
            bridge.nvof_bridge_execute_bgr_sequential.argtypes = [
                c_void_p, POINTER(c_uint8), c_uint32, POINTER(c_uint8), c_uint32,
                POINTER(c_float), c_uint32, c_int32, POINTER(NvOFBridgeTiming)]
            bridge.nvof_bridge_execute_bgr_sequential.restype = c_int32

        _NVOF_DEVICE_OUTPUT_EXPORT = hasattr(
            bridge, "nvof_bridge_execute_bgr_sequential_device")
        if _NVOF_DEVICE_OUTPUT_EXPORT:
            bridge.nvof_bridge_execute_bgr_sequential_device.argtypes = [
                c_void_p, POINTER(c_uint8), c_uint32, POINTER(c_uint8), c_uint32,
                c_int32, POINTER(NvOFBridgeTiming)]
            bridge.nvof_bridge_execute_bgr_sequential_device.restype = c_int32

        # A device RGBA8 row set from another CUDA context, so a frame that is
        # already on the GPU never comes back to host memory to reach NVOF. The
        # counters are read separately because they describe input copies, not
        # the flow output.
        _NVOF_DEVICE_RGBA_INPUT_EXPORT = (
            hasattr(bridge, "nvof_bridge_execute_rgba_device_sequential") and
            hasattr(bridge, "nvof_bridge_get_device_input_counters"))
        if _NVOF_DEVICE_RGBA_INPUT_EXPORT:
            bridge.nvof_bridge_execute_rgba_device_sequential.argtypes = [
                c_void_p, c_uint64, c_uint32, c_void_p, c_int32,
                POINTER(NvOFBridgeTiming)]
            bridge.nvof_bridge_execute_rgba_device_sequential.restype = c_int32
            bridge.nvof_bridge_get_device_input_counters.argtypes = [
                c_void_p, POINTER(c_uint64), POINTER(c_uint64)]
            bridge.nvof_bridge_get_device_input_counters.restype = c_int32

        bridge.nvof_bridge_get_last_timing.argtypes = [c_void_p, POINTER(NvOFBridgeTiming)]
        bridge.nvof_bridge_get_last_timing.restype = c_int32

        bridge.nvof_bridge_get_counters.argtypes = [c_void_p, POINTER(NvOFBridgeCounters)]
        bridge.nvof_bridge_get_counters.restype = c_int32

        bridge.nvof_bridge_destroy.argtypes = [c_void_p]
        bridge.nvof_bridge_destroy.restype = c_int32

        bridge.nvof_bridge_get_session_info.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p), POINTER(c_uint32)]
        bridge.nvof_bridge_get_session_info.restype = c_int32

        bridge.nvof_bridge_get_output_deviceptr.argtypes = [c_void_p, POINTER(c_uint64), POINTER(c_size_t)]
        bridge.nvof_bridge_get_output_deviceptr.restype = c_int32

        bridge.nvof_bridge_get_last_error.restype = c_char_p

        _NATIVE_BRIDGE = bridge
        return _NATIVE_BRIDGE
    except Exception as e:
        _NATIVE_BRIDGE_LOAD_ERROR = f"failed to load {dll_path}: {e}"
        print(f"[flow] Warning: {_NATIVE_BRIDGE_LOAD_ERROR}", flush=True)
        return None


def probe_nvof_capabilities() -> dict:
    """Probe hardware for real NVIDIA Optical Flow capability without creating a persistent session."""
    global _NATIVE_PROBE_CACHE
    if _NATIVE_PROBE_CACHE is not None:
        return _NATIVE_PROBE_CACHE

    bridge = _load_native_bridge()
    if not bridge:
        _NATIVE_PROBE_CACHE = {
            "available": False,
            "hardware": False,
            "device": None,
            "error": _NATIVE_BRIDGE_LOAD_ERROR or "nvof_bridge.dll could not be loaded",
            "supported_grids": [],
            "bgr_input": False,
            "device_output": False,
            "device_rgba_input": False,
        }
        return _NATIVE_PROBE_CACHE

    info = NvOFBridgeProbeInfo()
    res = bridge.nvof_bridge_probe(byref(info))
    if res != 0 or info.is_available != 1:
        err_msg = info.error_message.decode('utf-8', errors='replace').strip()
        if not err_msg:
            err_msg = bridge.nvof_bridge_get_last_error() or "Hardware probe failed"
        _NATIVE_PROBE_CACHE = {
            "available": False,
            "hardware": False,
            "device": None,
            "error": err_msg,
            "supported_grids": [],
            "bgr_input": False,
            "device_output": False,
            "device_rgba_input": False,
        }
        return _NATIVE_PROBE_CACHE

    gpu_name = info.gpu_name.decode('utf-8', errors='replace').strip()
    grids = [g for g in [1, 2, 4] if (info.supported_grid_mask & (1 << g))]

    _NATIVE_PROBE_CACHE = {
        "available": True,
        "hardware": True,
        "device": gpu_name,
        "driver_version": info.driver_version,
        "max_api_version": hex(info.max_api_version),
        "min_dims": (info.min_width, info.min_height),
        "max_dims": (info.max_width, info.max_height),
        "supported_grids": grids,
        # Whether the loaded bridge can take BGR24 host frames directly. A property of the
        # binary, read off the export, and what a caller negotiates the reduced transfer
        # path against.
        "bgr_input": bool(_NVOF_BGR_INPUT_EXPORT),
        "device_output": bool(_NVOF_DEVICE_OUTPUT_EXPORT),
        "device_rgba_input": bool(_NVOF_DEVICE_RGBA_INPUT_EXPORT),
        "error": None,
    }
    return _NATIVE_PROBE_CACHE


class NvOFHardwareEngine:
    """Real NVIDIA Optical Flow hardware engine backed by NVOF SDK C ABI Native Bridge.
    Strictly NO silent fallback to DIS when initialization or execution fails.
    """

    # What the most recent execute on this engine handed the bridge as frame input. Each
    # entry point restates it, so the value describes the call that produced the timing dict
    # rather than how the engine was built: a session driven by both input paths in one A/B
    # run cannot report a single format.
    nvof_input_format = "rgba8_host"

    def __init__(self, quality="1F", grid=None, preset=None):
        self.quality = quality
        self._session = None
        self._bridge = _load_native_bridge()
        if not self._bridge:
            raise RuntimeError("NVIDIA Optical Flow 初始化失败: 未找到 nvof_bridge.dll 硬件桥接库。")

        probe = probe_nvof_capabilities()
        if not probe["available"]:
            raise RuntimeError(f"NVIDIA Optical Flow 初始化失败: {probe.get('error', '硬件不可用')}")

        self._gpu_name = probe["device"]
        self._supported_grids = probe["supported_grids"]
        if not self._supported_grids:
            raise RuntimeError('NVIDIA Optical Flow 初始化失败: 驱动未报告支持的输出网格。')

        # Parse Grid and Preset
        self._grid_size, self._perf_level = self._resolve_grid_and_preset(quality, grid, preset)

        self._session = None
        self._session_dims = None  # (width, height)
        self._last_frame_seq = None
        self._force_reset_hints = True
        self.execute_count = 0
        self._last_timing = {}
        self._prev_counters = None  # For counter delta tracking

    def _resolve_grid_and_preset(self, quality: str, grid, preset) -> tuple:
        # If explicit grid & preset given:
        if grid is not None and preset is not None:
            g = int(grid)
            p_map = {"fast": 20, "medium": 10, "slow": 5}
            p = p_map.get(str(preset).lower(), 20)
            if g not in self._supported_grids:
                g = self._supported_grids[0]
            return g, p

        q = str(quality).upper().strip()
        # Mappings for all 9 combinations:
        # 1F -> (1, 20), 1M -> (1, 10), 1S -> (1, 5)
        # 2F -> (2, 20), 2M -> (2, 10), 2S -> (2, 5)
        # 4F -> (4, 20), 4M -> (4, 10), 4S -> (4, 5)
        if q == "1F":
            g, p = 1, 20
        elif q == "1M":
            g, p = 1, 10
        elif q == "1S":
            g, p = 1, 5
        elif q == "2F":
            g, p = 2, 20
        elif q == "2M":
            g, p = 2, 10
        elif q == "2S":
            g, p = 2, 5
        elif q == "4F":
            g, p = 4, 20
        elif q == "4M":
            g, p = 4, 10
        elif q == "4S":
            g, p = 4, 5
        else:
            g = 1 if 1 in self._supported_grids else self._supported_grids[0]
            p = 20

        if g not in self._supported_grids:
            g = self._supported_grids[0]
        return g, p

    def _ensure_session(self, width: int, height: int):
        if self._session is not None and self._session_dims == (width, height):
            return

        self._close_session()

        sess_ptr = c_void_p(0)
        res = self._bridge.nvof_bridge_create(
            c_uint32(width),
            c_uint32(height),
            c_uint32(self._grid_size),
            c_uint32(self._perf_level),
            byref(sess_ptr)
        )
        if res != 0 or not sess_ptr.value:
            err = self._bridge.nvof_bridge_get_last_error()
            err_str = err.decode('utf-8', errors='replace') if err else f"code {res}"
            raise RuntimeError(f"NVIDIA Optical Flow 会话创建失败 ({width}x{height}, grid={self._grid_size}): {err_str}")

        self._session = sess_ptr
        self._session_dims = (width, height)
        self._force_reset_hints = True
        # The bridge counters of a freshly created session start at zero while the
        # baseline taken against the previous one does not, so the first frame after a
        # resize would publish a negative delta. Re-baseline against the new session.
        self._prev_counters = self.get_counters()

        # Preallocate reusable frame and flow buffers to eliminate per-frame allocations
        self._cur_abgr_buf = np.empty((height, width, 4), dtype=np.uint8)
        self._prev_abgr_buf = np.empty((height, width, 4), dtype=np.uint8)
        self._flow_buf = np.empty((height, width, 2), dtype=np.float32)

    def _close_session(self):
        if self._session is not None and self._bridge:
            try:
                self._bridge.nvof_bridge_destroy(self._session)
            except Exception:
                pass
            self._session = None
            self._session_dims = None
            # No bridge counters exist to diff against until the next session opens.
            self._prev_counters = None

    def reset_temporal_hints(self):
        """Reset temporal hints (call on seek, jump, or non-consecutive frames)."""
        self._force_reset_hints = True

    def close(self):
        self._close_session()

    def __del__(self):
        self._close_session()

    def provider_info(self) -> dict:
        dev_idx = c_uint32(0)
        ctx_val = c_void_p(0)
        thread_id = c_uint32(0)
        if self._session and self._bridge:
            self._bridge.nvof_bridge_get_session_info(
                self._session, byref(dev_idx), byref(ctx_val), byref(thread_id)
            )

        return {
            "engine": "nvof",
            "provider": "nvidia-optical-flow",
            "hardware": True,
            "device": self._gpu_name,
            "available": True,
            "grid_size": self._grid_size,
            "perf_level": "fast" if self._perf_level == 20 else ("medium" if self._perf_level == 10 else "slow"),
            "device_index": dev_idx.value,
            "cuda_context": hex(ctx_val.value if ctx_val.value else 0),
            "thread_owner_id": thread_id.value,
            "sync_mode": "CRITICAL_SECTION + cuCtxPushCurrent",
        }

    def compute_flow(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray, reset_temporal_hints: bool = False) -> np.ndarray:
        """Compute backward optical flow (cur -> prev) using NVIDIA Optical Flow hardware.
        Explicitly converts OpenCV BGR24 to ABGR8 (RGBA32).
        Never copies raw BGR24 into NVOF hardware buffers.
        """
        if prev_bgr is None or cur_bgr is None:
            h, w = cur_bgr.shape[:2] if cur_bgr is not None else (720, 1280)
            return np.zeros((h, w, 2), dtype=np.float32)

        h, w = cur_bgr.shape[:2]
        self._ensure_session(w, h)
        self.nvof_input_format = "rgba8_host"

        t_py_start = time.perf_counter()

        # Explicitly convert OpenCV BGR24 to ABGR8 (RGBA in memory) using preallocated buffers
        t_cur_0 = time.perf_counter()
        if cur_bgr.ndim == 3 and cur_bgr.shape[2] == 3:
            transfer_profiler.convert_color(cur_bgr, cv2.COLOR_BGR2RGBA,
                                            dst=self._cur_abgr_buf,
                                            site="nvof_input")
            cur_abgr = self._cur_abgr_buf
        elif cur_bgr.ndim == 3 and cur_bgr.shape[2] == 4:
            cur_abgr = np.ascontiguousarray(cur_bgr, dtype=np.uint8)
        else:
            transfer_profiler.convert_color(cur_bgr, cv2.COLOR_GRAY2RGBA,
                                            dst=self._cur_abgr_buf,
                                            site="nvof_input")
            cur_abgr = self._cur_abgr_buf
        t_cur_1 = time.perf_counter()
        cur_bgr_to_rgba_ms = (t_cur_1 - t_cur_0) * 1000.0

        t_prev_0 = time.perf_counter()
        if prev_bgr.ndim == 3 and prev_bgr.shape[2] == 3:
            transfer_profiler.convert_color(prev_bgr, cv2.COLOR_BGR2RGBA,
                                            dst=self._prev_abgr_buf,
                                            site="nvof_input")
            prev_abgr = self._prev_abgr_buf
        elif prev_bgr.ndim == 3 and prev_bgr.shape[2] == 4:
            prev_abgr = np.ascontiguousarray(prev_bgr, dtype=np.uint8)
        else:
            transfer_profiler.convert_color(prev_bgr, cv2.COLOR_GRAY2RGBA,
                                            dst=self._prev_abgr_buf,
                                            site="nvof_input")
            prev_abgr = self._prev_abgr_buf
        t_prev_1 = time.perf_counter()
        prev_bgr_to_rgba_ms = (t_prev_1 - t_prev_0) * 1000.0

        bgr_to_abgr_ms = cur_bgr_to_rgba_ms + prev_bgr_to_rgba_ms

        out_flow = self._flow_buf

        cur_ptr = cur_abgr.ctypes.data_as(POINTER(c_uint8))
        prev_ptr = prev_abgr.ctypes.data_as(POINTER(c_uint8))
        flow_ptr = out_flow.ctypes.data_as(POINTER(c_float))

        do_reset = 1 if (reset_temporal_hints or self._force_reset_hints) else 0
        self._force_reset_hints = False

        timing = NvOFBridgeTiming()
        res = self._bridge.nvof_bridge_execute_timed(
            self._session, cur_ptr, prev_ptr, flow_ptr, 0, do_reset, byref(timing)
        )
        if res != 0:
            err = self._bridge.nvof_bridge_get_last_error()
            err_str = err.decode('utf-8', errors='replace') if err else f"code {res}"
            raise RuntimeError(f"NVIDIA Optical Flow 计算执行失败: {err_str}")

        t_py_end = time.perf_counter()
        python_total_ms = (t_py_end - t_py_start) * 1000.0

        self.execute_count += 1
        global _GLOBAL_NVOF_EXECUTE_COUNT
        _GLOBAL_NVOF_EXECUTE_COUNT += 1

        self._last_timing = {
            "bgr_to_abgr_ms": bgr_to_abgr_ms,
            "nvof_input_format": self.nvof_input_format,
            "cur_bgr_to_rgba_ms": cur_bgr_to_rgba_ms,
            "prev_bgr_to_rgba_ms": prev_bgr_to_rgba_ms,
            # One key per measured span. The spans the bridge cannot time by themselves are
            # reported unavailable instead of 0.0: the S10.5 conversion and the upscale are
            # one fused kernel, and the flow hardware's own execution time cannot be
            # separated from the synchronize that waits for it.
            "nvof_upload_ms": _timing_ms(timing.upload_ms),
            "nvof_submit_cpu_ms": _timing_ms(timing.submit_cpu_ms),
            "nvof_dependency_wait_ms": _timing_ms(timing.dependency_wait_ms),
            "nvof_download_ms": _timing_ms(timing.download_ms),
            "nvof_postprocess_wait_inclusive_ms": _timing_ms(timing.postprocess_wait_inclusive_ms),
            # Unmeasured on every RGBA-input entry point: no byte-map kernel runs, because the
            # CPU already did the reordering before the bridge was called.
            "nvof_input_convert_wait_ms": _timing_ms(timing.input_convert_wait_ms),
            "nvof_convert_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_gpu_execute_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_total_native_ms": _timing_ms(timing.total_native_ms),
            "python_total_ms": python_total_ms,
            "fps": 1000.0 / max(0.001, python_total_ms),
        }

        return out_flow

    def compute_flow_sequential(self, cur_bgr: np.ndarray, prev_bgr: np.ndarray = None,
                                reset_temporal_hints: bool = False, cur_rgba: np.ndarray = None) -> np.ndarray:
        """Compute backward optical flow (cur -> prev) with 1 H2D upload/frame in steady state.
        If cur_rgba is provided (pre-converted in FrameWorkspace), avoids any CPU color conversion.
        """
        if cur_bgr is None and cur_rgba is None:
            return np.zeros((720, 1280, 2), dtype=np.float32)

        if cur_rgba is not None:
            h, w = cur_rgba.shape[:2]
        else:
            h, w = cur_bgr.shape[:2]
        self._ensure_session(w, h)
        self.nvof_input_format = "rgba8_host"

        t_py_start = time.perf_counter()
        cur_bgr_to_rgba_ms = 0.0
        prev_bgr_to_rgba_ms = 0.0

        if cur_rgba is not None:
            cur_abgr = cur_rgba
        else:
            t_cur_0 = time.perf_counter()
            if cur_bgr.ndim == 3 and cur_bgr.shape[2] == 3:
                transfer_profiler.convert_color(cur_bgr, cv2.COLOR_BGR2RGBA,
                                                dst=self._cur_abgr_buf,
                                                site="nvof_input")
                cur_abgr = self._cur_abgr_buf
            elif cur_bgr.ndim == 3 and cur_bgr.shape[2] == 4:
                cur_abgr = np.ascontiguousarray(cur_bgr, dtype=np.uint8)
            else:
                transfer_profiler.convert_color(cur_bgr, cv2.COLOR_GRAY2RGBA,
                                                dst=self._cur_abgr_buf,
                                                site="nvof_input")
                cur_abgr = self._cur_abgr_buf
            t_cur_1 = time.perf_counter()
            cur_bgr_to_rgba_ms = (t_cur_1 - t_cur_0) * 1000.0

        do_reset = 1 if (reset_temporal_hints or self._force_reset_hints) else 0
        self._force_reset_hints = False

        prev_ptr = None
        if do_reset:
            if prev_bgr is not None:
                t_prev_0 = time.perf_counter()
                if prev_bgr.ndim == 3 and prev_bgr.shape[2] == 3:
                    transfer_profiler.convert_color(prev_bgr, cv2.COLOR_BGR2RGBA,
                                                    dst=self._prev_abgr_buf,
                                                    site="nvof_input")
                    prev_abgr = self._prev_abgr_buf
                elif prev_bgr.ndim == 3 and prev_bgr.shape[2] == 4:
                    prev_abgr = np.ascontiguousarray(prev_bgr, dtype=np.uint8)
                else:
                    transfer_profiler.convert_color(prev_bgr, cv2.COLOR_GRAY2RGBA,
                                                    dst=self._prev_abgr_buf,
                                                    site="nvof_input")
                    prev_abgr = self._prev_abgr_buf
                t_prev_1 = time.perf_counter()
                prev_bgr_to_rgba_ms = (t_prev_1 - t_prev_0) * 1000.0
                prev_ptr = prev_abgr.ctypes.data_as(POINTER(c_uint8))
            else:
                prev_ptr = cur_abgr.ctypes.data_as(POINTER(c_uint8))

        out_flow = self._flow_buf
        cur_ptr = cur_abgr.ctypes.data_as(POINTER(c_uint8))
        flow_ptr = out_flow.ctypes.data_as(POINTER(c_float))

        timing = NvOFBridgeTiming()
        res = self._bridge.nvof_bridge_execute_sequential(
            self._session, cur_ptr, prev_ptr, flow_ptr, 0, do_reset, byref(timing)
        )
        if res != 0:
            err = self._bridge.nvof_bridge_get_last_error()
            err_str = err.decode('utf-8', errors='replace') if err else f"code {res}"
            raise RuntimeError(f"NVIDIA Optical Flow sequential 计算执行失败: {err_str}")

        t_py_end = time.perf_counter()
        python_total_ms = (t_py_end - t_py_start) * 1000.0

        self.execute_count += 1
        global _GLOBAL_NVOF_EXECUTE_COUNT
        _GLOBAL_NVOF_EXECUTE_COUNT += 1

        self._last_timing = {
            "bgr_to_abgr_ms": cur_bgr_to_rgba_ms + prev_bgr_to_rgba_ms,
            "nvof_input_format": self.nvof_input_format,
            "cur_bgr_to_rgba_ms": cur_bgr_to_rgba_ms,
            "prev_bgr_to_rgba_ms": prev_bgr_to_rgba_ms,
            # The sequential path never times the post-execute synchronize on its own, so
            # its dependency wait arrives unmeasured and shows as unavailable; the whole
            # hardware wait sits inside the postprocess span below.
            "nvof_upload_ms": _timing_ms(timing.upload_ms),
            "nvof_submit_cpu_ms": _timing_ms(timing.submit_cpu_ms),
            "nvof_dependency_wait_ms": _timing_ms(timing.dependency_wait_ms),
            "nvof_download_ms": _timing_ms(timing.download_ms),
            "nvof_postprocess_wait_inclusive_ms": _timing_ms(timing.postprocess_wait_inclusive_ms),
            # Unmeasured on every RGBA-input entry point: no byte-map kernel runs, because the
            # CPU already did the reordering before the bridge was called.
            "nvof_input_convert_wait_ms": _timing_ms(timing.input_convert_wait_ms),
            "nvof_convert_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_gpu_execute_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_total_native_ms": _timing_ms(timing.total_native_ms),
            "python_total_ms": python_total_ms,
            "fps": 1000.0 / max(0.001, python_total_ms),
        }

        return out_flow

    def compute_flow_bgr24(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray,
                           reset_temporal_hints: bool = False) -> np.ndarray:
        """Backward flow (cur -> prev) from BGR24 host frames, reordered on the GPU.

        The CPU hands the bridge the working BGR buffer it already holds, so it never writes
        an NVOF copy of the frame and the upload is 3 bytes per pixel instead of 4; the
        channel order the hardware wants is filled by the bridge's byte-map kernel from
        persistent device staging. History semantics are the sequential path's: one upload on
        a steady frame, two on a reset.

        There is no fallback in here. A bridge without the export raises, because a caller that
        asked for the reduced transfer path has to be able to tell that it did not run.
        """
        if not probe_nvof_capabilities().get("bgr_input"):
            raise RuntimeError(
                "NVIDIA Optical Flow BGR input is unavailable: the loaded nvof_bridge exports "
                "no nvof_bridge_execute_bgr_sequential. Hand RGBA frames to "
                "compute_flow_sequential instead.")

        cur_bgr = _bgr24_host_view(cur_bgr, "cur_bgr")
        if prev_bgr is not None:
            prev_bgr = _bgr24_host_view(prev_bgr, "prev_bgr")

        h, w = cur_bgr.shape[:2]
        self._ensure_session(w, h)
        self.nvof_input_format = "bgr24_host"

        t_py_start = time.perf_counter()

        do_reset = 1 if (reset_temporal_hints or self._force_reset_hints) else 0
        self._force_reset_hints = False

        # Only a reset frame needs the reference on the way in; on a steady frame it is
        # already resident and the bridge swaps its own surfaces instead.
        prev_ptr = None
        prev_stride = 0
        if do_reset and prev_bgr is not None:
            prev_ptr = prev_bgr.ctypes.data_as(POINTER(c_uint8))
            prev_stride = int(prev_bgr.strides[0])

        out_flow = self._flow_buf
        timing = NvOFBridgeTiming()
        res = self._bridge.nvof_bridge_execute_bgr_sequential(
            self._session,
            cur_bgr.ctypes.data_as(POINTER(c_uint8)), int(cur_bgr.strides[0]),
            prev_ptr, prev_stride,
            out_flow.ctypes.data_as(POINTER(c_float)), 0, do_reset, byref(timing)
        )
        if res != 0:
            err = self._bridge.nvof_bridge_get_last_error()
            err_str = err.decode('utf-8', errors='replace') if err else f"code {res}"
            raise RuntimeError(f"NVIDIA Optical Flow BGR 输入计算执行失败: {err_str}")

        t_py_end = time.perf_counter()
        python_total_ms = (t_py_end - t_py_start) * 1000.0

        self.execute_count += 1
        global _GLOBAL_NVOF_EXECUTE_COUNT
        _GLOBAL_NVOF_EXECUTE_COUNT += 1

        self._last_timing = {
            # Zero because this entry point contains no conversion call to time. The
            # independent evidence is transfer_profiler's cpu_bgr_to_rgba@nvof_input_calls
            # delta, which stays 0 across a frame computed here; a test asserts the two
            # agree, so these are not a constant that was written instead of a measurement.
            "bgr_to_abgr_ms": 0.0,
            "nvof_input_format": self.nvof_input_format,
            "cur_bgr_to_rgba_ms": 0.0,
            "prev_bgr_to_rgba_ms": 0.0,
            "nvof_upload_ms": _timing_ms(timing.upload_ms),
            "nvof_submit_cpu_ms": _timing_ms(timing.submit_cpu_ms),
            "nvof_dependency_wait_ms": _timing_ms(timing.dependency_wait_ms),
            "nvof_download_ms": _timing_ms(timing.download_ms),
            "nvof_postprocess_wait_inclusive_ms": _timing_ms(timing.postprocess_wait_inclusive_ms),
            # The byte-map launches plus the synchronize that makes the surfaces safe for the
            # flow hardware to read, and nothing else: not the upload above, not the
            # postprocess below.
            "nvof_input_convert_wait_ms": _timing_ms(timing.input_convert_wait_ms),
            "nvof_convert_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_gpu_execute_ms": NVOF_TIMING_UNAVAILABLE,
            "nvof_total_native_ms": _timing_ms(timing.total_native_ms),
            "python_total_ms": python_total_ms,
            "fps": 1000.0 / max(0.001, python_total_ms),
        }

        return out_flow

    def compute_flow_bgr24_device(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray,
                                  reset_temporal_hints: bool = False) -> dict:
        """Run NVOF without downloading flow; return a borrowed CUDA resource descriptor.

        The device pointer belongs to this engine's session. It is valid only until the
        next execute, resize, or close. A native consumer must share this CUDA context
        or use explicit interop. It must never pass the pointer to dlssnr_process(),
        whose arguments are CPU pointers. This entry point is not used by the current
        host-based export path.
        """
        if not probe_nvof_capabilities().get("device_output"):
            raise RuntimeError("NVOF device-output entry point is unavailable")
        cur_bgr = _bgr24_host_view(cur_bgr, "cur_bgr")
        if prev_bgr is not None:
            prev_bgr = _bgr24_host_view(prev_bgr, "prev_bgr")
            if prev_bgr.shape != cur_bgr.shape:
                raise ValueError("prev_bgr and cur_bgr dimensions must match")

        h, w = cur_bgr.shape[:2]
        self._ensure_session(w, h)
        do_reset = 1 if (reset_temporal_hints or self._force_reset_hints) else 0
        prev_ptr = None
        prev_stride = 0
        if do_reset and prev_bgr is not None:
            prev_ptr = prev_bgr.ctypes.data_as(POINTER(c_uint8))
            prev_stride = int(prev_bgr.strides[0])

        timing = NvOFBridgeTiming()
        res = self._bridge.nvof_bridge_execute_bgr_sequential_device(
            self._session,
            cur_bgr.ctypes.data_as(POINTER(c_uint8)), int(cur_bgr.strides[0]),
            prev_ptr, prev_stride, do_reset, byref(timing))
        if res != 0:
            err = self._bridge.nvof_bridge_get_last_error()
            detail = err.decode("utf-8", errors="replace") if err else f"code {res}"
            raise RuntimeError(f"NVOF device-output execute failed: {detail}")

        dptr = c_uint64(0)
        pitch = c_size_t(0)
        device_index = c_uint32(0)
        context = c_void_p(0)
        owner_thread = c_uint32(0)
        if (self._bridge.nvof_bridge_get_output_deviceptr(
                self._session, byref(dptr), byref(pitch)) != 0 or
                self._bridge.nvof_bridge_get_session_info(
                    self._session, byref(device_index), byref(context),
                    byref(owner_thread)) != 0 or not dptr.value):
            raise RuntimeError("NVOF device-output descriptor unavailable after execute")

        self._force_reset_hints = False
        self.execute_count += 1
        global _GLOBAL_NVOF_EXECUTE_COUNT
        _GLOBAL_NVOF_EXECUTE_COUNT += 1
        self.nvof_input_format = "bgr24_host"
        self._last_timing = {
            "nvof_input_format": self.nvof_input_format,
            "nvof_upload_ms": _timing_ms(timing.upload_ms),
            "nvof_submit_cpu_ms": _timing_ms(timing.submit_cpu_ms),
            "nvof_postprocess_wait_inclusive_ms": _timing_ms(
                timing.postprocess_wait_inclusive_ms),
            "nvof_download_ms": _timing_ms(timing.download_ms),
            "nvof_total_native_ms": _timing_ms(timing.total_native_ms),
        }
        return {
            "device_ptr": dptr.value,
            "pitch_bytes": pitch.value,
            "cuda_context": context.value,
            "device_index": device_index.value,
            "owner_thread_id": owner_thread.value,
            "width": w,
            "height": h,
            "format": "float32x2",
        }

    @property
    def supports_bgr_nvof_input(self) -> bool:
        """Whether this engine's bridge can take BGR24 host frames directly."""
        return bool(probe_nvof_capabilities().get("bgr_input"))

    @property
    def supports_rgba_device_nvof_input(self) -> bool:
        """Whether this bridge can take an RGBA8 surface that never left the GPU."""
        return bool(probe_nvof_capabilities().get("device_rgba_input"))

    def device_input_counters(self) -> tuple:
        """Cumulative count and payload bytes of device-to-device input copies.

        Zero here is a measurement, not an absence: the copy is the only way a
        frame reaches NVOF on this entry point, so a zero would mean the frame
        crossed to the host instead.
        """
        if not self._session or not self._bridge or not (
                probe_nvof_capabilities().get("device_rgba_input")):
            return None, None
        copies = c_uint64(0)
        payload = c_uint64(0)
        if self._bridge.nvof_bridge_get_device_input_counters(
                self._session, byref(copies), byref(payload)) != 0:
            return None, None
        return copies.value, payload.value

    def compute_flow_rgba_device(self, cuda_context, rgba_device_ptr: int,
                                 rgba_pitch: int, width: int, height: int,
                                 reset_temporal_hints: bool = False) -> dict:
        """Run NVOF on an RGBA8 CUDA surface and keep the flow on device.

        The caller owns the input: it is borrowed from an allocation made by
        another bridge, in that bridge's context, and it must already be
        complete. Nothing in this call can download a frame, which is the point
        of the entry point.
        """
        if not probe_nvof_capabilities().get("device_rgba_input"):
            raise RuntimeError("NVOF device RGBA-input entry point is unavailable")
        if not cuda_context or not rgba_device_ptr or rgba_pitch < width * 4:
            raise ValueError("a matching RGBA8 CUDA surface and context are required")

        self._ensure_session(width, height)
        do_reset = 1 if (reset_temporal_hints or self._force_reset_hints) else 0
        timing = NvOFBridgeTiming()
        res = self._bridge.nvof_bridge_execute_rgba_device_sequential(
            self._session, int(rgba_device_ptr), int(rgba_pitch),
            cuda_context, do_reset, byref(timing))
        if res != 0:
            err = self._bridge.nvof_bridge_get_last_error()
            detail = err.decode("utf-8", errors="replace") if err else f"code {res}"
            raise RuntimeError(f"NVOF device RGBA execute failed: {detail}")

        dptr = c_uint64(0)
        pitch = c_size_t(0)
        device_index = c_uint32(0)
        context = c_void_p(0)
        owner_thread = c_uint32(0)
        if (self._bridge.nvof_bridge_get_output_deviceptr(
                self._session, byref(dptr), byref(pitch)) != 0 or
                self._bridge.nvof_bridge_get_session_info(
                    self._session, byref(device_index), byref(context),
                    byref(owner_thread)) != 0 or not dptr.value):
            raise RuntimeError("NVOF device flow descriptor unavailable after execute")

        self._force_reset_hints = False
        self.execute_count += 1
        global _GLOBAL_NVOF_EXECUTE_COUNT
        _GLOBAL_NVOF_EXECUTE_COUNT += 1
        self.nvof_input_format = "rgba8_device"
        self._last_timing = {
            "nvof_input_format": self.nvof_input_format,
            "nvof_upload_ms": _timing_ms(timing.upload_ms),
            "nvof_submit_cpu_ms": _timing_ms(timing.submit_cpu_ms),
            "nvof_postprocess_wait_inclusive_ms": _timing_ms(
                timing.postprocess_wait_inclusive_ms),
            "nvof_download_ms": _timing_ms(timing.download_ms),
            "nvof_total_native_ms": _timing_ms(timing.total_native_ms),
        }
        return {
            "device_ptr": dptr.value,
            "pitch_bytes": pitch.value,
            "cuda_context": context.value,
            "device_index": device_index.value,
            "owner_thread_id": owner_thread.value,
            "width": width,
            "height": height,
            "format": "float32x2",
        }

    def get_output_deviceptr(self) -> tuple:
        """Report the GPU address and pitch of the session's flow tensor.

        Diagnostic only. DLSSNR takes host pointers, so this value has no consumer in the
        pipeline and sharing it would not be zero-copy; NVOF flow reaches the CPU through
        the D2H copy in compute_flow().
        """
        if not self._session or not self._bridge:
            return 0, 0
        dptr = c_uint64(0)
        pitch = c_size_t(0)
        res = self._bridge.nvof_bridge_get_output_deviceptr(self._session, byref(dptr), byref(pitch))
        if res != 0:
            return 0, 0
        return dptr.value, pitch.value

    def get_last_timing(self) -> dict:
        return dict(self._last_timing)

    def get_counters(self) -> dict:
        # None wherever no native counter block was read: with no session there is no
        # staging allocation to report and no copy that happened, and 0 would claim a
        # measurement that did not happen. The bridge read failing is the same fact.
        unavailable_counters = {
            "h2d_copy_count": None,
            "h2d_bytes": None,
            "d2h_copy_count": None,
            "d2h_bytes": None,
            "reset_count": None,
            "bgr_staging_alloc_count": None,
            "bgr_input_call_count": None,
            "input_convert_launch_count": None,
        }
        if not self._session or not self._bridge:
            return {"execute_count": self.execute_count, **unavailable_counters}
        counters = NvOFBridgeCounters()
        res = self._bridge.nvof_bridge_get_counters(self._session, byref(counters))
        if res != 0:
            return {"execute_count": self.execute_count, **unavailable_counters}
        return {
            "execute_count": counters.execute_count,
            "h2d_copy_count": counters.h2d_copy_count,
            "h2d_bytes": counters.h2d_bytes,
            "d2h_copy_count": counters.d2h_copy_count,
            "d2h_bytes": counters.d2h_bytes,
            "reset_count": counters.reset_count,
            # Cumulative since the session was created. Staging is allocated twice by
            # nvof_bridge_create and never again, so a flat pair of deltas over a run of
            # frames is the evidence that no frame allocated or freed its own.
            "bgr_staging_alloc_count": counters.bgr_staging_alloc_count,
            "bgr_input_call_count": counters.bgr_input_call_count,
            "input_convert_launch_count": counters.input_convert_launch_count,
        }

    def get_counters_delta(self) -> dict:
        curr = self.get_counters()
        base = {} if self._prev_counters is None else self._prev_counters

        def _diff(key: str):
            here, there = curr.get(key), base.get(key, 0)
            if here is None or there is None:
                return None
            return here - there

        if self._prev_counters is None:
            first_execute = 1 if curr["execute_count"] > 0 else 0
        else:
            first_execute = None

        delta = {
            # The first snapshot has no baseline, so execute is reported as "ran at least
            # once" rather than as a cumulative count wearing a delta name.
            "nvof_execute_count_delta": (first_execute if first_execute is not None
                                         else _diff("execute_count")),
            "nvof_h2d_copy_count_delta": _diff("h2d_copy_count"),
            "nvof_h2d_bytes_delta": _diff("h2d_bytes"),
            "nvof_d2h_copy_count_delta": _diff("d2h_copy_count"),
            "nvof_d2h_bytes_delta": _diff("d2h_bytes"),
            "reset_count_delta": _diff("reset_count"),
            "nvof_bgr_staging_alloc_count_delta": _diff("bgr_staging_alloc_count"),
            "nvof_bgr_input_call_count_delta": _diff("bgr_input_call_count"),
            "nvof_input_convert_launch_count_delta": _diff("input_convert_launch_count"),
        }
        self._prev_counters = dict(curr)
        return delta


class DISOpticalFlowEngine:
    """OpenCV DIS CPU optical flow engine."""

    maintains_native_history = False

    def __init__(self, quality="Medium"):
        self.quality = quality
        self._dis_engine = None
        self._init_dis()

    def _init_dis(self):
        preset_map = {
            "FAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "SLOW": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            # Backward compatibility presets
            "4F": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "4M": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "4S": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            "2M": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            "2S": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            "PERFORMANCE": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "QUALITY": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        p = preset_map.get(str(self.quality).upper(), cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._dis_engine = cv2.DISOpticalFlow_create(p)
        if str(self.quality).upper() in ("2M", "2S", "SLOW"):
            self._dis_engine.setFinestScale(1)
            self._dis_engine.setPatchSize(8)
        else:
            self._dis_engine.setFinestScale(2)
            self._dis_engine.setPatchSize(8)

    def provider_info(self) -> dict:
        return {
            "engine": "dis",
            "provider": "opencv-dis-cpu",
            "hardware": False,
            "device": "cpu",
            "available": True,
        }

    def reset_temporal_hints(self):
        pass

    def compute_flow(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray, reset_temporal_hints: bool = False) -> np.ndarray:
        if prev_bgr is None or cur_bgr is None:
            h, w = cur_bgr.shape[:2] if cur_bgr is not None else (720, 1280)
            return np.zeros((h, w, 2), dtype=np.float32)

        h, w = cur_bgr.shape[:2]
        grid_scale = 0.5 if (w > 1920 or h > 1080) else 1.0

        if grid_scale < 1.0:
            gw, gh = int(w * grid_scale), int(h * grid_scale)
            g_prev = cv2.cvtColor(cv2.resize(prev_bgr, (gw, gh)), cv2.COLOR_BGR2GRAY)
            g_cur = cv2.cvtColor(cv2.resize(cur_bgr, (gw, gh)), cv2.COLOR_BGR2GRAY)
            flow_down = self._dis_engine.calc(g_cur, g_prev, None)
            flow = cv2.resize(flow_down, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= (1.0 / grid_scale)
            flow[..., 1] *= (1.0 / grid_scale)
        else:
            g_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
            g_cur = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
            flow = self._dis_engine.calc(g_cur, g_prev, None)

        return flow.astype(np.float32)

    def compute_flow_sequential(self, cur_bgr: np.ndarray, prev_bgr: np.ndarray = None,
                                reset_temporal_hints: bool = False, cur_rgba: np.ndarray = None) -> np.ndarray:
        return self.compute_flow(prev_bgr if prev_bgr is not None else cur_bgr, cur_bgr, reset_temporal_hints)

    def get_counters(self) -> dict:
        # A CPU engine reads no bridge counters, so every one of these is an absent
        # measurement rather than a measured zero.
        return {
            "execute_count": None,
            "h2d_copy_count": None,
            "h2d_bytes": None,
            "d2h_copy_count": None,
            "d2h_bytes": None,
            "reset_count": None,
        }

    def get_counters_delta(self) -> dict:
        return {
            "nvof_execute_count_delta": None,
            "nvof_h2d_copy_count_delta": None,
            "nvof_h2d_bytes_delta": None,
            "nvof_d2h_copy_count_delta": None,
            "nvof_d2h_bytes_delta": None,
            "reset_count_delta": None,
        }


def raft_analysis_size(width, height, edge):
    if int(edge) not in (512, 720, 1080) or width <= 0 or height <= 0:
        raise ValueError('Invalid RAFT long-edge analysis geometry')
    scale = min(1.0, edge / max(width, height))
    return tuple(max(8, round(v*scale/8)*8) for v in (width, height))


class RaftOpticalFlowEngine:
    """RAFT-Large deep neural network optical flow engine."""

    maintains_native_history = False

    def __init__(self, quality="Balanced", analysis_edge=None, iterations=None, *, shared_weights=None):
        self.quality = quality
        self.analysis_edge = int(analysis_edge if analysis_edge is not None else {'Fast': 512, 'Quality': 1080}.get(quality, 720))
        self.iterations = int(iterations if iterations is not None else {'Fast': 4, 'Quality': 12}.get(quality, 6))
        if self.analysis_edge not in (512, 720, 1080) or not 1 <= self.iterations <= 24:
            raise ValueError('RAFT analysis long edge must be 512/720/1080; updates must be 1–24')
        self._prepared_source = self._prepared_tensor = None
        self._timing = {}
        self._raft_model = None
        self._raft_transforms = None
        self._device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        if shared_weights is None:
            with torch_primary_context(self._device):
                self._init_raft()
        else:
            # Only weights are shared. Configuration and normalized pair cache
            # belong to this facade; its caller serializes model scratch use.
            self._raft_model = shared_weights

    def _init_raft(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to run RAFT-Large optical flow.")
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        weights = Raft_Large_Weights.DEFAULT
        self._raft_transforms = weights.transforms()

        # A weight the user supplies, so it is looked for beside the launcher rather
        # than inside the bundle's own contents directory. Same folder in the source
        # tree; where a portable build keeps the files an install can be given.
        local_pth = os.path.join(bundle_paths.app_dir(), "models", "raft_large_c_t.pth")
        if os.path.exists(local_pth):
            m = raft_large(weights=None).to(self._device).eval()
            m.load_state_dict(torch.load(local_pth, map_location="cpu"))
            self._raft_model = m
        else:
            if os.environ.get('LUNARC_STANDALONE') == '1' and not os.path.isfile(os.path.join(
                    os.environ.get('TORCH_HOME', ''), 'hub', 'checkpoints',
                    'raft_large_C_T_SKHT_V2-ff5fadd5.pth')):
                raise RuntimeError('完整运行环境中缺少 RAFT-Large 权重，请重新安装模型组件')
            self._raft_model = raft_large(weights=weights).to(self._device).eval()

    def provider_info(self) -> dict:
        is_hw = (self._device == "cuda")
        return {
            "engine": "raft",
            "provider": "torch-raft",
            "analysis_long_edge": self.analysis_edge,
            "iterations": self.iterations, "precision": "float32",
            "hardware": is_hw,
            "device": self._device,
            "available": True,
        }

    def reset_temporal_hints(self):
        self._prepared_source = self._prepared_tensor = None

    def get_last_timing(self):
        return dict(self._timing)

    def _prepare(self, bgr, size):
        small = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR) if bgr.shape[1::-1] != size else bgr
        # One normalized tensor transfer per new frame. Retain an owned exact
        # fingerprint so reused/mutated OpenCV arrays cannot masquerade as history.
        import hashlib
        fingerprint = hashlib.sha256(memoryview(np.ascontiguousarray(bgr))).digest()
        tensor = torch.from_numpy(np.ascontiguousarray(small[..., ::-1])).permute(2, 0, 1)[None]
        tensor = tensor.to(self._device, dtype=torch.float32).div_(255).mul_(2).sub_(1)
        return fingerprint, tensor

    def _infer(self, current, previous, size):
        import torch.nn.functional as functional
        fw, fh = size
        # torchvision's four correlation pyramid levels require at least 128.
        # Pad tiny/ultrawide inputs, never distort their aspect ratio to fill it.
        pad = (0, max(0, 128-fw), 0, max(0, 128-fh))
        if any(pad):
            current = functional.pad(current, pad, mode='replicate')
            previous = functional.pad(previous, pad, mode='replicate')
        with torch.inference_mode():
            try:
                flow = self._raft_model(current, previous, num_flow_updates=self.iterations)[-1]
            finally:
                # torchvision's correlation block is mutable scratch, not
                # temporal history. Release its full pair pyramid after use.
                self._raft_model.corr_block.corr_pyramid = []
        return flow[..., :fh, :fw]

    def compute_flow(self, prev_bgr, cur_bgr, reset_temporal_hints=False):
        # The CPU-facing adapter can follow NVOF on a reused request thread.
        # Its inputs are host arrays, but model/tensor allocations are CUDA;
        # they must not inherit NVOF's disposable session context.
        with torch_primary_context(self._device):
            return self._compute_flow(prev_bgr, cur_bgr, reset_temporal_hints)

    def _compute_flow(self, prev_bgr, cur_bgr, reset_temporal_hints=False):
        if cur_bgr is None:
            return np.zeros((720, 1280, 2), np.float32)
        h, w = cur_bgr.shape[:2]
        if prev_bgr is None:
            prev_bgr = cur_bgr
        if prev_bgr.shape != cur_bgr.shape:
            raise ValueError('RAFT frame pair extents must match')
        if reset_temporal_hints:
            self.reset_temporal_hints()
        size = raft_analysis_size(w, h, self.analysis_edge)
        started = time.perf_counter()
        import hashlib
        previous_key = hashlib.sha256(memoryview(np.ascontiguousarray(prev_bgr))).digest()
        reused = previous_key == self._prepared_source and self._prepared_tensor is not None
        previous = self._prepared_tensor if reused else self._prepare(prev_bgr, size)[1]
        key, current = self._prepare(cur_bgr, size)
        prepared = time.perf_counter()
        flow = self._infer(current, previous, size)
        host = flow[0].permute(1, 2, 0).float().cpu().numpy()
        completed = time.perf_counter()
        fw, fh = size
        if size != (w, h):
            host = cv2.resize(host, (w, h), interpolation=cv2.INTER_LINEAR)
            host[..., 0] *= w/fw
            host[..., 1] *= h/fh
        self._prepared_source, self._prepared_tensor = key, current
        self._timing = dict(analysis_dimensions=list(size), analysis_long_edge=self.analysis_edge,
            iterations=self.iterations, previous_tensor_reused=reused,
            prepare_ms=(prepared-started)*1000, inference_and_readback_ms=(completed-prepared)*1000,
            restore_ms=(time.perf_counter()-completed)*1000, direction='current_to_previous',
            units='input_pixels', precision='float32', host_flow_readback_bytes=flow.numel()*4)
        return np.ascontiguousarray(host, dtype=np.float32)

    def compute_flow_sequential(self, cur_bgr, prev_bgr=None, reset_temporal_hints=False, cur_rgba=None):
        return self.compute_flow(prev_bgr, cur_bgr, reset_temporal_hints)

    def get_counters(self) -> dict:
        # A CPU engine reads no bridge counters, so every one of these is an absent
        # measurement rather than a measured zero.
        return {
            "execute_count": None,
            "h2d_copy_count": None,
            "h2d_bytes": None,
            "d2h_copy_count": None,
            "d2h_bytes": None,
            "reset_count": None,
        }

    def get_counters_delta(self) -> dict:
        return {
            "nvof_execute_count_delta": None,
            "nvof_h2d_copy_count_delta": None,
            "nvof_h2d_bytes_delta": None,
            "nvof_d2h_copy_count_delta": None,
            "nvof_d2h_bytes_delta": None,
            "reset_count_delta": None,
        }


class OpticalFlowEngine:
    """Unified Optical Flow interface producing backward motion vectors suitable for DLSS."""

    _migration_warned = False

    def __init__(self, engine_type="nvof", quality="1F", grid=None, preset=None, raft_analysis_edge=None, raft_iterations=None):
        raw_engine = str(engine_type).lower()

        # Handle legacy "amd" migration
        if raw_engine in ("amd", "amdfx"):
            if not OpticalFlowEngine._migration_warned:
                print("[flow] Migration warning: legacy 'amd' flow_engine migrated to 'dis' (OpenCV DIS CPU)", flush=True)
                OpticalFlowEngine._migration_warned = True
            raw_engine = "dis"

        self.engine_type = raw_engine
        self.quality = quality
        self._engine = None

        if self.engine_type in ("nvof", "nvidia"):
            self._engine = NvOFHardwareEngine(quality=quality, grid=grid, preset=preset)
        elif self.engine_type in ("dis", "opencv", "cpu"):
            self._engine = DISOpticalFlowEngine(quality=quality)
        elif self.engine_type in ("raft", "raft_large"):
            self._engine = RaftOpticalFlowEngine(quality=quality, analysis_edge=raft_analysis_edge, iterations=raft_iterations)
        else:
            raise ValueError(f"Unknown optical flow engine: {engine_type!r}. Supported engines: 'nvof', 'dis', 'raft'")

    def provider_info(self) -> dict:
        return self._engine.provider_info()

    def get_last_timing(self) -> dict:
        if hasattr(self._engine, "get_last_timing"):
            return self._engine.get_last_timing()
        return {}

    def get_counters(self) -> dict:
        if hasattr(self._engine, "get_counters"):
            return self._engine.get_counters()
        return {
            "execute_count": self.execute_count,
            "h2d_copy_count": None,
            "h2d_bytes": None,
            "d2h_copy_count": None,
            "d2h_bytes": None,
            "reset_count": None,
        }

    @property
    def execute_count(self) -> int:
        return getattr(self._engine, "execute_count", 0)

    @property
    def maintains_native_history(self) -> bool:
        """Return whether the underlying engine maintains native GPU history."""
        if self.engine_type in ("nvof", "nvidia"):
            return True
        return False

    @property
    def supports_bgr_nvof_input(self) -> bool:
        """Whether the underlying engine can take BGR24 host frames as NVOF input.

        A capability of the loaded bridge plus the engine that was chosen, reported here so
        a caller negotiates one attribute instead of probing an engine it may not have.
        """
        return bool(getattr(self._engine, "supports_bgr_nvof_input", False))

    def get_counters_delta(self) -> dict:
        """Get delta counters from the underlying engine."""
        if hasattr(self._engine, "get_counters_delta"):
            return self._engine.get_counters_delta()
        return {
            "nvof_execute_count_delta": None,
            "nvof_h2d_copy_count_delta": None,
            "nvof_h2d_bytes_delta": None,
            "nvof_d2h_copy_count_delta": None,
            "nvof_d2h_bytes_delta": None,
            "reset_count_delta": None,
        }

    def reset_temporal_hints(self):
        if hasattr(self._engine, "reset_temporal_hints"):
            self._engine.reset_temporal_hints()

    def compute_flow(self, prev_bgr, cur_bgr, reset_temporal_hints: bool = False) -> np.ndarray:
        return self._engine.compute_flow(prev_bgr, cur_bgr, reset_temporal_hints=reset_temporal_hints)

    def compute_flow_sequential(self, cur_bgr: np.ndarray, prev_bgr: np.ndarray = None,
                                reset_temporal_hints: bool = False, cur_rgba: np.ndarray = None) -> np.ndarray:
        if hasattr(self._engine, "compute_flow_sequential"):
            return self._engine.compute_flow_sequential(cur_bgr, prev_bgr, reset_temporal_hints, cur_rgba)
        return self._engine.compute_flow(prev_bgr if prev_bgr is not None else cur_bgr, cur_bgr, reset_temporal_hints)

    def compute_flow_bgr24(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray,
                           reset_temporal_hints: bool = False) -> np.ndarray:
        """Backward flow from BGR24 host frames, for engines whose bridge can take them.

        No fallback: an engine without the capability raises, so a caller that asked for the
        reduced transfer path learns it did not run instead of measuring the CPU path.
        """
        if not hasattr(self._engine, "compute_flow_bgr24"):
            raise RuntimeError(
                f"Flow engine {self.engine_type!r} has no BGR24 NVOF input path; the reduced "
                f"transfer mode cannot run on it.")
        return self._engine.compute_flow_bgr24(prev_bgr, cur_bgr,
                                               reset_temporal_hints=reset_temporal_hints)

    def compute_flow_bgr24_device(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray,
                                  reset_temporal_hints: bool = False) -> dict:
        if not hasattr(self._engine, "compute_flow_bgr24_device"):
            raise RuntimeError(f"Flow engine {self.engine_type!r} has no device-output path")
        return self._engine.compute_flow_bgr24_device(
            prev_bgr, cur_bgr, reset_temporal_hints=reset_temporal_hints)

    @property
    def supports_rgba_device_nvof_input(self) -> bool:
        return bool(getattr(self._engine, "supports_rgba_device_nvof_input", False))

    def device_input_counters(self):
        if not hasattr(self._engine, "device_input_counters"):
            return None, None
        return self._engine.device_input_counters()

    def compute_flow_rgba_device(self, cuda_context, rgba_device_ptr: int,
                                 rgba_pitch: int, width: int, height: int,
                                 reset_temporal_hints: bool = False) -> dict:
        if not hasattr(self._engine, "compute_flow_rgba_device"):
            raise RuntimeError(
                f"Flow engine {self.engine_type!r} has no device RGBA input path")
        return self._engine.compute_flow_rgba_device(
            cuda_context, rgba_device_ptr, rgba_pitch, width, height,
            reset_temporal_hints=reset_temporal_hints)

    def close(self):
        if hasattr(self._engine, "close"):
            self._engine.close()


def get_all_flow_providers() -> list:
    """Return real hardware and driver status of all available optical flow providers."""
    nvof_probe = probe_nvof_capabilities()
    providers = []

    # 1. Real NVOF (Grid and Presets decoupled per Requirement 5)
    if nvof_probe["available"]:
        grids = []
        for g in nvof_probe.get("supported_grids", [1, 2, 4]):
            if g == 1:
                grids.append({"id": 1, "label": "1x1 full-resolution output grid (推荐)"})
            elif g == 2:
                grids.append({"id": 2, "label": "2x2 网格 · 高帧率 (Grid 2x2)"})
            elif g == 4:
                grids.append({"id": 4, "label": "4x4 网格 · 极低开销 (Grid 4x4)"})

        presets = [
            {"id": "fast", "label": "速度"},
            {"id": "medium", "label": "均衡"},
            {"id": "slow", "label": "质量"},
        ]

        # Backward compatibility composite qualities (all 9 Grid & Preset combinations)
        compat_qualities = [
            {"id": "1F", "label": "1x1 full-resolution output grid · 极速 (Fast · 推荐)", "grid": 1, "preset": "fast"},
            {"id": "1M", "label": "1x1 full-resolution output grid · 平衡 (Medium)", "grid": 1, "preset": "medium"},
            {"id": "1S", "label": "1x1 full-resolution output grid · 精度 (Slow)", "grid": 1, "preset": "slow"},
            {"id": "2F", "label": "2x2 网格 · 速度", "grid": 2, "preset": "fast"},
            {"id": "2M", "label": "2x2 网格 · 平衡 (Medium)", "grid": 2, "preset": "medium"},
            {"id": "2S", "label": "2x2 网格 · 精度 (Slow)", "grid": 2, "preset": "slow"},
            {"id": "4F", "label": "4x4 网格 · 速度", "grid": 4, "preset": "fast"},
            {"id": "4M", "label": "4x4 网格 · 平衡 (Medium)", "grid": 4, "preset": "medium"},
            {"id": "4S", "label": "4x4 网格 · 精度 (Slow)", "grid": 4, "preset": "slow"},
        ]

        providers.append({
            "engine": "nvof",
            "name": f"NVIDIA Optical Flow ({nvof_probe['device']})",
            "provider": "nvidia-optical-flow",
            "available": True,
            "hardware": True,
            "device": nvof_probe["device"],
            "grids": grids,
            "presets": presets,
            "qualities": compat_qualities,
        })
    else:
        providers.append({
            "engine": "nvof",
            "name": f"NVIDIA Optical Flow (不可用: {nvof_probe.get('error', '硬件不支持')})",
            "provider": "nvidia-optical-flow",
            "available": False,
            "hardware": False,
            "device": None,
            "error": nvof_probe.get("error"),
            "grids": [],
            "presets": [],
            "qualities": [],
        })

    # The GPU Transfer adapter owns SDK histories and needs the decoder context.
    # Do not create a session while merely listing UI capabilities.
    from fidelityfx_flow_engine import probe_bridge
    providers.append({"engine": "fidelityfx", "name": "AMD FidelityFX Optical Flow",
        "provider": "fidelityfx-d3d12-cuda", "hardware": True,
        "device": "D3D12/CUDA (checked per job)", "grids": [{"id": 8, "label": "8x8"}],
        "qualities": [], "presets": [], "requires_gpu_transfer": True,
        **probe_bridge()})

    # 2. OpenCV DIS CPU
    providers.append({
        "engine": "dis",
        "name": "OpenCV DIS (CPU)",
        "provider": "opencv-dis-cpu",
        "available": True,
        "hardware": False,
        "device": "cpu",
        "qualities": [
            {"id": "Fast", "label": "速度"},
            {"id": "Medium", "label": "均衡"},
            {"id": "Slow", "label": "质量"},
        ]
    })

    # 3. RAFT-Large
    is_torch_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
    providers.append({
        "engine": "raft",
        "name": f"深度学习模型 (RAFT-Large · {'CUDA' if is_torch_cuda else 'CPU'})",
        "provider": "torch-raft",
        "available": is_torch_cuda,
        "hardware": is_torch_cuda,
        "device": "cuda" if is_torch_cuda else "cpu",
        "qualities": [
            {"id": "Fast", "label": "速度 · 长边 512 / 4 次迭代"},
            {"id": "Balanced", "label": "均衡 · 长边 720 / 6 次迭代"},
            {"id": "Quality", "label": "质量 · 长边 1080 / 12 次迭代"},
        ]
    })

    return providers


_ENGINES = {}

def get_optical_flow_engine(engine_type="nvof", quality="1F", grid=None, preset=None,
                           raft_analysis_edge=None, raft_iterations=None) -> OpticalFlowEngine:
    norm_type = str(engine_type).lower()
    if norm_type in ("amd", "amdfx"):
        norm_type = "dis"
    key = (norm_type, str(quality), str(grid), str(preset))
    if norm_type in ('raft', 'raft_large'):
        key += (raft_analysis_edge, raft_iterations)
        if key not in _ENGINES:
            # Share immutable weights but never keep old pair/tensor histories.
            previous = next((e for k, e in _ENGINES.items() if isinstance(k, tuple)
                             and k[0] in ('raft', 'raft_large')), None)
            if previous is not None:
                engine = previous._engine
                edge = int(raft_analysis_edge if raft_analysis_edge is not None
                           else {'Fast':512, 'Quality':1080}.get(quality,720))
                iterations = int(raft_iterations if raft_iterations is not None
                                 else {'Fast':4, 'Quality':12}.get(quality,6))
                if edge not in (512,720,1080) or not 1 <= iterations <= 24:
                    raise ValueError('Invalid RAFT analysis configuration')
                engine.analysis_edge = edge
                engine.iterations = iterations
                engine.reset_temporal_hints()
                for k in list(_ENGINES):
                    if isinstance(k, tuple) and k[0] in ('raft', 'raft_large'):
                        del _ENGINES[k]
                _ENGINES[key] = previous

    # Bounded lifecycle: hardware engines hold GPU memory, contexts and native sessions.
    # When switching parameters for nvof (or requesting a new one), close and evict the previous instance.
    if norm_type in ("nvof", "nvidia"):
        active_key = _ENGINES.get("_active_nvof_key")
        if active_key != key:
            old_eng = _ENGINES.pop("nvof", None)
            if old_eng is not None:
                try:
                    old_eng.close()
                except Exception:
                    pass
            eng = OpticalFlowEngine(engine_type, quality=quality, grid=grid, preset=preset)
            pinfo = eng.provider_info()
            print(f"[flow] engine={key[0]!r} quality={key[1]!r} provider={pinfo['provider']} "
                  f"(hardware={pinfo['hardware']}, device={pinfo.get('device')})", flush=True)
            _ENGINES["nvof"] = eng
            _ENGINES["_active_nvof_key"] = key
        return _ENGINES["nvof"]

    # Non-hardware engines: CPU DIS or in-memory RAFT
    if key not in _ENGINES:
        eng = OpticalFlowEngine(engine_type, quality=quality, grid=grid, preset=preset,
                               raft_analysis_edge=raft_analysis_edge, raft_iterations=raft_iterations)
        pinfo = eng.provider_info()
        print(f"[flow] engine={key[0]!r} quality={key[1]!r} provider={pinfo['provider']} "
                  f"(hardware={pinfo['hardware']}, device={pinfo.get('device')})", flush=True)
        _ENGINES[key] = eng
    return _ENGINES[key]


class FrameGuidanceService:
    """Unified Frame Guidance Service (Magpie Architecture).

    Guarantees:
    1. Single Optical Flow Computation per Frame ID:
       Ensures NVOF/Motion is computed exactly once for each frame generation (idx/frame_id)
       across multiple consumers.
    2. Provider-Specific History Policy:
       - NVOF: uses native GPU hardware ping-pong (0 CPU frame copies in steady state).
       - DIS / CPU engines: preallocated previous frame buffer, distinct from workspace to prevent aliasing.
    3. Correct Guidance Backend Naming:
       - working_resolution_nvof_d2h: direct working resolution NVOF with host download.
       - full_resolution_nvof_cpu_adapter: full resolution NVOF with CPU displacement scaling.
       - cpu_dis / gpu_raft / disabled.
    4. Measured Native Counter Deltas:
       Reports exact per-frame hardware H2D/D2H copy deltas from driver counters.
    """

    def __init__(self, engine_type: str = "nvof", flow_quality: str = "4M", grid=None, preset=None,
                 *, dedicated: bool = False, raft_analysis_edge=None, raft_iterations=None):
        self.engine_type = engine_type
        self.flow_quality = flow_quality
        self.grid = grid
        self.preset = preset
        self.dedicated = bool(dedicated)
        self.raft_options = dict(raft_analysis_edge=raft_analysis_edge, raft_iterations=raft_iterations)
        self._flow_engine = None
        self._init_engine()

        self._last_frame_id = None
        self._last_src_flow = None
        self._prev_working_bgr = None
        self._adapter_cache = {}
        self._zero_buffers = {}

    def _init_engine(self):
        if self.engine_type in ("nvof", "nvidia", "dis", "opencv", "cpu", "raft", "raft_large"):
            try:
                factory = (OpticalFlowEngine if self.dedicated
                           else get_optical_flow_engine)
                kwargs = self.raft_options if self.engine_type in ('raft', 'raft_large') else {}
                self._flow_engine = factory(
                    self.engine_type, quality=self.flow_quality, grid=self.grid, preset=self.preset, **kwargs)
            except Exception as e:
                if self.engine_type in ('raft', 'raft_large'):
                    raise RuntimeError(f'RAFT initialization failed; no zero-flow fallback: {e}') from e
                print(f"[guidance] Warning: failed to init flow engine {self.engine_type}: {e}", flush=True)
                self._flow_engine = None
        else:
            self._flow_engine = None

    def get_zero_flow(self, w: int, h: int) -> np.ndarray:
        key = ("flow", w, h)
        if key not in self._zero_buffers:
            self._zero_buffers[key] = np.zeros((h, w, 2), dtype=np.float32)
        return self._zero_buffers[key]

    def reset(self):
        self._last_frame_id = None
        self._last_src_flow = None
        self._prev_working_bgr = None
        self._adapter_cache.clear()
        if self._flow_engine and hasattr(self._flow_engine, "reset_temporal_hints"):
            self._flow_engine.reset_temporal_hints()

    def update_settings(self, engine_type: str, flow_quality: str, grid=None, preset=None):
        if (self.engine_type, self.flow_quality, self.grid, self.preset) != (engine_type, flow_quality, grid, preset):
            self.engine_type = engine_type
            self.flow_quality = flow_quality
            self.grid = grid
            self.preset = preset
            self._init_engine()
            self.reset()

    @property
    def nvof_rgba_input_only(self) -> bool:
        """Whether this service's flow input has to be a CPU-converted RGBA copy.

        Exactly one shape says yes: an NVOF engine whose bridge has no BGR entry point. That
        is the one input a reduced transfer order cannot take, and everything else has no
        such conversion to remove - no engine at all, or a CPU framework engine that was
        handed BGR24 frames from the beginning. Whether the engine runs at all is the
        guidance mode's business, and the caller that asks this question knows it.
        """
        engine = self._flow_engine
        if engine is None:
            return False
        if not getattr(engine, "maintains_native_history", False):
            return False
        return not bool(getattr(engine, "supports_bgr_nvof_input", False))

    def get_guidance(self, frame_id: int, cur_bgr: np.ndarray, prev_bgr: np.ndarray = None,
                     target_w: int = None, target_h: int = None,
                     guidance_mode: int = 1, is_discontinuous: bool = False,
                     out_flow: np.ndarray = None,
                     input_format: str = "rgba8_host") -> tuple:
        """Returns (adapted_flow, timing_dict).
        
        If guidance_mode is 0 (Disabled): returns preallocated zero flow without running NVOF.
        If guidance_mode is 1 (Flow enabled): computes source flow once, then adapts to target extent.

        `input_format` describes the frame the caller is handing over, not a preference:
        "bgr24_host" says "this is BGR24 and I am not making an RGBA copy of it for you". On
        an NVOF engine that asks for the native BGR entry point, and an engine whose bridge
        cannot take it raises rather than converting behind the caller's back. A CPU or
        framework engine was always handed BGR24, so the request changes nothing for it.
        """
        if input_format not in ("rgba8_host", "bgr24_host"):
            raise ValueError(
                f"unknown flow input_format {input_format!r}; this service takes "
                f"'rgba8_host' or 'bgr24_host'")
        src_h, src_w = cur_bgr.shape[:2]
        tw = target_w or src_w
        th = target_h or src_h

        timing_dict = {}
        if guidance_mode not in (0, 1):
            raise ValueError("guidance_mode supports only 0 (off) or 1 (flow)")
        need_flow = guidance_mode == 1

        if not need_flow or self._flow_engine is None:
            flow = self.get_zero_flow(tw, th)
            timing_dict["guidance_backend"] = "disabled"
            return flow, timing_dict

        maintains_history = getattr(self._flow_engine, "maintains_native_history", False)
        # Recorded so a reported zero CPU previous-frame copy can be read as "the native
        # GPU history path ran", rather than as "the counter was never reached".
        timing_dict["nvof_native_history"] = bool(maintains_history)

        # 1. Compute Full/Working Resolution Source Flow once per frame_id
        if self._last_frame_id != frame_id or is_discontinuous:
            self._adapter_cache.clear()
            disc = is_discontinuous or (self._last_frame_id is None or frame_id != self._last_frame_id + 1)

            if maintains_history:
                if input_format == "bgr24_host":
                    # The working frame goes to the bridge as the caller holds it. No CPU
                    # copy of the frame is made for NVOF's benefit and the upload is three
                    # bytes per pixel; an engine without the entry point raises here rather
                    # than converting, which is what makes the zero at the conversion seam
                    # below mean "this path has no conversion" instead of "nothing counted".
                    src_flow = self._flow_engine.compute_flow_bgr24(
                        prev_bgr=prev_bgr, cur_bgr=cur_bgr, reset_temporal_hints=disc
                    )
                else:
                    src_flow = self._flow_engine.compute_flow_sequential(
                        cur_bgr=cur_bgr, prev_bgr=prev_bgr, reset_temporal_hints=disc
                    )
                timing_dict["cpu_previous_frame_copy_bytes"] = 0
            else:
                # CPU / Framework engine (DIS / RAFT): requires distinct previous CPU buffer
                if self._prev_working_bgr is None or self._prev_working_bgr.shape != cur_bgr.shape:
                    self._prev_working_bgr = np.empty_like(cur_bgr)
                    disc = True

                prev_input = cur_bgr if disc else self._prev_working_bgr
                src_flow = self._flow_engine.compute_flow(
                    prev_bgr=prev_input, cur_bgr=cur_bgr, reset_temporal_hints=disc
                )
                np.copyto(self._prev_working_bgr, cur_bgr)
                timing_dict["cpu_previous_frame_copy_bytes"] = cur_bgr.nbytes

            if hasattr(self._flow_engine, "get_last_timing"):
                timing_dict.update(self._flow_engine.get_last_timing())
            if hasattr(self._flow_engine, "get_counters_delta"):
                timing_dict.update(self._flow_engine.get_counters_delta())

            self._last_frame_id = frame_id
            self._last_src_flow = src_flow
        else:
            src_flow = self._last_src_flow
            if hasattr(self._flow_engine, "get_last_timing"):
                timing_dict.update(self._flow_engine.get_last_timing())
            if hasattr(self._flow_engine, "get_counters_delta"):
                timing_dict.update(self._flow_engine.get_counters_delta())

        # 2. Determine exact Guidance Backend string
        eng_name = str(getattr(self._flow_engine, "engine_type", self.engine_type)).lower()
        if (tw, th) == (src_w, src_h):
            adapted_flow = src_flow
            if eng_name in ("nvof", "nvidia"):
                timing_dict["guidance_backend"] = "working_resolution_nvof_d2h"
            elif eng_name in ("dis", "opencv", "cpu"):
                timing_dict["guidance_backend"] = "cpu_dis"
            elif eng_name in ("raft", "raft_large"):
                timing_dict["guidance_backend"] = "gpu_raft"
            else:
                timing_dict["guidance_backend"] = f"working_resolution_{eng_name}"
        else:
            # Multi-consumer adaptation: spatial interpolation and displacement scaling on CPU
            timing_dict["guidance_backend"] = "full_resolution_nvof_cpu_adapter"
            cache_key = (tw, th)
            if cache_key in self._adapter_cache:
                adapted_flow = self._adapter_cache[cache_key]
            else:
                t_adapt_0 = time.perf_counter()
                if out_flow is not None and out_flow.shape == (th, tw, 2):
                    cv2.resize(src_flow, (tw, th), dst=out_flow, interpolation=cv2.INTER_LINEAR)
                    out_flow[..., 0] *= (tw / float(src_w))
                    out_flow[..., 1] *= (th / float(src_h))
                    adapted_flow = out_flow
                else:
                    flow_down = cv2.resize(src_flow, (tw, th), interpolation=cv2.INTER_LINEAR)
                    flow_down[..., 0] *= (tw / float(src_w))
                    flow_down[..., 1] *= (th / float(src_h))
                    adapted_flow = flow_down
                timing_dict["guidance_adapt_ms"] = (time.perf_counter() - t_adapt_0) * 1000.0
                self._adapter_cache[cache_key] = adapted_flow

        timing_dict["nvof_input_width"] = src_w
        timing_dict["nvof_input_height"] = src_h
        timing_dict["nvof_output_width"] = tw
        timing_dict["nvof_output_height"] = th
        # Estimated from the format, never measured: the measured figure is the bridge's own
        # h2d_bytes counter, which this row cannot see. Bytes per pixel follows the input the
        # engine says it handed the bridge, so a BGR frame is not charged for an alpha channel
        # that never crossed PCIe.
        bytes_per_pixel = 3 if timing_dict.get("nvof_input_format") == "bgr24_host" else 4
        timing_dict["estimated_nvof_h2d_bytes"] = src_w * src_h * bytes_per_pixel * (2 if is_discontinuous else 1)
        timing_dict["estimated_nvof_d2h_bytes"] = src_w * src_h * 8

        return adapted_flow, timing_dict

    def get_guidance_device(self, frame_id: int, cur_bgr: np.ndarray,
                            target_w: int, target_h: int,
                            is_discontinuous: bool = False) -> tuple:
        """Return borrowed NVOF CUDA flow for an exact working-resolution consumer.

        No resize or CPU fallback is possible here. The consumer must finish with
        the descriptor before the next NVOF execute invalidates its contents.
        """
        if self.engine_type not in ("nvof", "nvidia") or self._flow_engine is None:
            raise RuntimeError("device guidance requires an available NVOF engine")
        height, width = cur_bgr.shape[:2]
        if (width, height) != (target_w, target_h):
            raise ValueError("device guidance requires exact working dimensions")
        if self._last_frame_id == frame_id and not is_discontinuous:
            resource = self._last_src_flow
        else:
            discontinuity = (is_discontinuous or self._last_frame_id is None or
                             frame_id != self._last_frame_id + 1)
            resource = self._flow_engine.compute_flow_bgr24_device(
                None, cur_bgr, reset_temporal_hints=discontinuity)
            self._last_frame_id = frame_id
            self._last_src_flow = resource
        timing = dict(self._flow_engine.get_last_timing())
        timing.update(self._flow_engine.get_counters_delta())
        timing.update({
            "guidance_backend": "working_resolution_nvof_device_d3d12",
            "nvof_native_history": True,
            "cpu_previous_frame_copy_bytes": 0,
            "nvof_input_width": width,
            "nvof_input_height": height,
            "nvof_output_width": width,
            "nvof_output_height": height,
            "estimated_nvof_d2h_bytes": 0,
        })
        return resource, timing

    def supports_device_rgba_input(self) -> bool:
        """Whether flow can be fed a surface that never left the GPU."""
        return bool(getattr(self._flow_engine, "supports_rgba_device_nvof_input",
                             False))

    def supports_device_bgr_flow(self) -> bool:
        """Whether ``get_guidance_device`` can answer, flow included.

        The entry hands the caller a CUDA descriptor for a frame read from host
        memory, which is something only an NVOF bridge does: a DIS or RAFT engine
        keeps its vectors where it computed them and returns an array. A caller
        that routes by this answer rather than by the wrapper class's method table
        can tell a plane that has to be uploaded from a descriptor that does not.
        """
        return (self.engine_type in ("nvof", "nvidia")
                and callable(getattr(self._flow_engine, "compute_flow_bgr24_device",
                                     None)))

    def device_input_counters(self) -> tuple:
        """Cumulative device-to-device NVOF input copies, or (None, None).

        None because an engine that was never asked has nothing to report; a
        zero from a session that ran the device entry point is a measurement.
        """
        getter = getattr(self._flow_engine, "device_input_counters", None)
        return getter() if callable(getter) else (None, None)

    def get_guidance_device_rgba(self, frame_id: int, cuda_context, color: dict,
                                 is_discontinuous: bool = False) -> tuple:
        """NVOF flow for a working frame that never left the GPU.

        ``color`` is the RGBA8 CUDA descriptor another bridge produced. There is
        no host copy on this path in either direction, so a caller that needs
        the frame to have stayed resident negotiates against it.
        """
        if self.engine_type not in ("nvof", "nvidia") or self._flow_engine is None:
            raise RuntimeError("device guidance requires an available NVOF engine")
        if not getattr(self._flow_engine, "supports_rgba_device_nvof_input", False):
            raise RuntimeError("NVOF device RGBA input is unavailable")
        if not isinstance(color, dict) or color.get("format") != "rgba8_cuda":
            raise ValueError("color must be an rgba8_cuda descriptor")
        width, height = int(color.get("width", 0)), int(color.get("height", 0))
        if not width or not height:
            raise ValueError("color descriptor must carry its extent")
        if self._last_frame_id == frame_id and not is_discontinuous:
            resource = self._last_src_flow
        else:
            discontinuity = (is_discontinuous or self._last_frame_id is None or
                             frame_id != self._last_frame_id + 1)
            resource = self._flow_engine.compute_flow_rgba_device(
                cuda_context or color.get("cuda_context"),
                int(color["device_ptr"]), int(color["pitch_bytes"]),
                width, height, reset_temporal_hints=discontinuity)
            self._last_frame_id = frame_id
            self._last_src_flow = resource
        timing = dict(self._flow_engine.get_last_timing())
        timing.update(self._flow_engine.get_counters_delta())
        copies, payload = self._flow_engine.device_input_counters()
        timing.update({
            "guidance_backend": "working_resolution_nvof_device_rgba",
            "nvof_native_history": True,
            "cpu_previous_frame_copy_bytes": 0,
            "nvof_input_width": width,
            "nvof_input_height": height,
            "nvof_output_width": width,
            "nvof_output_height": height,
            "estimated_nvof_d2h_bytes": 0,
            "estimated_nvof_h2d_bytes": 0,
            # Cumulative for the session: the per-frame figure is the difference
            # between two of these, which is what makes a zero a measurement.
            "nvof_device_input_copy_count_total": copies,
            "nvof_device_input_bytes_total": payload,
        })
        return resource, timing

    def close(self):
        self.reset()
        if self._flow_engine and hasattr(self._flow_engine, "close"):
            self._flow_engine.close()
            self._flow_engine = None


_GLOBAL_GUIDANCE_SERVICES = {}

def get_frame_guidance_service(engine_type="nvof", quality="4M", grid=None, preset=None,
                               raft_analysis_edge=None, raft_iterations=None) -> FrameGuidanceService:
    key = (str(engine_type).lower(), str(quality), str(grid), str(preset))
    kwargs = {}
    if key[0] in ('raft', 'raft_large'):
        key += (raft_analysis_edge, raft_iterations)
        kwargs = dict(raft_analysis_edge=raft_analysis_edge, raft_iterations=raft_iterations)
        for old in list(_GLOBAL_GUIDANCE_SERVICES):
            if old != key and old[0] in ('raft', 'raft_large'):
                del _GLOBAL_GUIDANCE_SERVICES[old]
    if key not in _GLOBAL_GUIDANCE_SERVICES:
        _GLOBAL_GUIDANCE_SERVICES[key] = FrameGuidanceService(engine_type, quality, grid, preset, **kwargs)
    return _GLOBAL_GUIDANCE_SERVICES[key]
