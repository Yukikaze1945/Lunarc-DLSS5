#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — optical-flow + DLSS + video export engine for the
standalone DLSS tool. Flow may be cached to disk for repeat runs.

Output layout, given a video at <video_dir>/<name>.mp4:
    <video_dir>/<name>_flow /<name>.flow /000000.flo  ...  Middlebury float32 HxWx2
    <video_dir>/<name>_dlss /000000.png ...               DLSS5 Feature18 output (needs SDK)

CLI:  python3 pipeline.py <video> [--flow] [--export] [--frames N]
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import queue
from native_watchdog import phase as native_phase
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

import bundle_paths
import transfer_profiler

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


# ---- external files: what a user supplies or replaces beside the program ----
# The launcher directory, not a bundle's contents directory: an install has to be
# givable an external torch tree. The DLLs the program loads itself are not here -
# dlss_gpu_engine resolves those, against the contents directory a PyInstaller bundle
# actually collected them into.
BASE = bundle_paths.app_dir()
os.environ.setdefault("TORCH_HOME", os.path.join(BASE, "torch_home"))
RAFT_PTH = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "raft_large_C_T_SKHT_V2-ff5fadd5.pth")

DEVICE = "cpu"                 # placeholder; resolved lazily when torch is present
_TORCH_OK = None

def _torch_available():
    """Return True if torch is importable on this machine (the exe may ship without it)."""
    global _TORCH_OK, DEVICE
    if _TORCH_OK is None:
        try:
            import torch
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _TORCH_OK = True
        except Exception:
            _TORCH_OK = False
    return _TORCH_OK

def _require_torch(what):
    if not _torch_available():
        raise RuntimeError(f"{what} 需要 RAFT 光流组件，请在软件设置中安装后重启软件")

from contextlib import contextmanager

@contextmanager
def amp():
    """fp16 mixed-precision autocast on GPU; no-op on CPU / when torch is absent."""
    if DEVICE == "cuda" and _torch_available():
        import torch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


# ----------------------------- helpers -----------------------------
def _bundle_tool(name):
    """Locate one FFmpeg command line tool: collected in the bundle when frozen, else
    from scoop / a standalone install / PATH / conda.

    The frozen branch is the contents directory, which is where a spec destination of
    '.' collects to - not the folder the launcher sits in. Both tools used to be walked
    by their own copy of this ladder, so an answer that had to be fixed in one place
    was two.
    """
    local_tool = os.path.join(os.environ.get('LUNARC_RUNTIME_DIR') or os.path.join(BASE, 'runtime'), 'tools', name)
    if os.path.isfile(local_tool):
        return local_tool
    if getattr(sys, 'frozen', False) and os.path.isfile(bundle_paths.collected(name)):
        return bundle_paths.collected(name)
    # A published standalone package must not silently borrow a different FFmpeg
    # from PATH (which may be a shim or lack the required filters/import DLLs).
    if os.environ.get('LUNARC_STANDALONE') == '1':
        return None
    user_home = os.path.expanduser("~")
    candidates = [
        os.path.join(user_home, "scoop", "shims", name),
        os.path.join(r"C:\ffmpeg-8.0-full_build\bin", name),
        os.path.join(r"C:\ffmpeg\bin", name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    p = shutil.which(name)
    if p:
        return p
    conda_p = os.path.join(sys.prefix, 'Library', 'bin', name)
    if os.path.exists(conda_p):
        return conda_p
    return None


def _bundle_ffmpeg():
    """Where ffmpeg.exe is, or None; the callers that need it say so."""
    return _bundle_tool('ffmpeg.exe')


def _bundle_ffprobe():
    """Where ffprobe.exe is, or None; the callers that need it say so."""
    return _bundle_tool('ffprobe.exe')


def _ffprobe_exact_packets(path, timeout=30):
    """Return exact video packet count via ffprobe -count_packets (slow but precise).
    Used when nb_frames is missing and dur*fps rounding would over-estimate
    (e.g. 5.023s*120=603 vs actual 600 decodable frames). Returns None on failure."""
    ffprobe = _bundle_ffprobe()
    if not ffprobe:
        return None
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets",
            "-count_packets",
            "-of", "json", path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=timeout)
        data = json.loads(res.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        nb = streams[0].get("nb_read_packets")
        if nb is not None and int(nb) > 0:
            return int(nb)
        return None
    except Exception:
        return None


def probe_pixel_info(path):
    """Return input bit-depth / color metadata (P1-01 honesty probe).

    Probe source precision independently of preview and export choice. The
    legacy pipe is 8-bit; GPU Transfer has separate SDR8, SDR10 BT.709 and HDR10
    domains. Unknown values remain unknown rather than becoming HDR.
    """
    info = {"pix_fmt": "", "bit_depth": 8, "color_space": "",
            "color_transfer": "", "color_primaries": "", "color_range": "", "is_hdr": False}
    ffprobe = _bundle_ffprobe()
    if not ffprobe:
        return info
    try:
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries",
                 "stream=codec_name,profile,pix_fmt,bits_per_raw_sample,color_range,color_space,color_transfer,color_primaries,width,height:stream_side_data",
               "-of", "json", path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        s = (json.loads(res.stdout).get("streams") or [{}])[0]
        pix_fmt = str(s.get("pix_fmt") or "")
        info["pix_fmt"] = pix_fmt
        info['codec_name'] = s.get('codec_name', '')
        info['profile'] = s.get('profile', '')
        depth = 8
        try:
            raw_bits = int(s.get("bits_per_raw_sample") or 0)
            if raw_bits > 0:
                depth = raw_bits
        except Exception:
            pass
        if depth == 8:
            m = re.search(r"p0?(\d+)|yuv4(?:44|22|20)p(\d+)|rgb(\d+)", pix_fmt)
            if m:
                try:
                    depth = max(int(g) for g in m.groups() if g)
                except Exception:
                    pass
        info["bit_depth"] = depth if depth in (8, 10, 12, 16) else 8
        info["color_space"] = str(s.get("color_space") or "")
        info["color_transfer"] = str(s.get("color_transfer") or "")
        info["color_primaries"] = str(s.get("color_primaries") or "")
        info["color_range"] = str(s.get("color_range") or "")
        trc = info["color_transfer"].lower()
        # Bit depth and BT.2020 do not establish HDR: 10/12-bit SDR is common.
        # Identify the source transfer independently from export opt-in settings.
        info["is_hdr"] = trc in ("smpte2084", "smpte-st-2084", "arib-std-b67")
        info.update(width=int(s.get('width') or 0), height=int(s.get('height') or 0),
                    side_data_list=list(s.get('side_data_list') or []))
        from source_colour import resolve_source_colour
        return resolve_source_colour(info)
    except Exception:
        return info


def _ffprobe_info(path):
    ffprobe = _bundle_ffprobe()
    if not ffprobe:
        return None
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name:format=duration",
            "-of", "json", path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        data = json.loads(res.stdout)
        if not data.get("streams"):
            return None
        s = data["streams"][0]
        w = int(s.get("width", 0))
        h = int(s.get("height", 0))
        fps_str = s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1"
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)
        nb = s.get("nb_frames")
        if nb and int(nb) > 0:
            n = int(nb)
        else:
            # nb_frames missing (common for MKV/H264): prefer exact packet count
            # over dur*fps rounding which over-estimates (e.g. 603 vs 600).
            exact = _ffprobe_exact_packets(path)
            if exact is not None and exact > 0:
                n = exact
            else:
                dur = float(s.get("duration") or data.get("format", {}).get("duration") or 0)
                n = int(round(dur * fps))
        codec = s.get("codec_name", "")
        return n, fps, w, h, codec
    except Exception:
        return None


_VIDEO_INFO_CACHE = {}

def _video_info_read(abs_p) -> tuple:
    """The two probes, in order: (the four figures, whether they may be remembered).

    An answer from ffprobe is worth caching; one scraped off OpenCV with no size behind
    it is not, because it says the file could not be read rather than what it holds.
    """
    probe = _ffprobe_info(abs_p)
    if probe is not None:
        n, fps, w, h, _ = probe
        if n > 0 and w > 0 and h > 0:
            return (n, fps, w, h), True
    cap = cv2.VideoCapture(abs_p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (n, fps, w, h), (w > 0 and h > 0)


def video_info_uncached(path):
    """(n, fps, w, h) read from the bytes on disk right now, with nothing remembered.

    video_info keeps its answer for the whole process because a source that is read
    frame by frame never changes underneath it. A finished export is exactly the file
    that did change - it was written after anything earlier probed that name - so the
    integrity check asks this instead and gets the count the encoder left behind.
    """
    return _video_info_read(os.path.abspath(path))[0]


def video_info(path):
    """Returns (n, fps, w, h). Robust to AV1 and modern video codecs with task-level caching."""
    abs_p = os.path.abspath(path)
    if abs_p in _VIDEO_INFO_CACHE:
        return _VIDEO_INFO_CACHE[abs_p]
    res, cacheable = _video_info_read(abs_p)
    if cacheable:
        _VIDEO_INFO_CACHE[abs_p] = res
    return res


def resolve_working_dims(src_w: int, src_h: int, scale: float = 1.0) -> tuple:
    """Calculate 8-aligned working dimensions for any continuous scale between 0.25 and 1.0 (Magpie architecture)."""
    s = float(np.clip(scale, 0.25, 1.0))
    if abs(s - 1.0) < 1e-4:
        return _even(src_w), _even(src_h)
    ww = max(16, int(round(src_w * s / 8.0) * 8))
    wh = max(16, int(round(src_h * s / 8.0) * 8))
    return ww, wh


class SmartCapture:
    """Unified video reader supporting OpenCV native, PyAV (streaming + fast seek), and FFmpeg CLI pipe.
    Provides identical interface to cv2.VideoCapture (get, set, read, isOpened, release).
    Thread-safe and optimized for 4K video scrubbing.
    """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        self._cap = None
        self._mode = "cv2"
        self._ffmpeg = _bundle_ffmpeg()
        self._ffprobe = _bundle_ffprobe()

        probe = _ffprobe_info(self.path)
        if probe is not None:
            self.n, self.fps, self.w, self.h, self.codec = probe
        else:
            self.n, self.fps, self.w, self.h, self.codec = 0, 30.0, 0, 0, ""

        self._pos = 0
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._max_cache_bytes = 256 * 1024 * 1024  # 256 MB RAM budget for 4K scrubbing

        self._av_container = None
        self._av_stream = None
        self._packet_iter = None
        self._frame_iter = None
        self._last_idx = None

        is_av1 = any(x in (self.codec or "").lower() for x in ["av1", "av01", "dav1d"]) or ("av1" in os.path.basename(self.path).lower())
        if not is_av1:
            try:
                cap = cv2.VideoCapture(self.path)
                if cap.isOpened():
                    ok, test_f = cap.read()
                    if ok and test_f is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._cap = cap
                        # Bug2 fix: keep the exact ffprobe count (e.g. 600);
                        # cv2 CAP_PROP_FRAME_COUNT over-estimates MKV (603) and
                        # must only fill gaps, never overwrite a good probe.
                        if self.w == 0 or self.h == 0:
                            self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if not self.fps or self.fps <= 0:
                            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        if not self.n or self.n <= 0:
                            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        self._mode = "cv2"
                        return
                cap.release()
            except Exception:
                pass

        self._mode = "ffmpeg"
        try:
            import av
            self._av_container = av.open(self.path)
            self._av_stream = self._av_container.streams.video[0]
            self._av_stream.thread_type = "AUTO"
            try:
                self._av_stream.codec_context.thread_count = max(4, min(16, (os.cpu_count() or 4)))
            except Exception:
                pass
            if self.w == 0:
                self.w = self._av_stream.codec_context.width
                self.h = self._av_stream.codec_context.height
                self.fps = float(self._av_stream.average_rate or 30.0)
                self.n = self._av_stream.frames or self.n
            self._mode = "av"
            self._reset_av_iter()
        except Exception:
            self._av_container = None
            self._av_stream = None
            self._mode = "ffmpeg"

    def _reset_av_iter(self):
        if self._av_container and self._av_stream:
            self._packet_iter = self._av_container.demux(self._av_stream)
            self._frame_iter = None

    def _next_av_frame(self):
        if self._packet_iter is None:
            self._reset_av_iter()
        while True:
            if self._frame_iter is not None:
                try:
                    return next(self._frame_iter)
                except StopIteration:
                    self._frame_iter = None
            try:
                packet = next(self._packet_iter)
                self._frame_iter = iter(packet.decode())
            except StopIteration:
                return None

    def _get_av_frame_idx(self, frame):
        if frame.pts is not None:
            c_sec = float(frame.pts * self._av_stream.time_base)
            return int(round(c_sec * self.fps))
        return (self._last_idx + 1) if self._last_idx is not None else 0

    def isOpened(self):
        with self._lock:
            if self._mode == "cv2":
                return self._cap is not None and self._cap.isOpened()
            return self.w > 0 and self.h > 0

    def get(self, prop_id):
        with self._lock:
            # Bug2 fix: FRAME_COUNT must return the exact ffprobe value (self.n),
            # never cv2's container estimate (603 vs 600). Same for FPS/dims when
            # the probe already supplied them.
            if prop_id == cv2.CAP_PROP_FRAME_COUNT and self.n > 0:
                return float(self.n)
            if self._mode == "cv2" and self._cap:
                return self._cap.get(prop_id)
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return float(self.n)
            elif prop_id == cv2.CAP_PROP_FPS:
                return float(self.fps)
            elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.w)
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.h)
            elif prop_id == cv2.CAP_PROP_POS_FRAMES:
                return float(self._pos)
            return 0.0

    def set(self, prop_id, value):
        with self._lock:
            if self._mode == "cv2" and self._cap:
                return self._cap.set(prop_id, value)
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._pos = max(0, int(value))
                return True
            return False

    def read(self):
        with self._lock:
            if self._mode == "cv2" and self._cap:
                ok, f = self._cap.read()
                if ok:
                    return True, f

            cur_idx = self._pos
            if cur_idx in self._cache:
                self._pos += 1
                return True, self._cache[cur_idx]

            if self._mode == "av" and self._av_container is not None:
                img = self._read_frame_av(cur_idx)
                if img is not None:
                    self._store_cache(cur_idx, img)
                    self._pos = cur_idx + 1
                    return True, img

            img = self._read_frame_ffmpeg_cli(cur_idx)
            if img is not None:
                self._store_cache(cur_idx, img)
                self._pos = cur_idx + 1
                return True, img

            return False, None

    def _store_cache(self, idx, img):
        if img is None:
            return
        if idx in self._cache:
            return
        img_bytes = img.nbytes if hasattr(img, "nbytes") else (img.shape[0] * img.shape[1] * 3)
        while self._cache and (self._cache_bytes + img_bytes > self._max_cache_bytes):
            _, old_img = self._cache.popitem(last=False)
            self._cache_bytes -= (old_img.nbytes if hasattr(old_img, "nbytes") else (old_img.shape[0] * old_img.shape[1] * 3))
        self._cache[idx] = img
        self._cache_bytes += img_bytes

    def _read_frame_av(self, target_idx):
        try:
            if target_idx in self._cache:
                return self._cache[target_idx]

            # 1. Forward step without seek if target is close ahead (up to 240 frames / ~2 sec at 120fps)
            if self._last_idx is not None and 0 <= target_idx - self._last_idx <= 240:
                if target_idx == self._last_idx and target_idx in self._cache:
                    return self._cache[target_idx]
                while True:
                    f = self._next_av_frame()
                    if f is None:
                        break
                    idx = self._get_av_frame_idx(f)
                    self._last_idx = idx
                    if target_idx - idx <= 2:
                        bgr = f.to_ndarray(format="bgr24")
                        self._store_cache(idx, bgr)
                        if idx >= target_idx:
                            return bgr

            # 2. Seek backward to keyframe (only when jumping backward or leaping > 240 frames)
            target_sec = max(0.0, target_idx / max(1.0, self.fps))
            pts = int(round(target_sec / self._av_stream.time_base))
            try:
                self._av_container.seek(pts, any_frame=False, backward=True, stream=self._av_stream)
                if hasattr(self._av_stream.codec_context, "flush_buffers"):
                    self._av_stream.codec_context.flush_buffers()
            except Exception:
                pass
            self._reset_av_iter()

            # 3. Decode forward; defer to_ndarray conversion to intermediate target frames only
            found = None
            while True:
                f = self._next_av_frame()
                if f is None:
                    break
                idx = self._get_av_frame_idx(f)
                self._last_idx = idx
                if target_idx - idx <= 2:
                    bgr = f.to_ndarray(format="bgr24")
                    self._store_cache(idx, bgr)
                    if idx >= target_idx:
                        found = bgr
                        break
            return found
        except Exception:
            return None

    def _read_frame_ffmpeg_cli(self, target_idx):
        if not self._ffmpeg:
            return None
        t = max(0.0, target_idx / max(1.0, self.fps))
        cmd = [
            self._ffmpeg, "-v", "error",
            "-hwaccel", "auto",
            "-ss", f"{t:.4f}",
            "-i", self.path,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
            expected = self.w * self.h * 3
            if len(res.stdout) >= expected:
                return np.frombuffer(res.stdout[:expected], dtype=np.uint8).reshape((self.h, self.w, 3))
        except Exception:
            pass
        return None

    def release(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self._av_container is not None:
                try:
                    self._av_container.close()
                except Exception:
                    pass
                self._av_container = None
            self._packet_iter = None
            self._frame_iter = None
            self._cache.clear()


def open_capture(path):
    """Open video or one decoded still picture through the capture interface."""
    from image_source import is_image, Capture
    if is_image(path):
        return Capture(path)
    return SmartCapture(path)


class JobCancelled(RuntimeError):
    """Raised when the user cancels a running job while it is blocked in IO."""
    pass


def _is_cancelled(stop_event=None, cancel=None):
    try:
        if stop_event is not None and stop_event.is_set():
            return True
    except Exception:
        pass
    try:
        if callable(cancel) and cancel():
            return True
    except Exception:
        pass
    return False


class _DecoderPipeReader:
    """One bounded background reader per FFmpeg stdout pipe.

    Blocking reads happen once on this dedicated reader thread. The bounded
    queue prevents decoded frames from accumulating when processing falls behind.
    """
    def __init__(self, stream):
        self.stream = stream
        self.fd = stream.fileno()
        self.items = queue.Queue(maxsize=2)
        self.requests = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.pending = b""
        self.eof = False
        self.thread = threading.Thread(target=self._run, name="FFmpegPipeReader", daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    remaining = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                while remaining and not self.stop_event.is_set():
                    target = min(262144, remaining)
                    data = bytearray()
                    while len(data) < target and not self.stop_event.is_set():
                        chunk = os.read(self.fd, target - len(data))
                        if not chunk:
                            if data:
                                self._put((data, None))
                            self._put((None, None))
                            return
                        data.extend(chunk)
                    if not self._put((data, None)):
                        return
                    remaining -= len(data)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._put((None, exc))

    def read(self, n, wait):
        if self.pending:
            if len(self.pending) <= n:
                chunk, self.pending = self.pending, b""
                return chunk
            chunk, self.pending = self.pending[:n], self.pending[n:]
            return chunk
        if self.eof:
            return b""
        try:
            data, error = self.items.get(timeout=wait)
        except queue.Empty:
            return None
        if error is not None:
            raise error
        if data is None:
            self.eof = True
            return b""
        if len(data) <= n:
            return data
        chunk, self.pending = data[:n], data[n:]
        return chunk

    def request(self, n):
        if not self.eof:
            self.requests.put_nowait(n)

    def close(self):
        self.stop_event.set()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=0.25)


def _decoder_pipe_reader(proc, stream):
    if proc is None:
        return None
    current = getattr(proc, "_lunarc_decoder_pipe_reader", None)
    if current is not None and current.stream is stream:
        return current
    try:
        current = _DecoderPipeReader(stream)
    except (AttributeError, OSError, ValueError):
        return None
    try:
        proc._lunarc_decoder_pipe_reader = current
    except Exception:
        current.close()
        return None
    return current


def _stop_decoder_pipe_reader(proc):
    reader = getattr(proc, "_lunarc_decoder_pipe_reader", None) if proc is not None else None
    if reader is not None:
        reader.close()
        proc._lunarc_decoder_pipe_reader = None


def _close_decoder_process(proc):
    # End its writer before joining the reader or closing a pipe it is reading.
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=1)
    except Exception:
        pass
    _stop_decoder_pipe_reader(proc)
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _cancel_or(error, stop_event=None, cancel=None):
    """Relabel an IO error that happened during a cancel as that cancellation.

    Killing a child closes its pipes, so EOF, a broken pipe and a non-zero exit code
    are all expected consequences of cancelling. Without the flag set the error is
    returned untouched: a real FFmpeg failure must never be reported as a user action.
    """
    if isinstance(error, JobCancelled) or not _is_cancelled(stop_event, cancel):
        return error
    return JobCancelled(f"用户已取消任务 ({error})")


def _write_frame_bytes(pipe_stdin, frame_ndarray, chunk_size=1048576, cancel_fn=None, proc=None, frame_idx=0) -> int:
    """Write frame bytes to an encoder pipe via 1D byte-casted memoryview chunking.

    Guarantees:
    - Zero intermediate buffer duplication (no .tobytes() full-frame copy).
    - True 1D byte indexing (memoryview.cast('B')), ensuring each chunk <= chunk_size bytes.
    - Responsive cancellation and BrokenPipe detection on each chunk boundary.
    - Returns total written byte count.
    """
    arr = np.ascontiguousarray(frame_ndarray)
    mv = memoryview(arr).cast("B")
    total_bytes = len(mv)
    off = 0
    step = max(4096, int(chunk_size))

    try:
        while off < total_bytes:
            if callable(cancel_fn) and cancel_fn():
                raise JobCancelled(f"用户已取消任务 (第 {frame_idx} 帧管道写入时中止)")
            if proc is not None and proc.poll() is not None:
                raise _cancel_or(RuntimeError(f"FFmpeg 编码进程在第 {frame_idx} 帧写入中退出 (退出码 {proc.returncode})"),
                                 cancel=cancel_fn)

            chunk_len = min(step, total_bytes - off)
            pipe_stdin.write(mv[off:off + chunk_len])
            off += chunk_len
    except JobCancelled:
        raise
    except (BrokenPipeError, OSError) as e:
        raise _cancel_or(RuntimeError(f"FFmpeg 编码管道在第 {frame_idx} 帧破裂 (BrokenPipe): {e}"),
                         cancel=cancel_fn)

    return off


def _read_exactly(stream, n, proc=None, stop_event=None, cancel=None, timeout=None, what="frame data"):
    """Read exactly n bytes without spawning a thread per frame while staying responsive to cancel.

    Returns short bytes only on graceful EOF (exit code 0 / untracked stream);
    an abnormal exit code raises. Raises JobCancelled on cancel.
    """
    deadline = time.monotonic() + float(timeout) if timeout is not None else None
    pipe_reader = _decoder_pipe_reader(proc, stream)
    if pipe_reader is not None:
        pipe_reader.request(n)
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        if _is_cancelled(stop_event, cancel):
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
                _stop_decoder_pipe_reader(proc)
            raise JobCancelled("用户已取消任务 (等待 %s 时中止)" % what)
        if deadline is not None and time.monotonic() >= deadline:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
                _stop_decoder_pipe_reader(proc)
            raise RuntimeError("读取 %s 超时 (%.1f 秒)，已终止 FFmpeg 解码进程" % (what, timeout))
        try:
            wait = 0.01 if deadline is None else min(0.01, max(0.0, deadline - time.monotonic()))
            chunk = (pipe_reader.read(min(262144, n - pos), wait) if pipe_reader is not None
                     else stream.read(min(262144, n - pos)))
        except Exception as e:
            raise _cancel_or(RuntimeError("读取 %s 失败: %s" % (what, e)), stop_event, cancel)
        if chunk is None:
            continue
        if not chunk:
            if proc is None:
                break
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.005)
            continue
        view[pos:pos + len(chunk)] = chunk
        pos += len(chunk)

    if pos < n:
        if proc is not None:
            rc = proc.poll()
            if rc not in (None, 0):
                raise _cancel_or(RuntimeError("FFmpeg 解码进程异常退出 (等待 %s，退出码 %s)" % (what, rc)),
                                 stop_event, cancel)
        return bytes(view[:pos])
    return bytes(buf)


def _popen_read_first(proc, nbytes, timeout=25):
    """Read the first probe packet with a bounded, cancellable pipe read.

    Prevents infinite block when the CUDA filter graph never produces output.
    Returns b"" on timeout/failure (caller falls back or raises).
    """
    try:
        return _read_exactly(proc.stdout, nbytes, proc=proc, timeout=timeout,
                             what="CUDA 首帧探测")
    except RuntimeError:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        return b""


_CUDA_DECODE_CACHE = {}


def probe_cuda_decode_support(ffmpeg=None) -> tuple:
    """Static capability probe for FFmpeg CUDA hardware acceleration and scale_cuda filter.
    Returns (is_supported: bool, reason: str).
    """
    return probe_cuda_decode_capability(ffmpeg)


def probe_cuda_decode_for_input(video_path: str, ffmpeg=None) -> tuple:
    """Dynamic input-specific probe: attempts to decode 1 frame with CUDA hwaccel.
    Returns (is_supported: bool, reason: str, info_dict: dict).
    """
    ffmpeg = ffmpeg or _bundle_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found", {"decode_actual": "cpu_fallback"}

    is_static_ok, static_reason = probe_cuda_decode_support(ffmpeg)
    if not is_static_ok:
        return False, static_reason, {"decode_actual": "cpu_fallback"}

    proc = None
    try:
        n, fps, w, h = video_info(video_path)
        if w <= 0 or h <= 0:
            return False, "Invalid video resolution", {"decode_actual": "cpu_fallback"}

        expected_bytes = w * h * 3
        cmd = [
            ffmpeg, "-v", "error",
            "-hwaccel", "cuda",
            "-i", os.path.abspath(video_path),
            "-vframes", "1",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        raw = _popen_read_first(proc, expected_bytes, timeout=10)
        proc.poll()
        if proc.returncode not in (None, 0) or len(raw) < expected_bytes:
            # stderr.read() waits for EOF too: stop the writer before reading it.
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1)
            err = proc.stderr.read().decode('utf-8', 'replace') if proc.stderr else "Incomplete frame read"
            return False, f"CUDA decode failed for input: {err}", {"decode_actual": "cpu_fallback"}

        return True, "CUDA hardware decode verified", {
            "decode_actual": "cuda", "width": w, "height": h, "fps": fps
        }
    except Exception as e:
        return False, f"Dynamic probe error: {e}", {"decode_actual": "cpu_fallback"}
    finally:
        if proc is not None:
            _close_decoder_process(proc)


def probe_cuda_decode_capability(ffmpeg=None) -> tuple:
    """Probe whether FFmpeg supports CUDA hardware decoding (hwaccel cuda + scale_cuda filter).
    Returns (is_supported: bool, reason: str).
    """
    ffmpeg = ffmpeg or _bundle_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg executable not found"

    if ffmpeg in _CUDA_DECODE_CACHE:
        return _CUDA_DECODE_CACHE[ffmpeg]

    try:
        res = subprocess.run([ffmpeg, "-hwaccels"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        hwaccels = res.stdout.lower() if res.stdout else ""
        if "cuda" not in hwaccels:
            _CUDA_DECODE_CACHE[ffmpeg] = (False, "FFmpeg does not support 'cuda' hwaccel")
            return _CUDA_DECODE_CACHE[ffmpeg]

        res_filter = subprocess.run([ffmpeg, "-filters"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        filters = res_filter.stdout.lower() if res_filter.stdout else ""
        if "scale_cuda" not in filters:
            _CUDA_DECODE_CACHE[ffmpeg] = (False, "FFmpeg lacks 'scale_cuda' filter")
            return _CUDA_DECODE_CACHE[ffmpeg]

        _CUDA_DECODE_CACHE[ffmpeg] = (True, "CUDA hwaccel and scale_cuda available")
        return _CUDA_DECODE_CACHE[ffmpeg]
    except Exception as e:
        _CUDA_DECODE_CACHE[ffmpeg] = (False, f"Probe error: {e}")
        return _CUDA_DECODE_CACHE[ffmpeg]


def iter_frames(path, limit=None, decode_mode="auto", stop_event=None, cancel=None,
                on_proc_started=None, on_proc_exited=None, decode_info=None):
    """Yields (i, frame_bgr) sequentially.
    decode_mode:
      - 'cpu': Explicitly use CPU decode (OpenCV or FFmpeg CPU).
      - 'cuda': Strictly require CUDA hardware decode. Never silently fallback to CPU.
      - 'auto': Attempt CUDA first; if CUDA fails, allow fallback to CPU.
    decode_info: Optional mutable dict to receive {'requested': str, 'actual': str, 'probe_reason': str}.
    """
    path = os.path.abspath(path)
    decode_mode = str(decode_mode or "auto").lower().strip()
    if decode_mode not in ("cpu", "cuda", "auto"):
        decode_mode = "auto"

    ffmpeg = _bundle_ffmpeg()
    cuda_supported, probe_reason = probe_cuda_decode_capability(ffmpeg)

    if isinstance(decode_info, dict):
        decode_info["requested"] = decode_mode
        decode_info["actual"] = "unknown"
        decode_info["probe_reason"] = probe_reason

    n, fps, w, h = video_info(path)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"无法获取视频分辨率: {path}")

    frame_bytes_bgr = w * h * 3
    frame_bytes_nv12 = w * h * 3 // 2

    # 1. decode_mode == 'cpu': Strictly use managed FFmpeg CPU software decode child process
    if decode_mode == "cpu":
        if not ffmpeg:
            raise RuntimeError(f"无法解码视频 {path}，且未找到 ffmpeg")
        if isinstance(decode_info, dict):
            decode_info["actual"] = "ffmpeg_cpu"
        cmd_cpu = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        proc = subprocess.Popen(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if callable(on_proc_started):
            try: on_proc_started(proc)
            except Exception: pass
        i = 0
        try:
            while True:
                if _is_cancelled(stop_event, cancel):
                    raise JobCancelled("用户已取消任务 (解码中)")
                if limit is not None and i >= limit:
                    break
                raw = _read_exactly(proc.stdout, frame_bytes_bgr, proc=proc, stop_event=stop_event, cancel=cancel, timeout=90, what="CPU BGR 帧")
                if len(raw) < frame_bytes_bgr:
                    break
                transfer_profiler.DECODER_PIPE.note_frame(i, len(raw))
                f = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                yield i, f
                i += 1
        finally:
            _close_decoder_process(proc)
            if callable(on_proc_exited):
                try: on_proc_exited(proc)
                except Exception: pass
        return

    # 2. decode_mode in ('cuda', 'auto'): Try CUDA Hardware Decode first
    if not ffmpeg:
        raise RuntimeError(f"无法解码视频 {path}，且未找到 ffmpeg")

    force_cuda = (decode_mode == "cuda")
    if force_cuda and not cuda_supported:
        if isinstance(decode_info, dict):
            decode_info["actual"] = "failed"
        raise RuntimeError(f"强制 CUDA 硬件解码失败 (decode_mode=cuda): {probe_reason}")

    proc = None
    is_nv12 = False
    first_raw = None

    if cuda_supported:
        try:
            cmd_nv12 = [
                ffmpeg, "-y", "-loglevel", "error",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", path,
                "-vf", "scale_cuda=format=nv12,hwdownload,format=nv12",
                "-f", "rawvideo",
                "-pix_fmt", "nv12",
                "-"
            ]
            test_proc = subprocess.Popen(cmd_nv12, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            if callable(on_proc_started):
                try: on_proc_started(test_proc)
                except Exception: pass
            test_raw = _popen_read_first(test_proc, frame_bytes_nv12, timeout=25)
            if len(test_raw) == frame_bytes_nv12:
                proc = test_proc
                is_nv12 = True
                first_raw = test_raw
                if isinstance(decode_info, dict):
                    decode_info["actual"] = "cuda"
            else:
                _close_decoder_process(test_proc)
                if callable(on_proc_exited):
                    try: on_proc_exited(test_proc)
                    except Exception: pass
                if force_cuda:
                    if isinstance(decode_info, dict):
                        decode_info["actual"] = "failed"
                    raise RuntimeError(
                        "强制 CUDA 硬件解码失败 (decode_mode=cuda)：25s 内未收到首帧 "
                        "(scale_cuda 初始化挂起或该格式/硬件不支持 NVDEC)，已中止以免无限卡 0 帧")
        except JobCancelled:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            if force_cuda:
                if isinstance(decode_info, dict):
                    decode_info["actual"] = "failed"
                raise RuntimeError(f"强制 CUDA 硬件解码失败 (decode_mode=cuda): {e}")

    if proc is None and force_cuda:
        if isinstance(decode_info, dict):
            decode_info["actual"] = "failed"
        raise RuntimeError(f"强制 CUDA 硬件解码失败 (decode_mode=cuda)，系统不支持 NVDEC 或 scale_cuda 管道不可用: {path}")

    # 3. decode_mode == 'auto' and CUDA failed: Fallback to CPU
    if proc is None:
        if isinstance(decode_info, dict):
            decode_info["actual"] = "cpu_fallback"

        # Use managed FFmpeg software decoder child process (NOT OpenCV)
        if not ffmpeg:
            raise RuntimeError(f"无法解码视频 {path}，且未找到 ffmpeg")
        
        cmd_cpu = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        proc = subprocess.Popen(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if callable(on_proc_started):
            try: on_proc_started(proc)
            except Exception: pass
        is_nv12 = False

    i = 0
    try:
        while True:
            if _is_cancelled(stop_event, cancel):
                raise JobCancelled("用户已取消任务 (解码中)")
            if limit is not None and i >= limit:
                break
            if first_raw is not None:
                raw = first_raw
                first_raw = None
            else:
                read_size = frame_bytes_nv12 if is_nv12 else frame_bytes_bgr
                raw = _read_exactly(proc.stdout, read_size, proc=proc,
                                    stop_event=stop_event, cancel=cancel,
                                    timeout=90, what="解码第 %d 帧" % i)
                if len(raw) < read_size:
                    break

            if is_nv12:
                transfer_profiler.DECODER_PIPE.note_frame(i, len(raw))
                mat = np.frombuffer(raw, dtype=np.uint8).reshape((h * 3 // 2, w))
                frame = transfer_profiler.convert_color(
                    mat, cv2.COLOR_YUV2BGR_NV12, site="decode")
            else:
                transfer_profiler.DECODER_PIPE.note_frame(i, len(raw))
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            yield i, frame
            i += 1
    finally:
        _close_decoder_process(proc)
        if callable(on_proc_exited):
            try:
                on_proc_exited(proc)
            except Exception:
                pass


def write_flo(path, flow):
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(b"PIEH")
        f.write(np.array([w, h], dtype=np.int32).tobytes())
        f.write(flow.astype(np.float32).tobytes())


def read_flo(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"PIEH"
        w, h = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32).reshape(h, w, 2)


FLOW_ITERS = 6   # RAFT iterative refinement steps (default 12; 6 ~= 2x faster, fine for DLSS guidance)

def _round8(x):
    """Round to the nearest multiple of 8 (min 8) so RAFT accepts the dims."""
    return max(8, int(round(x / 8.0)) * 8)

def imread_u(path, flags=cv2.IMREAD_COLOR):
    """Unicode-safe image reader for Windows (supports non-ASCII paths)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_u(path, img, params=None):
    """Unicode-safe image writer for Windows (supports non-ASCII paths)."""
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img, params)
        if ok:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return False


class FlowProvider:
    """Motion guidance for one frame, delivered at the working extent."""
    name = "abstract"

    def get_flow(self, frame_idx: int, cur_bgr: np.ndarray, prev_bgr, width: int, height: int,
                 is_discontinuous: bool = False, out: np.ndarray = None,
                 input_format: str = "rgba8_host") -> np.ndarray:
        raise NotImplementedError

    def requires_rgba_input(self) -> bool:
        """Whether this provider needs an RGBA copy of the working frame as its own input.

        The reduced transfer order has nowhere to put such a copy - each stage is handed the
        frame as the pipeline already holds it - so a provider that needs one is offered
        legacy. True is the default so that a provider which has not answered the question
        is not handed the reduction on an assumption.
        """
        return True

    def timing(self) -> dict:
        return {}

    def get_flow_device_rgba(self, frame_idx, color, cuda_context,
                             is_discontinuous=False):
        """Flow for a working frame that is a CUDA surface rather than an array.

        Only an engine whose bridge reads device memory can answer, so the base
        says no instead of downloading the frame to satisfy the call and letting
        the caller believe it stayed resident.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no device-resident flow input path")

    def supports_device_rgba_flow(self) -> bool:
        return False

    def device_input_counters(self) -> tuple:
        return None, None


class ZeroFlowProvider(FlowProvider):
    """guidance_mode 0 and 2: DLSSNR runs without motion guidance."""
    name = "zero"

    def get_flow(self, frame_idx, cur_bgr, prev_bgr, width, height,
                 is_discontinuous=False, out=None, input_format="rgba8_host"):
        if out is not None:
            out.fill(0.0)
            return out
        return np.zeros((height, width, 2), dtype=np.float32)

    def requires_rgba_input(self) -> bool:
        # Reads no frame at all, so there is no input format of the working frame to convert.
        return False


class GuidanceServiceFlowProvider(FlowProvider):
    """Adapts an optical_flow frame guidance service (NVOF, DIS or RAFT).

    This adapter returns only the motion plane and timing metadata.
    """

    def __init__(self, service, guidance_mode: int = 1):
        self._service = service
        self._guidance_mode = int(guidance_mode)
        self._timing = {}
        self.name = f"service:{type(service).__name__}"

    def new_independent_layer(self):
        """Own NVOF engine and history for one experimental DLSS layer."""
        from optical_flow import FrameGuidanceService
        source = self._service
        if self._guidance_mode != 1 or source.engine_type not in ("nvof", "nvidia"):
            raise ValueError("independent layer requires a device NVOF provider")
        service = FrameGuidanceService(source.engine_type, source.flow_quality,
                                       source.grid, source.preset, dedicated=True)
        return GuidanceServiceFlowProvider(service, guidance_mode=1)

    def close_independent_layer(self):
        if getattr(self._service, "dedicated", False):
            self._service.close()

    def get_flow(self, frame_idx, cur_bgr, prev_bgr, width, height,
                 is_discontinuous=False, out=None, input_format="rgba8_host"):
        flow, timing = self._service.get_guidance(
            frame_id=frame_idx,
            cur_bgr=cur_bgr,
            prev_bgr=prev_bgr,
            target_w=width,
            target_h=height,
            guidance_mode=self._guidance_mode,
            is_discontinuous=is_discontinuous,
            out_flow=out,
            input_format=input_format,
        )
        self._timing = timing or {}
        return flow

    def get_flow_device(self, frame_idx, cur_bgr, width, height,
                        is_discontinuous=False):
        if self._guidance_mode != 1:
            raise ValueError("device flow requires guidance mode 1")
        resource, timing = self._service.get_guidance_device(
            frame_id=frame_idx, cur_bgr=cur_bgr,
            target_w=width, target_h=height,
            is_discontinuous=is_discontinuous)
        self._timing = timing
        return resource

    def get_flow_device_rgba(self, frame_idx, color, cuda_context,
                             is_discontinuous=False):
        """NVOF flow for a device-resident working frame.

        ``color`` is the RGBA8 CUDA descriptor the GPU export backend got out of
        the Lanczos stage; the returned flow descriptor is borrowed and dies at
        the next NVOF call, which is why nothing here is cached or buffered.
        """
        if self._guidance_mode != 1:
            raise ValueError("device RGBA flow requires guidance mode 1")
        resource, timing = self._service.get_guidance_device_rgba(
            frame_id=frame_idx, cuda_context=cuda_context, color=color,
            is_discontinuous=is_discontinuous)
        self._timing = timing or {}
        return resource

    def supports_device_rgba_flow(self) -> bool:
        """Whether the engine behind this provider can be fed device memory."""
        if self._guidance_mode != 1:
            return False
        checker = getattr(self._service, "supports_device_rgba_input", None)
        return bool(checker()) if callable(checker) else False

    def supports_device_flow(self) -> bool:
        """Whether the engine behind this provider can hand flow over on the card.

        The adapter's own method table proves nothing here: one class wraps every
        flow engine, so it offers ``get_flow_device`` for DIS exactly as it does for
        NVOF, and only the engine knows whether its vectors can stay resident. A
        service that will not answer is left with the method-table reading, which is
        what every provider that is not this adapter already gets.
        """
        if self._guidance_mode != 1:
            return False
        checker = getattr(self._service, "supports_device_bgr_flow", None)
        return bool(checker()) if callable(checker) else True

    def device_input_counters(self) -> tuple:
        getter = getattr(self._service, "device_input_counters", None)
        return getter() if callable(getter) else (None, None)

    def requires_rgba_input(self) -> bool:
        """Whether the service behind this provider has to be handed an RGBA copy.

        A guidance mode that never reaches the engine runs no engine at all, so its input
        format is not a thing; otherwise the answer is the service's own, which says yes only
        for an NVOF engine whose bridge has no BGR entry point.
        """
        if self._guidance_mode != 1:
            return False
        return bool(getattr(self._service, "nvof_rgba_input_only", True))

    def timing(self) -> dict:
        return self._timing


class CachedFloFlowProvider(FlowProvider):
    """Motion vectors precomputed by the offline RAFT pass: ``<video>_flow/000000.flo``.

    Uses the project's existing ``read_flo`` rather than restating the Middlebury
    format. A missing frame yields zeros and is counted so the caller can tell the
    user guidance degraded, instead of the pipeline silently looking guided.
    """

    def __init__(self, flow_dir: str):
        self.flow_dir = flow_dir or ""
        self.name = f"cached_flo:{self.flow_dir}" if self.flow_dir else "cached_flo:<none>"
        self.loaded_frame_count = 0
        self.missing_frame_count = 0
        self._timing = {"guidance_backend": "cached_flo"}

    def requires_rgba_input(self) -> bool:
        # The vectors come off disk; cur_bgr and prev_bgr are never read, so there is no
        # input format of the working frame for this provider to ask for.
        return False

    def has_flow(self, frame_idx: int) -> bool:
        return bool(self.flow_dir) and os.path.exists(os.path.join(self.flow_dir, f"{frame_idx:06d}.flo"))

    def get_flow(self, frame_idx, cur_bgr, prev_bgr, width, height,
                 is_discontinuous=False, out=None, input_format="rgba8_host"):
        path = os.path.join(self.flow_dir, f"{frame_idx:06d}.flo") if self.flow_dir else None
        flow = None
        if path and os.path.exists(path):
            try:
                flow = read_flo(path)
            except Exception:
                flow = None
        if flow is None:
            self.missing_frame_count += 1
            if out is not None:
                out.fill(0.0)
                return out
            return np.zeros((height, width, 2), dtype=np.float32)
        self.loaded_frame_count += 1
        if flow.shape[:2] != (height, width):
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        if out is not None:
            np.copyto(out, flow)
            return out
        return np.ascontiguousarray(flow, dtype=np.float32)

    def timing(self) -> dict:
        return self._timing


def colorize_flow(flow):
    mag = np.linalg.norm(flow, axis=2)
    ang = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag / (np.percentile(mag, 98) + 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def out_dirs(video):
    """Return (flow_dir, dlss_dir) for a given video path."""
    base = os.path.dirname(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    return (os.path.join(base, stem + "_flow"), os.path.join(base, stem + "_dlss"))


def _all_cached(dirpath, total, ext):
    for i in range(total):
        if not os.path.exists(os.path.join(dirpath, f"{i:06d}.{ext}")):
            return False
    return total > 0


# ----------------------------- flow -----------------------------
_flow_model = None
def get_flow_model():
    global _flow_model
    if _flow_model is None:
        _require_torch("光流生成")
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        w = Raft_Large_Weights.DEFAULT
        # Offline: construct the (identical) architecture then load the bundled
        # state dict directly, so torchvision never tries to download from the net.
        m = raft_large(weights=None).to(DEVICE).eval()
        m.load_state_dict(torch.load(RAFT_PTH, map_location="cpu"))
        _flow_model = (m, w.transforms())
    return _flow_model


def generate_flow(video, frame_limit=None, progress=None, cancel=None, force=False, edge=720):
    """Cache flow/*.flo for every frame. Frame 0 = zero flow.
    The SHORT edge is clamped to `edge` (720) regardless of aspect ratio; flow is computed
    downscaled and then upscaled back to the full frame size (values scaled by 1/scale).
    This bounds RAFT's O((W*H)^2) correlation volume so high-res input doesn't blow up VRAM.
    A video whose short edge is already <= `edge` is never upscaled."""
    flow_dir, _ = out_dirs(video)
    os.makedirs(flow_dir, exist_ok=True)
    import torch
    total = video_info(video)[0]
    n = frame_limit if frame_limit else total
    if not force and _all_cached(flow_dir, n, "flo"):
        if progress: progress(n, n, "cached")
        return n
    (model, transforms) = get_flow_model()

    def fed_dims(img):
        h, w = img.shape[:2]
        short_edge = min(h, w)
        scale = min(1.0, edge / short_edge) if edge else 1.0
        fw = _round8(min(w, max(int(w * scale), 8)))
        fh = _round8(min(h, max(int(h * scale), 8)))
        return fw, fh

    def infer(prev_bgr, cur_bgr):
        h, w = cur_bgr.shape[:2]
        fw, fh = fed_dims(cur_bgr)
        if (fw, fh) != (w, h):
            p1 = cv2.resize(prev_bgr, (fw, fh)); p2 = cv2.resize(cur_bgr, (fw, fh))
        else:
            p1, p2 = prev_bgr, cur_bgr
        img1 = torch.from_numpy(p1[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img2 = torch.from_numpy(p2[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img1, img2 = transforms(img1, img2)
        with torch.no_grad(), amp():
            fl = model(img1.to(DEVICE), img2.to(DEVICE), num_flow_updates=FLOW_ITERS)[-1]
        fl = fl[0].permute(1, 2, 0).cpu().float().numpy()   # (fh, fw, 2)
        if (fw, fh) != (w, h):
            fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR)
            fl[..., 0] *= w / fw                            # displacement in full-res px
            fl[..., 1] *= h / fh
        return -fl                                          # backward = DLSS motion vector

    prev = None
    for i, cur in iter_frames(video, frame_limit if frame_limit else None):
        if cancel and cancel():
            break
        if prev is not None:
            p = os.path.join(flow_dir, f"{i:06d}.flo")
            if not (os.path.exists(p) and not force):
                write_flo(p, infer(prev, cur))
            if progress: progress(i + 1, total, "ok")
        else:
            p = os.path.join(flow_dir, f"000000.flo")
            if not os.path.exists(p):
                write_flo(p, np.zeros((cur.shape[0], cur.shape[1], 2), dtype=np.float32))
            if progress: progress(1, total, "ok")
        prev = cur
    return total


# ----------------------------- export -----------------------------
_AV1_NVENC_SUPPORTED = None

def _supports_av1_nvenc(ffmpeg):
    global _AV1_NVENC_SUPPORTED
    if _AV1_NVENC_SUPPORTED is not None:
        return _AV1_NVENC_SUPPORTED
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-h", "encoder=av1_nvenc"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        _AV1_NVENC_SUPPORTED = (r.returncode == 0)
    except Exception:
        _AV1_NVENC_SUPPORTED = False
    return _AV1_NVENC_SUPPORTED


_TUNE_UHQ_SUPPORTED = {}

def _supports_tune_uhq(ffmpeg, codec="av1_nvenc"):
    if not ffmpeg or codec == "h264_nvenc":
        return False
    key = (ffmpeg, codec)
    if key in _TUNE_UHQ_SUPPORTED:
        return _TUNE_UHQ_SUPPORTED[key]
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-h", f"encoder={codec}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        _TUNE_UHQ_SUPPORTED[key] = (r.returncode == 0 and "uhq" in (r.stdout or ""))
    except Exception:
        _TUNE_UHQ_SUPPORTED[key] = False
    return _TUNE_UHQ_SUPPORTED[key]


def export_video(video, kind, frames=None, fps=30.0, crf=24, with_audio=True, progress=None, cancel=None):
    """Export a preview/output video. kind in {flow, dlss}. Returns mp4 path.

    Uses ffmpeg with NVIDIA NVENC AV1 hardware encoding (-c:v av1_nvenc -preset p7 -tune hq
    -multipass qres -rc:v vbr -cq:v 24 -b:v 0 -g {round(fps*2)}) and direct audio streamcopy
    (-c:a copy).
    Falls back to libx264 or OpenCV mp4v when NVENC / ffmpeg is unavailable or frame dimensions
    are odd. crf/cq: 0-63, lower = higher quality, default 24.
    """
    flow_dir, dlss_dir = out_dirs(video)
    base = os.path.dirname(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    n = video_info(video)[0]

    if kind == "flow":
        src_dir, out_name, loader = flow_dir, stem + "_flow.mkv", \
            lambda p: colorize_flow(read_flo(p))
    else:
        # DLSS output (raw RGBA off the host) saved as PNGs by the GUI's cached-frame
        # render, and read back here as PNGs.
        src_dir, out_name, loader = dlss_dir, stem + "_dlss.mkv", lambda p: imread_u(p)
    out_path = os.path.join(base, out_name)

    if frames is None:
        frames = n

    ffmpeg = _bundle_ffmpeg() if crf is not None else None
    cq = int(max(0, min(63, crf))) if crf is not None else 24
    w, h = 0, 0
    first = True
    proc = None       # ffmpeg Popen (rawvideo stdin pipe) when active
    writer = None     # OpenCV VideoWriter fallback

    for i in range(frames):
        if cancel and cancel():
            break
        ext = ".flo" if kind == "flow" else ".png"
        p = os.path.join(src_dir, f"{i:06d}" + ext)
        if not os.path.exists(p):
            continue
        img = loader(p)
        if img is None:
            continue
        if first:
            h, w = img.shape[:2]
            first = False
            if ffmpeg and not (w % 2 or h % 2):
                cmd = [ffmpeg, "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
                has_audio = with_audio and os.path.exists(video)
                if has_audio:
                    cmd += ["-i", video, "-map", "0:v:0", "-map", "1:a:0?"]
                
                if _supports_av1_nvenc(ffmpeg):
                    gop = max(1, int(round(fps * 2.0)))
                    cmd += [
                        "-c:v", "av1_nvenc",
                        "-preset", "p7",
                        "-tune", "hq",
                        "-multipass", "qres",
                        "-rc:v", "vbr",
                        "-cq:v", str(cq),
                        "-b:v", "0",
                        "-g", str(gop),
                        "-pix_fmt", "yuv420p"
                    ]
                else:
                    cmd += [
                        "-c:v", "libx264",
                        "-crf", str(cq),
                        "-preset", "medium",
                        "-pix_fmt", "yuv420p"
                    ]
                if has_audio:
                    cmd += ["-c:a", "copy", "-shortest"]
                if out_path.lower().endswith(".mp4"):
                    cmd += ["-movflags", "+faststart"]
                cmd += [out_path]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                ffmpeg = None   # fall back once; every further frame uses OpenCV
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if proc is not None:
            try:
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
            except (BrokenPipeError, OSError):
                break
        else:
            writer.write(img)
        if progress and (i % 5 == 0 or i == frames - 1):
            progress(i + 1, frames, f"导出 {kind}")

    if proc is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            rc = proc.wait()
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            if rc != 0:
                raise RuntimeError("ffmpeg 导出失败 (rc=%s)\n%s" % (rc, err[-600:]))
        except Exception as ex:
            if "ffmpeg 导出失败" not in str(ex):
                raise
    else:
        if writer is not None:
            writer.release()
    if not first:
        return out_path
    return None


def _even(v):
    r = int(round(float(v)))
    if r % 2 != 0:
        r -= 1
    return max(2, r)


def resolve_output_dims(src_w, src_h, transcode_settings=None):
    """Bug5 fix: wire resolution_mode / custom_width / custom_height /
    custom_factor / keep_aspect to real output dimensions (mirrors frontend
    DeliverPage outputDimensions logic). Returns (dst_w, dst_h)."""
    ts = transcode_settings or {}
    mode = str(ts.get("resolution_mode", "original"))
    try:
        if mode in ("original", "source", "", "none", None):
            return _even(src_w), _even(src_h)
        if mode == "double":
            return _even(src_w * 2), _even(src_h * 2)
        if mode == "half":
            return _even(src_w / 2), _even(src_h / 2)
        if mode == "7680x4320":
            return 7680, 4320
        if mode == "5120x2880":
            return 5120, 2880
        if mode == "3840x2160":
            return 3840, 2160
        if mode == "2560x1440":
            return 2560, 1440
        if mode == "1920x1080":
            return 1920, 1080
        if mode == "1280x720":
            return 1280, 720
        if mode == "custom_factor":
            factor = float(ts.get("custom_factor", 1.0))
            factor = max(0.1, min(8.0, factor))
            return _even(src_w * factor), _even(src_h * factor)
        if mode == "custom_size":
            cw = float(ts.get("custom_width", src_w))
            ch = float(ts.get("custom_height", src_h))
            if ts.get("keep_aspect", True):
                ratio = min(cw / max(1, src_w), ch / max(1, src_h))
                return _even(src_w * ratio), _even(src_h * ratio)
            return _even(cw), _even(ch)
    except Exception:
        pass
    return _even(src_w), _even(src_h)


def scale_algorithm_to_cv2(name):
    """Bug5 fix: wire scale_algorithm to a real interpolation."""
    n = str(name or "lanczos").lower()
    if n == "nearest":
        return cv2.INTER_NEAREST
    if n == "bilinear":
        return cv2.INTER_LINEAR
    if n == "bicubic":
        return cv2.INTER_CUBIC
    return cv2.INTER_LANCZOS4


def scale_algorithm_to_sws(name):
    n = str(name or "lanczos").lower()
    return {"lanczos": "lanczos", "bicubic": "bicubic", "bilinear": "bilinear", "nearest": "neighbor"}.get(n, "lanczos")


def color_preset_to_ffmpeg_args(transcode_settings=None):
    """Bug5 fix: wire color_preset to real FFmpeg color metadata args."""
    ts = transcode_settings or {}
    preset = str(ts.get("color_preset", "preserve"))
    if preset in ("preserve", "", "none", None):
        return []
    if preset == "bt709_limited":
        return ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if preset == "bt709_full":
        return ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "pc"]
    if preset == "bt2020_sdr":
        return ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "bt709", "-color_range", "tv"]
    if preset == "bt2020_pq":
        return ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-color_range", "tv"]
    if preset == "bt2020_hlg":
        return ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-color_range", "tv"]
    if preset == "hdr_to_sdr":
        return ["-vf", "tonemap=libplacebo:desat=0:tonemapping=bt2390:peak=100:format=yuv420p"]
    if preset == "custom":
        args = []
        csp = str(ts.get("custom_color_space", "")).strip()
        prim = str(ts.get("custom_color_primaries", "")).strip()
        trc = str(ts.get("custom_color_transfer", "")).strip()
        rng = str(ts.get("custom_color_range", "")).strip()
        if csp:
            args += ["-colorspace", csp]
        if prim:
            args += ["-color_primaries", prim]
        if trc:
            args += ["-color_trc", trc]
        if rng:
            args += ["-color_range", rng]
        return args
    return []


class OutputOverwritesInputError(ValueError):
    """The destination a job names is the very file it reads.

    A job whose output resolves to its input would destroy the source: the pipe
    path runs FFmpeg with -y, which truncates the destination the moment the
    process starts, while the same file is FFmpeg's second input and OpenCV's
    decoder; the GPU path muxes over the source the same way. Neither is a slow
    corruption that can be noticed and stopped - the first frame written replaces
    the frames still being read - so the pairing is refused outright rather than
    run with a warning.
    """


def _comparable_path(path) -> str:
    """One spelling of a path, with the aliases the filesystem itself ignores gone.

    Windows answers the same file for a short 8.3 name, another letter case, a
    trailing dot or space, a ``\\?\\`` prefix or a symlinked directory, so two
    strings that differ can be one file. ``realpath`` folds those on the way to a
    comparison, and the component trimming covers what a resolved path still keeps.
    The result is only ever used to compare, never to open anything, so a path this
    cannot resolve simply stays in its own spelling.
    """
    text = str(path or "").strip()
    if not text:
        return ""
    if os.name == "nt":
        if text.lower().startswith("\\\\?\\unc\\"):
            text = "\\\\" + text[len("\\\\?\\UNC\\"):]
        elif text.lower().startswith("\\\\?\\"):
            text = text[len("\\\\?\\"):]
    try:
        resolved = os.path.realpath(text)
    except OSError:
        resolved = text
    folded = os.path.normcase(os.path.normpath(resolved))
    if os.name == "nt":
        folded = "\\".join(seg.rstrip(". ") for seg in folded.split("\\"))
    return folded


def same_underlying_file(left, right) -> bool:
    """Whether two paths reach one file - and only ever answering yes on evidence.

    Identity is proven by the resolved spelling, or, when both sides exist, by the
    (device, file-index) pair the OS reports. That second test is what catches a
    hard link: two directory entries, one record, and no alias for realpath to
    fold. A filesystem that reports no file index falls back to the path test
    alone, which can miss a hard link but cannot call two different files one.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    a, b = _comparable_path(left), _comparable_path(right)
    if a and a == b:
        return True
    try:
        stat_left, stat_right = os.stat(left), os.stat(right)
    except OSError:
        return False
    if stat_left.st_ino and stat_right.st_ino:
        return ((stat_left.st_dev, stat_left.st_ino)
                == (stat_right.st_dev, stat_right.st_ino))
    return False


def assert_output_not_input(video, out_path) -> None:
    """Refuse a job whose destination is its source, before anything is opened.

    Every export entry point calls this with the paths it is about to hand to a
    decoder and an encoder, so the refusal lands before either has the file open.
    """
    if not video or not out_path:
        return
    if same_underlying_file(video, out_path):
        raise OutputOverwritesInputError(
            f"导出目标与输入素材解析为同一个文件：{out_path}。"
            "编码会在源素材仍被读取时就地覆盖它，本任务已在解码器与编码器启动之前被拒绝，"
            "源素材未被改动。请把【导出目标文件夹】或【文件名】改成与源素材不同的值；"
            "路径仅大小写、斜杠写法或 8.3 短名不同而实为同一个文件，仍会被拒绝，"
            "换一个容器后缀则得到另一个文件，属于有效改法。")


def build_transcode_command(video, out_path, w, h, fps, transcode_settings=None, cq=24, with_audio=True):
    """Build FFmpeg NVENC streaming command based on transcode_settings from RTXVideoTranscoderSlint."""
    assert_output_not_input(video, out_path)
    ffmpeg = _bundle_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg 可执行文件，无法进行硬件流式编码")

    ts_early = transcode_settings or {}
    # Bug6 fix: respect skip_existing / overwrite_output instead of unconditional -y.
    # skip_existing=true means never truncate an existing file (use -n).
    skip_existing = bool(ts_early.get("skip_existing", False))
    overwrite_output = ts_early.get("overwrite_output", True)
    if isinstance(overwrite_output, str):
        overwrite_output = overwrite_output.lower() not in ("false", "0", "no", "off")
    overwrite_flag = "-n" if (skip_existing or not overwrite_output) else "-y"

    # Bug5 fix: resolution_mode / custom_* determine the piped frame size.
    # export_dlss_stream resizes DLSS output to dst before piping, so -s must be dst.
    dst_w, dst_h = resolve_output_dims(w, h, transcode_settings)
    scale_algo = str(ts_early.get("scale_algorithm", "lanczos"))

    cmd = [
        ffmpeg, overwrite_flag, "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{dst_w}x{dst_h}", "-r", f"{fps:.6f}", "-i", "-"
    ]

    has_input = os.path.exists(video) if video else False
    if has_input:
        cmd += ["-i", video]

    cmd += ["-map", "0:v:0"]

    if not transcode_settings:
        if with_audio and has_input:
            cmd += ["-map", "1:a:0?"]
        if _supports_av1_nvenc(ffmpeg):
            gop = max(1, int(round(fps * 2.0)))
            cmd += [
                "-c:v", "av1_nvenc",
                "-preset", "p7",
                "-tune", "hq",
                "-multipass", "qres",
                "-rc:v", "vbr",
                "-cq:v", str(int(max(0, min(63, cq)))),
                "-b:v", "0",
                "-g", str(gop),
                "-pix_fmt", "yuv420p"
            ]
        else:
            cmd += [
                "-c:v", "libx264",
                "-crf", str(int(max(0, min(51, cq)))),
                "-preset", "medium",
                "-pix_fmt", "yuv420p"
            ]
        if with_audio and has_input:
            cmd += ["-c:a", "copy", "-shortest"]
        if out_path.lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
        cmd += [out_path]
        return cmd

    ts = transcode_settings
    audio_mode = ts.get("audio_mode", "copy")
    # with_audio is the job's own answer to "keep the sound?", and the caller that turned
    # it off means it: an encoder setting left at its default must not put the audio back
    # on, so the request wins over the mode and lands on the -an branch below.
    if not with_audio:
        audio_mode = "remove"
    if audio_mode != "remove" and has_input:
        cmd += ["-map", "1:a?"]

    subtitle_mode = ts.get("subtitle_mode", "copy")
    if subtitle_mode != "drop" and has_input:
        cmd += ["-map", "1:s?"]

    copy_attachments = ts.get("copy_attachments", True)
    if copy_attachments and out_path.lower().endswith(".mkv") and has_input:
        cmd += ["-map", "1:t?"]

    cmd += [
        "-map_metadata", "1" if ts.get("copy_metadata", True) and has_input else "-1",
        "-map_chapters", "1" if ts.get("copy_chapters", True) and has_input else "-1"
    ]

    codec = ts.get("codec", "av1_nvenc")
    preset = ts.get("preset", "p7")
    cmd += ["-c:v", codec, "-preset", preset]

    tune = ts.get("tune", "auto")
    if tune == "uhq":
        if codec == "h264_nvenc":
            tune = "hq"
        elif not _supports_tune_uhq(ffmpeg, codec):
            tune = "hq"
    if tune != "auto":
        cmd += ["-tune", tune]

    multipass = ts.get("multipass", "auto")
    if preset != "p7" and multipass != "auto":
        cmd += ["-multipass", multipass]

    rate_mode = ts.get("rate_mode", "cq")
    if rate_mode == "cq":
        cq_val = ts.get("cq", 16)
        cmd += ["-rc:v", "vbr", "-cq:v", str(cq_val), "-b:v", "0"]
    elif rate_mode == "constqp":
        qp_val = ts.get("qp", 20)
        cmd += ["-rc:v", "constqp", "-qp", str(qp_val)]
    elif rate_mode == "cbr":
        br = str(ts.get("video_bitrate", "80M"))
        cmd += ["-rc:v", "cbr", "-b:v", br, "-maxrate", br, "-bufsize", br]
    else:
        br = str(ts.get("video_bitrate", "80M"))
        cmd += ["-rc:v", "vbr", "-b:v", br]

    if rate_mode in ("cq", "vbr") and ts.get("peak_policy") == "custom":
        maxrate = str(ts.get("maxrate", "380M"))
        bufsize = str(ts.get("bufsize", "1520M"))
        cmd += ["-maxrate", maxrate, "-bufsize", bufsize]

    gop_mode = ts.get("gop_mode", "auto")
    if gop_mode == "auto":
        gop = max(1, int(round(fps * 2.0)))
    else:
        gop = max(1, int(ts.get("gop", 240)))
    cmd += ["-g", str(gop)]

    b_frames = str(ts.get("b_frames", "auto"))
    if b_frames != "auto":
        cmd += ["-bf", b_frames]

    b_ref_mode = ts.get("b_ref_mode", "auto")
    if b_ref_mode != "auto":
        cmd += ["-b_ref_mode", b_ref_mode]

    lookahead = str(ts.get("lookahead", "auto"))
    if lookahead != "auto":
        cmd += ["-rc-lookahead", lookahead]

    lookahead_level = str(ts.get("lookahead_level", "auto"))
    if lookahead_level != "auto":
        cmd += ["-lookahead_level", lookahead_level]

    aq_mode = ts.get("aq_mode", "spatial")
    if aq_mode == "spatial":
        aq_str = str(ts.get("aq_strength", 8))
        cmd += ["-spatial-aq", "1", "-temporal-aq", "0", "-aq-strength", aq_str]
    elif aq_mode == "temporal":
        cmd += ["-spatial-aq", "0", "-temporal-aq", "1"]
    else:
        cmd += ["-spatial-aq", "0", "-temporal-aq", "0"]

    if ts.get("weighted_prediction", False):
        cmd += ["-weighted_pred", "1"]

    temporal_filter = str(ts.get("temporal_filter", "auto"))
    if temporal_filter != "auto":
        cmd += ["-tf_level", temporal_filter]

    split_encode = str(ts.get("split_encode", "auto"))
    if split_encode != "auto":
        cmd += ["-split_encode_mode", split_encode]

    gpu = str(ts.get("gpu", "-1"))
    if gpu != "-1":
        cmd += ["-gpu", gpu]

    profile = str(ts.get("profile", "auto"))
    if profile != "auto":
        cmd += ["-profile:v", profile]

    level = str(ts.get("level", "auto"))
    if level != "auto":
        cmd += ["-level", level]

    tier = str(ts.get("tier", "auto"))
    if tier != "auto" and codec in ("av1_nvenc", "hevc_nvenc"):
        cmd += ["-tier", tier]

    pix_fmt = ts.get("pixel_format", "auto")
    if pix_fmt != "auto":
        cmd += ["-pix_fmt", pix_fmt]
    else:
        cmd += ["-pix_fmt", "yuv420p"]

    # Bug5 fix: wire scale_algorithm (sws flags) and color_preset (color metadata).
    # Order matches frontend preview (right after pix_fmt, before audio).
    cmd += ["-sws_flags", scale_algorithm_to_sws(scale_algo)]
    cmd += color_preset_to_ffmpeg_args(ts)

    # Audio
    if audio_mode == "copy":
        cmd += ["-c:a", "copy", "-shortest"]
    elif audio_mode == "aac":
        cmd += ["-c:a", "aac", "-b:a", ts.get("audio_bitrate", "320k"), "-shortest"]
    elif audio_mode == "opus":
        cmd += ["-c:a", "libopus", "-b:a", ts.get("audio_bitrate", "320k"), "-shortest"]
    elif audio_mode == "flac":
        cmd += ["-c:a", "flac", "-shortest"]
    elif audio_mode == "pcm":
        cmd += ["-c:a", "pcm_s24le", "-shortest"]
    elif audio_mode == "remove":
        cmd += ["-an"]

    # Subtitles
    if subtitle_mode == "copy":
        cmd += ["-c:s", "copy"]
    elif subtitle_mode == "convert":
        sub_codec = "webvtt" if out_path.lower().endswith(".webm") else "srt" if out_path.lower().endswith(".mkv") else "mov_text"
        cmd += ["-c:s", sub_codec]
    elif subtitle_mode == "drop":
        cmd += ["-sn"]

    if copy_attachments and out_path.lower().endswith(".mkv"):
        cmd += ["-c:t", "copy"]

    if codec == "av1_nvenc" and (out_path.lower().endswith(".mp4") or out_path.lower().endswith(".mov")):
        cmd += ["-tag:v", "av01"]
    elif codec == "hevc_nvenc" and (out_path.lower().endswith(".mp4") or out_path.lower().endswith(".mov")):
        cmd += ["-tag:v", "hvc1"]

    if out_path.lower().endswith(".mp4") or out_path.lower().endswith(".mov"):
        cmd += ["-movflags", "+faststart"]

    extra_args = ts.get("extra_args", "").strip()
    if extra_args:
        import shlex
        try:
            cmd += shlex.split(extra_args)
        except Exception:
            pass

    cmd += [out_path]
    return cmd


class StageProfiler:
    """Low-overhead stage profiler for export pipeline diagnostics."""
    def __init__(self):
        self.records = []
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def record(self, stage_dict: dict):
        self.records.append(stage_dict)

    def summary(self) -> dict:
        if not self.records:
            return {}
        keys = list(self.records[0].keys())
        res = {}
        for k in keys:
            vals = [r[k] for r in self.records if k in r and isinstance(r[k], (int, float))]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                res[k] = {
                    "mean": float(np.mean(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0.0
        n_frames = len(self.records)
        res["fps"] = n_frames / max(0.001, total_time)
        res["total_time_sec"] = total_time
        res["frame_count"] = n_frames
        return res

    def print_summary(self, title="EXPORT STAGE PROFILER REPORT"):
        s = self.summary()
        if not s:
            print(f"[{title}] No records captured.", flush=True)
            return
        print("=" * 85, flush=True)
        print(f" {title} (Frames: {s.get('frame_count', 0)}, FPS: {s.get('fps', 0.0):.2f})", flush=True)
        print("=" * 85, flush=True)
        print(f"{'Stage':<24} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Min (ms)':<9} | {'Max (ms)':<9}", flush=True)
        print("-" * 85, flush=True)
        for k, v in s.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"{k:<24} | {v['mean']:<10.2f} | {v['p50']:<10.2f} | {v['p95']:<10.2f} | {v['min']:<9.2f} | {v['max']:<9.2f}", flush=True)
        print("=" * 85, flush=True)


@dataclass(frozen=True)
class AxisPlan:
    """One axis of a separable resampling filter.

    ``indices`` and ``weights`` are padded to a rectangular ``max_taps`` window;
    ``tap_counts[d]`` says how many leading columns are real taps. Padded columns
    carry weight 0.0, so a weighted sum over the full row equals the filtered value.

    Indices are already clamp-to-edge resolved (they repeat the border sample
    instead of dropping the tap), matching the Magpie reference kernels.
    """
    src_len: int
    dst_len: int
    kernel: str
    indices: np.ndarray    # int32 [dst_len, max_taps]
    weights: np.ndarray    # float32 [dst_len, max_taps]
    tap_counts: np.ndarray  # int32 [dst_len]

    @property
    def max_taps(self) -> int:
        return int(self.indices.shape[1])


# Incremented by every AxisPlan build. Working dimensions are fixed for the whole
# Export Job, so this must stay flat across frames of one job.
filter_plan_build_count = 0


def _lanczos2_kernel(x: float) -> float:
    """Lanczos-2, radius 2. Magpie: abs(x) < 2.0 ? Sinc(x) * Sinc(x * 0.5) : 0.0"""
    if abs(x) >= 2.0:
        return 0.0
    return _lanczos2_sinc(x) * _lanczos2_sinc(x * 0.5)


def _lanczos2_sinc(x: float) -> float:
    x = x * math.pi
    if abs(x) < 1e-5:
        return 1.0
    return math.sin(x) / x


def _catmull_rom_kernel(x: float) -> float:
    """Keys cubic convolution with B = 0, C = 0.5 (Catmull-Rom). Negative lobes kept."""
    x = abs(x)
    if x < 1.0:
        return ((1.5 * x - 2.5) * x) * x + 1.0
    if x < 2.0:
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    return 0.0


# centre-1 .. centre+2, and nothing about the window depends on the ratio: the
# position is where the destination lands, the taps are the four source samples
# around it. So one 4-tap kernel describes every Catmull-Rom upscale, where the
# Lanczos window above grows as the downscale deepens.
CATMULL_ROM_TAPS = 4


def _finalize_axis_rows(src_len: int, dst_len: int, kernel: str, rows) -> AxisPlan:
    """Pad per-destination tap rows into the rectangular AxisPlan buffers."""
    max_taps = max(len(r) for r in rows)
    indices = np.zeros((dst_len, max_taps), dtype=np.int32)
    weights = np.zeros((dst_len, max_taps), dtype=np.float32)
    tap_counts = np.zeros(dst_len, dtype=np.int32)
    for d, taps in enumerate(rows):
        tap_counts[d] = len(taps)
        for t, (src_index, weight) in enumerate(taps):
            indices[d, t] = src_index
            weights[d, t] = weight
    return AxisPlan(
        src_len=int(src_len), dst_len=int(dst_len), kernel=kernel,
        indices=indices, weights=weights, tap_counts=tap_counts,
    )


def _build_catmull_rom_axis_plan(src_len: int, dst_len: int) -> AxisPlan:
    """Catmull-Rom (B=0, C=0.5) axis plan, one fixed 4-tap window per destination.

    Reference: Magpie DLSSNRFilter.cpp UpsampleResidualHorizontal and
    CompositeResidualVertical.
      position   = (d + 0.5) * src_len / dst_len - 0.5
      center     = floor(position)
      taps       = center-1 .. center+2
      weight     = CatmullRom(position - tap_index)        # negative lobes kept
      index      = clamp(tap_index, 0, src_len - 1)        # clamp-to-edge, taps never dropped
      normalized = weight / total_weight (abs guard 1e-6)
    """
    global filter_plan_build_count
    filter_plan_build_count += 1
    rows = []
    for d in range(dst_len):
        position = (d + 0.5) * src_len / float(dst_len) - 0.5
        center = math.floor(position)
        taps = [(min(max(center + o, 0), src_len - 1), _catmull_rom_kernel(position - (center + o)))
                for o in range(-1, CATMULL_ROM_TAPS - 1)]
        total = sum(w for _, w in taps)
        if abs(total) > 1e-6:
            taps = [(i, w / total) for i, w in taps]
        rows.append(taps)
    return _finalize_axis_rows(src_len, dst_len, "catmull-rom", rows)


def _apply_axis_plan(values: np.ndarray, plan: AxisPlan, axis: int) -> np.ndarray:
    """Filter one axis of an FP32 image with an AxisPlan. Never clamps."""
    if values.dtype != np.float32:
        raise TypeError(f"_apply_axis_plan needs float32, got {values.dtype}")
    if plan.src_len != values.shape[axis]:
        raise ValueError(
            f"AxisPlan {plan.kernel} maps {plan.src_len}->{plan.dst_len} but "
            f"axis {axis} has length {values.shape[axis]}"
        )
    idx = plan.indices
    w = plan.weights
    if axis == 0:
        out = np.zeros((plan.dst_len,) + values.shape[1:], dtype=np.float32)
        for t in range(plan.max_taps):
            out += w[:, t][:, None, None] * values[idx[:, t], :, :]
    else:
        out = np.zeros((values.shape[0], plan.dst_len, values.shape[2]), dtype=np.float32)
        for t in range(plan.max_taps):
            out += w[:, t][None, :, None] * values[:, idx[:, t], :]
    return out


def _lanczos2_axis_windows(src_len: int, dst_len: int):
    """Every destination row's source window, as (first, last, position, scale).

    The plan builder and the width query below share this arithmetic, so a gate
    that asks how wide a scale's filter is cannot disagree with the filter it
    asked about. Magpie: position = (d + 0.5) / scale - 0.5, support = 2.0 /
    scale, window = ceil(position - support) .. floor(position + support).
    """
    scale = dst_len / float(src_len)
    support = 2.0 / scale
    for d in range(dst_len):
        position = (d + 0.5) / scale - 0.5
        yield (math.ceil(position - support), math.floor(position + support),
               position, scale)


def lanczos2_axis_taps(src_len: int, dst_len: int) -> int:
    """Taps in the widest row of the Lanczos-2 plan for this axis, unsimplified.

    Same window as ``_build_lanczos2_axis_plan`` without evaluating a single
    weight, and it does not build a plan: ``filter_plan_build_count`` exists to
    catch re-planning per frame, and a width query is neither a plan nor a frame.
    Equals ``AxisPlan.max_taps`` for the plan of the same axis, because that is
    the widest row padded out.
    """
    if src_len < 1 or dst_len < 1:
        raise ValueError(f"a Lanczos axis maps {src_len}->{dst_len}; both lengths "
                         "have to be at least 1")
    return max(last - first + 1
               for first, last, _, _ in _lanczos2_axis_windows(src_len, dst_len))


def _build_lanczos2_axis_plan(src_len: int, dst_len: int) -> AxisPlan:
    """Anti-aliased Lanczos-2 axis plan with a scale-derived, variable tap count.

    Reference: Magpie DLSSNRFilter.cpp DownsampleColorVertical /
    DownsampleColorHorizontal.
      scale    = dst_len / src_len
      position = (d + 0.5) / scale - 0.5
      support  = 2.0 / scale          # widens as 1/scale when downscaling
      taps     = ceil(position - support) .. floor(position + support)
      weight   = Lanczos2((i - position) * scale)   # so |arg| < 2 <=> |i - position| < support
      index    = clamp(i, 0, src_len - 1)           # clamp-to-edge, taps never dropped
      normalized = weight / total_weight (abs guard 1e-6)

    A 50% downscale therefore gets 8 taps, a 25% downscale 16 — never a fixed 4.
    """
    global filter_plan_build_count
    filter_plan_build_count += 1
    rows = []
    for first, last, position, scale in _lanczos2_axis_windows(src_len, dst_len):
        taps = [(min(max(i, 0), src_len - 1), _lanczos2_kernel((i - position) * scale))
                for i in range(first, last + 1)]
        total = sum(w for _, w in taps)
        if abs(total) > 1e-6:
            taps = [(i, w / total) for i, w in taps]
        rows.append(taps)
    return _finalize_axis_rows(src_len, dst_len, "lanczos2", rows)


def downsample_lanczos2(src_bgr: np.ndarray, target_dims: tuple, dst: np.ndarray = None,
                        workspace=None) -> np.ndarray:
    """Anti-aliased working-resolution color downsample, Magpie Lanczos-2 AA.

    Vertical axis first (src_h -> dst_h), then horizontal (src_w -> dst_w), which is
    the dispatch order of the reference kernels. When ``dst`` is supplied the result is
    written into it and that same object is returned.
    """
    tw, th = target_dims
    sh, sw = src_bgr.shape[:2]
    if (tw, th) == (sw, sh):
        if dst is not None:
            np.copyto(dst, src_bgr)
            return dst
        return src_bgr

    plan_y, plan_x = resolve_lanczos2_axis_plans(sw, sh, tw, th, workspace)

    vert = _apply_axis_plan(src_bgr.astype(np.float32), plan_y, axis=0)
    horiz = _apply_axis_plan(vert, plan_x, axis=1)
    # The reference writes to an unorm8 target, which rounds; astype() alone truncates
    # and would bias every working-resolution pixel down by half a step.
    result = np.rint(np.clip(horiz, 0.0, 255.0)).astype(src_bgr.dtype)

    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


class ResamplerPlan:
    """The four AxisPlans for one Export Job, built once when working size is fixed.

    Color goes src -> working with Lanczos-2, signed residual goes working -> src
    with Catmull-Rom. Frames reuse these objects; rebuilds are counted globally in
    ``filter_plan_build_count`` so a regression that re-plans per frame is visible.
    """

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.live_w = int(live_w)
        self.live_h = int(live_h)
        built_before = filter_plan_build_count
        self.lanczos_y = _build_lanczos2_axis_plan(self.src_h, self.live_h)
        self.lanczos_x = _build_lanczos2_axis_plan(self.src_w, self.live_w)
        self.catmull_y = _build_catmull_rom_axis_plan(self.live_h, self.src_h)
        self.catmull_x = _build_catmull_rom_axis_plan(self.live_w, self.src_w)
        self.axis_plan_builds = filter_plan_build_count - built_before


def resolve_lanczos2_axis_plans(src_w: int, src_h: int, live_w: int, live_h: int,
                                workspace=None):
    """Return (plan_y, plan_x) for src -> working, from the job cache when it fits."""
    rp = getattr(workspace, "resampler_plan", None) if workspace is not None else None
    if rp is not None and (rp.src_w, rp.src_h, rp.live_w, rp.live_h) == (src_w, src_h, live_w, live_h):
        return rp.lanczos_y, rp.lanczos_x
    return (_build_lanczos2_axis_plan(src_h, live_h), _build_lanczos2_axis_plan(src_w, live_w))


def resolve_catmull_rom_axis_plans(live_w: int, live_h: int, src_w: int, src_h: int,
                                   workspace=None):
    """Return (plan_y, plan_x) for working -> src, from the job cache when it fits."""
    rp = getattr(workspace, "resampler_plan", None) if workspace is not None else None
    if rp is not None and (rp.src_w, rp.src_h, rp.live_w, rp.live_h) == (src_w, src_h, live_w, live_h):
        return rp.catmull_y, rp.catmull_x
    return (_build_catmull_rom_axis_plan(live_h, src_h), _build_catmull_rom_axis_plan(live_w, src_w))


class ResamplerContext:
    """Filter plans and FP32 scratch for one (source, working) size pair.

    Allocated when the size pair is created and reused for every frame of the job;
    released by :meth:`close`. This is the state a backend owns, not per-frame state.
    """

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.live_w = int(live_w)
        self.live_h = int(live_h)
        self.resampler_plan = ResamplerPlan(src_w, src_h, live_w, live_h)
        self.res_diff_low_f32 = np.empty((self.live_h, self.live_w, 3), dtype=np.float32)
        self.res_diff_horiz_f32 = np.empty((self.live_h, self.src_w, 3), dtype=np.float32)
        self.res_diff_high_f32 = np.empty((self.src_h, self.src_w, 3), dtype=np.float32)
        self.scratch_bytes = sum(
            int(a.nbytes) for a in (self.res_diff_low_f32, self.res_diff_horiz_f32,
                                    self.res_diff_high_f32))

    def close(self):
        self.resampler_plan = None
        self.res_diff_low_f32 = None
        self.res_diff_horiz_f32 = None
        self.res_diff_high_f32 = None


class ResidualBackend:
    """The one working-color downsample plus signed-residual contract.

    Preview, export and benchmarks obtain a backend from create_residual_backend() and
    then share exactly the same filter, residual formula, scratch and plans. A backend
    is created per size pair and reused across frames; it is not per-frame state.
    """
    name = "abstract"
    version = "0"

    def prepare_working_color(self, source_bgr: np.ndarray, dst_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(self, source_full_bgr: np.ndarray, source_low_bgr: np.ndarray,
                    dlss_low_bgr: np.ndarray, strength: float, dst_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass

    def backend_info(self) -> dict:
        return {"residual_backend": self.name, "residual_backend_version": self.version}

    def diagnostics(self) -> dict:
        return {"residual_backend": self.name}


class PythonReferenceBackend(ResidualBackend):
    """The NumPy reference implementation: AxisPlan Lanczos-2 and Catmull-Rom.

    Correctness reference, fallback and the thing the native backend is compared
    against. It is not a fast path: it converts and allocates per frame in Python.
    """
    name = "cpu_reference_python"
    version = "axisplan-1"

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int):
        self.context = ResamplerContext(src_w, src_h, live_w, live_h)
        self.downsample_call_count = 0
        self.reconstruct_call_count = 0
        self.closed = False

    @property
    def dims(self):
        c = self.context
        return (c.src_w, c.src_h, c.live_w, c.live_h)

    def prepare_working_color(self, source_bgr: np.ndarray, dst_bgr: np.ndarray) -> np.ndarray:
        c = self.context
        if source_bgr.shape[:2] != (c.src_h, c.src_w):
            raise ValueError(f"backend built for {c.src_w}x{c.src_h}, got source {source_bgr.shape[1]}x{source_bgr.shape[0]}")
        self.downsample_call_count += 1
        return downsample_lanczos2(source_bgr, (c.live_w, c.live_h), dst=dst_bgr, workspace=c)

    def reconstruct(self, source_full_bgr: np.ndarray, source_low_bgr: np.ndarray,
                    dlss_low_bgr: np.ndarray, strength: float, dst_bgr: np.ndarray) -> np.ndarray:
        c = self.context
        if source_full_bgr.shape[:2] != (c.src_h, c.src_w):
            raise ValueError(f"backend built for {c.src_w}x{c.src_h}, got source {source_full_bgr.shape[1]}x{source_full_bgr.shape[0]}")
        self.reconstruct_call_count += 1
        return reconstruct_residual(source_full_bgr, source_low_bgr, dlss_low_bgr,
                                    strength=strength, workspace=c, out_recon=dst_bgr)

    def close(self):
        self.context.close()
        self.closed = True

    def backend_info(self) -> dict:
        c = self.context
        return {"residual_backend": self.name, "residual_backend_version": self.version,
                "source_width": c.src_w, "source_height": c.src_h,
                "working_width": c.live_w, "working_height": c.live_h}

    def diagnostics(self) -> dict:
        return {"residual_backend": self.name,
                "residual_backend_version": self.version,
                "downsample_call_count": self.downsample_call_count,
                "reconstruct_call_count": self.reconstruct_call_count,
                "python_scratch_bytes": self.context.scratch_bytes,
                "axis_plan_builds": self.context.resampler_plan.axis_plan_builds,
                "closed": self.closed}


def create_residual_backend(requested: str, src_w: int, src_h: int, live_w: int, live_h: int,
                            diagnostics_sink: dict = None) -> ResidualBackend:
    """The single place that decides which residual implementation runs.

    "python" forces the reference. "native" demands the C++ hot path and fails if it is
    unavailable. "auto" prefers native and records why it fell back, so a machine
    without the built bridge still exports, but never claims to be native.
    """
    requested = str(requested or "auto").lower()
    if requested in ("cpu_reference_python", "python", "reference"):
        backend = PythonReferenceBackend(src_w, src_h, live_w, live_h)
        actual = backend.name
        reason = "requested_python"
    else:
        backend = None
        actual = None
        reason = None
        if requested in ("cpu_reference_native", "native"):
            backend = _load_native_backend(src_w, src_h, live_w, live_h)
            if backend is None:
                raise RuntimeError(
                    "residual_backend=native was requested but residual_bridge could not be loaded")
            actual = backend.name
        else:
            backend = _load_native_backend(src_w, src_h, live_w, live_h)
            if backend is not None:
                actual = backend.name
                reason = "native_available"
            else:
                backend = PythonReferenceBackend(src_w, src_h, live_w, live_h)
                actual = backend.name
                reason = "native_load_failed"
    if diagnostics_sink is not None:
        diagnostics_sink["residual_backend_requested"] = requested
        diagnostics_sink["residual_backend_actual"] = actual
        diagnostics_sink["residual_backend_fallback_reason"] = reason
    return backend


def _load_native_backend(src_w, src_h, live_w, live_h):
    """Return a native backend, or None when the bridge is not built on this machine."""
    try:
        import residual_native
    except Exception:
        return None
    try:
        return residual_native.NativeResidualBackend(src_w, src_h, live_w, live_h)
    except Exception:
        return None


class FrameWorkspace:
    """Reusable per-frame buffers for the DLSS session and the guidance planes."""

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int, dst_w: int, dst_h: int,
                 reduced_transfer: bool = False, with_flow_plane: bool = True):
        self.src_w = src_w
        self.src_h = src_h
        self.live_w = live_w
        self.live_h = live_h
        self.dst_w = dst_w
        self.dst_h = dst_h

        # RGBA buffer for Live input (at live resolution)
        self.live_rgba = np.empty((live_h, live_w, 4), dtype=np.uint8)
        # The motion plane a host-pointer session is handed. Reserving one for a
        # frame no flow consumer reads would be 66 MB of zeros at 4K for an
        # argument that never gets read, so this is the allocation the caller's
        # "no flow" claim is measured against.
        self.live_flow = (np.zeros((live_h, live_w, 2), dtype=np.float32)
                          if with_flow_plane else None)

        # DLSS output buffer (at live resolution). One pass, so one buffer: the native
        # host ABI has no per-pass session handle.
        self.dlss_out_rgba = np.empty((live_h, live_w, 4), dtype=np.uint8)
        # The BGR twin of that buffer exists for one reason: the residual composite reads
        # three channels and the host wrote four. A composite that reads the RGBA store in
        # place has no use for it, so the reduced order does not reserve 6,220,800 bytes at
        # 1080p for a conversion that never happens. Only ProductionExportCore sets the flag,
        # and it withholds the reduced order unless a downscaled residual stage exists, which
        # is what keeps the alias below from pointing a full-size frame at None.
        self.dlss_out_bgr = None if reduced_transfer else np.empty((live_h, live_w, 3), dtype=np.uint8)

        # Full resolution buffers if downscaled
        if (live_w, live_h) != (src_w, src_h):
            self.down_cur_bgr = np.empty((live_h, live_w, 3), dtype=np.uint8)
            self.recon_bgr = np.empty((src_h, src_w, 3), dtype=np.uint8)
        else:
            self.down_cur_bgr = None
            self.recon_bgr = self.dlss_out_bgr

        # Final target output buffer
        self.dst_bgr = np.empty((dst_h, dst_w, 3), dtype=np.uint8)

def reconstruct_residual(source_bgr: np.ndarray, down_cur_bgr: np.ndarray, dlss_out_bgr: np.ndarray,
                         strength: float = 1.0, workspace=None, out_recon=None) -> np.ndarray:
    """Signed-residual reconstruction, Magpie Catmull-Rom (B=0, C=0.5) upsampling.

        residual_low  = dlss_out_bgr - down_cur_bgr      # signed FP32, working size
        residual_full = catmull_rom_y(catmull_rom_x(residual_low))   # still signed
        out           = clip(source_bgr + residual_full * strength, 0, 255)

    Horizontal axis first then vertical, matching UpsampleResidualHorizontal and
    CompositeResidualVertical. Only the composite may clamp; clamping the residual
    would destroy the negative lobes the kernel exists to produce.
    """
    src_h, src_w = source_bgr.shape[:2]
    live_h, live_w = down_cur_bgr.shape[:2]
    if workspace is None or getattr(workspace, "res_diff_low_f32", None) is None:
        raise ValueError(
            "reconstruct_residual needs a downscaled FrameWorkspace; "
            "there is no residual stage at working_scale 1.0 to fall back to."
        )

    # 1. Signed FP32 residual at working resolution.
    np.subtract(dlss_out_bgr.astype(np.float32), down_cur_bgr.astype(np.float32),
                out=workspace.res_diff_low_f32)

    # 2. Separable Catmull-Rom back up to source resolution, no clamp anywhere.
    plan_y, plan_x = resolve_catmull_rom_axis_plans(live_w, live_h, src_w, src_h, workspace)
    np.copyto(workspace.res_diff_horiz_f32,
              _apply_axis_plan(workspace.res_diff_low_f32, plan_x, axis=1))
    np.copyto(workspace.res_diff_high_f32,
              _apply_axis_plan(workspace.res_diff_horiz_f32, plan_y, axis=0))

    # 3. Strength, then the single clamp of the whole chain.
    if abs(strength - 1.0) > 1e-4:
        workspace.res_diff_high_f32 *= strength

    dst = out_recon if out_recon is not None else workspace.recon_bgr
    res_composite = np.rint(np.clip(source_bgr.astype(np.float32) + workspace.res_diff_high_f32, 0.0, 255.0))
    np.copyto(dst, res_composite.astype(np.uint8))
    return dst


class ProductionExportCore:
    """The single production frame path used by export and the benchmarks.

    One data flow per stage, and every stage refuses to guess:
    1. Working color is prepared exactly once per frame by the residual backend, and
       the same array feeds guidance, DLSSNR and the residual.
    2. Flow comes from a FlowProvider, delivered whichever way that provider can
       deliver it: a CUDA descriptor goes to the GPU host's device-flow entry, a CPU
       float32x2 plane goes to its host-motion entry, and either goes to the legacy
       host's one entry when the session is that host. Which of them a run uses is
       reported as ``dlss_entry`` and ``flow_transport``. Except on the GPU host with
       guidance off: that frame has no provider, because the host answers motion from
       its own resident zero texture. Depth guidance is unavailable; the legacy host
       receives one fixed zero plane and the GPU host owns a zero texture.
    3. Exactly one DLSS pass. The native host ABI has no independent session handle, so
       multi-pass is not supported and is rejected rather than looped over.
    4. Stage timing and counter metadata per frame.

    There are two transfer orders through those stages, chosen once at construction and
    reported per frame. The legacy one converts the working plane to RGBA for the DLSS host,
    hands NVOF a CPU-reordered copy of it, and converts the host's output back to BGR before
    the composite reads it. The reduced one deletes all three: the fused filter pass writes
    both the BGR plane and the RGBA plane, the BGR plane goes to NVOF as it stands, and the
    composite reads the host's RGBA output in place. Reduced is offered only when every one
    of those three is available, and a request for it that cannot be met raises - the mode
    is a claim about which bytes moved, so half of it is not half a saving but a wrong
    description of the whole frame.
    """

    def __init__(self, src_w: int, src_h: int, dst_w: int, dst_h: int,
                 passes, job_settings: dict = None, flow_provider=None,
                 profiler=None, lock=None, residual_backend=None):
        self.src_w = src_w
        self.src_h = src_h
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.passes = passes if isinstance(passes, (list, tuple)) else [passes]
        if len(self.passes) != 1:
            raise ValueError(
                f"the production pipeline supports exactly one DLSS pass, got {len(self.passes)}; "
                "multi-pass needs a native session handle per pass and is not implemented.")
        self.primary_live = self.passes[0]
        # The one DLSS host this build ships a loader for is the project GPU host, and
        # what identifies it is the entry that binds motion on the card. Read off the
        # session rather than off a setting, so a run cannot report a host it was not
        # given. A session without it is not a host this pipeline can drive: every
        # downstream decision - which entry draws the frame, where depth comes from,
        # what the byte accounting says crossed a boundary - is written for a host that
        # owns textures on the card.
        if not callable(getattr(self.primary_live, "process_device_flow", None)):
            raise ValueError(
                "渲染会话没有 DLSS GPU 宿主的入口点（process_device_flow）："
                f"传入的是 {type(self.primary_live).__name__}。本版本唯一的原生 DLSS 宿主是 "
                "native/dlss_gpu_host.dll（dlss_backend=gpu_flow_d3d12）；旧宿主 "
                "dlssnr_host.dll 已随其 ctypes 封装一并移除，本管线里已不存在能用 "
                "process() 驱动的宿主。")
        # Which native host and ABI answered, taken from the session that opened:
        # which host drew a frame is invisible in the frame, so the run's own record
        # has to carry it.
        self.dlss_host_dll = getattr(self.primary_live, "host_dll", None)
        self.dlss_host_abi = getattr(self.primary_live, "host_abi", None)
        self.live_w = getattr(self.primary_live, "_w", src_w)
        self.live_h = getattr(self.primary_live, "_h", src_h)
        self.job_settings = job_settings or {}
        self.working_scale = float(self.job_settings.get("working_scale", 1.0))
        self.residual_strength = float(self.job_settings.get("residual_strength", 1.0))
        self.guidance_mode = int(self.job_settings.get("guidance_mode", 0))
        self.profiler = profiler
        self.lock = lock
        self.uses_flow = self.guidance_mode == 1
        if self.guidance_mode not in (0, 1):
            raise ValueError("guidance_mode supports only 0 (off) or 1 (flow)")
        # The GPU host answers an unguided frame from the zero motion texture it
        # bound at open, so a frame on this branch has no motion input of any kind.
        # That is a shape of frame in its own right, not a guided frame with the
        # provider swapped out, and it is decided here once.
        self.gpu_zero_motion = not self.uses_flow
        # A guided frame on the same host takes one of two entries, and which one
        # is a property of the provider, not of a setting: a device descriptor
        # arrives at the entry that binds device memory, a CPU float32x2 plane at
        # the entry that stages host memory. Decided once, here, before frame zero.
        self.gpu_device_flow = False
        self.gpu_host_motion = False
        if self.gpu_zero_motion:
            if flow_provider is not None:
                raise ValueError(
                    "guidance_mode=0 on the DLSS GPU host takes no flow provider: "
                    "the host answers motion from its own resident zero texture, "
                    "so a provider here is an engine nobody asked for.")
            self._require_zero_motion_host()
            self.flow_provider = None
        else:
            if flow_provider is None:
                if self.uses_flow:
                    raise ValueError(
                        f"guidance_mode={self.guidance_mode} needs a flow_provider; "
                        "refusing to denoise against a silent zero-flow plane.")
                flow_provider = ZeroFlowProvider()
            self.flow_provider = flow_provider
            self._route_guided_flow()

        self.dlss_entry = ("process_zero_motion" if self.gpu_zero_motion
                           else "process_device_flow" if self.gpu_device_flow
                           else "process_host_motion")
        # Where the motion this frame is guided by physically comes from, kept
        # distinct from the entry name because the byte accounting reads it: a
        # plane the host uploads and a descriptor already resident on the card are
        # different transfers, and only one of them is a round trip.
        self.flow_transport = ("resident_zero_texture" if self.gpu_zero_motion
                               else "nvof_device_descriptor" if self.gpu_device_flow
                               else "host_plane_upload")
        # The one branch that reserves no motion plane: the host's zero texture is
        # already bound, so there is no array to fill and nothing to upload.
        self.flow_plane_requested = not self.gpu_zero_motion

        self.scale_algo = str(self.job_settings.get("scale_algorithm", "lanczos"))
        self.scale_interp = scale_algorithm_to_cv2(self.scale_algo)

        # A working resolution below the source is what makes a residual stage exist.
        self.downscaled = (self.live_w, self.live_h) != (self.src_w, self.src_h)
        self.use_pyramid = bool(self.job_settings.get("residual_pyramid", True)) and self.downscaled
        self.backend_diagnostics = {}
        if residual_backend is not None:
            self.residual_backend = residual_backend
            self.backend_diagnostics["residual_backend_requested"] = "injected"
        elif self.downscaled:
            # Default is "auto": the native hot path when residual_bridge is built, the
            # NumPy reference otherwise, with the reason recorded either way. Proven by
            # the end-to-end gate (50% working beats 100% working), so it ships as the
            # default rather than needing to be requested per job.
            self.residual_backend = create_residual_backend(
                self.job_settings.get("residual_backend"),
                self.src_w, self.src_h, self.live_w, self.live_h,
                diagnostics_sink=self.backend_diagnostics)
        else:
            self.residual_backend = None

        self.transfer_diagnostics = {}
        self.transfer_reduced = False
        self._negotiate_transfer_mode()

        self.workspace = FrameWorkspace(
            src_w=self.src_w, src_h=self.src_h,
            live_w=self.live_w, live_h=self.live_h,
            dst_w=self.dst_w, dst_h=self.dst_h,
            reduced_transfer=self.transfer_reduced,
            with_flow_plane=self.flow_plane_requested
        )
        self.first_proc = True
        self.last_idx = None

    def _require_zero_motion_host(self):
        """Refuse a GPU host that binds no zero motion texture, before frame zero.

        An ABI-2 dlss_gpu_host.dll has no guidance-off entry point. The frame has
        nowhere else to go: this build ships one DLSS host, so the answer is a
        rebuild of that host, said in as many words.
        """
        probe = getattr(self.primary_live, "supports_zero_motion", None)
        if not (callable(probe) and probe()):
            raise ValueError(
                "guidance_mode=0 on dlss_backend=gpu_flow_d3d12 needs a GPU host that "
                "binds a resident zero motion texture (native/dlss_gpu_host.dll ABI 3); "
                "the loaded session exposes no such entry point. Rebuild "
                "native/dlss_gpu_host.dll from native/. There is no second DLSS host "
                "in this build to run the frame on instead.")

    def _flow_delivery(self) -> str:
        """How the supplied provider can hand over motion, asked of the provider.

        "device" answers with a CUDA descriptor from ``get_flow_device``, which only
        an engine reading device memory can produce; "host_plane" answers with a
        float32x2 array from ``get_flow``; "both" can do either; "neither" is a class
        that was handed to a frame it cannot serve. Probed as methods actually
        implemented, never off a class name, so a provider is classified by what it
        can be made to return rather than by what it is called.

        A provider that delegates to something else answers for that thing instead:
        one adapter class wraps every flow engine, so its own method table says the
        same thing about NVOF and about DIS, and the engine underneath is the only
        one that knows whether the vectors can be left on the card.
        """
        provider = self.flow_provider
        device = callable(getattr(provider, "get_flow_device", None))
        answers_for_its_engine = getattr(provider, "supports_device_flow", None)
        if device and callable(answers_for_its_engine):
            device = bool(answers_for_its_engine())
        host_plane = type(provider).get_flow is not FlowProvider.get_flow
        return ("both" if device and host_plane else "device" if device
                else "host_plane" if host_plane else "neither")

    def _route_guided_flow(self):
        """Pick the entry a guided frame runs on, or refuse the pairing before frame zero.

        A device descriptor and a host plane are not interchangeable at the ABI
        boundary: the first is bound where it lies and the second is staged into an
        upload buffer by the host, so handing one to the other's entry is either a
        download nobody asked for or a crash on the first frame. Where a provider can
        do both, the descriptor wins on the GPU host, because a caller that can keep
        the vectors resident should not be made to round-trip them to satisfy a host
        entry. What is left is a pairing with no entry at all: a CPU plane in front of
        a loaded host that predates the entry taking one. That is a refusal with the
        reason in it, not a substitution onto another host.
        """
        delivery = self._flow_delivery()
        if delivery == "neither":
            raise ValueError(
                f"flow provider {type(self.flow_provider).__name__} implements "
                "neither get_flow nor get_flow_device, so this frame has no motion "
                "input for any host entry to bind.")
        if delivery in ("device", "both"):
            self.gpu_device_flow = True
            return
        self._require_host_motion_host()
        self.gpu_host_motion = True

    def _require_host_motion_host(self):
        """Refuse a GPU host that cannot take a motion plane, before frame zero.

        ``dlss_gpu_host_process_host_motion`` is an additive entry at ABI 3, so a
        build that predates it exposes no such symbol and no version check catches
        that. The only other host this project ever had is gone from the build, so
        the answer to a stale native is a rebuild, stated as the thing to do.
        """
        probe = getattr(self.primary_live, "supports_host_motion", None)
        if not (callable(probe) and probe()):
            raise ValueError(
                "a CPU float32x2 flow provider on dlss_backend=gpu_flow_d3d12 needs a "
                "GPU host with a host-motion entry point "
                "(native/dlss_gpu_host.dll, dlss_gpu_host_process_host_motion); the "
                "loaded session exposes no such symbol. Rebuild native/dlss_gpu_host.dll "
                "from native/. There is no second DLSS host in this build to run the "
                "frame on instead.")

    def _transfer_capabilities(self) -> dict:
        """What the reduced order needs, each item answered by the thing that must provide it.

        Nothing here is inferred from a class name or a version string: the fused store and
        the RGBA-direct composite are capabilities the loaded bridge reports about itself,
        the flow answer comes from the provider that would have to be handed the frame, and
        the residual stage is the one case where a working downsample exists at all.
        """
        backend_caps = getattr(self.residual_backend, "capabilities", None) or {}
        return {
            "residual_stage_exists": bool(self.downscaled and self.use_pyramid),
            "fused_working_bgr_and_rgba_store": bool(backend_caps.get("prepare_bgr_rgba")),
            "rgba_read_in_place_by_reconstruct": bool(backend_caps.get("reconstruct_rgba")),
            # Vacuously true on the unguided GPU host: there is no flow input to
            # copy, which is the whole point of that branch.
            "flow_input_needs_no_rgba_copy": (
                True if self.gpu_zero_motion
                else not self.flow_provider.requires_rgba_input()),
        }

    def _negotiate_transfer_mode(self):
        """Choose between the legacy and the reduced transfer order, and record why.

        The reduced order is the one where the working frame goes to NVOF as the BGR24 plane
        it already is, the DLSS input arrives as the second output of the same Lanczos pass,
        and the composite reads the host's RGBA output where the host wrote it. All three
        have to be there together, because a path that quietly loses one of them is the old
        path wearing a new name: it still makes the copy it claims not to, and only a
        counter would notice, which is precisely why the mode is negotiated up front from
        what the binaries report rather than assumed per frame.
        """
        requested = str(self.job_settings.get("transfer_mode") or "auto").lower()
        if requested not in ("auto", "legacy", "reduced"):
            raise ValueError(
                f"transfer_mode={requested!r} is not a mode this pipeline offers; the "
                f"choices are 'auto', 'legacy' and 'reduced'")
        capabilities = self._transfer_capabilities()
        missing = sorted(name for name, present in capabilities.items() if not present)
        if not missing:
            # Nothing had to be given up, so an explicit request is honoured either way.
            actual = "legacy" if requested == "legacy" else "reduced"
            reason = "requested_legacy" if requested == "legacy" else None
        elif requested == "reduced":
            raise RuntimeError(
                f"transfer_mode=reduced was requested and cannot be provided: "
                f"{', '.join(missing)}. Refusing rather than running the legacy order, "
                f"because a caller that asked for the reduced transfers has to be able to "
                f"tell from the run that it did not get them.")
        elif requested == "legacy":
            actual, reason = "legacy", "requested_legacy"
        else:
            actual, reason = "legacy", "missing_capability: " + ", ".join(missing)
        self.transfer_reduced = actual == "reduced"
        self.flow_input_format = "bgr24_host" if self.transfer_reduced else "rgba8_host"
        self.transfer_diagnostics = {
            "transfer_mode_requested": requested,
            "transfer_mode_actual": actual,
            "transfer_mode_fallback_reason": reason,
            "transfer_capabilities": capabilities,
            "transfer_flow_input_requested": self.flow_input_format,
            "dlss_backend_actual": "gpu_flow_d3d12",
            # The host file and ABI version that answered, taken off the session this
            # run opened. A backend name says which host was asked for; these say
            # which binary drew the frames.
            "dlss_host_dll": self.dlss_host_dll,
            "dlss_host_abi": self.dlss_host_abi,
            # Which host entry point answers every frame of this run. Read off the
            # session, not off a setting, so a job that claimed a host it did not
            # get is contradicted by the run's own diagnostics.
            "dlss_entry": self.dlss_entry,
            # And where the motion it binds comes from, which is what separates a
            # CPU flow plane uploaded to the GPU host from an NVOF descriptor that
            # never left the card, and both from the resident zero texture.
            "flow_transport": self.flow_transport,
            "flow_provider": "none" if self.flow_provider is None else self.flow_provider.name,
            "flow_plane_allocated": self.flow_plane_requested,
        }

    def close(self):
        if self.residual_backend is not None:
            self.residual_backend.close()
            self.residual_backend = None

    def process_frame(self, idx: int, fr: np.ndarray, is_discontinuous: bool = False,
                      decode_wait_ms: float = 0.0) -> tuple:
        """Process a single frame through the production pipeline.
        
        Returns (final_bgr, record_dict).
        """
        t_frame_start = time.perf_counter()
        # Read before the frame, not after: the counters below are then a measurement of
        # what this frame's own call sites did, not of what the code happens to contain.
        conversions_before = transfer_profiler.conversion_snapshot()

        # 1. Prepare Working Color (downsampled exactly once, by the residual backend)
        t_prep_0 = time.perf_counter()
        if self.transfer_reduced:
            # One filter pass publishes both orders this frame needs: the BGR plane the
            # residual composite and NVOF read, and the RGBA plane the DLSS host reads. The
            # BGR side is bit-identical to what the two-pass path writes, which is gated in
            # test_native_residual rather than assumed from the shared kernel.
            self.residual_backend.prepare_working_color_rgba(
                fr, self.workspace.down_cur_bgr, self.workspace.live_rgba)
            cur_work = self.workspace.down_cur_bgr
        elif self.downscaled:
            self.residual_backend.prepare_working_color(fr, self.workspace.down_cur_bgr)
            cur_work = self.workspace.down_cur_bgr
        else:
            cur_work = fr
        t_prep_1 = time.perf_counter()
        working_color_prepare_ms = (t_prep_1 - t_prep_0) * 1000.0

        # 2. The RGBA the host reads, plus motion guidance at the working extent
        # A guided frame holds the session lock from here through the inference below. The
        # reason is a lifetime, not a rate limit: what the device route hands stage 3 is a
        # descriptor the NVOF engine owns and its *next* execute destroys, so any other
        # caller that computes flow in between (a preview frame, a second chain sharing the
        # process-wide engine) leaves this frame guided by memory that is already gone -
        # wrong pixels, no error. The engine's temporal history is one object for the whole
        # process too, so an interleaved execute rewrites the hints this frame reset or did
        # not reset to use. Holding across both stages is the only span that covers it; the
        # unguided route has no stage two work, so it keeps the narrow hold around the
        # inference alone.
        guidance_held = False
        if self.uses_flow and self.lock is not None:
            if not self.lock.acquire(timeout=15):
                raise RuntimeError(
                    f"DLSS 会话锁 (_live_lock) 获取超时 15s：第 {idx} 帧无法进入制导")
            guidance_held = True
        t_g_0 = time.perf_counter()
        if not self.transfer_reduced:
            transfer_profiler.convert_color(cur_work, cv2.COLOR_BGR2RGBA,
                                            dst=self.workspace.live_rgba, site="dlss_input")
        rgba = self.workspace.live_rgba
        if self.gpu_zero_motion:
            # No stage two work at all: the host reads its own resident zero
            # texture, so there is no provider to ask and no plane to hand it.
            flow = None
        elif self.gpu_device_flow:
            flow = self.flow_provider.get_flow_device(
                idx, cur_work, self.live_w, self.live_h,
                is_discontinuous=is_discontinuous)
        else:
            # A CPU float32x2 plane: the workspace buffer the host stages into for
            # process_host_motion, which is the only entry that takes one.
            flow = self.flow_provider.get_flow(idx, cur_work, None,
                                               self.live_w, self.live_h,
                                               is_discontinuous=is_discontinuous,
                                               out=self.workspace.live_flow,
                                               input_format=self.flow_input_format)
        g_timing = dict(self.flow_provider.timing()) if self.flow_provider else {}
        t_g_1 = time.perf_counter()
        guidance_ms = (t_g_1 - t_g_0) * 1000.0

        # 3. DLSS neural inference: exactly one pass, session lock timed
        t_dlss_0 = time.perf_counter()
        out_target = self.workspace.dlss_out_rgba
        reset = self.first_proc or is_discontinuous
        def _run_dlss():
            if self.gpu_zero_motion:
                return self.primary_live.process_zero_motion(
                    rgba, reset=reset, out=out_target)
            if self.gpu_device_flow:
                return self.primary_live.process_device_flow(
                    rgba, flow, reset=reset, out=out_target)
            return self.primary_live.process_host_motion(
                rgba, flow, reset=reset, out=out_target)

        def _run_under_lock():
            if self.lock is None or guidance_held:
                # Either nothing shares this session, or stage two already holds the one
                # lock there is - a second acquire of a plain Lock would deadlock.
                return _run_dlss()
            if not self.lock.acquire(timeout=15):
                raise RuntimeError(
                    f"DLSS 会话锁 (_live_lock) 获取超时 15s：第 {idx} 帧无法进入推理")
            try:
                return _run_dlss()
            finally:
                try:
                    self.lock.release()
                except Exception:
                    pass
        try:
            o = _run_under_lock()
        finally:
            if guidance_held:
                try:
                    self.lock.release()
                except Exception:
                    pass
        if o is None:
            raise RuntimeError(f"DLSS 神经网络在第 {idx} 帧推理失败: "
                               f"{getattr(self.primary_live, 'last_error', '未知错误')}")
        self.first_proc = False
        t_dlss_1 = time.perf_counter()
        dlss_process_ms = (t_dlss_1 - t_dlss_0) * 1000.0

        # 4. Residual reconstruction by the same backend that prepared the color
        t_res_0 = time.perf_counter()
        if not self.transfer_reduced:
            transfer_profiler.convert_color(o, cv2.COLOR_RGBA2BGR,
                                            dst=self.workspace.dlss_out_bgr,
                                            site="residual_input")
        if self.use_pyramid:
            if self.transfer_reduced:
                # The host wrote four channels and the composite reads them where they are,
                # stepping over alpha. Making a three-channel copy first is the round trip
                # this order exists to delete, and it is the reason dlss_out_bgr was never
                # allocated for a reduced run.
                recon_bgr = self.residual_backend.reconstruct_from_rgba(
                    fr, self.workspace.down_cur_bgr, o, self.residual_strength,
                    self.workspace.recon_bgr)
            else:
                recon_bgr = self.residual_backend.reconstruct(
                    fr, self.workspace.down_cur_bgr, self.workspace.dlss_out_bgr,
                    self.residual_strength, self.workspace.recon_bgr)
            residual_filter = self.residual_backend.name
        elif self.downscaled:
            recon_bgr = cv2.resize(self.workspace.dlss_out_bgr, (self.src_w, self.src_h),
                                   dst=self.workspace.recon_bgr, interpolation=self.scale_interp)
            residual_filter = "none"
        else:
            recon_bgr = self.workspace.dlss_out_bgr
            residual_filter = "none"
        t_res_1 = time.perf_counter()
        residual_ms = (t_res_1 - t_res_0) * 1000.0

        # 5. Target Resolution Resizing
        t_conv_0 = time.perf_counter()
        if (recon_bgr.shape[1], recon_bgr.shape[0]) != (self.dst_w, self.dst_h):
            final_bgr = cv2.resize(recon_bgr, (self.dst_w, self.dst_h), dst=self.workspace.dst_bgr, interpolation=self.scale_interp)
        else:
            final_bgr = recon_bgr
        t_conv_1 = time.perf_counter()
        output_convert_ms = (t_conv_1 - t_conv_0) * 1000.0

        # Byte accounting for the frame that just ran. The conversion figures are deltas
        # of counters incremented by the call sites above, so a value of 0 here means the
        # seam was reached and no such conversion happened.
        transfer_fields, _ = transfer_profiler.conversion_deltas(conversions_before)
        transfer_fields.update({
            "nvof_input_format": g_timing.get("nvof_input_format"),
            "nvof_h2d_copy_count": g_timing.get("nvof_h2d_copy_count_delta"),
            "nvof_h2d_bytes_measured": g_timing.get("nvof_h2d_bytes_delta"),
            "nvof_d2h_copy_count": g_timing.get("nvof_d2h_copy_count_delta"),
            "nvof_d2h_bytes_measured": g_timing.get("nvof_d2h_bytes_delta"),
            "dlss_input_color_bytes_logical": int(rgba.nbytes),
            # Not "unmeasured" but absent: the unguided GPU host is handed no motion.
            # The device descriptor's figure is the extent it covers, not a transfer:
            # those vectors were never in host memory, so no H2D of them exists. The
            # plane the host-motion entry is handed is a real H2D the moment the card
            # stages it, and this is its logical size - the loaded host counts colour
            # staging but has no motion counter, so the crossing itself stays
            # unmeasured and is not to be read as a zero-copy claim.
            "dlss_input_flow_bytes_logical": (
                0 if self.gpu_zero_motion
                else int(flow["width"] * flow["height"] * 8) if self.gpu_device_flow
                else int(flow.nbytes)),
            # The GPU host owns an immutable zero depth texture it created at open:
            # no depth bytes cross from this process on any entry point.
            "dlss_input_depth_bytes_logical": 0,
            "dlss_output_bytes_logical": int(o.nbytes),
            # This backend never reads the GPU host's transfer counters, so there is no
            # number here; the GPU-transfer backend does, and publishes them. The logical
            # sizes above are the host buffers this host was handed, not evidence about
            # what it did with them.
            "dlss_gpu_transfer_bytes_measured": None,
            "encoder_gpu_upload_bytes_measured": None,
            # The two pipes belong to the export loop, which is where the read and the
            # write happen; the frame record carries them empty rather than absent.
            **transfer_profiler.unmeasured_pipe_fields(),
        })

        record_dict = {
            "source_width": self.src_w,
            "source_height": self.src_h,
            "requested_working_scale": self.working_scale,
            "actual_working_width": self.live_w,
            "actual_working_height": self.live_h,
            "actual_dlss_feature_width": self.live_w,
            "actual_dlss_feature_height": self.live_h,
            "guidance_backend": (g_timing.get("guidance_backend")
                                 or ("none" if self.flow_provider is None
                                     else self.flow_provider.name)),
            "dlss_backend_actual": "gpu_flow_d3d12",
            "dlss_host_dll": self.dlss_host_dll,
            "dlss_host_abi": self.dlss_host_abi,
            "dlss_entry": self.dlss_entry,
            "flow_transport": self.flow_transport,
            "residual_backend": residual_filter,
            "residual_backend_requested": self.backend_diagnostics.get("residual_backend_requested", "auto"),
            "residual_backend_actual": self.backend_diagnostics.get(
                "residual_backend_actual", "none" if self.residual_backend is None else self.residual_backend.name),
            "residual_backend_version": getattr(self.residual_backend, "version", "none"),
            "flow_provider": "none" if self.flow_provider is None else self.flow_provider.name,
            # The only depth this pipeline can hand a frame: the zero texture the GPU
            # host created at open and never writes.
            "depth_source": "gpu_zero_texture",
            "flow_missing_frames": getattr(self.flow_provider, "missing_frame_count", 0),
            "filter_plan_build_count": filter_plan_build_count,
            "flow_grid": self.job_settings.get("flow_grid", 4),
            "flow_preset": self.job_settings.get("flow_quality", "4M"),
            "decode_wait": decode_wait_ms,
            "working_color_prepare_ms": working_color_prepare_ms,
            "guidance_total": guidance_ms,
            # One key per measured NVOF span, named for the span the bridge timed. A stage
            # the bridge cannot time on its own arrives as "unavailable"; StageProfiler
            # ignores non-numeric values, so it drops out of the summary instead of being
            # averaged as a zero that looks like a fast stage.
            "nvof_upload_ms": g_timing.get("nvof_upload_ms"),
            "nvof_submit_cpu_ms": g_timing.get("nvof_submit_cpu_ms"),
            "nvof_dependency_wait_ms": g_timing.get("nvof_dependency_wait_ms"),
            "nvof_postprocess_wait_inclusive_ms": g_timing.get("nvof_postprocess_wait_inclusive_ms"),
            "nvof_download_ms": g_timing.get("nvof_download_ms"),
            "nvof_total_native_ms": g_timing.get("nvof_total_native_ms"),
            # The byte-map kernel and the synchronize that waits for it, measured only when the
            # bridge was handed BGR24; on an RGBA-input execute it arrives as "unavailable",
            # which StageProfiler drops instead of averaging it as a zero.
            "nvof_input_convert_wait_ms": g_timing.get("nvof_input_convert_wait_ms"),
            # Which of the two orders ran, and why: "auto" that could not provide the reduced
            # transfers is a different fact from "legacy" that was asked for, and a reader of a
            # byte figure has to be able to tell them apart from the record alone.
            "transfer_mode_requested": self.transfer_diagnostics.get("transfer_mode_requested"),
            "transfer_mode_actual": self.transfer_diagnostics.get("transfer_mode_actual"),
            "transfer_mode_fallback_reason": self.transfer_diagnostics.get("transfer_mode_fallback_reason"),
            "nvof_bgr_staging_alloc_count_delta": g_timing.get("nvof_bgr_staging_alloc_count_delta"),
            "nvof_bgr_input_call_count_delta": g_timing.get("nvof_bgr_input_call_count_delta"),
            "nvof_input_convert_launch_count_delta": g_timing.get("nvof_input_convert_launch_count_delta"),
            "dlss_process": dlss_process_ms,
            "residual_pyramid": residual_ms,
            "output_convert": output_convert_ms,
            "cpu_previous_frame_copy_bytes": g_timing.get("cpu_previous_frame_copy_bytes", 0),
            # The service publishes these keys only when it read the bridge counter block,
            # and it publishes None there when the read failed or no session existed. A
            # default here would turn an absent measurement into a reported zero, which is
            # indistinguishable from a frame that really downloaded nothing.
            "nvof_h2d_bytes_delta": g_timing.get("nvof_h2d_bytes_delta"),
            "nvof_d2h_bytes_delta": g_timing.get("nvof_d2h_bytes_delta"),
            "nvof_h2d_copy_count_delta": g_timing.get("nvof_h2d_copy_count_delta"),
            "nvof_d2h_copy_count_delta": g_timing.get("nvof_d2h_copy_count_delta"),
            "nvof_execute_count_delta": g_timing.get("nvof_execute_count_delta"),
            # 1 while the flow engine keeps previous-frame history on the GPU, so the
            # zero above is read as "no copy was possible", not "nothing was measured".
            "nvof_native_history": 1 if g_timing.get("nvof_native_history") else 0,
            **transfer_fields,
        }
        return final_bgr, record_dict


def verify_export_output(out_path: str, frames: int, probe=None, cv2_probe=None) -> int:
    """Refuse to call an export done until the file on disk says so.

    Both backends hand their finished file to this one check, so a GPU-resident
    export cannot be delivered on looser terms than a piped one. The frame count
    is probed from the container, falling back to OpenCV when the probe cannot
    read it, and an unverifiable file is a failure rather than a success with a
    shrug. ``frames`` is what was asked for; the tolerance is the container's
    rounding, not a licence to truncate. ``probe`` and ``cv2_probe`` are seams
    for the tests, which check the verdict rules without a decoder attached.
    """
    probe = probe or video_info_uncached
    if cv2_probe is None:
        def cv2_probe(path):
            cap = cv2.VideoCapture(path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count

    if not os.path.isfile(out_path):
        raise RuntimeError(f"渲染成品严苛校验失败：目标文件不存在: {out_path}")
    out_size = os.path.getsize(out_path)
    if out_size <= 1024:
        raise RuntimeError(f"渲染成品严苛校验失败：目标文件大小异常 ({out_size} 字节): {out_path}")

    tolerance = max(5, int(round(frames * 0.01)))
    # P2-01: if NEITHER probe yields a positive count the file is
    # unverifiable -> fail loudly instead of marking completed.
    try:
        actual_n, _, _, _ = probe(out_path)
        if actual_n <= 0:
            raise RuntimeError(
                f"渲染成品帧数强校验不通过！ffprobe 无法确认 {out_path} 的有效视频帧数（{actual_n}），拒绝交付！"
            )
        if actual_n + tolerance < frames:
            raise RuntimeError(
                f"渲染成品帧数强校验不通过！预期导出 {frames} 帧，实际生成视频物理仅包含 {actual_n} 帧（存在截断缺损），拒绝标记完成并拒绝交付！"
            )
        return actual_n
    except Exception as probe_err:
        if "渲染成品帧数强校验不通过" in str(probe_err):
            raise
    # If the probe could not read it (e.g. MKV index still writing), verify with cv2
    try:
        real_count = cv2_probe(out_path)
    except Exception as cv_err:
        if "渲染成品帧数强校验不通过" in str(cv_err):
            raise
        real_count = 0
    if real_count <= 0:
        raise RuntimeError(
            f"渲染成品帧数强校验不通过！ffprobe 与 OpenCV 均无法确认 {out_path} 的有效视频帧数，拒绝交付！"
        )
    if real_count + tolerance < frames:
        raise RuntimeError(
            f"渲染成品帧数强校验不通过！预期导出 {frames} 帧，实际物理仅包含 {real_count} 帧，拒绝交付！"
        )
    return real_count


EXPORT_BACKENDS = ("auto", "legacy", "gpu_transfer")

# Which native DLSS host a session is opened on. There is one answer now: the
# project GPU host, reached through D3D12 and shipped as native/dlss_gpu_host.dll.
# A value outside this pair means nothing here and is refused rather than quietly
# served by whatever host happens to be the default.
DLSS_BACKENDS = ("gpu_flow_d3d12",)

# The hosts this build cannot run and will not pretend to run. dlssnr_host.dll was
# retired along with its ctypes wrapper, its packaging entry and its frontend
# selector, so a caller that still names it is reading a settings file, a saved job
# or a script from before that change. It gets the name of what is gone and the one
# setting that replaces it - never a frame drawn on a host it did not ask for.
RETIRED_DLSS_BACKENDS = ("host_pointer",)


def retired_dlss_backend_message(requested: str) -> str:
    """The refusal for a DLSS host this build no longer ships, with the migration.

    Written once here so the API boundary, the queue and the preview all answer the
    same retired name with the same words.
    """
    return (
        f"dlss_backend={requested!r} 已被移除：旧宿主 dlssnr_host.dll（连同 dlss_engine.py "
        "封装、打包清单条目和界面选项）不再随本版本发布，本进程无法加载它，也不会把帧悄悄交给"
        "另一个宿主。迁移方式：删除该设置，或显式设 dlss_backend=gpu_flow_d3d12 —— "
        "项目 GPU 宿主 native/dlss_gpu_host.dll 现在服务两种引导模式（0/1）、"
        "CPU 与设备光流，以及预览、静帧、视频渲染与两条导出链路。")


def export_gpu_transfer_stream(video, live, frames, fps, cq, with_audio, out_path,
                               transcode_settings, job_settings, flow_provider,
                               lock=None, progress=None, cancel=None,
                               on_proc_started=None, on_proc_exited=None,
                               diagnostics_out=None, profiler=None,
                               transfer_collector=None, module=None,
                               hdr_module=None):
    """Run one export on the GPU-resident chain, or refuse it and say why.

    The whole of the branch is one ffprobe, one library load and one negotiation
    before a single pixel moves, so a task the chain cannot serve is known before
    anything is committed. There is deliberately no fall-through to the pipe:
    a caller who asked for the transfers to be gone has no way to tell that they
    were not if the refusal turns into a successful legacy export.

    ``module`` exists for the tests, which negotiate against a stub instead of
    loading five real bridges.

    One job setting routes out of this function before any of that: a job that
    explicitly opted into the experimental HDR chain is handed to
    ``export_gpu_hdr_transfer_stream``, which negotiates against that chain's own
    gate. The SDR path below is reached by every other job, exactly as before.
    """
    # Direct callers of this branch get the same refusal as the pipe, and for the
    # same reason: the chain's final copy pass muxes with FFmpeg -y onto out_path
    # while reading the source for the streams the bridge could not carry.
    assert_output_not_input(video, out_path)
    gtx = module or __import__("gpu_transfer_export")
    js = dict(job_settings or {})
    _, source_fps, w, h = video_info(video)
    import native_effects
    js = native_effects.execution_settings(js)
    nr_w, nr_h = native_effects.processing_size(js, (w, h))
    live_w, live_h = resolve_working_dims(nr_w, nr_h, native_effects.core_working_scale(js))
    dst_w, dst_h = resolve_output_dims(w, h, transcode_settings)
    dims = {"src_w": w, "src_h": h, "live_w": live_w, "live_h": live_h,
            "dst_w": dst_w, "dst_h": dst_h, "fps": fps or source_fps}
    import native_effects
    if str((transcode_settings or {}).get('resolution_mode', 'original')) == 'original':
        dims['dst_w'], dims['dst_h'] = native_effects.output_size(js, (w, h))

    htx = hdr_module or __import__("gpu_hdr_transfer_export")
    if 'truehdr' in native_effects.stages(js) and probe_pixel_info(video).get('color_transfer') == 'smpte2084':
        # TrueHDR explicitly requests a new HDR output. Its HDR source must first
        # enter the existing FP16/PQ decoder, then be mapped to SDR at that stage.
        js['experimental_hdr_scrgb'] = True
    sdr10 = (not htx.hdr_requested(js) and
             getattr(htx, "sdr10_source", lambda source: False)(probe_pixel_info(video)))
    if htx.hdr_requested(js) or sdr10:
        # The HDR chain's gate is a different gate: it insists on a 10-bit PQ source,
        # an FP16 Feature 18 session and an encoder that carries HDR10 metadata, and it
        # refuses the SDR chain's own 8-bit rules would accept. Running this
        # negotiation on such a job would refuse the job for being HDR.
        return export_gpu_hdr_transfer_stream(
            video=video, live=live, dims=dims, frames=frames, fps=fps, cq=cq,
            with_audio=with_audio, out_path=out_path,
            transcode_settings=transcode_settings, job_settings=js,
            flow_provider=flow_provider, lock=lock, progress=progress,
            cancel=cancel, on_proc_started=on_proc_started,
            on_proc_exited=on_proc_exited, diagnostics_out=diagnostics_out,
            profiler=profiler, transfer_collector=transfer_collector,
            module=htx, transfer_module=gtx,
            **({"colour_mode": "sdr10_bt709"} if sdr10 else {}))

    # Which bridges this job can even load is part of what its mode asks for:
    # guidance_mode=1 needs an optical flow engine, guidance_mode=0 takes its
    # motion from the DLSS host's resident zero texture and so has no reason to
    # meet one. Asked of the same helper the gate reads the mode with, so the
    # load and the refusal cannot disagree about what was requested.
    requires_flow = gtx.guidance_requires_flow(js)
    bridges, bridge_error = None, None
    try:
        if str(js.get("reconstruction_mode", "single")) in ("dual", "triple"):
            bridges = gtx.load_bridges(requires_flow=requires_flow,
                                       requires_multiscale=True)
        else:
            bridges = gtx.load_bridges(requires_flow=requires_flow)
    except Exception as load_error:
        bridge_error = str(load_error)

    source, source_error = {}, None
    try:
        source = gtx.probe_source(video)
    except Exception as probe_error:
        source_error = f"source probe failed: {probe_error}"

    negotiation = gtx.negotiate(
        dims, source, js, transcode_settings, cq, with_audio, out_path, live,
        option_available=(bridges or {}).get("option_available"),
        bridges_ok=bridges is not None, bridge_error=bridge_error,
        flow_provider=flow_provider,
        tune_available=(bridges or {}).get("tune_available"),
        capabilities=(bridges or {}).get("capabilities"),
        extra_reasons=[source_error] if source_error else None)

    if not negotiation["supported"]:
        detail = "; ".join(negotiation["reasons"])
        if diagnostics_out is not None:
            diagnostics_out.update({
                "export_backend_requested": "gpu_transfer",
                "export_backend_actual": "refused",
                "export_backend_fallback_reason": detail,
                "export_backend_refusal_reasons": list(negotiation["reasons"]),
            })
        raise RuntimeError(
            f"GPU 显存直传导出后端 (export_backend=gpu_transfer) 拒绝执行本任务："
            f"{detail}。未回退到 legacy 管道：静默改用 FFmpeg 管道会让调用方以为"
            f"帧确实留在了显存里。"
        )

    finished, report = gtx.export_gpu_transfer(
        video=video, live=live, flow_provider=flow_provider, out_path=out_path,
        dims=dims, frames=frames, cq=cq, with_audio=with_audio,
        transcode_settings=transcode_settings, job_settings=js,
        negotiation=negotiation, progress=progress, cancel=cancel,
        on_proc_started=on_proc_started, on_proc_exited=on_proc_exited,
        diagnostics_out=diagnostics_out, profiler=profiler,
        transfer_collector=transfer_collector, lock=lock,
        # The set the negotiation already asked this mode for and read its capability
        # answers out of. Left unsent, the chain loads a second set and the bridges that
        # decided this run's shape are orphaned - and a guided load probes an optical flow
        # engine, so the work is paid for twice.
        bridges=bridges)
    print("[gpu transfer] " + json.dumps(report, default=str), flush=True)
    return finished


def export_gpu_hdr_transfer_stream(video, live, dims, frames, fps, cq, with_audio,
                                   out_path, transcode_settings, job_settings,
                                   flow_provider, lock=None, progress=None,
                                   cancel=None, on_proc_started=None,
                                   on_proc_exited=None, diagnostics_out=None,
                                   profiler=None, transfer_collector=None,
                                   module=None, transfer_module=None, colour_mode="hdr10"):
    """Run one experimental HDR10 export on the GPU-resident chain, or refuse it.

    The same shape as ``export_gpu_transfer_stream`` - one ffprobe, one library load,
    one capability read and one negotiation before a pixel moves - against the HDR
    chain's own gate, which asks different questions of the source, of the Feature 18
    session and of the encoder. The refusal text is the chain's list of reasons and
    nothing is softened into a fallback: a job that asked for HDR10 and did not get it
    has to be readable as a failed HDR10 job, because an 8-bit frame drawn from a PQ
    source looks exactly like a successful HDR export in every other field.

    ``module`` is the test hook, for the same reason the SDR entry has one, and
    ``transfer_module`` is the same seam for the bridge set this chain adds entries to.

    Ownership of what this function loads is its own until the chain takes it: the
    bridge set carries the DLL-directory cookies that made its libraries resolvable,
    ``export_gpu_hdr_transfer`` releases them in its ``finally``, and a job that never
    reaches that hand-over - refused by the gate, or raised on the way to it - releases
    them here rather than leaving a handle behind per refusal.
    """
    assert_output_not_input(video, out_path)
    htx = module or __import__("gpu_hdr_transfer_export")
    # The bridge set comes from the SDR module's loader, so the same injected seam that
    # stands in for the HDR chain also stands in for the loading of the libraries.
    gtx = transfer_module or __import__("gpu_transfer_export")
    js = dict(job_settings or {})
    sdr10 = colour_mode == "sdr10_bt709"

    # The bridge set is shared: the HDR chain adds its entries onto the libraries the
    # GPU-transfer backend already loaded, in the load order that set proved, so a job
    # that could not load them fails here with their reason rather than at a frame.
    requires_flow = gtx.guidance_requires_flow(js)
    bridges, bridge_error = None, None
    try:
        if str(js.get("reconstruction_mode", "single")) in ("dual", "triple"):
            # A second layer is a second set of libraries, not a second call: the
            # multiscale observation bridge and the typed solver are only loaded for
            # a job that named more than one layer, so a single-layer HDR export
            # still loads exactly the set it always loaded.
            bridges = gtx.load_bridges(requires_flow=requires_flow,
                                       requires_multiscale=True)
        else:
            bridges = gtx.load_bridges(requires_flow=requires_flow)
    except Exception as load_error:
        bridge_error = str(load_error)

    source, source_error = {}, None
    try:
        # The HDR route asks for two facts the shared probe does not report by default:
        # the real chapter count from -show_chapters, because format.nb_chapters is
        # missing on a Matroska file that has chapters and an admission built on it would
        # drop them, and the video stream's own tags and disposition, because the encoder
        # writes a stream that cannot inherit them. The SDR entry above asks for neither
        # and probes exactly what it always probed.
        source = gtx.probe_source(video, include_chapters=True,
                                  include_video_metadata=True)
    except Exception as probe_error:
        source_error = f"source probe failed: {probe_error}"

    hdr_caps = {}
    if bridges is not None:
        try:
            capability_reader = htx.load_sdr10_capabilities if sdr10 else htx.load_hdr_capabilities
            hdr_caps = capability_reader(
                bridges, int(dims["src_w"]), int(dims["src_h"]),
                **({'codec': transcode_settings['codec']} if transcode_settings and
                   transcode_settings.get('codec') == 'av1_nvenc' else {}))
        except Exception as caps_error:
            hdr_caps = {"missing": [f"the HDR entries could not be read: {caps_error}"]}

    negotiation = None
    try:
        gate = htx.negotiate_sdr10 if sdr10 else htx.negotiate_hdr
        negotiation = gate(
            dims, source, js, transcode_settings, cq, with_audio, out_path, live,
            hdr_caps=hdr_caps,
            option_available=(bridges or {}).get("option_available"),
            bridges_ok=bridges is not None, bridge_error=bridge_error,
            flow_provider=flow_provider,
            tune_available=(bridges or {}).get("tune_available"),
            extra_reasons=[source_error] if source_error else None)
    except BaseException:
        # Nothing below this line has taken the bridge set, so its loader cookies are
        # still this function's to give back - and an exception on the way to a
        # negotiation is not a reason to leak one per attempt.
        htx.release_loader_cookies(bridges)
        raise

    if not negotiation["supported"]:
        # Same ownership: the refusal raises, the chain never sees the set, and the
        # export's own finally - which is what releases cookies on a run that gets a
        # frame - never runs.
        htx.release_loader_cookies(bridges)
        detail = "; ".join(negotiation["reasons"])
        if diagnostics_out is not None:
            diagnostics_out.update({
                "export_backend_requested": "gpu_transfer",
                "export_backend_actual": "refused",
                "export_backend_fallback_reason": detail,
                "export_backend_refusal_reasons": list(negotiation["reasons"]),
                "transfer_mode_requested": "gpu_sdr10_transfer" if sdr10 else "gpu_hdr_transfer",
                "transfer_mode_actual": "refused",
                "experimental_hdr_scrgb": not sdr10,
                "sdr10_negotiation" if sdr10 else "hdr_negotiation": negotiation,
                "sdr10_capabilities" if sdr10 else "hdr_capabilities": hdr_caps,
            })
        if sdr10:
            raise RuntimeError(f"GPU SDR 10-bit 链路拒绝执行本任务：{detail}。未降为 8-bit，未切换导出链路。")
        raise RuntimeError(
            f"GPU 显存直传 {'SDR 10-bit' if sdr10 else 'HDR10'} 链路 (export_backend=gpu_transfer + "
            f"experimental_hdr_scrgb=true) 拒绝执行本任务：{detail}。未回退到 SDR 显存"
            "链路，也未回退到 legacy 管道：用 8-bit 通路导出的 PQ 源在每一个字段上都会"
            "看起来像一次成功的 HDR 导出。"
        )

    finished, report = htx.export_gpu_hdr_transfer(
        video=video, live=live, flow_provider=flow_provider, out_path=out_path,
        dims=dims, frames=frames, cq=cq, with_audio=with_audio,
        transcode_settings=transcode_settings, job_settings=js,
        negotiation=negotiation, progress=progress, cancel=cancel,
        on_proc_started=on_proc_started, on_proc_exited=on_proc_exited,
        diagnostics_out=diagnostics_out, profiler=profiler,
        transfer_collector=transfer_collector, lock=lock, bridges=bridges,
        hdr_caps=hdr_caps)
    print("[gpu hdr transfer] " + json.dumps(report, default=str), flush=True)
    return finished


def export_dlss_stream(video, live, lock=None, frames=None, fps=None, cq=24, with_audio=True,
                       flow_provider=None, progress=None, cancel=None,
                       out_path=None, transcode_settings=None,
                       on_proc_started=None, on_proc_exited=None,
                       job_settings=None, profiler=None,
                       diagnostics_out=None, transfer_collector=None):
    """Streaming export: frames are rendered on CPU and piped to an FFmpeg NVENC process.

    What actually happens per frame: a decoder thread yields decoded frames, this thread
    renders them through ProductionCore, and the resulting BGR24 buffer is written in 1 MiB
    chunks into the FFmpeg child process' stdin, which FFmpeg reads back as rawvideo and
    hands to the encoder. The encoding itself is NVENC hardware work, and nothing is ever
    written to disk as intermediate images, which is the point of the pipe. The pipe is
    not a GPU path: every rendered frame crosses CPU memory twice, and FFmpeg uploads it
    afterwards on its own. Removing those round trips is a separate piece of work, not a
    property of this function - it is what ``export_backend=gpu_transfer`` in job_settings
    is for, and that branch leaves this function before the pipe is built. On the pipe
    path, decoding runs on a background producer thread so it overlaps the rendering; the
    GPU branch is single threaded because its bridges hand each other borrowed memory.

    What this loop also does is count: the bytes it reads from the decoder, the bytes the
    encoder write returned, and the channel-order conversions the frame performed. Those
    go to transfer_profiler, which keeps them apart from the timings - a duration says how
    long a stage took, a byte count says what crossed a boundary, and only the second one
    can show a round trip being removed.
    """
    import queue

    # A caller that named a destination is answered before this function probes the
    # source at all; one that did not is answered below, where the default is derived.
    if out_path:
        assert_output_not_input(video, out_path)

    n, v_fps, w, h = video_info(video)
    if fps is None:
        fps = v_fps
    if frames is None:
        frames = n

    ts_exp = transcode_settings or {}
    dst_w, dst_h = resolve_output_dims(w, h, transcode_settings)
    decode_mode = str(ts_exp.get("decode_mode", "auto"))

    if not out_path:
        base = os.path.dirname(video)
        stem = os.path.splitext(os.path.basename(video))[0]
        ext = transcode_settings.get("container", ".mkv") if transcode_settings else ".mkv"
        if not ext.startswith("."):
            ext = "." + ext
        out_path = os.path.join(base, stem + "_dlss" + ext)

    # Checked here, at the one point both chains pass through, so a destination that
    # is the source is refused before the branch below and before FFmpeg is spawned:
    # the -y this builder emits when the job allows overwriting truncates the file
    # the decoder is still reading.
    assert_output_not_input(video, out_path)

    # Opt-in backend switch. Everything below this point is the FFmpeg stdin pipe;
    # the branch happens before the encoder process is built because the GPU chain
    # has no pipe to feed. "auto" is legacy on purpose: the backend is opt-in, and
    # a job that did not ask for it must not find itself on a path it was not
    # tested against.
    requested_backend = str((job_settings or {}).get("export_backend") or "auto").lower()
    from post_control import controls_from_settings
    if not controls_from_settings(job_settings).neutral and requested_backend != "gpu_transfer":
        raise ValueError("GPU residual controls require export_backend=gpu_transfer; no fallback")
    from temporal_control import temporal_from_settings
    if temporal_from_settings(job_settings).enabled and requested_backend != "gpu_transfer":
        raise ValueError("GPU scene/temporal controls require export_backend=gpu_transfer; no fallback")
    if requested_backend not in EXPORT_BACKENDS:
        raise ValueError(
            f"export_backend={requested_backend!r} is not a backend this pipeline "
            f"offers; the choices are {', '.join(repr(b) for b in EXPORT_BACKENDS)}")
    if (str((job_settings or {}).get("reconstruction_mode", "single")) in
            ("dual", "triple") and requested_backend != "gpu_transfer"):
        raise ValueError("Dual / Triple independent layers require export_backend=gpu_transfer")
    if requested_backend == "gpu_transfer":
        return export_gpu_transfer_stream(
            video=video, live=live, frames=frames, fps=fps, cq=cq,
            with_audio=with_audio, out_path=out_path,
            transcode_settings=transcode_settings, job_settings=job_settings,
            flow_provider=flow_provider, lock=lock, progress=progress,
            cancel=cancel, on_proc_started=on_proc_started,
            on_proc_exited=on_proc_exited, diagnostics_out=diagnostics_out,
            profiler=profiler, transfer_collector=transfer_collector)
    if diagnostics_out is not None:
        diagnostics_out.update({
            "export_backend_requested": requested_backend,
            "export_backend_actual": "legacy_pipe",
            "export_backend_fallback_reason": None if requested_backend == "legacy"
            else "gpu_transfer_is_opt_in: set export_backend=gpu_transfer to request it",
        })

    cmd = build_transcode_command(
        video=video,
        out_path=out_path,
        w=w,
        h=h,
        fps=fps,
        transcode_settings=transcode_settings,
        cq=cq,
        with_audio=with_audio
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if callable(on_proc_started):
        try:
            on_proc_started(proc)
        except Exception:
            pass

    # Clean Producer/Consumer queue: Producer only pushes raw decoded (idx, fr)
    # to avoid buffer sharing and aliasing across queue items.
    frame_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()

    decode_info = {}
    def _producer():
        try:
            last_read_idx = -1
            for idx, fr in iter_frames(video, decode_mode=decode_mode,
                                       stop_event=stop_event, cancel=cancel,
                                       on_proc_started=on_proc_started,
                                       on_proc_exited=on_proc_exited,
                                       decode_info=decode_info):
                if stop_event.is_set() or (cancel and cancel()):
                    break
                if idx >= frames:
                    break
                if fr is None:
                    raise RuntimeError(f"视频解码器在第 {idx} 帧返回空帧，素材可能损坏或解码异常中断")
                last_read_idx = idx

                while not stop_event.is_set():
                    if cancel and cancel():
                        return
                    try:
                        frame_queue.put((idx, fr), timeout=0.05)
                        break
                    except queue.Full:
                        continue
            if not stop_event.is_set() and not (cancel and cancel()) and (last_read_idx + 1) < frames:
                shortfall = frames - (last_read_idx + 1)
                tolerance = max(5, int(round(frames * 0.01)))
                if shortfall <= tolerance:
                    print(f"[pipeline] 输入帧数估计 {frames} 与实际可解码 {last_read_idx + 1} 相差 {shortfall} 帧 (容差 {tolerance})，视为容器舍入误差，继续导出")
                else:
                    raise RuntimeError(f"输入视频解码提前终止！预期读取 {frames} 帧，但在第 {last_read_idx + 1} 帧结束（输入文件可能损坏或解码中断）")
        except Exception as e:
            # A decoder that dies because its pipe was closed by the cancel is part of
            # the cancellation, not a real failure; without the flag set it still is one.
            frame_queue.put(_cancel_or(e, stop_event, cancel))
        finally:
            frame_queue.put(None)

    producer_t = threading.Thread(target=_producer, name="VideoDecoderProducer", daemon=True)
    producer_t.start()

    passes = live if isinstance(live, (list, tuple)) else [live]
    core = ProductionExportCore(
        src_w=w, src_h=h, dst_w=dst_w, dst_h=dst_h,
        passes=passes, job_settings=job_settings,
        flow_provider=flow_provider,
        profiler=profiler, lock=lock
    )

    # Read once, here, rather than after the last frame: which transfer order this run is in
    # is decided at construction and nothing changes it, so a record written by a cancelled
    # or failing run still says what ran.
    if diagnostics_out is not None:
        diagnostics_out.update(core.transfer_diagnostics)

    if profiler is None:
        profiler = StageProfiler()
    if transfer_collector is None:
        transfer_collector = transfer_profiler.TransferProfiler()
    profiler.start()

    processed = 0
    last_idx = None
    t0 = time.time()
    last_frame_t = time.time()
    WATCHDOG_TIMEOUT_SEC = 30.0

    try:
        while True:
            if cancel and cancel():
                stop_event.set()
                raise JobCancelled("用户已取消任务")

            # Watchdog: detect complete pipeline stall / driver hang
            if time.time() - last_frame_t > WATCHDOG_TIMEOUT_SEC:
                if cancel and cancel():
                    raise JobCancelled("用户已取消任务")
                raise RuntimeError(
                    f"渲染管线在第 {processed} 帧严重停滞（已超过 {WATCHDOG_TIMEOUT_SEC} 秒无有效帧产出），"
                    f"显卡 D3D12/Feature18 或编码器可能已发生死锁或崩溃"
                )

            t_wait_0 = time.perf_counter()
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if not producer_t.is_alive():
                    break
                continue
            t_wait_1 = time.perf_counter()
            t_decode_wait = (t_wait_1 - t_wait_0) * 1000.0

            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            idx, fr = item
            is_discontinuous = (last_idx is None or idx != last_idx + 1)
            last_idx = idx

            t_frame_start = time.perf_counter()

            # Execute unified production processing core
            native_phase(f'处理导出第 {idx + 1} 帧', 300)
            final_bgr, record_dict = core.process_frame(
                idx=idx, fr=fr, is_discontinuous=is_discontinuous, decode_wait_ms=t_decode_wait
            )

            # Assert FFmpeg encoder subprocess is still alive before pipe write
            if proc.poll() is not None:
                if cancel and cancel():
                    raise JobCancelled("用户已取消任务")
                raise RuntimeError(f"FFmpeg 编码进程在第 {idx} 帧前已异常退出 (退出码 {proc.returncode})")

            # Production chunked byte write with cancellation detection
            t_pipe_0 = time.perf_counter()
            written = _write_frame_bytes(proc.stdin, final_bgr, chunk_size=1<<20, cancel_fn=cancel, proc=proc, frame_idx=idx)
            t_pipe_1 = time.perf_counter()
            pipe_write_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            transfer_profiler.ENCODER_PIPE.note_frame(idx, written)

            t_frame_end = time.perf_counter()
            frame_total_ms = (t_frame_end - t_frame_start) * 1000.0

            record_dict["pipe_write"] = pipe_write_ms
            record_dict["frame_total"] = frame_total_ms
            record_dict["decode_requested"] = decode_mode
            record_dict["decode_actual"] = decode_info.get("actual", decode_mode)
            record_dict.update(transfer_profiler.pipe_fields(
                idx, gpu_decoded=(record_dict["decode_actual"] == "cuda")))

            transfer_collector.record(record_dict)
            profiler.record(record_dict)

            processed += 1
            last_frame_t = time.time()

            if progress:
                elapsed = time.time() - t0
                fps_cur = processed / max(0.001, elapsed)
                rem_sec = int((frames - processed) / max(0.1, fps_cur))
                rem_str = f"{rem_sec // 3600:02d}:{(rem_sec % 3600) // 60:02d}:{rem_sec % 60:02d}"
                progress(processed, frames, f"[流式导出 DLSS] {processed}/{frames} ({processed/frames*100:.1f}%) | {fps_cur:.1f} fps | 剩余 {rem_str}")
    finally:
        profiler.stop()
        profiler.print_summary(f"DLSS STREAM EXPORT PROFILER ({processed} FRAMES)")
        stop_event.set()
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        producer_t.join(timeout=2.0)
        if core.residual_backend is not None:
            backend_diagnostics = core.residual_backend.diagnostics()
            print("[residual backend] " + json.dumps(backend_diagnostics, default=str), flush=True)
            if diagnostics_out is not None:
                diagnostics_out.update(backend_diagnostics)
        core.close()
        native_phase('等待 FFmpeg 编码结束', None)
        if callable(on_proc_exited):
            try:
                on_proc_exited(proc)
            except Exception:
                pass

        err = ""
        if proc.poll() is None:
            if cancel and cancel():
                try:
                    proc.kill()
                except Exception:
                    pass
            elif proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                _, stderr_data = proc.communicate(timeout=5 if (cancel and cancel()) else 10)
                err = stderr_data.decode("utf-8", "replace") if stderr_data else ""
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    _, stderr_data = proc.communicate()
                    err = stderr_data.decode("utf-8", "replace") if stderr_data else ""
                except Exception:
                    pass
        else:
            try:
                _, stderr_data = proc.communicate(timeout=2)
                err = stderr_data.decode("utf-8", "replace") if stderr_data else ""
            except Exception:
                pass
        rc = proc.returncode
        if isinstance(rc, int) and rc != 0 and not (cancel and cancel()):
            raise RuntimeError(f"FFmpeg 流式导出失败 (rc={rc})\n{err[-600:]}")

    # Cancelled jobs skip completeness checks and file verification entirely.
    if cancel and cancel():
        raise JobCancelled("用户已取消任务")

    if not (cancel and cancel()) and processed < frames:
        shortfall = frames - processed
        tolerance = max(5, int(round(frames * 0.01)))
        if shortfall > tolerance:
            raise RuntimeError(f"渲染未完整完成！预期处理 {frames} 帧，实际仅送入 {processed} 帧，已中止以防生成缺损视频")

    # Strict End-to-End Output File Verification
    if processed > 0:
        verify_export_output(out_path, frames)
        return out_path
    return None


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--flow", action="store_true", help="run flow (cached)")
    ap.add_argument("--export", type=str, default=None, choices=["flow", "dlss"],
                    help="export a preview video")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--edge", type=int, default=720,
                    help="clamp SHORT edge for flow compute (short edge=720p; default 720)")
    args = ap.parse_args()

    n, fps, w, h = video_info(args.video)
    print(f"[info] {args.video}: {n} frames @ {fps:.2f} fps, {w}x{h}, device={DEVICE}")

    if args.flow:
        t0 = time.time()
        generate_flow(args.video, args.frames, force=args.force, edge=args.edge)
        print(f"[flow] done ({time.time()-t0:.1f}s) -> {out_dirs(args.video)[0]}")

    if args.export:
        p = export_video(args.video, args.export, args.frames, fps)
        print(f"[export] {args.export} -> {p}")


if __name__ == "__main__":
    main()
