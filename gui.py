#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py — tkinter frontend for the test3 DLSS tool.

Pipeline: 导入视频 → 生成光流 → 运行 DLSS (Feature 18) → 导出视频 (flow/dlss).

Preview: 右下角视图切换 (原图/光流/DLSS) + 帧进度条，预览窗保持长宽比(letterbox)。

Run:  python3 gui.py
"""
import hashlib
import os
import queue
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

import cv2
import numpy as np

import pipeline
import dlss_gpu_engine
from dlss_errors import DLSSInferenceError

VIEWS = ["原图", "光流", "DLSS", "对比"]


class DlssHostUnavailable(RuntimeError):
    """No DLSS host can serve still images: the GPU host is missing, is ABI 2, or failed to open."""


class _PreviewWorker:
    """Background worker for rendering preview frames asynchronously.
    Prevents 4K video decoding and DLSS processing from freezing the Tkinter GUI thread.
    """
    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending = None  # (frame, view, split_x, cw, ch, settings, from_play, req_time)
        self._stopped = False
        self._paused = False
        self._live_cache = None
        self._last_dlss_frame = -1
        self._live_session = None
        self.last_error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="PreviewWorker")
        self._thread.start()

    def request(self, frame, view, split_x, cw, ch, settings, from_play=False, zoom_1x=False, pan_xy=(0.5, 0.5)):
        with self._cond:
            self._pending = (frame, view, split_x, cw, ch, settings, from_play, time.time(), zoom_1x, pan_xy)
            self._cond.notify_all()

    def pause(self):
        with self._cond:
            self._paused = True
            self._cond.notify_all()

    def resume(self):
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def close_live(self):
        self._live_cache = None
        self._last_dlss_frame = -1
        self._live_session = None

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def _live_dlss_image(self, frame, settings):
        """One video preview frame through the GPU host, in the mode guidance asks for.

        The session lookup and the frame call share one _dlss_lock hold: which of the
        host's two entries a frame goes through is decided by the same settings that
        chose the session, so a lookup that released the lock would let another route
        take the slot and close the session this frame is about to draw on.
        """
        sk = (settings['preset'], settings['style'], settings['intensity'], settings['local_tone'],
              settings['local_struct'], settings['skin_struct'], settings['use_auto_mask'],
              settings['ui_correction'], settings['guidance_mode'],
              settings['motion_scale_x'], settings['motion_scale_y'])
        self.last_error = None
        if self._live_cache and self._live_cache[0] == frame and self._live_cache[1] == sk:
            return self._live_cache[2]

        gm = settings['guidance_mode']
        need_flow = gm == 1
        fm, _ = pipeline.out_dirs(self.app.video)
        fpath = os.path.join(fm, f"{frame:06d}.flo") if need_flow else None
        if need_flow and not os.path.exists(fpath):
            self._live_cache = None
            return None

        cap = getattr(self.app, '_cap', None)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, fr = cap.read()
        if not ok or fr is None:
            return None

        h, w = fr.shape[:2]
        rgba = cv2.cvtColor(fr, cv2.COLOR_BGR2RGBA)
        # A guided frame hands the host the cached .flo plane itself, at the size the
        # session is opened: the card narrows float32 to the bound motion texture, so
        # nothing here copies, scales or zeroes it. An unguided frame gets no plane at
        # all, which is the whole point of the zero-motion entry.
        flow = None
        if need_flow:
            flow = pipeline.read_flo(fpath)
            if flow.shape[:2] != (h, w):
                flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)

        kind = self.app.video_preview_kind(settings)
        try:
            with self.app._dlss_lock:
                live = self.app._get_or_create_session_locked(w, h, settings, kind)
                if live is None:
                    return None
                # A seek, a jump, or a session someone else rebuilt all start a new
                # temporal sequence; only a continuing one on the same session does not.
                reset = (frame != self._last_dlss_frame + 1) or (live is not self._live_session)
                o = (live.process_host_motion(rgba, flow, reset=reset) if need_flow
                     else live.process_zero_motion(rgba, reset=reset))
        except Exception as ex:
            # A frame that the host cannot produce is an exception, not a silent None;
            # a preview says what failed and shows the fallback frame rather than
            # stopping the stream.
            self.last_error = str(ex)
            traceback.print_exc()
            self.app.logln(f"[预览 DLSS] {ex}")
            self._live_cache = None
            self._live_session = None
            self._last_dlss_frame = -1
            return None

        self._live_session = live
        self._last_dlss_frame = frame
        if o is None:
            self._live_cache = None
            return None
        bgr = cv2.cvtColor(o, cv2.COLOR_RGBA2BGR)
        self._live_cache = (frame, sk, bgr)
        return bgr

    def _load_raw(self, view, frame, settings):
        if not self.app.video:
            return None
        fm, lm = pipeline.out_dirs(self.app.video)
        if view == "原图":
            cap = getattr(self.app, "_cap", None)
            if cap is None:
                cap = pipeline.open_capture(self.app.video)
                self.app._cap = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, f = cap.read()
            return f if ok else None
        if view == "光流":
            p = os.path.join(fm, f"{frame:06d}.flo")
            if os.path.exists(p):
                return pipeline.colorize_flow(pipeline.read_flo(p))
        if view == "DLSS":
            img = self._live_dlss_image(frame, settings)
            if img is not None:
                return img
            p = os.path.join(lm, f"{frame:06d}.png")
            if os.path.exists(p):
                return pipeline.imread_u(p)
        return None

    def _run(self):
        while True:
            with self._cond:
                while (self._pending is None or self._paused) and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    break
                req = self._pending
                self._pending = None

            frame, view, split_x, cw, ch, settings, from_play, req_time, zoom_1x, pan_xy = req
            try:
                if self._stopped or self._paused:
                    continue
                res = self._render(frame, view, split_x, cw, ch, settings, zoom_1x=zoom_1x, pan_xy=pan_xy)
                if res is not None:
                    with self._cond:
                        if self._pending is not None:
                            continue
                    elapsed = time.time() - req_time
                    self.app.post_ui("preview", res, from_play, elapsed)
            except Exception as ex:
                traceback.print_exc()
                self.app.logln(f"[预览异常] {ex}")

    def _render(self, frame, view, split_x, cw, ch, settings, zoom_1x=False, pan_xy=(0.5, 0.5)):
        if not self.app.video:
            return None
        if view == "对比":
            orig = self._load_raw("原图", frame, settings)
            dlss = self._load_raw("DLSS", frame, settings)
            if orig is None:
                return {"type": "msg", "text": f"帧 {frame} 原图读取失败"}
            if dlss is None:
                ih, iw = orig.shape[:2]
                if zoom_1x:
                    crop_w, crop_h = min(cw, iw), min(ch, ih)
                    px, py = pan_xy
                    x0 = max(0, min(int(px * iw - crop_w / 2), iw - crop_w))
                    y0 = max(0, min(int(py * ih - crop_h / 2), ih - crop_h))
                    nimg = orig[y0:y0 + crop_h, x0:x0 + crop_w]
                    nw, nh = crop_w, crop_h
                else:
                    scale = min(cw / iw, ch / ih)
                    nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
                    interp = cv2.INTER_AREA if (nw < iw and nh < ih) else cv2.INTER_LANCZOS4
                    nimg = cv2.resize(orig, (nw, nh), interpolation=interp)
                ox, oy = (cw - nw) // 2, (ch - nh) // 2
                return {"type": "image", "frame": frame, "view": view, "split": False,
                        "bgr": nimg, "ox": ox, "oy": oy, "nw": nw, "nh": nh,
                        "msg": "先运行 DLSS 才能对比 (或将引导模式设为关闭)"}
            ih, iw = orig.shape[:2]
            if zoom_1x:
                crop_w, crop_h = min(cw, iw), min(ch, ih)
                px, py = pan_xy
                x0 = max(0, min(int(px * iw - crop_w / 2), iw - crop_w))
                y0 = max(0, min(int(py * ih - crop_h / 2), ih - crop_h))
                so = orig[y0:y0 + crop_h, x0:x0 + crop_w]
                sd = dlss[y0:y0 + crop_h, x0:x0 + crop_w]
                nw, nh = crop_w, crop_h
            else:
                scale = min(cw / iw, ch / ih)
                nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
                interp = cv2.INTER_AREA if (nw < iw and nh < ih) else cv2.INTER_LANCZOS4
                so = cv2.resize(orig, (nw, nh), interpolation=interp)
                sd = cv2.resize(dlss, (nw, nh), interpolation=interp)
            o = so.copy()
            sx = int(split_x * nw)
            o[:, sx:] = sd[:, sx:]
            o[:, max(sx - 1, 0):min(sx + 1, nw)] = [0, 255, 255]
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            return {"type": "image", "frame": frame, "view": view, "split": True,
                    "bgr": o, "ox": ox, "oy": oy, "nw": nw, "nh": nh,
                    "split_orig": so, "split_dlss": sd,
                    "msg": "🔍 1:1 原画细节模式 (按住右键拖动视野，双击切换)" if zoom_1x else None}

        img = self._load_raw(view, frame, settings)
        if img is None:
            msg = f"{view}：帧 {frame} 尚未生成 (先点对应按钮)"
            if view == "DLSS":
                # A frame the GPU host refused or failed says so on the canvas; the
                # hint about missing flow only means something when nothing failed.
                msg = ("DLSS 预览失败：" + self.last_error if self.last_error else
                       "DLSS 预览：请先生成光流，或把【引导模式】设为【关闭】即可实时生成")
            return {"type": "msg", "text": msg}

        ih, iw = img.shape[:2]
        if zoom_1x:
            crop_w, crop_h = min(cw, iw), min(ch, ih)
            px, py = pan_xy
            x0 = max(0, min(int(px * iw - crop_w / 2), iw - crop_w))
            y0 = max(0, min(int(py * ih - crop_h / 2), ih - crop_h))
            nimg = img[y0:y0 + crop_h, x0:x0 + crop_w]
            nw, nh = crop_w, crop_h
        else:
            scale = min(cw / iw, ch / ih)
            nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
            interp = cv2.INTER_AREA if (nw < iw and nh < ih) else cv2.INTER_LANCZOS4
            nimg = cv2.resize(img, (nw, nh), interpolation=interp)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        return {"type": "image", "frame": frame, "view": view, "split": False,
                "bgr": nimg, "ox": ox, "oy": oy, "nw": nw, "nh": nh,
                "msg": "🔍 1:1 原画细节模式 (按住右键拖动视野，双击切换)" if zoom_1x else None}


class App:
    def __init__(self, root):
        self.root = root
        root.title("DLSS5 离线工具 — 原图/光流/DLSS 预览 + 导出")
        root.geometry("1060x840")
        root.minsize(880, 680)
        self.video = None
        self.nframes = 0
        self.fps = 30.0
        self.v_w = 1920
        self.v_h = 1080
        self.thread = None
        self.current_is_image = False   # 单张图片导入模式
        self.image_path = None
        self.image_bgr = None
        self.image_dlss = None
        self._dlss_lock = threading.Lock()
        self._dlss_session = None
        self._dlss_session_kind = None
        self._image_batch_running = False
        # Which kind a running video task holds the single session slot for, so a
        # preview that names the other host cannot close the session mid-task.
        self._video_task_kind = None

        # ---- video row ----
        t = ttk.Frame(root); t.pack(fill="x", padx=8, pady=6)
        ttk.Button(t, text="导入视频", command=self.import_video).pack(side="left")
        ttk.Button(t, text="处理图片文件夹", command=self.process_images).pack(side="left", padx=3)
        ttk.Button(t, text="导入图片", command=self.import_image).pack(side="left", padx=3)
        self.vlabel = ttk.Label(t, text="(未选择视频)", anchor="w")
        self.vlabel.pack(side="left", padx=8, fill="x", expand=True)

        # ---- view + frame row ----
        v = ttk.Frame(root); v.pack(fill="x", padx=8, pady=2)
        ttk.Label(v, text="显示:").pack(side="left")
        self.view_var = tk.StringVar(value="对比")
        self.view_cb = ttk.Combobox(v, textvariable=self.view_var, values=VIEWS,
                                    state="readonly", width=7)
        self.view_cb.pack(side="left", padx=3)
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self.on_view_change())
        self.split_x = 0.5
        self.playing = False
        self._live_debounce = None
        self._pending_export = False
        self._play_after = None

        self.v_zoom_1x = tk.BooleanVar(value=False)
        ttk.Checkbutton(v, text="🔍 1:1 原画细节", variable=self.v_zoom_1x,
                        command=self.display_view).pack(side="left", padx=(10, 6))
        self.pan_x = 0.5
        self.pan_y = 0.5
        self._pan_drag_start = None

        self._drag_nw = 1
        self._drag_nh = 1
        self._drag_offsetx = 0
        self._drag_offsety = 0
        self._split_orig = None
        self._split_dlss = None
        self._split_frame = -1
        self._pilimg = None
        self._photo = None
        self._ui_queue = queue.Queue()
        self._closing = False
        self._poll_ui_queue()
        self._preview_worker = _PreviewWorker(self)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        ttk.Label(v, text="  帧:").pack(side="left")
        self.fslider = tk.Scale(v, from_=0, to=1, orient="horizontal", showvalue=False,
                                resolution=1, length=300, command=lambda e: self.on_frame())
        self.fslider.pack(side="left", padx=3)
        self.flabel = ttk.Label(v, text="0", width=6)
        self.flabel.pack(side="left")
        ttk.Button(v, text="▶ 播放", command=self.play).pack(side="left", padx=(6, 0))
        ttk.Button(v, text="⏸ 暂停", command=self.pause).pack(side="left", padx=3)

        # ---- preview canvas (aspect-ratio preserved via letterbox) ----
        self.canvas = tk.Canvas(root, bg="#161616", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", lambda e: self.display_view())
        self.canvas.bind("<B1-Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-1>", self.on_canvas_motion)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_down)
        self.canvas.bind("<B3-Motion>", self.on_canvas_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_right_up)

        # ---- action buttons ----
        b = ttk.Frame(root); b.pack(fill="x", padx=8, pady=4)
        ttk.Button(b, text="生成光流", command=lambda: self.run_worker("flow")).pack(side="left", padx=3)
        ttk.Button(b, text="运行 DLSS", command=lambda: self.run_worker("dlss")).pack(side="left", padx=3)
        self.stop_btn = ttk.Button(b, text="🛑 停止任务", command=self.stop_worker, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 3))
        ttk.Label(b, text="  (断点自动续算，中途停止无损，缺少数据平滑降级)").pack(side="left")

        # ---- DLSS settings panel ----
        sf = ttk.LabelFrame(root, text="DLSS 设置")
        sf.pack(fill="x", padx=8, pady=4)
        self._build_settings(sf)

        # ---- export row ----
        e = ttk.Frame(root); e.pack(fill="x", padx=8, pady=4)
        ttk.Label(e, text="导出:").pack(side="left")
        self.export_kind = ttk.Combobox(e, values=["DLSS", "光流"], state="readonly", width=8)
        self.export_kind.set("DLSS"); self.export_kind.pack(side="left", padx=3)
        ttk.Label(e, text="CQ:").pack(side="left", padx=(6, 0))
        self.v_export_crf = tk.IntVar(value=24)
        ttk.Spinbox(e, from_=0, to=63, increment=1, width=4, wrap=False,
                    textvariable=self.v_export_crf).pack(side="left", padx=(3, 6))
        ttk.Label(e, text="(0-63, NVENC AV1 CQ, 默认24)").pack(side="left")
        self.v_export_audio = tk.IntVar(value=1)
        ttk.Checkbutton(e, text="保留原声", variable=self.v_export_audio).pack(side="left", padx=(0, 10))
        self.export_btn = ttk.Button(e, text="导出视频", command=self.export)
        self.export_btn.pack(side="left", padx=3)

        # ---- progress ----
        p = ttk.Frame(root); p.pack(fill="x", padx=8, pady=2)
        self.pbar = ttk.Progressbar(p, maximum=100)
        self.pbar.pack(fill="x", expand=True)
        self.status = ttk.Label(p, text="就绪")
        self.status.pack(fill="x")

        # ---- log ----
        self.log = scrolledtext.ScrolledText(root, height=7, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=False, padx=8, pady=4)
        self._update_export_btn()
        self._cached_settings = self._collect_settings()
        self._cached_crf = self._export_crf()
        self._cached_audio = self._export_audio()
        self.v_export_crf.trace_add("write", lambda *_: self._export_crf())
        self.v_export_audio.trace_add("write", lambda *_: self._export_audio())

    # Each kind the project GPU host serves, the capability that says the loaded
    # host has the entry it runs on, what the operator is told when it does not, and
    # the line that names the session that answered. A missing entry point means a
    # stale native build. There is no second host to draw the frame behind the
    # caller's back instead: which host produced a picture is invisible in the picture.
    _GPU_HOST_KINDS = {
        "image": ("supports_zero_motion",
                  "dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
                  "rebuild native/dlss_gpu_host.dll to run images on the GPU host",
                  "[图片 DLSS] GPU 宿主零运动会话已就绪"),
        "gpu_zero_motion": ("supports_zero_motion",
                            "dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
                            "rebuild native/dlss_gpu_host.dll to preview guidance_mode=0 "
                            "on the GPU host",
                            "[预览 DLSS] GPU 宿主零运动会话已就绪"),
        "gpu_host_motion": ("supports_host_motion",
                            "dlss_gpu_host.dll exposes no host-motion entry point; rebuild "
                            "native/dlss_gpu_host.dll to preview guidance_mode=1 on the "
                            "GPU host",
                            "[预览 DLSS] GPU 宿主运动向量会话已就绪"),
        # The streaming export's two entries are the preview's two entries at the same
        # extent for the same reason, and named apart for one reason of the slot's:
        # _video_task_kind only refuses a request that names a *different* kind, so an
        # export that borrowed the preview's name would let a preview frame that slipped
        # past pause() reset the temporal sequence mid-run, or close the session the
        # encoder is being fed from.
        "export_zero_motion": ("supports_zero_motion",
                               "dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
                               "rebuild native/dlss_gpu_host.dll to stream-export "
                               "guidance_mode=0 on the GPU host",
                               "[导出 DLSS] GPU 宿主零运动会话已就绪"),
        "export_host_motion": ("supports_host_motion",
                               "dlss_gpu_host.dll exposes no host-motion entry point; rebuild "
                               "native/dlss_gpu_host.dll to stream-export guidance_mode=1 on "
                               "the GPU host",
                               "[导出 DLSS] GPU 宿主运动向量会话已就绪"),
        # The cached-frame render runs on the same two entries the export does, at the same
        # extent, and is named apart for the same reason the export is named apart from the
        # preview: the slot is shared and keyed by kind, so a render that borrowed the
        # export's name would let one session serve two loops that each believe they own its
        # temporal sequence. The kind is also the only record of which task opened a session,
        # which is what the ready line tells the operator.
        "render_zero_motion": ("supports_zero_motion",
                               "dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
                               "rebuild native/dlss_gpu_host.dll to render guidance_mode=0 "
                               "video frames on the GPU host",
                               "[DLSS 渲染] GPU 宿主零运动会话已就绪"),
        "render_host_motion": ("supports_host_motion",
                               "dlss_gpu_host.dll exposes no host-motion entry point; rebuild "
                               "native/dlss_gpu_host.dll to render guidance_mode=1 video "
                               "frames on the GPU host",
                               "[DLSS 渲染] GPU 宿主运动向量会话已就绪"),
    }

    @classmethod
    def video_preview_kind(cls, settings):
        """The GPU host entry a video preview frame runs on, read off guidance.

        Both motion modes have a GPU host entry now, so moving the preview needed no
        new selector: guidance off asks for the resident zero motion texture, guidance
        on asks for the host-memory plane the cached .flo already holds.
        """
        return "gpu_host_motion" if int(settings.get('guidance_mode', 0)) == 1 \
            else "gpu_zero_motion"

    @classmethod
    def video_export_kind(cls, settings):
        """The GPU host entry a streaming export frame runs on, read off guidance.

        The same two entries the preview uses, asked for by the same control: guidance
        off wants the resident zero motion texture and asks for no motion input at all,
        guidance on wants the plane the cached .flo already holds. It is a separate
        selector only because the session slot is shared and named by kind, which is how
        a running export keeps a preview frame out of its temporal sequence.
        """
        return "export_host_motion" if int(settings.get('guidance_mode', 0)) == 1 \
            else "export_zero_motion"

    @classmethod
    def video_render_kind(cls, settings):
        """The GPU host entry a cached-frame render runs on, read off guidance.

        The third caller of the same two entries: guidance off asks the host for its
        resident zero motion texture and hands it no motion at all, guidance on asks it
        to stage the cached `.flo` plane. Nothing here needed a new control, because the
        panel's guidance box already said which of the two entries a frame belongs on --
        what this selector adds is the name the shared slot is claimed under.
        """
        return "render_host_motion" if int(settings.get('guidance_mode', 0)) == 1 \
            else "render_zero_motion"

    def get_dlss_session(self, w, h, settings, kind):
        """Unified thread-safe DLSS session. Reuses the existing D3D12 device and NGX
        feature across preview and export without tearing down or deadlocking
        nvngx_dlssnr.dll.

        kind names the GPU-host entry the caller needs: "image" is the zero-motion
        entry a still picture asks for because image mode already forces guidance
        off, "gpu_zero_motion" / "gpu_host_motion" are the two entries for a video
        preview frame, "export_zero_motion" / "export_host_motion" are the same two
        for a streaming export frame and "render_zero_motion" / "render_host_motion"
        the same two for a cached-frame render. One slot serves all of them, because
        dlss_gpu_host_open refuses a second session while one is active, so a request
        for another entry closes the cached session rather than driving it through an
        entry it does not have. There is no second host to name: the retired
        dlssnr_host.dll had a "legacy" kind for its own session, and a caller that
        asks for anything outside _GPU_HOST_KINDS is refused.
        """
        with self._dlss_lock:
            try:
                return self._get_or_create_session_locked(w, h, settings, kind)
            except Exception as ex:
                self.logln("[DLSS 引擎] " + str(ex))
                traceback.print_exc()
                return None

    def _get_or_create_session_locked(self, w, h, settings, kind):
        """Session bookkeeping for one GPU-host entry. The caller holds self._dlss_lock."""
        if kind not in self._GPU_HOST_KINDS:
            raise DLSSInferenceError(
                f"未知 DLSS 会话类型 kind={kind!r}：本版本唯一的原生 DLSS 宿主是 "
                f"native/dlss_gpu_host.dll，其入口只有 {', '.join(self._GPU_HOST_KINDS)}。")
        if kind != "image" and self._image_batch_running:
            # A paused preview worker can still be mid-frame, and handing this
            # request the slot would close the GPU session the batch is running on
            # while it holds it. A dropped preview frame is the cheaper answer.
            return None
        if self._video_task_kind is not None and kind != self._video_task_kind:
            # A DLSS render or video export owns the slot on the entry it opened.
            # Another caller names a different entry, so left alone it would evict
            # the session that task is mid-frame on and leave it calling a closed
            # handle for the rest of the run. A dropped preview frame is the cheaper
            # answer.
            return None
        session = self._dlss_session
        reuse = (session is not None and self._dlss_session_kind == kind
                 and getattr(session, "_w", 0) == w and getattr(session, "_h", 0) == h)
        if reuse:
            # The GPU host reads its options once, at open, and exposes no live
            # update entry, so an option that moved means a session that must go.
            reuse = getattr(session, "_gpu_options_key", None) == \
                dlss_gpu_engine.baked_option_key(settings)
        if reuse:
            return session

        if session is not None:
            try:
                session.close()
            except Exception:
                pass
            self._dlss_session = None
            self._dlss_session_kind = None

        live = self._open_gpu_live(w, h, settings, kind)
        self._dlss_session = live
        self._dlss_session_kind = kind
        return live

    def _open_gpu_live(self, w, h, settings, kind):
        """Open the GPU host session for this kind, or refuse it.

        A host that does not bind the entry the kind runs on is refused before any
        frame is drawn, and the caller is told to rebuild it. This build ships no
        second DLSS host, so a missing entry point has no answer but the rebuild:
        drawing the frame somewhere else would report a GPU-host run that did not
        happen.
        """
        capability, refusal, ready = self._GPU_HOST_KINDS[kind]
        live = dlss_gpu_engine.Live(w, h, settings)
        if not getattr(live, capability)():
            try:
                live.close()
            except Exception:
                pass
            raise DLSSInferenceError(refusal)
        live._gpu_options_key = dlss_gpu_engine.baked_option_key(settings)
        self.logln(f"{ready}: {getattr(live, 'host_dll', 'dlss_gpu_host.dll')}"
                   f" abi={getattr(live, 'host_abi', '?')} @ {w}x{h}")
        return live

    def close_dlss_session(self):
        with self._dlss_lock:
            if self._dlss_session is not None:
                try:
                    self._dlss_session.close()
                except Exception:
                    pass
                self._dlss_session = None
                self._dlss_session_kind = None

    # ---------- helpers & thread-safe ui dispatch ----------
    def post_ui(self, msg_type, *args):
        self._ui_queue.put((msg_type, *args))

    def call_ui(self, fn, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            fn(*args, **kwargs)
        else:
            self._ui_queue.put(("call", fn, args, kwargs))

    def _poll_ui_queue(self):
        try:
            while True:
                item = self._ui_queue.get_nowait()
                msg_type = item[0]
                if msg_type == "preview":
                    _, res, from_play, elapsed = item
                    self._apply_preview_result(res, from_play, elapsed)
                elif msg_type == "status":
                    _, text = item
                    self.status.config(text=text)
                elif msg_type == "log":
                    _, text = item
                    self.log.config(state="normal")
                    self.log.insert("end", text + "\n")
                    self.log.see("end")
                    self.log.config(state="disabled")
                elif msg_type == "progress":
                    _, i, total, extra = item
                    self._update_progress_ui(i, total, extra)
                elif msg_type == "call":
                    _, fn, args, kwargs = item
                    fn(*args, **kwargs)
                self._ui_queue.task_done()
        except queue.Empty:
            pass
        except Exception as ex:
            traceback.print_exc()
        finally:
            if not getattr(self, "_closing", False):
                self.root.after(20, self._poll_ui_queue)

    def set_status(self, msg):
        if threading.current_thread() is threading.main_thread():
            self.status.config(text=msg)
        else:
            self.post_ui("status", msg)

    def logln(self, msg):
        if threading.current_thread() is threading.main_thread():
            self.log.config(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.config(state="disabled")
        else:
            self.post_ui("log", msg)

    def _update_progress_ui(self, i, total, extra=""):
        if not total:
            return
        self.pbar["maximum"] = total
        self.pbar["value"] = i
        pct = (i / total) * 100.0 if total > 0 else 0.0

        now = time.time()
        last_task = getattr(self, "_progress_last_task", None)
        start_time = getattr(self, "_progress_start_time", None)

        if last_task != extra or i <= 1 or start_time is None:
            self._progress_start_time = now
            self._progress_last_task = extra
            self._progress_start_i = i
            speed_info = ""
        else:
            elapsed = now - start_time
            done_frames = i - getattr(self, "_progress_start_i", 0)
            if elapsed >= 0.5 and done_frames > 0:
                fps = done_frames / max(elapsed, 0.001)
                rem_sec = max(0.0, (total - i) / max(fps, 0.001))
                rem_m, rem_s = divmod(int(rem_sec), 60)
                rem_h, rem_m = divmod(rem_m, 60)
                eta_str = f"{rem_h:02d}:{rem_m:02d}:{rem_s:02d}" if rem_h else f"{rem_m:02d}:{rem_s:02d}"
                speed_info = f" | {fps:.1f} fps | 剩余 {eta_str}"
            else:
                speed_info = " | 估算中..."

        tag = f"[{extra}] " if extra else ""
        self.status.config(text=f"{tag}{i}/{total} ({pct:.1f}%){speed_info}")

    def set_progress(self, i, total, extra=""):
        if threading.current_thread() is threading.main_thread():
            self._update_progress_ui(i, total, extra)
        else:
            self.post_ui("progress", i, total, extra)

    # ---------- preview (asynchronous worker) ----------
    def request_view(self, frame=None, from_play=False):
        if self.current_is_image:
            self._display_image()
            return
        if not self.video:
            return
        if frame is None:
            frame = int(self.fslider.get())
        view = self.view_var.get()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cw = w if w > 10 else 780
        ch = h if h > 10 else 400
        s = self._collect_settings()
        zoom_1x = bool(getattr(self, "v_zoom_1x", None) and self.v_zoom_1x.get())
        self._preview_worker.request(frame, view, self.split_x, cw, ch, s,
                                     from_play=from_play, zoom_1x=zoom_1x,
                                     pan_xy=(self.pan_x, self.pan_y))

    def display_view(self):
        self.request_view()

    def on_frame(self):
        if getattr(self, "_updating_slider", False):
            return
        f = int(self.fslider.get())
        self.flabel.config(text=str(f))
        self.request_view(frame=f)

    def on_view_change(self):
        if self.view_var.get() == "对比":
            self.split_x = 0.5
        self.request_view()

    def on_canvas_double_click(self, event):
        new_zoom = not self.v_zoom_1x.get()
        self.v_zoom_1x.set(new_zoom)
        if new_zoom:
            # 双击放大时，居中到点击的目标区域
            if getattr(self, "_drag_nw", 0) > 0 and getattr(self, "_drag_nh", 0) > 0:
                ox = getattr(self, "_drag_offsetx", 0)
                oy = getattr(self, "_drag_offsety", 0)
                nw = self._drag_nw
                nh = self._drag_nh
                if ox <= event.x <= ox + nw and oy <= event.y <= oy + nh:
                    self.pan_x = max(0.0, min(1.0, (event.x - ox) / nw))
                    self.pan_y = max(0.0, min(1.0, (event.y - oy) / nh))
        self.display_view()

    def on_canvas_right_down(self, event):
        self._pan_drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_canvas_right_drag(self, event):
        if not self.v_zoom_1x.get() or not self._pan_drag_start:
            return
        sx, sy, orig_px, orig_py = self._pan_drag_start
        dx = event.x - sx
        dy = event.y - sy
        # 1:1 模式下鼠标拖动画布平移视野
        vw = max(getattr(self, "v_w", 1920), 1)
        vh = max(getattr(self, "v_h", 1080), 1)
        self.pan_x = max(0.0, min(1.0, orig_px - (dx / vw)))
        self.pan_y = max(0.0, min(1.0, orig_py - (dy / vh)))
        self.request_view()

    def on_canvas_right_up(self, event):
        self._pan_drag_start = None

    def _apply_preview_result(self, res, from_play=False, elapsed=0.0):
        if self.current_is_image or not self.video:
            return
        cur_slider = int(self.fslider.get())
        frame = res.get("frame")
        if frame is not None and frame != cur_slider and not from_play:
            return

        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cw = w if w > 10 else 780
        ch = h if h > 10 else 400

        if res.get("type") == "msg":
            self.canvas.create_text(cw // 2, ch // 2, text=res.get("text", ""),
                                    fill="#888888", font=("Microsoft YaHei", 11))
            return

        bgr = res["bgr"]
        ox, oy = res["ox"], res["oy"]
        nw, nh = res["nw"], res["nh"]
        self._last_rendered_frame = frame

        self._drag_nw = nw
        self._drag_nh = nh
        self._drag_offsetx = ox
        self._drag_offsety = oy

        if res.get("split"):
            self._split_orig = res["split_orig"]
            self._split_dlss = res["split_dlss"]
            self._split_frame = frame
        else:
            self._split_orig = None
            self._split_dlss = None
            self._split_frame = -1

        self._pilimg = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(self._pilimg)
        self.canvas.image = self._photo
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

        if res.get("msg"):
            self.canvas.create_text(cw // 2, 16, text=res["msg"], fill="#ffcc00", font=("Microsoft YaHei", 9))

        if from_play and self.playing:
            target_interval = max(16, int(1000.0 / max(self.fps, 1.0)))
            delay = max(5, int(target_interval - (elapsed * 1000)))
            self._play_after = self.root.after(delay, self._play_step)

    def _redraw_split_only(self):
        cur_f = int(self.fslider.get())
        if self._split_frame != cur_f or self._split_orig is None or self._split_dlss is None:
            self.request_view()
            return
        nw, nh = self._drag_nw, self._drag_nh
        o = self._split_orig.copy()
        sx = int(self.split_x * nw)
        o[:, sx:] = self._split_dlss[:, sx:]
        o[:, max(sx - 1, 0):min(sx + 1, nw)] = [0, 255, 255]
        self._pilimg = Image.fromarray(cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(self._pilimg)
        self.canvas.image = self._photo
        self.canvas.delete("all")
        self.canvas.create_image(self._drag_offsetx, self._drag_offsety, anchor="nw", image=self._photo)
        if getattr(self, "v_zoom_1x", None) and self.v_zoom_1x.get():
            self.canvas.create_text(self.canvas.winfo_width() // 2, 16,
                                    text="🔍 1:1 原画细节模式 (按住右键拖动视野，双击切换)",
                                    fill="#ffcc00", font=("Microsoft YaHei", 9))

    def on_canvas_motion(self, event):
        if self.view_var.get() != "对比":
            return
        if not hasattr(self, "_drag_nw") or not hasattr(self, "_drag_offsetx"):
            return
        frac = (event.x - self._drag_offsetx) / max(self._drag_nw, 1)
        self.split_x = max(0.0, min(1.0, frac))
        self._redraw_split_only()

    # ---------- playback ----------
    def play(self):
        if not self.video:
            messagebox.showwarning("提示", "请先导入视频")
            return
        self.playing = True
        if getattr(self, "_play_after", None):
            self.root.after_cancel(self._play_after)
            self._play_after = None
        self._play_step()

    def _play_step(self):
        if not self.playing:
            return
        nxt = int(self.fslider.get()) + 1
        if nxt > int(self.fslider.cget("to")):
            nxt = 0
        self._updating_slider = True
        try:
            self.fslider.set(nxt)
            self.flabel.config(text=str(nxt))
        finally:
            self._updating_slider = False
        self.request_view(frame=nxt, from_play=True)

    def pause(self):
        self.playing = False
        if getattr(self, "_play_after", None):
            self.root.after_cancel(self._play_after)
            self._play_after = None

    def on_settings_change(self, event=None):
        self._collect_settings()
        if getattr(self, "_live_debounce", None):
            self.root.after_cancel(self._live_debounce)
        self._live_debounce = self.root.after(60, self._refresh_dlss)

    def _refresh_dlss(self):
        self._live_debounce = None
        self._preview_worker.close_live()
        self._split_frame = -1
        if self.view_var.get() in ("DLSS", "对比"):
            self.request_view()

    def on_close(self):
        self._closing = True
        self.pause()
        if hasattr(self, "_preview_worker"):
            self._preview_worker.stop()
        if getattr(self, "_cap", None):
            try:
                self._cap.release()
            except Exception:
                pass
        self.close_dlss_session()
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


    # ---------- DLSS settings ----------
    def _build_settings(self, parent):
        self.v_preset     = tk.StringVar(value="Preset #1")
        self.v_style      = tk.StringVar(value="默认")
        self.v_intensity  = tk.DoubleVar(value=1.0)
        self.v_localTone  = tk.DoubleVar(value=1.0)
        self.v_localStruct= tk.DoubleVar(value=1.0)
        self.v_skinStruct = tk.DoubleVar(value=1.0)
        self.v_autoMask   = tk.IntVar(value=1)   # 默认打开自动遮罩
        self.v_uiCorr     = tk.IntVar(value=0)
        self.v_guidance   = tk.StringVar(value="关闭")
        self.v_motionSX   = tk.DoubleVar(value=1.0)
        self.v_motionSY   = tk.DoubleVar(value=1.0)
        S=tk.Scale
        ttk.Label(parent,text="预设:").grid(row=0,column=0,sticky="e",padx=(6,2),pady=2)
        ttk.Combobox(parent,textvariable=self.v_preset,values=["Preset #1","Preset #2","Preset #3"],state="readonly",width=9).grid(row=0,column=1,padx=(0,8))
        ttk.Label(parent,text="风格:").grid(row=0,column=2,sticky="e",padx=(6,2))
        ttk.Combobox(parent,textvariable=self.v_style,values=["默认","自然","电影","风格3"],state="readonly",width=8).grid(row=0,column=3,padx=(0,8))
        ttk.Label(parent,text="强度:").grid(row=0,column=4,sticky="e",padx=(6,2))
        S(parent,from_=0,to=1,resolution=0.05,orient="horizontal",showvalue=True,variable=self.v_intensity,length=100).grid(row=0,column=5,padx=(0,8))
        ttk.Label(parent,text="运动缩放X:").grid(row=0,column=6,sticky="e",padx=(6,2))
        S(parent,from_=0,to=2,resolution=0.1,orient="horizontal",showvalue=True,variable=self.v_motionSX,length=90).grid(row=0,column=7,padx=(0,8))
        ttk.Label(parent,text="Y:").grid(row=0,column=8,sticky="e")
        S(parent,from_=0,to=2,resolution=0.1,orient="horizontal",showvalue=True,variable=self.v_motionSY,length=70).grid(row=0,column=9,padx=(0,8))
        ttk.Label(parent,text="本地色调:").grid(row=1,column=0,sticky="e",padx=(6,2))
        S(parent,from_=0,to=5,resolution=0.1,orient="horizontal",showvalue=True,variable=self.v_localTone,length=80).grid(row=1,column=1,padx=(0,8))
        ttk.Label(parent,text="本地结构:").grid(row=1,column=2,sticky="e",padx=(6,2))
        S(parent,from_=0,to=5,resolution=0.1,orient="horizontal",showvalue=True,variable=self.v_localStruct,length=80).grid(row=1,column=3,padx=(0,8))
        ttk.Label(parent,text="皮肤结构:").grid(row=1,column=4,sticky="e",padx=(6,2))
        S(parent,from_=0,to=5,resolution=0.1,orient="horizontal",showvalue=True,variable=self.v_skinStruct,length=80).grid(row=1,column=5,padx=(0,8))
        ttk.Checkbutton(parent,text="自动遮罩",variable=self.v_autoMask).grid(row=2,column=0,columnspan=2,sticky="w",padx=(6,2))
        ttk.Checkbutton(parent,text="UI校正",variable=self.v_uiCorr).grid(row=2,column=2,columnspan=2,sticky="w",padx=(6,2))
        ttk.Label(parent,text="引导模式:").grid(row=2,column=4,sticky="e",padx=(6,2))
        ttk.Combobox(parent,textvariable=self.v_guidance,values=["仅光流","关闭"],state="readonly",width=8).grid(row=2,column=5,padx=(0,8))
        # live-tweak: any setting change re-renders the current DLSS frame (debounced)
        for child in parent.winfo_children():
            if isinstance(child, tk.Scale):
                child.config(command=lambda e: self.on_settings_change())
            elif isinstance(child, ttk.Combobox):
                child.bind("<<ComboboxSelected>>", lambda e: self.on_settings_change())
            elif isinstance(child, tk.Checkbutton):
                child.config(command=lambda: self.on_settings_change())

    def _collect_settings(self):
        if threading.current_thread() is not threading.main_thread():
            return dict(getattr(self, "_cached_settings", {}))
        style_map={"默认":0,"自然":1,"电影":2,"风格3":3}
        guid_map={"仅光流":1,"关闭":0}
        try:
            preset=int(self.v_preset.get().split("#")[-1])
        except Exception:
            preset=1
        s = {
            'preset': preset,
            'style': style_map.get(self.v_style.get(),0),
            'intensity': float(self.v_intensity.get()),
            'local_tone': float(self.v_localTone.get()),
            'local_struct': float(self.v_localStruct.get()),
            'skin_struct': float(self.v_skinStruct.get()),
            'use_auto_mask': int(self.v_autoMask.get()),
            'ui_correction': int(self.v_uiCorr.get()),
            'guidance_mode': guid_map.get(self.v_guidance.get(),0),
            'motion_scale_x': float(self.v_motionSX.get()),
            'motion_scale_y': float(self.v_motionSY.get()),
        }
        self._cached_settings = dict(s)
        return s

    # ---------- actions ----------
    def import_video(self):
        p = filedialog.askopenfilename(filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if not p:
            return
        self.pause()
        self._preview_worker.close_live()
        self._split_frame = -1          # 换片后清掉旧的对比缓存，强制重建(否则视窗显示旧片)
        self.current_is_image = False
        self._update_export_btn()
        self.video = os.path.abspath(p)
        if getattr(self, "_cap", None):
            self._cap.release()
        self._cap = pipeline.open_capture(self.video)
        n, fps, w, h = pipeline.video_info(self.video)
        self.nframes, self.fps = n, fps
        self.v_w, self.v_h = w, h
        self.vlabel.config(text=f"{os.path.basename(self.video)}  ({n} 帧 @ {fps:.0f}fps {w}x{h})")
        self.fslider.config(to=max(n - 1, 1))
        self.fslider.set(0)
        self.flabel.config(text="0")
        try:
            self.request_view(frame=0)
        except Exception as ex:
            self.logln(f"[preview] {ex}")
        self.logln(f"已导入: {self.video}  ({n} 帧)")
        self.set_status("就绪 (可拖动滑块预览任意帧)")

    def _update_export_btn(self):
        self.export_btn.config(text="导出图片" if self.current_is_image else "导出视频")

    # ---------- single-image mode (guidance=off) ----------
    def import_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("所有文件", "*.*")])
        if not p:
            return
        self.pause(); self._preview_worker.close_live()
        img = pipeline.imread_u(p, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法读取该图片"); return
        self.video = None
        self.current_is_image = True
        self.image_path = os.path.abspath(p)
        self.image_bgr = img
        self.image_dlss = None
        h, w = img.shape[:2]
        self.nframes = 1; self.fps = 1.0
        self.v_w, self.v_h = w, h
        self.vlabel.config(text=os.path.basename(self.image_path) + "  (图片)")
        self.flabel.config(text="0"); self.fslider.config(to=1); self.fslider.set(0)
        self.view_var.set("DLSS")
        self._update_export_btn()
        self.set_status("图片 DLSS 生成中...")
        self._in_thread(self._image_worker)

    def _image_worker(self):
        try:
            self.image_dlss = self._image_dlss(self.image_bgr)
            if self.image_dlss is None:
                self.logln("图片 DLSS 失败，显示原图")
            self.call_ui(lambda: (self.set_status("就绪"), self.display_view()))
        except Exception as ex:
            self.logln("图片 DLSS 错误: " + str(ex)); traceback.print_exc()
            self.set_status("出错: " + str(ex)[:80])

    def _image_dlss(self, img):
        s = self._collect_settings(); s['guidance_mode'] = 0   # 无引导
        try:
            return self._dlss_image(img, s)
        except DlssHostUnavailable as ex:
            self.logln("[图片 DLSS] GPU 宿主不可用: " + str(ex))
            traceback.print_exc()
            return None

    def _dlss_image(self, img, settings):
        """One BGR image through the GPU host's resident zero-motion texture.

        Padding, the RGBA assembly, the session and the call live here so the
        single-image and the batch path are the same path. The session lookup and
        the call share one _dlss_lock hold, so a concurrent video rebuild cannot
        close the session between them. reset=True every time: each still image is
        its own sequence and must not inherit another image's temporal state.
        Returns None when this image failed; raises DlssHostUnavailable when the
        host itself cannot serve images, which no per-image retry can fix.
        """
        h, w = img.shape[:2]
        # pad UP to a multiple of 8 (DLSS texture stability), trim back after
        tw = max(8, ((w + 7) // 8) * 8); th = max(8, ((h + 7) // 8) * 8)
        pt = (th - h) // 2; pb = (th - h) - pt; pl = (tw - w) // 2; pr = (tw - w) - pl
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (tw, th) != (w, h):
            rgb = cv2.copyMakeBorder(rgb, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        rgba = np.dstack([rgb, np.full((th, tw), 255, np.uint8)])
        with self._dlss_lock:
            try:
                live = self._get_or_create_session_locked(tw, th, settings, "image")
            except Exception as ex:
                raise DlssHostUnavailable(str(ex)) from ex
            if live is None:
                raise DlssHostUnavailable("DLSS 会话不可用")
            try:
                o = live.process_zero_motion(rgba, reset=True)
            except Exception as ex:
                self.logln("[图片 DLSS] 推理失败: " + str(ex))
                return None
        if o is None:
            self.logln("[图片 DLSS] 无推理结果")
            return None
        bgr = cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR)
        if (tw, th) != (w, h):
            bgr = bgr[pt:pt + h, pl:pl + w]
        return bgr

    def _display_image(self):
        view = self.view_var.get()
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cw = w if w > 10 else 780
        ch = h if h > 10 else 400
        if view == "原图":
            img = self.image_bgr
        elif view == "对比":
            if self.image_bgr is not None and self.image_dlss is not None:
                img = np.hstack([self.image_bgr, self.image_dlss])
            else:
                img = self.image_bgr
        else:  # DLSS 及其它 -> DLSS 结果
            img = self.image_dlss
        if img is None:
            self.canvas.create_text(cw // 2, ch // 2,
                                    text="图片模式：仅支持 原图 / DLSS / 对比",
                                    fill="#888888", font=("Microsoft YaHei", 11)); return
        ih, iw = img.shape[:2]
        zoom_1x = bool(getattr(self, "v_zoom_1x", None) and self.v_zoom_1x.get())
        if zoom_1x:
            crop_w, crop_h = min(cw, iw), min(ch, ih)
            px, py = getattr(self, "pan_x", 0.5), getattr(self, "pan_y", 0.5)
            x0 = max(0, min(int(px * iw - crop_w / 2), iw - crop_w))
            y0 = max(0, min(int(py * ih - crop_h / 2), ih - crop_h))
            nimg = img[y0:y0 + crop_h, x0:x0 + crop_w]
            nw, nh = crop_w, crop_h
        else:
            scale = min(cw / iw, ch / ih)
            nw = max(int(iw * scale), 1); nh = max(int(ih * scale), 1)
            interp = cv2.INTER_AREA if (nw < iw and nh < ih) else cv2.INTER_LANCZOS4
            nimg = cv2.resize(img, (nw, nh), interpolation=interp)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self._drag_nw = nw
        self._drag_nh = nh
        self._drag_offsetx = ox
        self._drag_offsety = oy
        from PIL import Image, ImageTk
        self._pilimg = Image.fromarray(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(self._pilimg)
        self.canvas.image = self._photo
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        if zoom_1x:
            self.canvas.create_text(cw // 2, 16, text="🔍 1:1 原画细节模式 (按住右键拖动视野，双击切换)",
                                    fill="#ffcc00", font=("Microsoft YaHei", 9))

    def _export_image(self):
        if self.image_dlss is None:
            messagebox.showinfo("提示", "还没有可导出的 DLSS 结果"); return
        base = os.path.splitext(os.path.basename(self.image_path or "image"))[0]
        p = filedialog.asksaveasfilename(title="导出 png", defaultextension=".png",
                                         initialfile=base + "_dlss.png", filetypes=[("PNG", "*.png")])
        if not p:
            return
        # A single rendered frame is written straight to the path the user picked, with
        # no decoder and no encoder to refuse a destination that resolves back to the
        # source image, so the guard the export entry points use has to answer here too.
        try:
            pipeline.assert_output_not_input(self.image_path, p)
        except pipeline.OutputOverwritesInputError as exc:
            messagebox.showerror("导出被拒绝", str(exc))
            return
        pipeline.imwrite_u(p, self.image_dlss)
        self.logln("导出图片: " + p)
        messagebox.showinfo("导出", "已导出: " + p)

    def _is_cancelled(self):
        return getattr(self, "_cancel_requested", False)

    def stop_worker(self):
        if self.thread and self.thread.is_alive():
            self._cancel_requested = True
            self.set_status("正在中止，等待当前帧保存...")
            self.logln("已发送中止信号，正在安全停止任务...")
            self.stop_btn.config(state="disabled")

    def _in_thread(self, fn):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("忙", "上一个任务还没结束"); return
        self._cancel_requested = False
        self.stop_btn.config(state="normal")
        def wrapper():
            try:
                fn()
            finally:
                self.call_ui(lambda: self.stop_btn.config(state="disabled"))
                self._cancel_requested = False
        self.thread = threading.Thread(target=wrapper, daemon=True)
        self.thread.start()

    def run_worker(self, kind):
        if not self.video:
            messagebox.showwarning("提示", "请先导入视频"); return
        if kind in ("flow", "dlss"):
            self._in_thread(lambda: self._do(kind))

    def _do(self, kind):
        try:
            self._preview_worker.pause()
            self.logln(f"开始: {kind} ...")
            self.set_status(f"{kind} 运行中...")
            if kind == "flow":
                pipeline.generate_flow(
                    self.video,
                    progress=lambda i, t, s: self.set_progress(i, t, "光流"),
                    cancel=self._is_cancelled
                )
                if self._is_cancelled():
                    self.logln("光流生成已由用户中止，已处理的帧已完整保存")
                    self.set_status("已中止 (断点已保存)")
                    return
                self.logln("光流完成 -> " + pipeline.out_dirs(self.video)[0])
            elif kind == "dlss":
                dlss_ok = self._run_dlss_gui()
                if self._is_cancelled():
                    self.logln("DLSS 渲染已由用户中止，已生成的帧已完整保存")
                    self.set_status("已中止 (断点已保存)")
                    return
                if dlss_ok and getattr(self, "_pending_export", False):
                    self._pending_export = False
                    self.logln("DLSS 渲染完成，正在封装为 MKV 视频...")
                    self.set_status("正在封装 MKV 视频...")
                    p = pipeline.export_video(
                        self.video, "dlss", self.nframes, self.fps,
                        crf=self._export_crf(),
                        with_audio=self._export_audio(),
                        progress=lambda i, t, s: self.set_progress(i, t, "封装 MKV"),
                        cancel=self._is_cancelled
                    )
                    if self._is_cancelled():
                        self.logln("视频导出封装已中止")
                        self.set_status("封装已中止")
                        return
                    if p:
                        self.logln("导出成功: " + p)
                        self.set_status("导出完成")
                        self.call_ui(lambda: messagebox.showinfo("导出", f"已成功导出 DLSS 视频:\n{p}"))
                    else:
                        self.logln("DLSS 导出失败 (无有效帧)")
                        self.set_status("导出失败")
                        self.call_ui(lambda: messagebox.showwarning("导出", "DLSS 导出失败 (无有效帧)"))
                else:
                    self._pending_export = False
            self.set_status("完成")
            self.call_ui(lambda: self.pbar.config(value=0))
            self.call_ui(self.request_view)   # refresh preview after generation
        except Exception as ex:
            self.logln("错误: " + str(ex))
            traceback.print_exc()
            self.set_status("出错: " + str(ex)[:80])
        finally:
            self._preview_worker.resume()

    def _guidance_inputs(self, idx, fr, gm):
        """Colour for the host, plus the motion this frame asks for.

        Guidance off produces no motion plane at all: the host answers from the zero
        texture it bound at open, and the array this helper used to build for the old
        host's argument slot is something on the card that has no entry reading it.

        Guidance on hands over the cached `.flo` resized to the extent the frame is drawn
        at. A frame with no cache, or with a cache that will not parse, keeps the
        documented smooth degrade and gets a zero plane, which is what it always got; it
        comes back flagged so the loop can say the frame was fed zero motion vectors
        rather than claim it as a guided result.
        """
        h, w = fr.shape[:2]
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])

        flow, degraded = None, False
        if gm == 1:
            fm, _ = pipeline.out_dirs(self.video)
            fpath = os.path.join(fm, f"{idx:06d}.flo")
            if os.path.exists(fpath):
                try:
                    flow = pipeline.read_flo(fpath)
                    if flow.shape[:2] != (h, w):
                        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    flow = None
            if flow is None:
                flow = np.zeros((h, w, 2), np.float32)
                degraded = True
        return rgba, flow, degraded

    def _run_dlss_gui(self):
        s = self._collect_settings()
        gm = s['guidance_mode']
        gm_label = {0: "关闭", 1: "仅光流"}.get(gm, gm)
        n, fps, w, h = pipeline.video_info(self.video)
        dlss_dir = pipeline.out_dirs(self.video)[1]
        os.makedirs(dlss_dir, exist_ok=True)

        is_export = getattr(self, "_pending_export", False)
        task_label = "导出 DLSS" if is_export else "DLSS"
        self.logln(f"启动 DLSS5 神经渲染 ({w}x{h}, 预设={s.get('preset', 1)}, 引导={gm_label}) ...")
        self.set_progress(0, n, task_label)

        # Every frame of this run is drawn by the project GPU host, on the entry the guidance
        # control names: the preview moved there first, the streaming export after it, and
        # this loop uses the same two entries at the same extent. What is apart is the name
        # the one session slot is claimed under, because that name is the only thing standing
        # between this loop and a preview frame that closed the session mid-run.
        kind = self.video_render_kind(s)
        with self._dlss_lock:
            self._video_task_kind = kind
        try:
            live = self.get_dlss_session(w, h, s, kind=kind)
            if live is None:
                self.logln("[DLSS] 无法初始化 DLSS 引擎")
                return False
            rendered_through = None    # the last frame this session drew
            flow_missing = 0
            for idx, fr in pipeline.iter_frames(self.video):
                if self._is_cancelled():
                    self.logln(f"DLSS 处理于第 {idx} 帧停止")
                    return False
                out_png = os.path.join(dlss_dir, f"{idx:06d}.png")
                if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
                    # 断点续算：已有帧跳过推理
                    self.set_progress(idx + 1, n, task_label)
                    continue
                rgba, flow, degraded = self._guidance_inputs(idx, fr, gm)
                if degraded:
                    if not flow_missing:
                        self.logln(f"[平滑降级] 第 {idx} 帧缺少光流缓存，缺少缓存的帧以零运动向量继续渲染")
                    flow_missing += 1
                with self._dlss_lock:
                    # A run that resumed past cached frames, a run that restarted into a
                    # session the slot kept, and a session someone else rebuilt all begin a
                    # new temporal sequence; only the frame straight after the one this
                    # session drew joins the sequence already accumulating in it.
                    reset = rendered_through is None or idx != rendered_through + 1
                    o = (live.process_host_motion(rgba, flow, reset=reset)
                         if flow is not None
                         else live.process_zero_motion(rgba, reset=reset))
                rendered_through = idx
                pipeline.imwrite_u(out_png, cv2.cvtColor(o, cv2.COLOR_RGBA2BGR))
                self.set_progress(idx + 1, n, task_label)
            if flow_missing:
                # The count, not a claim: these frames went through the motion entry with
                # nothing in it, so the run cannot describe them as guided results.
                self.logln(f"[平滑降级] {flow_missing} 帧缺少光流缓存，这些帧以零运动向量送入渲染")
            self.logln("DLSS 完成 -> " + dlss_dir)
            self.call_ui(lambda: self.view_var.set("DLSS"))
            return True
        except Exception as ex:
            self.logln("[DLSS] " + str(ex))
            traceback.print_exc()
            self.set_status("DLSS 出错: " + str(ex)[:80])
            return False
        finally:
            with self._dlss_lock:
                self._video_task_kind = None

    # ---------- image-folder batch processing (guidance=off) ----------
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def process_images(self):
        """Pick an image folder; process every image with DLSS (no guidance) and write
        PNGs to a sibling '<folder>_dlss' directory next to the picked folder on disk."""
        d = filedialog.askdirectory(title="选择图片文件夹")
        if not d:
            return
        d = os.path.abspath(d)
        files = sorted(f for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in self.IMG_EXTS
                       and os.path.isfile(os.path.join(d, f)))
        if not files:
            messagebox.showinfo("提示", "该文件夹没有可处理的图片"); return
        self._in_thread(lambda: self._do_images(d, files))

    def _do_images(self, d, files):
        # A folder batch runs in video mode too, where the preview worker holds the one
        # DLSS session slot at the video's size. Each image needs a session at its own
        # size, so left alone the two evict each other on every file and re-open the
        # host for the whole run. Park the worker and hold the slot.
        worker = getattr(self, "_preview_worker", None)
        with self._dlss_lock:
            self._image_batch_running = True
        if worker is not None:
            worker.pause()
        try:
            out_dir = os.path.join(os.path.dirname(d.rstrip("/\\")),
                                   os.path.basename(d) + "_dlss")
            os.makedirs(out_dir, exist_ok=True)
            self.logln("处理图片文件夹: " + d)
            self.logln("输出          -> " + out_dir)
            self.set_status("图片批处理 ...")
            s = self._collect_settings(); s['guidance_mode'] = 0   # 强制无引导
            total = len(files)
            done = 0
            for i, fn in enumerate(files):
                if self._is_cancelled():
                    self.logln(f"图片批处理已由用户中止于第 {i} 张")
                    self.set_status("批处理已中止")
                    break
                out_path = os.path.join(out_dir, os.path.splitext(fn)[0] + ".png")
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    done += 1
                    self.set_progress(i + 1, total, "图片")
                    continue
                p = os.path.join(d, fn)
                img = pipeline.imread_u(p, cv2.IMREAD_COLOR)
                if img is None:
                    self.logln("跳过(无法读取): " + fn); continue
                try:
                    bgr = self._dlss_image(img, s)
                except DlssHostUnavailable as ex:
                    # A host that cannot open will not open for the next file
                    # either; say so once instead of once per remaining image.
                    self.logln("图片批处理中止：GPU 宿主不可用: " + str(ex))
                    traceback.print_exc()
                    self.set_status("GPU 宿主不可用，批处理已中止")
                    break
                if bgr is None:
                    self.logln("DLSS 失败: " + fn); continue
                pipeline.imwrite_u(out_path, bgr)
                done += 1
                self.set_progress(i + 1, total, "图片")
            if not self._is_cancelled():
                self.logln(f"图片批处理完成: {done}/{total} -> " + out_dir)
                self.set_status(f"完成: 处理 {done}/{total}")
        except Exception as ex:
            self.logln("图片批处理错误: " + str(ex))
            traceback.print_exc()
            self.set_status("出错: " + str(ex)[:80])
        finally:
            with self._dlss_lock:
                self._image_batch_running = False
            if worker is not None:
                worker.resume()

    def _export_crf(self):
        """Return the user-set NVENC AV1 CQ (0-63; lower = higher quality)."""
        if threading.current_thread() is not threading.main_thread():
            return getattr(self, "_cached_crf", 24)
        try:
            crf = int(self.v_export_crf.get())
        except Exception:
            crf = 24
        crf = max(0, min(63, crf))
        self._cached_crf = crf
        return crf

    def _export_audio(self):
        """Return whether audio track should be kept in exported video."""
        if threading.current_thread() is not threading.main_thread():
            return getattr(self, "_cached_audio", True)
        try:
            val = bool(self.v_export_audio.get())
        except Exception:
            val = True
        self._cached_audio = val
        return val

    def export(self):
        if self.current_is_image:
            self._export_image(); return
        if not self.video:
            messagebox.showwarning("提示", "请先导入视频"); return
        kind_label = self.export_kind.get()
        kind_map = {"DLSS": "dlss", "光流": "flow", "dlss": "dlss", "flow": "flow"}
        kind = kind_map.get(kind_label, "dlss")

        flow_dir, dlss_dir = pipeline.out_dirs(self.video)
        if kind == "flow":
            has_flow = os.path.isdir(flow_dir) and any(f.endswith(".flo") for f in os.listdir(flow_dir)) if os.path.isdir(flow_dir) else False
            if not has_flow:
                messagebox.showwarning("提示", "未找到生成的光流数据。\n\n请先点击上方【生成光流】按钮生成后再导出光流视频。")
                return
        elif kind == "dlss":
            existing_pngs = [f for f in os.listdir(dlss_dir) if f.endswith(".png")] if os.path.isdir(dlss_dir) else []
            has_dlss_all = len(existing_pngs) >= self.nframes and self.nframes > 0
            if not has_dlss_all:
                s = self._collect_settings()
                gm = s.get('guidance_mode', 0)
                gm_label = {0: "关闭", 1: "仅光流"}.get(gm, str(gm))
                need_flow = gm == 1
                has_flow = os.path.isdir(flow_dir) and any(f.endswith(".flo") for f in os.listdir(flow_dir)) if os.path.isdir(flow_dir) else False
                if need_flow and not has_flow:
                    missing = []
                    if need_flow and not has_flow: missing.append("光流")
                    m_str = "、".join(missing)
                    ok = messagebox.askyesno(
                        "平滑降级提示",
                        f"当前引导模式为【{gm_label}】，尚未完整生成【{m_str}】数据。\n\n"
                        f"程序支持【平滑降级】：缺少数据的部分将自动回退为无引导基础模式继续处理并导出。\n\n"
                        f"是否立即开始渲染并导出？\n(点【否】可先去生成对应数据)"
                    )
                    if not ok:
                        return
                if len(existing_pngs) > 0:
                    self.logln(f"检测到已有 {len(existing_pngs)}/{self.nframes} 帧缓存，但未完全覆盖。正在启动 DLSS 高速流式硬件导出...")
                else:
                    self.logln("未找到完整缓存帧，正在启动 DLSS 高速流式硬件渲染并直接导出...")
                self._in_thread(self._do_dlss_stream_export)
                return

        self._in_thread(lambda: self._do_export_only(kind, kind_label))

    def _do_dlss_stream_export(self):
        try:
            self._preview_worker.pause()
            self.logln("启动 DLSS 高速流式导出 (渲染帧经宿主内存管道送入 FFmpeg NVENC 硬件编码，免写中间图片)...")
            self.set_status("正在进行 DLSS 高速流式导出...")
            cq = self._export_crf()
            with_audio = self._export_audio()
            s = self._collect_settings()
            n, fps, w, h = pipeline.video_info(self.video)
            gm = s.get('guidance_mode', 0)
            kind = self.video_export_kind(s)
            # Own the one session slot, on the host and entry this run actually opened,
            # for the whole run: a preview frame handed the slot would either reset the
            # temporal sequence the encoder is being fed from, or close the session it is
            # being fed through.
            with self._dlss_lock:
                self._video_task_kind = kind
            live = self.get_dlss_session(w, h, s, kind=kind)
            if live is None:
                self.logln("无法初始化 DLSS 会话")
                self.set_status("DLSS 会话初始化失败")
                return

            flow_dir, _ = pipeline.out_dirs(self.video)
            # Guidance off takes no provider at all: the host answers motion from the
            # zero texture it bound at open, and the export core refuses a plane nobody
            # asked for. Guidance on hands it the cached .flo directory, which is the
            # host-motion entry's host-memory input.
            flow_provider = (pipeline.CachedFloFlowProvider(flow_dir)
                             if gm == 1 else None)
            job_settings = {"guidance_mode": gm, "dlss_backend": "gpu_flow_d3d12",
                            "export_backend": "legacy"}
            backend_diagnostics = {}

            def _progress_cb(cur, total, text):
                self.set_progress(cur, total, text)

            p = pipeline.export_dlss_stream(
                self.video,
                live=live,
                lock=self._dlss_lock,
                frames=self.nframes,
                fps=self.fps,
                cq=cq,
                with_audio=with_audio,
                flow_provider=flow_provider,
                job_settings=job_settings,
                progress=_progress_cb,
                cancel=self._is_cancelled,
                diagnostics_out=backend_diagnostics
            )
            route = backend_diagnostics.get("dlss_entry")
            if route:
                # Which host answered is invisible in the finished file, so the run says
                # it: the entry the frames went through and where their motion came from.
                self.logln(f"[流式导出 DLSS] 宿主入口 {route}，"
                           f"运动来源 {backend_diagnostics.get('flow_transport')}")
            if getattr(flow_provider, "missing_frame_count", 0):
                self.logln(f"[平滑降级] {flow_provider.missing_frame_count} 帧缺少光流缓存，"
                           f"这些帧以零运动向量送入 DLSSNR")

            if self._is_cancelled():
                self.logln("DLSS 高速流式导出已由用户中止 (已生成的部分视频已安全保存)")
                self.set_status("导出已中止")
                return
            if p:
                self.logln("DLSS 高速流式导出成功: " + p)
                self.set_status("导出完成")
                self.call_ui(lambda: messagebox.showinfo("导出成功", f"已成功完成 DLSS 高速流式导出:\n{p}"))
            else:
                self.logln("DLSS 导出失败 (无有效帧)")
                self.set_status("导出失败")
        except pipeline.JobCancelled:
            # The pipe's own cancellation is an abort, not an error: it raises rather than
            # returning, and a dialog that says 导出错误 over a user's own Stop click is a
            # report of a failure that did not happen.
            self.logln("DLSS 高速流式导出已由用户中止 (已生成的部分视频已安全保存)")
            self.set_status("导出已中止")
        except Exception as ex:
            self.logln("流式导出错误: " + str(ex))
            traceback.print_exc()
            self.set_status("流式导出出错: " + str(ex)[:80])
            self.call_ui(lambda message=str(ex): messagebox.showerror("导出错误", message))
        finally:
            self.call_ui(lambda: self.pbar.config(value=0))
            with self._dlss_lock:
                self._video_task_kind = None
            self._preview_worker.resume()

    def _do_export_only(self, kind, kind_label):
        try:
            self._preview_worker.pause()
            self.logln(f"开始导出【{kind_label}】视频...")
            self.set_status(f"正在导出【{kind_label}】视频...")
            crf = self._export_crf()
            with_audio = self._export_audio()
            p = pipeline.export_video(
                self.video, kind, self.nframes, self.fps,
                crf=crf,
                with_audio=with_audio,
                progress=lambda i, t, s: self.set_progress(i, t, f"导出 {kind_label}"),
                cancel=self._is_cancelled
            )
            if self._is_cancelled():
                self.logln(f"导出【{kind_label}】视频已中止")
                self.set_status("导出已中止")
                return
            if p:
                self.logln(f"导出成功: {p}")
                self.set_status("导出完成")
                self.call_ui(lambda: messagebox.showinfo("导出", f"已成功导出【{kind_label}】视频:\n{p}"))
            else:
                self.logln(f"导出失败: 未找到可导出的【{kind_label}】帧")
                self.set_status("导出失败")
                self.call_ui(lambda: messagebox.showwarning("导出", f"未找到可导出的【{kind_label}】帧。\n\n请先在上方点击对应的生成步骤。"))
        except Exception as ex:
            self.logln("导出错误: " + str(ex))
            traceback.print_exc()
            self.set_status("导出出错: " + str(ex)[:80])
            self.call_ui(lambda message=str(ex): messagebox.showerror("导出错误", message))
        finally:
            self.call_ui(lambda: self.pbar.config(value=0))
            self._preview_worker.resume()


def _selftest(report_path="_selftest.txt"):
    """Headless DLSS smoke test (--selftest); writes the result to _selftest.txt for a
    GUI-less verify, and prints it so a terminal run does not have to go looking.

    Guidance is off in a probe, so the frame asks for the project GPU host's
    zero-motion entry: the one `App._GPU_HOST_KINDS["image"]` describes, and the one
    the window's own still pictures run on. Capability and refusal are read off that
    table rather than restated here, so this probe cannot go on describing an entry
    the window has since moved -- which is exactly how it rotted. An older version of
    this probe passed a depth plane as a third positional argument, which collided
    with the `reset=True` keyword it also passed and ended every run in a TypeError
    before a frame was drawn. What it draws here is one zero-motion frame on the host
    the window actually uses: a GPU host with no such entry point is a stale native
    build and is reported as one, because this build ships no other DLSS host to draw
    it instead.
    """
    capability, refusal, _ = App._GPU_HOST_KINDS["image"]
    stage = "host-load"
    try:
        try:
            import torch
            tn = "torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available())
        except Exception as te:
            tn = "torch=ERR " + repr(te)
        W, H = 320, 240
        try:
            live = dlss_gpu_engine.Live(W, H, {'preset': 1, 'guidance_mode': 0})
        except OSError as e:
            raise RuntimeError(
                f"默认 DLSS 宿主 native/dlss_gpu_host.dll 无法加载：{e}"
                "（本版本唯一的 DLSS 宿主就是它，旧宿主 dlssnr_host.dll 已移除）") from e
        stage = "capability"
        try:
            if not getattr(live, capability)():
                raise DLSSInferenceError(refusal)
            rgba = np.zeros((H, W, 4), np.uint8)
            rgba[..., 3] = 255
            stage = "frame"
            o = live.process_zero_motion(rgba, reset=True)
        finally:
            live.close()
        stage = "output"
        if not isinstance(o, np.ndarray) or o.shape != (H, W, 4) or o.dtype != np.uint8:
            raise RuntimeError(
                f"GPU 宿主返回的帧不是 uint8 {H}x{W}x4，而是 "
                f"{type(o).__name__} "
                f"{getattr(o, 'shape', None)} {getattr(o, 'dtype', None)}")
        rgb = o[..., :3]
        detail = (f"host=native/dlss_gpu_host.dll entry=process_zero_motion reset=1 "
                  f"rgb_mean={rgb.mean():.3f} rgb_max={int(rgb.max())} "
                  f"alpha_mean={o[..., 3].mean():.3f} "
                  f"sha1={hashlib.sha1(o.tobytes()).hexdigest()[:12]}")
        msg = f"DLSS_OK {str((o.shape[0], o.shape[1], o.shape[2]))} | {detail} | {tn}"
    except Exception as ex:
        msg = f"DLSS_ERR [{stage}] " + repr(ex) + "\n" + traceback.format_exc()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(msg)
    print(msg)
    return msg


GPU_TRANSFER_SELFTEST_FLAG = "--gpu-transfer-selftest"
# Bounded by construction: this entry is a correctness probe, and a frame count
# outside this window is a benchmark the probe has no business being.
GPU_TRANSFER_SELFTEST_FRAMES = (3, 8)
GPU_TRANSFER_SELFTEST_SIZE = (1280, 720)
GPU_TRANSFER_SELFTEST_FPS = 24
# The ratio the probe runs when nobody names one. Every other ratio it can run has
# to be asked for, and is checked against the shapes the gate serves before a device
# session is opened.
GPU_TRANSFER_SELFTEST_SCALE = 0.5
# The temp file the copy-pass chain writes, muxes over, and unlinks. The prefix is
# the whole signature of a run that leaked a device-produced file into the temp dir.
GPU_TRANSFER_SELFTEST_INTERMEDIATE_PREFIX = "dlss_gpu_transfer_"
# The precondition set the chain asks for and the source-tree gate ran: guided,
# NVOF motion, GPU residual, this build's one DLSS host. ``working_scale`` is the
# default below and is overwritten per run by the validated scale.
GPU_TRANSFER_SELFTEST_SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 0.5, "residual_pyramid": True, "residual_strength": 1.0,
    "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
    "export_backend": "gpu_transfer",
}
# Read out of the finished run's own record: what answered, and the boundaries a
# frame may not have crossed. A None counter means uninstrumented, so every field
# here has to answer with a number for the probe to call the run clean. The NVOF
# device-input counters are deliberately absent: on this chain they count copies
# the flow engine makes on the card, and a guided run is expected to move them.
GPU_TRANSFER_SELFTEST_ZERO_COUNTERS = (
    "decode_host_copy_bytes_measured",
    "decode_host_copy_count_measured",
    "dlss_host_color_stage_bytes_measured",
    "dlss_output_readback_bytes_measured",
    "residual_output_readback_bytes_measured",
    "encoder_host_pixel_bytes_measured",
)
GPU_TRANSFER_SELFTEST_RUN_COUNTERS = (
    "decode_frames_decoded_measured",
    "decode_device_frame_count_measured",
    "nv12_to_rgba_convert_count_measured",
    "working_color_process_count_measured",
    "encoder_frames_submitted_measured",
    "encoder_packets_written_measured",
    "gpu_transfer_dlss_flow_calls",
)
# The two fields only a downscaled chain can answer for. On the 1.0 identity chain
# neither the working-colour bridge nor the residual bridge is opened at all, so the
# honest report is None for both: a zero there would be a number nobody measured.
# The source-tree identity gate refuses to make that claim, and so does this probe.
GPU_TRANSFER_SELFTEST_DOWNSCALED_RUN_COUNTERS = ("working_color_process_count_measured",)
GPU_TRANSFER_SELFTEST_DOWNSCALED_ZERO_COUNTERS = ("residual_output_readback_bytes_measured",)


def _gpu_transfer_served_range(gpu_transfer_export):
    """The working scales this backend serves, as the module that serves them states
    them: one whole-percentage range. A probe that enumerated the ratios it liked
    would go on refusing percentages the gate has since accepted, which is the
    direction this drifted in before the range existed.
    """
    return (gpu_transfer_export.WORKING_PERCENT_MIN,
            gpu_transfer_export.WORKING_PERCENT_MAX)


def _gpu_transfer_selftest_scale(value):
    """The requested ratio as a float, or None if it is not a number at all."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_transfer_selftest_branch(gpu_transfer_export, src_w, src_h, scale):
    """Which chain shape the production gate will accept this request on, or None.

    Asked of the functions the gate itself uses -- ``working_percent`` for the
    range, ``pipeline.resolve_working_dims`` for the extents,
    ``plan_driven_scale`` for the plan-driven percentages, and the same two
    extent equalities ``negotiate`` matches the fixed shapes on -- so this probe
    holds a finished run to the classification the chain makes rather than to a
    second copy of it. No tap table, kernel or resample is re-derived here.

    The tap-width test is the other half of the same arithmetic: a percentage
    whose plan is wider than the bridge walks is refused by the gate before it
    decodes a frame, and this says so before it opens a DLSS session. The window
    the kernel walks is sized for the widest plan any percentage in the range can
    resolve to on any source the bridge opens, so on ordinary 4K or 720p extents
    this answers None for a request the range does not contain (a third of the
    source) or for a source too small for 8-alignment to leave a downsample to
    do -- not for a ratio somebody had not measured yet. It is measured against
    ``gpu_transfer_export.LANCZOS_MAX_TAPS``, the window this build's own kernel
    compiles: a packaged run against the older bridge, which exports no
    ``gpu_lanczos_max_taps`` and walks 16, classifies here by the wider figure and
    is refused by the gate it then hands the job to. That drift is a failed selftest
    rather than a delivered file, which is the direction this probe is written to
    fail in.

    A classification that drifts from the gate's shows up here as a refusal or as a
    failed check, never as a pass: that direction is the point.
    """
    percent = gpu_transfer_export.working_percent(scale)
    live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
    dims = [int(live_w), int(live_h)]
    if percent == 50 and (live_w, live_h) == (src_w // 2, src_h // 2):
        return {"resampling": "exact_two_x", "working_dims": dims, "downscaled": True}
    if (percent == gpu_transfer_export.WORKING_PERCENT_MAX
            and (live_w, live_h) == (src_w, src_h)):
        return {"resampling": "identity", "working_dims": dims, "downscaled": False}
    if gpu_transfer_export.plan_driven_scale(src_w, src_h, live_w, live_h, scale) is not None:
        color_taps, upscale_taps = gpu_transfer_export.plan_tap_widths(
            src_w, src_h, live_w, live_h)
        if color_taps > gpu_transfer_export.LANCZOS_MAX_TAPS:
            return None
        if upscale_taps > gpu_transfer_export.RESIDUAL_MAX_TAPS:
            return None
        return {"resampling": "plan_driven", "working_dims": dims, "downscaled": True}
    return None


def _gpu_transfer_selftest_transcode_settings(path):
    """The encoder payload a caller hands this probe, read and validated.

    The Delivery page builds ``transcode_settings`` from its own state and puts it on
    the queue; the queue is what passes it to the chain, and a standalone bundle has
    no queue. So the only way a packaged build can be shown to accept the payload an
    operator's panel actually emits is to hand this entry the payload itself, read off
    a request the interface made.

    It is validated rather than trusted, and validated narrowly: it has to be a file
    that parses as a JSON *object*. What the keys mean is the gate's business --
    ``gpu_transfer_export.negotiate`` already refuses an encoding choice this backend
    cannot apply, and a probe that pre-filtered the dictionary would be a second,
    weaker copy of that gate. A refusal here therefore happens before a device session
    opens, and never by dropping a key on the floor.
    """
    import hashlib
    import json

    path = os.path.abspath(path)
    with open(path, "rb") as handle:
        blob = handle.read()
    provenance = {"path": path, "bytes": len(blob),
                  "sha1": hashlib.sha1(blob).hexdigest()[:12]}
    try:
        loaded = json.loads(blob.decode("utf-8"))
    except UnicodeDecodeError as bad:
        raise ValueError(f"transcode_settings file {path} is not UTF-8 JSON: {bad}")
    except ValueError as bad:
        raise ValueError(f"transcode_settings file {path} is not JSON: {bad}")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"transcode_settings file {path} holds a "
            f"{type(loaded).__name__}, not the JSON object the queue carries; this "
            "entry takes a file containing the transcode_settings dictionary itself")
    return loaded, provenance


def _gpu_transfer_selftest_tune_held(gpu_transfer_export, pipeline, codec,
                                     requested, reported):
    """Whether the tune the run reported is the tune the loaded encoder answers for.

    ``plan_encoder`` changes a requested ``uhq`` to ``hq`` when the encoder that has to
    honour it does not have it, and it asks the installation the bridges bound to rather
    than whichever ``ffmpeg`` PATH would answer. This re-asks that same predicate, of
    the same CLI, for the same codec: an encoder that received ``uhq`` has to advertise
    it, and one that received something else has to have been asked for something it
    does not advertise, or for nothing at all. Answering the downgrade question here
    would be a second copy of the plan's rule, so this only ever asks the encoder.
    """
    advertises = bool(pipeline._supports_tune_uhq(
        gpu_transfer_export.encoder_cli(), codec))
    if reported == "uhq":
        return advertises
    if requested == "uhq":
        return reported == "hq" and not advertises
    return True


def _selftest_argv_value(argv, flag):
    """The ``key=value`` tokens that follow ``flag``, so the frozen CLI takes a
    path or a frame count without carrying an argument parser."""
    args = {}
    if flag in argv:
        for token in argv[argv.index(flag) + 1:]:
            key, sep, value = token.partition("=")
            if sep and key:
                args[key.lstrip("-").lower()] = value
    return args


def _under(path, root):
    path = os.path.normcase(os.path.abspath(path or ""))
    root = os.path.normcase(os.path.abspath(root or ""))
    return path == root or path.startswith(root + os.sep)


def _gpu_transfer_selftest(video=None, out_path=None, frames=6, working_scale=None,
                           report_path=None, transcode_settings_path=None):
    """Run one real GPU-transfer export inside this layout, and report what answered.

    The window cannot reach this backend: ``export_backend`` is a queue setting and
    the queue lives in the service, which the standalone bundle does not ship. So the
    bundle has never exported on the device chain, and "the frozen build works" has
    rested on one still frame from ``--selftest``. This closes that gap the same way
    the source-tree gate does -- ``pipeline.export_dlss_stream`` with
    ``export_backend=gpu_transfer`` -- and adds nothing of its own: the fixture is
    generated with this layout's FFmpeg, the chain is the production chain, and the
    verdict is read off the run's own counters and an ffprobe of the deliverable.

    ``working_scale`` is one of the two settings this probe moves, and it is validated
    rather than trusted: it has to be a number, and the extents it resolves to on
    this fixture have to be a chain the production gate serves. A ratio outside that
    set is a refusal naming the served ones, because an unvalidated scale reaching
    the chain would be refused by the gate anyway -- with a DLSS session already
    opened, and with this probe reporting only the tail of the reason.

    ``transcode_settings_path`` is the other one, and it is the payload the delivery
    panel builds rather than a dictionary this file restates: the queue is what hands
    a job its ``transcode_settings``, a standalone bundle ships no queue, and so the
    packaged build has never been shown to accept what the interface emits. Read with
    ``_gpu_transfer_selftest_transcode_settings``, handed unchanged to the same
    ``pipeline.export_dlss_stream`` the queue would hand it to, and written back out
    beside the file it came from. With no path given the run negotiates against an
    empty dictionary, which is the encoder plan this probe has always used.

        DLSS5Tool.exe --gpu-transfer-selftest [frames=6] [scale=0.5] [video=path]
                                                     [out=path] [report=path]
                                                     [transcode_settings=path]
    """
    import hashlib
    import json
    import subprocess
    import tempfile

    import bundle_paths
    import gpu_transfer_export
    import optical_flow
    import transfer_profiler

    report_path = report_path or os.path.join(tempfile.gettempdir(),
                                             "dlss5_gpu_transfer_selftest.json")
    workdir = os.path.abspath(tempfile.mkdtemp(prefix="dlss5_gpu_transfer_selftest_"))
    contents = os.path.abspath(bundle_paths.contents_dir())
    facts = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": os.path.abspath(sys.executable),
        "cwd": os.path.abspath(os.getcwd()),
        "contents_dir": contents,
        "workdir": workdir,
        "requested_frames": frames,
        "requested_working_scale": working_scale,
        "requested_transcode_settings": transcode_settings_path,
    }
    ok = False
    try:
        frames = max(GPU_TRANSFER_SELFTEST_FRAMES[0],
                     min(GPU_TRANSFER_SELFTEST_FRAMES[1], int(frames)))
        facts["frames"] = frames
        scale = (GPU_TRANSFER_SELFTEST_SCALE if working_scale is None
                 or str(working_scale).strip() == ""
                 else _gpu_transfer_selftest_scale(working_scale))
        served = _gpu_transfer_served_range(gpu_transfer_export)
        if scale is None:
            raise ValueError(
                f"scale={working_scale!r} is not a number; this entry takes a "
                f"working_scale from {served[0]}% to {served[1]}%")
        if gpu_transfer_export.working_percent(scale) is None:
            raise ValueError(
                f"working_scale={scale:g} is not a whole percentage between "
                f"{served[0]} and {served[1]}, which is the range this backend "
                "serves; each percentage in it resolves through "
                "pipeline.resolve_working_dims and what a given source can still "
                "refuse depends on those extents and the tap window they need")
        facts["working_scale"] = scale
        # Read before a fixture is generated or a session is opened: a file that is not
        # a JSON object is a refusal, and a refusal that first ran FFmpeg into a temp
        # directory would leave the artifact of a run that never happened.
        if transcode_settings_path:
            transcode_settings, transcode_settings_file = (
                _gpu_transfer_selftest_transcode_settings(transcode_settings_path))
        else:
            transcode_settings, transcode_settings_file = {}, None
        facts["transcode_settings_file"] = transcode_settings_file
        facts["transcode_settings_accepted"] = dict(transcode_settings)
        ffmpeg = pipeline._bundle_ffmpeg()
        ffprobe = pipeline._bundle_ffprobe()
        facts["ffmpeg_path"] = ffmpeg
        facts["ffprobe_path"] = ffprobe
        facts["bridge_dll_directories"] = gpu_transfer_export.ffmpeg_dll_directories()
        facts["native_dir"] = gpu_transfer_export.NATIVE_DIR
        # Every binary this run resolves has to belong to the layout it was started
        # from. A PATH answer inside a bundle is the fault this probe would otherwise
        # report as success: the export would have run against the checkout's tools.
        outside = [name for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe),
                                           ("native_dir", gpu_transfer_export.NATIVE_DIR))
                   if not _under(path, contents)]
        facts["binaries_outside_contents"] = outside

        if not video:
            width, height = GPU_TRANSFER_SELFTEST_SIZE
            video = os.path.join(workdir, "fixture.mkv")
            command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                       "-f", "lavfi",
                       "-i", ("testsrc2=size=%dx%d:rate=%d:duration=%.6f"
                              % (width, height, GPU_TRANSFER_SELFTEST_FPS,
                                 frames / GPU_TRANSFER_SELFTEST_FPS)),
                       "-frames:v", str(frames), "-c:v", "libx264",
                       "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", video]
            subprocess.run(command, check=True, timeout=300,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with open(video, "rb") as handle:
                blob = handle.read()
            facts["fixture_command"] = command
            facts["fixture"] = {"generated": True, "path": video, "bytes": len(blob),
                                "sha1": hashlib.sha1(blob).hexdigest()[:12]}
        else:
            facts["fixture"] = {"generated": False, "path": os.path.abspath(video)}
        if not out_path:
            out_path = os.path.join(workdir, "gpu_transfer_selftest.mkv")
        facts["output"] = os.path.abspath(out_path)

        # The audio request the probe hands the chain, and the encoder plan read
        # above. Both are named once because the copy-pass expectation below is
        # computed from these same values: an expectation built from different inputs
        # than the run got would be a second opinion, not a check. The probe asks for
        # no audio; a payload that names an audio mode gets it through the copy pass
        # only if the source has a stream to copy, which this fixture does not.
        with_audio = False
        n, fps, w, h = pipeline.video_info(video)
        branch = _gpu_transfer_selftest_branch(gpu_transfer_export, w, h, scale)
        if branch is None:
            raise ValueError(
                f"working_scale={scale:g} resolves to "
                f"{list(pipeline.resolve_working_dims(w, h, scale))} from {w}x{h}, "
                "which is no chain this backend serves for these source extents; the "
                "gate would refuse the job before it decoded a frame")
        facts["branch"] = branch
        settings = dict(GPU_TRANSFER_SELFTEST_SETTINGS)
        settings["working_scale"] = scale
        facts["settings"] = dict(settings)
        live_w, live_h = branch["working_dims"]
        facts["source"] = {"frames": n, "fps": fps, "width": w, "height": h,
                           "working": [live_w, live_h],
                           "probe": gpu_transfer_export.probe_source(video, ffprobe)}
        # A copy-pass run writes a temp intermediate, muxes over it and unlinks it;
        # a leak is the one way this entry can leave a device-produced file behind in
        # a directory the caller never named, so the set is taken before the run.
        temps_before = {name for name in os.listdir(tempfile.gettempdir())
                        if name.startswith(GPU_TRANSFER_SELFTEST_INTERMEDIATE_PREFIX)}

        service = optical_flow.get_frame_guidance_service(
            settings["flow_engine"], settings["flow_quality"])
        provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)
        collector = transfer_profiler.TransferProfiler()
        diagnostics = {}
        live = dlss_gpu_engine.Live(live_w, live_h, settings)
        started = time.time()
        try:
            finished = pipeline.export_dlss_stream(
                video=video, live=live, frames=frames, fps=fps, cq=24,
                with_audio=with_audio, out_path=out_path,
                transcode_settings=transcode_settings,
                job_settings=settings, flow_provider=provider,
                diagnostics_out=diagnostics, transfer_collector=collector)
        finally:
            live.close()
            temps_after = {name for name in os.listdir(tempfile.gettempdir())
                           if name.startswith(GPU_TRANSFER_SELFTEST_INTERMEDIATE_PREFIX)}
        facts["elapsed_sec"] = round(time.time() - started, 3)
        facts["leaked_intermediates"] = sorted(temps_after - temps_before)
        facts["returned_path"] = finished
        facts["diagnostics"] = diagnostics
        # What the encoder was actually handed, as the run itself says. A request and
        # a resolution are different facts -- one value the loaded build does not have
        # is downgraded by the plan, and a container, an audio mode or a suffix never
        # reaches the encoder at all -- so the probe reports the answer rather than
        # restating the question and calling it a check.
        facts["encoder_options"] = {
            "codec": diagnostics.get("gpu_transfer_encoder_codec"),
            "av_options": diagnostics.get("gpu_transfer_encoder_av_options"),
            "struct": diagnostics.get("gpu_transfer_encoder_options"),
            "bridge_dll_directory": diagnostics.get("gpu_transfer_bridge_dll_directory"),
        }
        # The directory the decode and encode bridges got their FFmpeg libraries
        # from is the one place a bundle can quietly lose to the machine: the two
        # import avcodec at load time, and a PATH hit registers ahead of nothing.
        facts["binaries_outside_contents"] += [
            "bridge_dll_directory=" + directory
            for directory in (diagnostics.get("gpu_transfer_bridge_dll_directory") or [])
            if not _under(directory, contents)]
        facts["per_frame_backends"] = sorted({
            str(r.get("export_backend_actual")) for r in collector.records})
        facts["per_frame_hosts"] = sorted({
            str(r.get("dlss_backend_actual")) for r in collector.records})
        # The per-frame half of the chain shape. The session report says what the
        # run decided once; this says no frame was served by a stage the branch has
        # none of, which is where a mixed run would otherwise hide.
        facts["per_frame_residual_backends"] = sorted({
            str(r.get("residual_backend")) for r in collector.records})
        facts["per_frame_flow_copies"] = sum(
            int(r.get("nvof_device_input_copy_count") or 0) for r in collector.records)
        facts["frames_recorded"] = len(collector.records)

        dump = json.loads(subprocess.check_output(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
             "-show_entries", "packet=pts_time", "-count_packets",
             "-select_streams", "v:0", facts["output"]], timeout=300
        ).decode("utf-8", "replace"))
        stream = (dump.get("streams") or [{}])[0]
        packets = [p.get("pts_time") for p in dump.get("packets") or []]
        facts["ffprobe"] = {
            "codec": stream.get("codec_name"),
            "size": "%sx%s" % (stream.get("width"), stream.get("height")),
            "pix_fmt": stream.get("pix_fmt"), "range": stream.get("color_range"),
            "matrix": stream.get("color_space"),
            "packets": int(stream.get("nb_read_packets") or 0),
            "first_pts": packets[0] if packets else None,
            "last_pts": packets[-1] if packets else None,
            "bytes": os.path.getsize(facts["output"]),
        }

        # The retired host may not be anywhere in the layout that just ran, and the
        # modules that ran may not have been read out of a source checkout. A bundle
        # is walked whole; a source tree is walked at the directories this build
        # actually loads from, because its root holds a checkout's history.
        roots = [contents, bundle_paths.native_dir(),
                 os.path.dirname(facts["executable"])]
        found = []
        for root in roots:
            if not facts["frozen"]:
                found += [os.path.join(root, name) for name in
                          (os.listdir(root) if os.path.isdir(root) else [])
                          if "dlssnr_host" in name.lower()]
                continue
            for dirpath, _dirs, names in os.walk(root):
                found += [os.path.join(dirpath, name) for name in names
                          if "dlssnr_host" in name.lower()]
        facts["retired_host_files"] = sorted(set(found))
        facts["module_paths_outside_contents"] = sorted(
            f"{module.__name__}={getattr(module, '__file__', None)}"
            for module in (pipeline, gpu_transfer_export, optical_flow,
                           transfer_profiler, dlss_gpu_engine)
            if getattr(module, "__file__", None)
            and not _under(module.__file__, contents))

        checks = {}
        checks["return_is_output"] = os.path.abspath(finished or "") == facts["output"]
        checks["deliverable_exists"] = os.path.isfile(facts["output"])
        downscaled = branch["downscaled"]
        expected_residual = ("gpu_residual_bridge_d3d12" if downscaled else "none")
        # Which chain the run says it ran, held to the classification the gate made
        # when this probe asked for the ratio. A run that answered another resampling
        # shape, another working extent or another residual backend ran a chain this
        # probe did not ask for -- and every counter below would still be satisfied
        # by it, which is exactly why the branch is asserted and not inferred.
        checks["reported_working_scale"] = diagnostics.get(
            "gpu_transfer_working_scale") == scale
        checks["reported_resampling"] = (
            diagnostics.get("gpu_transfer_resampling") == branch["resampling"])
        checks["reported_working_dims"] = list(
            diagnostics.get("gpu_transfer_working_dims") or []) == branch["working_dims"]
        checks["reported_residual_backend"] = diagnostics.get(
            "gpu_transfer_residual_backend") == expected_residual
        checks["per_frame_residual_backend"] = (
            facts["per_frame_residual_backends"] == [expected_residual])
        # The container question, asked of the production function that answers it and
        # with the inputs this run was actually handed. It is not a scale question:
        # what decides it is the output container, the encoder plan and whether the
        # source has anything worth rewrapping for.
        expected_copy_pass = bool(gpu_transfer_export.needs_copy_pass(
            facts["source"]["probe"], transcode_settings, with_audio, facts["output"]))
        checks["reported_copy_pass"] = (
            diagnostics.get("gpu_transfer_copy_pass") == expected_copy_pass)
        checks["requested_gpu_transfer"] = (
            diagnostics.get("export_backend_requested") == "gpu_transfer")
        checks["ran_gpu_transfer"] = diagnostics.get("export_backend_actual") == "gpu_transfer"
        checks["no_fallback_reason"] = diagnostics.get(
            "export_backend_fallback_reason") is None
        checks["host_is_gpu_flow"] = facts["per_frame_hosts"] == ["gpu_flow_d3d12"]
        checks["nvof_bridge_loaded"] = diagnostics.get("gpu_transfer_nvof_bridge_loaded") is True
        checks["flow_drove_every_frame"] = (
            diagnostics.get("gpu_transfer_dlss_flow_calls") == frames
            and diagnostics.get("gpu_transfer_dlss_zero_motion_calls") == 0)
        checks["probe_saw_gpu_transfer_frames"] = facts["per_frame_backends"] == ["gpu_transfer"]
        checks["packets_equal_frames"] = facts["ffprobe"]["packets"] == frames
        checks["encoder_output_shape"] = (
            facts["ffprobe"]["codec"] == "av1"
            and facts["ffprobe"]["size"] == "%dx%d" % (w, h)
            and facts["ffprobe"]["pix_fmt"] == "yuv420p")
        # Asked only of a run that was handed a payload: with an empty dictionary
        # nothing was requested, so there is no request to hold the answer against and
        # these checks would be a pass nobody earned. The resolved struct is the run's
        # own record of what the bridge received.
        resolved = diagnostics.get("gpu_transfer_encoder_options") or {}
        rate_mode = str(transcode_settings.get("rate_mode", "cq"))
        if transcode_settings:
            if "codec" in transcode_settings:
                checks["requested_encoder_codec"] = (
                    diagnostics.get("gpu_transfer_encoder_codec")
                    == str(transcode_settings["codec"]))
            if "cq" in transcode_settings and rate_mode == "cq":
                checks["requested_encoder_cq"] = (
                    resolved.get("constant_quality") == int(transcode_settings["cq"]))
            if "tune" in transcode_settings:
                requested_tune = str(transcode_settings["tune"])
                reported_tune = resolved.get("tune")
                # An encoder that received nothing has to have been asked for nothing;
                # one that received something has to say which. A silent drop -- a tune
                # requested and no tune reported, or reported to a different encoder
                # than the one that answered -- is how a payload would stop being a
                # payload and go back to being this probe's own defaults.
                checks["requested_encoder_tune_reported"] = bool(
                    reported_tune) == (requested_tune not in ("", "auto", "None"))
                checks["encoder_tune_matches_encoder"] = _gpu_transfer_selftest_tune_held(
                    gpu_transfer_export, pipeline,
                    str(diagnostics.get("gpu_transfer_encoder_codec")
                        or transcode_settings.get("codec", "av1_nvenc")),
                    requested_tune, reported_tune)
        for name in GPU_TRANSFER_SELFTEST_RUN_COUNTERS:
            if not downscaled and name in GPU_TRANSFER_SELFTEST_DOWNSCALED_RUN_COUNTERS:
                continue
            checks["counter:" + name] = diagnostics.get(name) == frames
        for name in GPU_TRANSFER_SELFTEST_ZERO_COUNTERS:
            if not downscaled and name in GPU_TRANSFER_SELFTEST_DOWNSCALED_ZERO_COUNTERS:
                continue
            checks["zero:" + name] = diagnostics.get(name) == 0
        if not downscaled:
            # On the identity chain these two stages never opened, so None is the
            # only honest figure. A zero would be a measurement nobody took, and
            # accepting one here is the same mistake as accepting a frame that
            # crossed to host memory.
            for name in (GPU_TRANSFER_SELFTEST_DOWNSCALED_RUN_COUNTERS +
                         GPU_TRANSFER_SELFTEST_DOWNSCALED_ZERO_COUNTERS):
                checks["unmeasured:" + name] = diagnostics.get(name) is None
        checks["no_leaked_intermediate"] = not facts["leaked_intermediates"]
        # The scan above is written as a promise ("the retired host may not be anywhere in
        # the layout that just ran") but nothing read its answer, so a tree that still
        # carried dlssnr_host.dll reported GPU_TRANSFER_OK. A second host in the layout is
        # exactly what this gate exists to catch.
        checks["no_retired_host_files"] = not facts["retired_host_files"]
        # Only a bundle is claimed to carry its own tools. A source tree runs
        # against the machine's FFmpeg install by design, so demanding the same
        # containment there would fail a layout that is working as intended.
        checks["modules_from_this_layout"] = (not facts["frozen"]
                                             or not facts["module_paths_outside_contents"])
        checks["binaries_from_this_layout"] = (not facts["frozen"]
                                               or not facts["binaries_outside_contents"])
        facts["checks"] = checks
        ok = all(checks.values())
        facts["result"] = "GPU_TRANSFER_OK" if ok else "GPU_TRANSFER_INCOMPLETE"
        facts["failed_checks"] = sorted(name for name, value in checks.items() if not value)
    except Exception as ex:
        facts["result"] = "GPU_TRANSFER_ERR"
        facts["error"] = repr(ex)
        facts["traceback"] = traceback.format_exc()
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(facts, handle, indent=2, default=str, ensure_ascii=False)
    facts["report_path"] = os.path.abspath(report_path)
    print("%s frames=%s scale=%s packets=%s report=%s%s" % (
        facts.get("result"), frames,
        ("%g" % facts["working_scale"]) if facts.get("working_scale") is not None
        else (facts.get("requested_working_scale") or "?"),
        (facts.get("ffprobe") or {}).get("packets"),
        facts["report_path"],
        (" failed=" + ",".join(facts.get("failed_checks") or []))
        if facts.get("failed_checks") else ""), flush=True)
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        _selftest(); return
    if GPU_TRANSFER_SELFTEST_FLAG in sys.argv:
        args = _selftest_argv_value(sys.argv, GPU_TRANSFER_SELFTEST_FLAG)
        sys.exit(_gpu_transfer_selftest(video=args.get("video"),
                                        out_path=args.get("out"),
                                        frames=args.get("frames", 6),
                                        working_scale=args.get("scale",
                                                               args.get("working_scale")),
                                        report_path=args.get("report"),
                                        transcode_settings_path=args.get(
                                            "transcode_settings")))
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
