#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — High-Performance Localhost Backend Server for DLSS5 Studio.

Exposes REST and WebSocket endpoints for:
- Frame decoding and real-time DLSS neural processing
- Multi-Engine Optical Flow (NVIDIA Optical Flow Hardware / OpenCV DIS CPU / RAFT-Large in-memory)
- High-speed WebSocket streaming playback
- DaVinci-style render queue management
"""
import os
import re
import secrets
import sys
import time
import json
import base64
import asyncio
import traceback
import atexit
from typing import Optional
from contextlib import asynccontextmanager

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from dlss_service import dlss_service, diagnostic_logger
from queue_payload import diagnostic_json, queue_job, queue_message
import pipeline
from runtime_environment import quick_system_info, environment_report

# ---------------- Local control-plane security (P0-01) ----------------
# The backend binds localhost but any *web page* can still reach 127.0.0.1.
# Two layers: (1) Origin/Host gate blocks non-local web origins outright;
# (2) per-launch session token required for everything except readiness.
LOCAL_TOKEN = os.environ.get("DLSS_LOCAL_TOKEN") or secrets.token_urlsafe(32)
if not os.environ.get("DLSS_LOCAL_TOKEN"):
    print("[security] Generated a local session token.", flush=True)

_TOKEN_FILE = os.path.join(os.environ.get('LUNARC_DATA_DIR') or os.path.dirname(os.path.abspath(__file__)), ".session_token")


def _save_session_token(token: str) -> None:
    try:
        with open(_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token)
        if hasattr(os, "chmod"):
            try:
                os.chmod(_TOKEN_FILE, 0o600)
            except Exception:
                pass
    except Exception as e:
        print(f"[security] Warning: could not write session token to {_TOKEN_FILE}: {e}", flush=True)

    try:
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        udir = os.environ.get('LUNARC_DATA_DIR') or os.path.join(appdata, "DaVinci_DLSS5_Studio")
        os.makedirs(udir, exist_ok=True)
        user_token_file = os.path.join(udir, ".session_token")
        with open(user_token_file, "w", encoding="utf-8") as f:
            f.write(token)
        if hasattr(os, "chmod"):
            try:
                os.chmod(user_token_file, 0o600)
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_session_token() -> None:
    try:
        if os.path.exists(_TOKEN_FILE):
            with open(_TOKEN_FILE,encoding='utf-8') as token_file:
                matches=token_file.read().strip()==LOCAL_TOKEN
            if matches:os.remove(_TOKEN_FILE)
    except Exception:
        pass


atexit.register(_cleanup_session_token)

_LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", re.IGNORECASE)
_ALLOWED_EXACT_ORIGINS = {"app://studio"}


def _origin_allowed(origin: str | None) -> bool:
    # Browsers always send Origin on cross-origin fetch/WS. Absent covers
    # curl/python clients and native same-process fetches without an Origin header.
    # We strictly allow loopback http(s) and the exact production Electron scheme app://studio.
    # NOTE: "null", "file://", "devtools://", and broad "app://.*" are strictly disallowed.
    if not origin:
        return True
    o = origin.strip().lower()
    if o in _ALLOWED_EXACT_ORIGINS:
        return True
    return bool(_LOCAL_ORIGIN_RE.match(origin.strip()))


def _host_allowed(host: str | None) -> bool:
    # DNS-rebinding guard: Host must be this machine's loopback.
    if not host:
        return True
    h = host.strip().lower().split(",")[0].strip().strip("[]")
    name = h.rsplit(":", 1)[0] if ":" in h and h.count(":") == 1 else h
    return name in ("localhost", "127.0.0.1", "::1")


def _token_ok(token: str | None) -> bool:
    return bool(token) and secrets.compare_digest(token, LOCAL_TOKEN)


def _request_token(request: Request) -> str | None:
    return request.headers.get("x-dlss-token") or request.query_params.get("token")


# Readiness probe stays open (Electron main + test harness poll it tokenless).
_TOKEN_OPEN = {("GET", "/api/system")}

# Active WebSocket connections for queue progress
queue_ws_clients = set()
main_loop: Optional[asyncio.AbstractEventLoop] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()
    _save_session_token(LOCAL_TOKEN)
    try:
        async with _mcp_http.router.lifespan_context(_mcp_http):
            yield
    finally:
        try:
            engine=getattr(dlss_service,'_gpu_preview',None)
            if engine is not None:
                await asyncio.to_thread(engine.shutdown)
                dlss_service._gpu_preview=None
        finally:
            _cleanup_session_token()


app = FastAPI(title="DLSS 5 Studio Backend API", version="2.0.0", lifespan=lifespan)

from automation import Automation, create_router as automation_router
automation = Automation(dlss_service)

app.add_middleware(
    CORSMiddleware,
    # Never "*" together with credentials: only this machine's web origins
    # and the exact Electron production scheme (app://studio) may read responses.
    # No file://, no null, no devtools://, and no broad app:// wildcards.
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|app://studio)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=['X-Comparison-Actual-Time', 'X-Comparison-Actual-PTS',
                    'X-Comparison-Delta', 'Content-Range', 'Accept-Ranges'],
)


_comparison_service = None


def comparison_service():
    global _comparison_service
    if _comparison_service is None:
        from comparison_media import MediaComparisonService
        _comparison_service = MediaComparisonService()
    return _comparison_service


@app.post('/api/compare/open')
async def compare_open(request: Request):
    try:
        data = await request.json()
    except ValueError as error:
        raise HTTPException(400, 'Invalid comparison request') from error
    if not isinstance(data, dict):
        raise HTTPException(400, 'Comparison request must contain two file paths')
    try:
        return await asyncio.to_thread(comparison_service().open_pair,
                                       data.get('original'), data.get('exported'),
                                       data.get('original_offset_seconds', 0), data.get('duration_seconds'))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get('/api/compare/frame')
async def compare_frame(session: str, time: float = 0, side: str = 'original'):
    try:
        result = await asyncio.to_thread(comparison_service().get_frame, session, time, side)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Response(result['jpeg'], media_type='image/jpeg', headers={
        'X-Comparison-Actual-Time': str(result['actual_time']),
        'X-Comparison-Actual-PTS': str(result['actual_pts']),
        'X-Comparison-Delta': str(result['delta_seconds']), 'Cache-Control': 'no-store'})


@app.get('/api/compare/timeline')
async def compare_timeline(session: str, time: float = 0):
    try:
        return await asyncio.to_thread(comparison_service().get_timeline, session, time)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.delete('/api/compare/{session}')
async def compare_close(session: str):
    return {'closed': comparison_service().close_session(session)}


@app.get('/api/compare/audio')
async def compare_audio(request: Request, session: str, side: str = 'original'):
    from comparison_audio import wav_range, audio_chunks
    service = comparison_service()
    try:
        source = service.audio_source(session, side)
        header, start, end, total = wav_range(source, request.headers.get('range'))
    except ValueError as error:
        raise HTTPException(416, str(error)) from error
    ranged = bool(request.headers.get('range'))
    headers = {'Accept-Ranges': 'bytes', 'Content-Length': str(end-start+1),
               'Cache-Control': 'no-store'}
    if ranged:
        headers['Content-Range'] = f'bytes {start}-{end}/{total}'
    return StreamingResponse(audio_chunks(service.ffmpeg, source, header, start, end),
                             media_type='audio/wav', status_code=206 if ranged else 200,
                             headers=headers)


@app.middleware("http")
async def _local_only_gate(request: Request, call_next):
    # NOTE: "http" middleware does not cover websocket routes; WS handlers
    # enforce the same checks inline via _check_ws_access().
    if not _origin_allowed(request.headers.get("origin")):
        return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
    if not _host_allowed(request.headers.get("host")):
        return JSONResponse({"detail": "Forbidden host (possible DNS rebinding)"}, status_code=403)
    if request.url.path.rstrip('/')=='/mcp':
        # URL-only native clients need no copied secret. Only this explicit local
        # endpoint is exempt: REST and settings keep the existing token checks.
        import ipaddress
        try:loopback=ipaddress.ip_address(request.client.host).is_loopback
        except (AttributeError,ValueError):loopback=False
        if not loopback:
            return JSONResponse({'detail':'MCP only accepts loopback connections'},status_code=403)
        if (request.headers.get('origin') or request.headers.get('sec-fetch-site')) and not _token_ok(_request_token(request)):
            return JSONResponse({'detail':'Browser MCP requests require the local token'},status_code=403)
        if not automation.control.enabled:
            return JSONResponse({'detail':'MCP 访问已关闭，请在软件设置中开启。'},status_code=403)
        return await call_next(request)
    # CORS preflight (OPTIONS) does not carry custom credentials/headers like X-DLSS-Token.
    # Once origin and host checks pass, allow preflight through to CORSMiddleware.
    if request.method == "OPTIONS":
        return await call_next(request)
    if (request.method, request.url.path) not in _TOKEN_OPEN and not _token_ok(_request_token(request)):
        origin = request.headers.get("origin")
        resp_headers = {}
        if origin and _origin_allowed(origin):
            resp_headers["Access-Control-Allow-Origin"] = origin
            resp_headers["Access-Control-Allow-Credentials"] = "true"
        return JSONResponse(
            {"detail": "Missing or invalid local token (X-DLSS-Token header or ?token=)"},
            status_code=401,
            headers=resp_headers,
        )
    return await call_next(request)


def _check_ws_access(websocket: WebSocket) -> str | None:
    """Return None when allowed, else a human-readable rejection reason."""
    if not _origin_allowed(websocket.headers.get("origin")):
        return "forbidden origin"
    if not _host_allowed(websocket.headers.get("host")):
        return "forbidden host"
    if not _token_ok(websocket.query_params.get("token")):
        return "missing or invalid token"
    return None


async def broadcast_queue_update(job_or_data):
    """Direct async broadcast to queue WebSockets."""
    automation.notify(job_or_data)
    if not queue_ws_clients:
        return
    if isinstance(job_or_data, dict) and "type" in job_or_data:
        msg = json.dumps(queue_message(job_or_data))
    else:
        msg = json.dumps(queue_message({"type": "job_update", "job": job_or_data,
                          "queue_running": dlss_service.queue_is_running()}))

    for ws in list(queue_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass


def broadcast_queue_update_from_thread(job_or_data):
    """Thread-safe broadcast called from background rendering threads."""
    automation.notify(job_or_data)
    if not queue_ws_clients:
        return
    if isinstance(job_or_data, dict) and "type" in job_or_data:
        msg = json.dumps(queue_message(job_or_data))
    else:
        # A row update cannot answer "may I reset this row?" on its own: between the
        # last row turning terminal and the worker exiting, nothing is running while
        # the queue still owns the GPU session. That is precisely when a retry has to
        # stay disabled, so every update carries the gate the button reads.
        msg = json.dumps(queue_message({"type": "job_update", "job": job_or_data,
                          "queue_running": dlss_service.queue_is_running()}))

    global main_loop
    target_loop = main_loop
    if target_loop and not target_loop.is_closed():
        for ws in list(queue_ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), target_loop)
            except Exception:
                pass


@app.get("/api/system")
def get_system_info(request: Request):
    """Return GPU, VRAM and platform status."""
    health=dlss_service.backend_health()
    return {**quick_system_info(), "client_origin": request.headers.get("origin"),
            "backend_ready":health['healthy'], 'queue_storage_error':health['queue_storage_error']}


@app.get('/api/backend/health')
async def backend_health():
    # Does not acquire the GPU lock or enter a native library, even after a hang.
    return dlss_service.backend_health()


@app.get('/api/gpu/memory')
async def gpu_memory():
    from gpu_memory import snapshot
    return {'memory': snapshot()}


@app.get('/api/environment')
def get_environment(refresh: bool = False):
    return environment_report(refresh, {
        'LUNARC_DLSS_DLL': dlss_service.settings.get('dlss_dll_path'),
        'LUNARC_VSR_RUNTIME': dlss_service.settings.get('vsr_runtime_path'),
    })


@app.get("/api/flow/providers")
def get_flow_providers():
    """Return real capability and status of optical flow engines."""
    import optical_flow
    return {
        "providers": optical_flow.get_all_flow_providers(),
        "nvof_probe": optical_flow.probe_nvof_capabilities(),
        "provider_contracts": __import__("flow_contracts").provider_contracts(),
    }


class RuntimeSourceRequest(BaseModel):
    kind: str
    location: str
    label: str
    sha256: str = ''
    confirmed: bool = False
    archive_sha256: str = ''


class ImageModelRequest(BaseModel):
    key: str
    path: Optional[str] = None


@app.get('/api/image/models')
def image_model_status():
    from image_model_store import store
    return store.status()


@app.post('/api/image/models/download')
def image_model_download(body: ImageModelRequest):
    from image_model_store import store
    try:return store.start(body.key)
    except ValueError as error:raise HTTPException(400,str(error)) from error


@app.post('/api/image/models/import')
def image_model_import(body: ImageModelRequest):
    from image_model_store import store
    try:
        if not body.path:raise ValueError('请选择模型文件')
        return store.import_model(body.key,body.path)
    except (ValueError,OSError) as error:raise HTTPException(400,str(error)) from error


class RuntimeImportRequest(BaseModel):
    path: Optional[str] = None
    source_id: Optional[str] = None
    sha256: str = ''
    catalog_id: Optional[str] = None
    confirmed: bool = False
    activate: bool = True


class RuntimeSelectionRequest(BaseModel):
    id: Optional[str] = None
    allow_unrecognized: bool = False


def _runtime_call(fn):
    from dlssnr_runtime import RuntimeErrorDetail
    try:
        return fn()
    except RuntimeErrorDetail as exc:
        raise HTTPException(409 if exc.code in ('operation_busy', 'active_version') else 400,
                            detail={'code': exc.code, 'message': str(exc)}) from exc
    except OSError as exc:
        raise HTTPException(400, detail={'code': 'cache_io', 'message': '运行库缓存不可读写，请检查权限、空间与文件占用'}) from exc


@app.get('/api/runtime/dlssnr')
def get_dlssnr_runtime():
    from dlssnr_runtime import manager
    return _runtime_call(manager().status)


@app.post('/api/runtime/dlssnr/sources')
def add_dlssnr_source(body: RuntimeSourceRequest):
    from dlssnr_runtime import manager
    return _runtime_call(lambda: manager().add_source(body.kind, body.location, body.label, body.sha256, body.confirmed, body.archive_sha256))


@app.delete('/api/runtime/dlssnr/sources/{source_id}')
def remove_dlssnr_source(source_id: str):
    from dlssnr_runtime import manager
    _runtime_call(lambda: manager().remove_source(source_id))
    return {'ok': True}


@app.post('/api/runtime/dlssnr/import')
def import_dlssnr(body: RuntimeImportRequest):
    from dlssnr_runtime import manager
    value = manager()
    record = _runtime_call(lambda: value.import_file(body.path, body.sha256, body.source_id,
                        catalog_id=body.catalog_id, confirmed=body.confirmed))
    if body.activate:
        _runtime_session_change(lambda: value.activate(record['id']))
    return record


def _runtime_session_change(fn):
    # Queue reserves a running job under this lock before opening its session.
    # Hold it through the switch so a newly started export cannot race selection.
    with dlss_service._queue_lock:
        if dlss_service.active_job_id is not None:
            raise HTTPException(409, detail={'code': 'export_busy', 'message': '导出进行中，请完成或停止导出后切换运行库'})
        if not dlss_service._live_lock.acquire(timeout=2):
            raise HTTPException(409, detail={'code': 'preview_busy', 'message': '预览正在处理，请稍后重试'})
        try:
            dlss_service._close_gpu_preview()  # close on the GPU preview owner thread
            if dlss_service._live_session:
                dlss_service._live_session.close()
                dlss_service._live_session = None
                dlss_service._live_session_dims = dlss_service._live_session_kind = None
            return _runtime_call(fn)
        finally:
            dlss_service._live_lock.release()


@app.post('/api/runtime/dlssnr/activate')
def activate_dlssnr(body: RuntimeSelectionRequest):
    from dlssnr_runtime import manager
    _runtime_session_change(lambda: manager().activate(body.id, body.allow_unrecognized))
    return {'ok': True}


@app.delete('/api/runtime/dlssnr/versions/{version_id}')
def delete_dlssnr(version_id: str):
    from dlssnr_runtime import manager
    _runtime_session_change(lambda: manager().delete(version_id))
    return {'ok': True}


@app.post('/api/runtime/dlssnr/verify')
def verify_dlssnr(body: RuntimeSelectionRequest):
    from dlssnr_runtime import manager
    value = manager()
    return _runtime_call(lambda: value.verified_path(body.id or value.load()['active'], force_signature=True)[1])


@app.post('/api/runtime/dlssnr/cancel')
def cancel_dlssnr_import():
    from dlssnr_runtime import manager
    manager().cancel()
    return {'ok': True}


@app.post("/api/dialog/open_file")
def dialog_open_file():
    """Trigger native Windows file dialog to choose media file."""
    import tkinter as tk
    from tkinter import filedialog
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="选择导入素材 (视频 / 图像)",
            filetypes=[
                ("所有支持的媒体", "*.mp4;*.mkv;*.mov;*.avi;*.wmv;*.flv;*.webm;*.m4v;*.mpg;*.mpeg;*.ts;*.mts;*.m2ts;*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff;*.avif;*.heic;*.heif;*.jxl;*.jxr;*.wdp;*.hdp"),
                ("视频文件", "*.mp4;*.mkv;*.mov;*.avi;*.wmv;*.flv;*.webm;*.m4v;*.mpg;*.mpeg;*.ts;*.mts;*.m2ts"),
                ("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff;*.avif;*.heic;*.heif;*.jxl;*.jxr;*.wdp;*.hdp"),
                ("所有文件", "*.*")
            ]
        )
        root.destroy()
        return {"path": path if path else None}
    except Exception as e:
        return {"path": None, "error": str(e)}


@app.post("/api/dialog/open_dir")
def dialog_open_dir():
    """Trigger native Windows folder picker."""
    import tkinter as tk
    from tkinter import filedialog
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="选择导出保存目录")
        root.destroy()
        return {"path": path if path else None}
    except Exception as e:
        return {"path": None, "error": str(e)}


class LoadVideoRequest(BaseModel):
    path: str


@app.post("/api/video/load")
def load_video(req: LoadVideoRequest):
    try:
        info = dlss_service.load_video(req.path)
        return {"status": "ok", "video": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/video/current")
def current_video():
    return {
        "video": dlss_service.video_info,
        "loaded": dlss_service.video_info is not None
    }


class ImageExportRequest(BaseModel):
    path: str
    output_path: str
    format: str = 'png'
    quality: int = 95
    preserve_alpha: bool = True
    overwrite: bool = False
    color_mode: str = 'sdr'
    render_source: str = 'rerender'
    result_id: Optional[str] = None
    allow_stale: bool = False
    frame_index: Optional[int] = Field(default=None,ge=0)


@app.post('/api/image/export')
def export_image(req: ImageExportRequest):
    try:
        result = dlss_service.export_image(req.path, req.output_path, req.format,
            req.quality, req.preserve_alpha, req.overwrite, req.color_mode,
            req.render_source,req.result_id,req.allow_stale,frame_index=req.frame_index)
        return {'status': 'ok', 'image': result}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


class ImageRenderRequest(BaseModel):
    path: str
    memory_mode: str = 'memory'
    native: bool = False
    frame_index: Optional[int] = Field(default=None,ge=0)


@app.post('/api/image/render')
def render_image(req: ImageRenderRequest):
    try:
        return diagnostic_json(dlss_service.render_image(req.path,req.memory_mode,req.native,req.frame_index))
    except Exception as error:
        raise HTTPException(400,str(error)) from error


@app.get('/api/image/results')
def image_results(path: str, frame_index: Optional[int] = Query(None,ge=0)):
    try:return diagnostic_json(dlss_service.image_result_state(path,frame_index))
    except (ValueError,OSError) as error:raise HTTPException(400,str(error)) from error


@app.get('/api/image/results/preview')
def image_result_preview(result_id: str, reference_id: Optional[str] = None):
    try:payload=dlss_service.image_result_preview(result_id,reference_id)
    except ValueError as error:raise HTTPException(400,str(error)) from error
    return Response(payload,media_type='image/png',headers={'Cache-Control':'no-store'})


class ImageCompareRequest(BaseModel):
    result_id: str
    reference_id: Optional[str] = None


@app.post('/api/image/results/compare')
def compare_image_results(req: ImageCompareRequest):
    try:return dlss_service.compare_image_results(req.result_id,req.reference_id)
    except (ValueError,RuntimeError,OSError) as error:raise HTTPException(400,str(error)) from error


@app.get('/api/image/thumbnail')
def image_thumbnail(path: str, edge: int = Query(720,ge=0,le=8192)):
    from image_source import is_image, read_rgba
    import cv2
    if not is_image(path):
        raise HTTPException(status_code=400, detail='请选择图片素材')
    try:
        rgba = read_rgba(path)
        ratio = min(1, edge / max(rgba.shape[:2])) if edge else 1
        if ratio < 1:
            rgba = cv2.resize(rgba, (max(1,round(rgba.shape[1]*ratio)),max(1,round(rgba.shape[0]*ratio))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise ValueError('图片编码失败')
        return Response(encoded.tobytes(), media_type='image/png')
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get('/api/image/formats')
def image_formats():
    from image_source import export_formats
    from image_hdr_export import capabilities
    return {'formats': export_formats(), 'hdr_formats': capabilities()}


@app.get("/api/preview/diagnostics")
def preview_diagnostics():
    return dlss_service.preview_diagnostics()


@app.get('/api/preview/memory')
def preview_memory_status():
    return dlss_service.preview_memory_state()


@app.post('/api/preview/memory/release')
async def release_preview_memory():
    from preview_player import render_gate,notify_memory
    try:dlss_service.begin_preview_release()
    except (ValueError,RuntimeError) as error:raise HTTPException(409,str(error)) from error
    notify_memory(dlss_service,pause=True)
    try:
        # Do not cancel an in-flight CUDA/NGX call. Retain the global render slot
        # through completion before acquiring the service GPU ownership lock.
        async with render_gate():
            task=asyncio.create_task(asyncio.to_thread(dlss_service.release_preview_memory))
            try:report=await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
        return dict(dlss_service.preview_memory_state(),releasing=False,report=report)
    except Exception as error:
        dlss_service._preview_release_error=str(error)
        raise HTTPException(409,str(error)) from error
    finally:
        dlss_service._preview_release_busy=False
        notify_memory(dlss_service)


@app.post('/api/preview/memory/resume')
async def resume_preview_memory():
    from preview_player import render_gate,notify_memory
    async with render_gate():
        try:state=dlss_service.resume_preview_memory()
        except (ValueError,RuntimeError) as error:raise HTTPException(409,str(error)) from error
    notify_memory(dlss_service)
    return state


@app.get('/api/source/frame')
def source_frame(path: str, frame: int = Query(0, ge=0), edge: int = Query(1600, ge=256, le=1920)):
    try:
        payload = dlss_service.source_preview(path, frame, edge)
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(400, str(error)) from error
    return Response(content=payload, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.get("/api/frame")
def get_frame(
    frame: int = Query(0),
    view: str = Query("dlss"),
    split_x: float = Query(0.5),
    zoom: float = Query(1.0),
    pan_x: float = Query(0.5),
    pan_y: float = Query(0.5),
):
    """Return rendered frame as JPEG image bytes."""
    frame = max(0, int(frame))
    split_x = max(0.0, min(1.0, float(split_x)))
    zoom = max(0.1, min(10.0, float(zoom)))
    pan_x = max(0.0, min(1.0, float(pan_x)))
    pan_y = max(0.0, min(1.0, float(pan_y)))
    jpeg_bytes, msg = dlss_service.render_preview(
        frame_idx=frame,
        view=view,
        split_x=split_x,
        zoom=zoom,
        pan=(pan_x, pan_y)
    )
    if jpeg_bytes is None:
        raise HTTPException(status_code=400, detail=msg)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/settings")
def get_settings():
    return dlss_service.settings


@app.post("/api/settings")
def update_settings(new_settings: dict):
    # A refused write has to say why. Raised bare, a retired dlss_backend name answered
    # HTTP 500 with no body, and the setting the caller named is the one thing the
    # answer has to make a migration out of - same reason /api/queue/add catches this.
    try:
        with dlss_service._settings_lock:
            updated = dlss_service.update_settings(new_settings)
            revision = dlss_service._preview_revision
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "settings": updated, "preview_revision": revision}


# ---------------- Queue Endpoints ----------------
def _queue_state():
    """The one shape the queue is reported in, over HTTP and over the socket.

    ``queue_running`` belongs here because the retry action is gated on it: a row
    reset while a round is in flight would hand the running worker a job it already
    passed over, which is the implicit re-run the explicit action replaces. The UI
    cannot infer it from ``active_id`` - between two jobs that reads as idle.
    """
    return {
        "jobs": [queue_job(job) for job in dlss_service.get_jobs()],
        "active_id": dlss_service.active_job_id,
        "queue_running": dlss_service.queue_is_running(),
    }


@app.get("/api/queue")
async def get_queue():
    return _queue_state()


from preview_segments import PreviewSegments
_preview_segments = PreviewSegments(dlss_service)


@app.post('/api/preview/segments/list')
async def preview_segments_list(data: dict):
    try:
        return {'segments': await asyncio.to_thread(_preview_segments.list,
            data.get('source'), data.get('settings') or {}, data.get('transcode_settings') or {})}
    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@app.post('/api/preview/segments')
async def preview_segments_create(data: dict):
    try:
        record = await asyncio.to_thread(_preview_segments.create, data.get('source'),
            data.get('settings') or {}, data.get('transcode_settings') or {}, data.get('frame_range'))
    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error)) from error
    if not record.get('ready'):
        dlss_service.start_preview_queue(progress_callback=broadcast_queue_update_from_thread)
    await broadcast_queue_update({'type': 'update', **_queue_state()})
    return record


@app.post("/api/queue/add")
async def add_to_queue(job_data: dict):
    # A refused job has to say why: raised bare, a ValueError here answered
    # HTTP 500 with no body, so an opt-in rejection the user could act on
    # (unknown or unpaired export backend) reached the UI as "Internal Server Error".
    try:
        job = dlss_service.add_job(job_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_queue_update({"type": "job_added", "job": job, **_queue_state()})
    return {"status": "ok", "job": queue_job(job)}


@app.post("/api/queue/remove")
async def remove_from_queue(req: dict):
    job_id = req.get("id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job id")
    try:
        dlss_service.remove_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    state = _queue_state()
    await broadcast_queue_update({"type": "job_removed", "id": job_id, **state})
    return {"status": "ok", **state}


@app.post("/api/queue/retry")
async def retry_queue_job(req: dict):
    """Put one named job back to pending - the only way a past failure ever re-renders.

    The queue runs pending jobs and nothing else, so this action exists rather than a
    rerun-on-start side effect. The service decides the refusal under the queue lock,
    and the reason has to reach the row that showed it: a bare ValueError would answer
    HTTP 500 with a body saying nothing about the round still in flight.
    """
    job_id = req.get("id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job id")
    try:
        job = dlss_service.retry_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"作业 [{job_id}] 已不在渲染队列中")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    state = _queue_state()
    await broadcast_queue_update({
        "type": "job_retried", "job": job, "jobs": state["jobs"],
        "active_id": state["active_id"], "queue_running": state["queue_running"],
    })
    return {"status": "ok", "job": queue_job(job), **state}


@app.post("/api/queue/start")
async def start_queue():
    try:
        started = dlss_service.start_queue(progress_callback=broadcast_queue_update_from_thread)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if started:
        # Says the round has begun before the first row turns running, so a client that
        # did not press the button reaches the same locked state as the one that did.
        await broadcast_queue_update({"type": "queue_started", **_queue_state()})
    return {"status": "started" if started else "already_running", **_queue_state()}


@app.post("/api/queue/cancel")
async def cancel_queue():
    dlss_service.cancel_active_job()
    return {"status": "cancelling"}


app.include_router(automation_router(automation, broadcast_queue_update_from_thread))

# ---------------- Diagnostics Endpoints ----------------
@app.get("/api/diagnostics/logs")
def get_diagnostic_logs(limit: int = Query(200)):
    return {"logs": diagnostic_logger.get_logs(limit=limit)}


@app.get("/api/diagnostics/job/{job_id}")
def get_job_diagnostic(job_id: str, summary: bool = False):
    jobs = dlss_service.get_jobs()
    for j in jobs:
        if j["id"] == job_id:
            # The job carries the DLSS host that ran or refused it, and that host's own
            # last error. There is no host log here: the retired dlssnr_host.dll wrote
            # dlssnr_run.log next to itself and a GPU-host failure used to be reported
            # with that file's tail as if it were the GPU host's log.
            return diagnostic_json({
                "job": j,
                "error": j.get("error"),
                "error_traceback": j.get("error_traceback"),
                "dlss_host": j.get("dlss_host"),
                "dlss_host_error": j.get("dlss_host_error")
            }, summary=summary)
    raise HTTPException(status_code=404, detail="Job not found")


@app.post('/api/preview/native')
async def configure_native_preview(request: Request):
    data = await request.json()
    try:
        return await asyncio.to_thread(dlss_service.configure_native_preview,data)
    except (ValueError,RuntimeError) as error:
        from gpu_faults import recovery_reason
        reason=recovery_reason(error)
        if reason:dlss_service._mark_native_degraded(reason)
        raise HTTPException(400,str(error)) from error


@app.get('/api/preview/native')
async def native_preview_status():
    engine=getattr(dlss_service,'_gpu_preview',None)
    if engine is None:return dict(native=False,pending=False)
    try:return await asyncio.to_thread(engine.native_status)
    except RuntimeError as error:
        from gpu_faults import recovery_reason
        reason=recovery_reason(error)
        if reason:dlss_service._mark_native_degraded(reason)
        raise HTTPException(400,str(error)) from error


# ---------------- WebSocket Player ----------------
@app.websocket('/ws/compare')
async def websocket_comparison(websocket: WebSocket, session: str,
                               time: float = 0, rate: float = 1):
    if _check_ws_access(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    from comparison_playback import ComparisonPlayback
    player = None
    async def stopped():
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            pass
    disconnect = asyncio.create_task(stopped())
    try:
        player = await asyncio.to_thread(ComparisonPlayback, comparison_service(), session, time, rate)
        # Decode before starting the clock: startup is not elapsed media time.
        packet = await asyncio.to_thread(player.packet, time)
        await websocket.send_bytes(packet)
        started = asyncio.get_running_loop().time()
        while not disconnect.done():
            elapsed = asyncio.get_running_loop().time() - started
            at = min(player.duration, time + elapsed * rate)
            packet = await asyncio.to_thread(player.packet, at)
            if disconnect.done():
                break
            await websocket.send_bytes(packet)
            if at >= player.duration:
                await websocket.send_json({'type': 'ended', 'time': player.duration})
                break
            wait = max(0., 1 / min(60., player.fps * rate) - (asyncio.get_running_loop().time() - started - elapsed))
            try:
                await asyncio.wait_for(asyncio.shield(disconnect), timeout=wait)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as error:
        if not disconnect.done():
            try:
                await websocket.send_json({'type': 'error', 'detail': str(error)})
            except (WebSocketDisconnect, RuntimeError):
                pass
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)
        if player:
            await asyncio.to_thread(player.close)
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


@app.websocket("/ws/player")
async def websocket_player(websocket: WebSocket):
    """Bounded native work and revision-safe realtime preview."""
    global main_loop
    if main_loop is None or main_loop.is_closed():
        main_loop = asyncio.get_running_loop()
    if _check_ws_access(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    from preview_player import PreviewPlayer
    player = PreviewPlayer(websocket, dlss_service)
    task = asyncio.create_task(player.run())
    try:
        while True:
            command = json.loads(await websocket.receive_text())
            player.command(command)
            if task.done():
                task.result()
    except WebSocketDisconnect:
        pass
    finally:
        player.closed = True
        player.wake.set()
        # A socket closing is not a license to recycle live native resources.
        try:
            await task
        except WebSocketDisconnect:
            pass


@app.websocket("/ws/queue")
async def websocket_queue(websocket: WebSocket):
    global main_loop
    if main_loop is None or main_loop.is_closed():
        main_loop = asyncio.get_running_loop()
    denied = _check_ws_access(websocket)
    if denied:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue_ws_clients.add(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({"type": "init", **_queue_state()}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        queue_ws_clients.discard(websocket)


from lunarc_mcp import create_http_app
_mcp_http=create_http_app(app,LOCAL_TOKEN)
automation.control.runtime_error=getattr(_mcp_http.state,'mcp_error',None)
# Last route, preserving all existing API/WebSocket routes and exact /mcp URL.
app.mount('/',_mcp_http)


def preflight_check():
    """P2-02: fail fast with an actionable message when runtime deps or
    external tools are missing, instead of a bare ModuleNotFoundError mid-run."""
    missing = []
    for mod in ("numpy", "cv2", "fastapi", "uvicorn", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"[preflight] 缺少运行时依赖: {', '.join(missing)}。请先 pip install -r requirements.txt")
    for opt, why in (("torch", "RAFT 光流需要此组件；原生 GPU 功能不需要"),
                      ("mcp", "智能体连接需要此组件；素材读取、预览和导出不受影响"),
                      ("httpx", "智能体连接需要此组件；素材读取、预览和导出不受影响"),
                      ("av", "PyAV 缺失将回退到 FFmpeg CLI 解码")):
        try:
            __import__(opt)
        except ImportError:
            print(f"[preflight] 可选依赖缺失 ({opt})：{why}", flush=True)
    if not pipeline._bundle_ffmpeg():
        print("[preflight] 未找到 ffmpeg，可移植解码/编码路径将不可用", flush=True)
    if not pipeline._bundle_ffprobe():
        print("[preflight] 未找到 ffprobe，精确帧数探测将降级", flush=True)


def main():
    dlss_service.enable_queue_storage(os.path.join(
        os.environ.get('LUNARC_DATA_DIR') or os.path.join(os.environ.get('APPDATA') or os.path.expanduser('~'),
            'DaVinci_DLSS5_Studio'), 'render-queue.json'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18923)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    preflight_check()

    # Wait for port to be available if previously occupied (e.g. Windows TIME_WAIT / process teardown)
    import socket
    for attempt in range(15):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((args.host, args.port))
            break
        except OSError:
            if attempt < 14:
                print(f"[server] Port {args.port} busy, waiting for release (attempt {attempt + 1}/15)...", flush=True)
                time.sleep(1)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_max_size=33554432,       # 32MB max message size for 4K/8K frame streaming
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0
    )


if __name__ == "__main__":
    main()
