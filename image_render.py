"""Single-image rendering and two immutable, full-resolution host results.

The memory mode checkpoints exact RGBA8 / linear FP16 between tiles in the
processing order. It never chooses model tiling, lower precision or JPEG.
"""
import ctypes as C
import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np


def image_settings(settings):
    import native_effects
    value = native_effects.execution_settings(settings)
    value.update(guidance_mode=0, frame_generation='off', export_backend='gpu_transfer',
                 temporal_controls={'scene_detection': False, 'mode': 'off'})
    value['render_pipeline'] = [s for s in native_effects.stages(value) if s != 'vfg']
    if not value['render_pipeline']:
        value['_pipeline_bypass'] = True
    return value


def source_key(path):
    stat = os.stat(path)
    return os.path.normcase(os.path.realpath(path)), stat.st_mtime_ns, stat.st_size


def video_frame_settings(settings):
    import native_effects
    value = native_effects.execution_settings(settings)
    value.update(frame_generation='off',export_backend='gpu_transfer')
    value['render_pipeline'] = [stage for stage in native_effects.stages(value) if stage != 'vfg']
    if not value['render_pipeline']:value['_pipeline_bypass']=True
    return value


def signature(settings, video_frame=False):
    value = video_frame_settings(settings) if video_frame else image_settings(settings)
    for key in ('image_memory_mode', 'image_preview_realtime', 'output_dir', 'output_path',
                'crf', 'overwrite', 'transcode_settings', 'frame_multiplier', 'framegen_quality',
                'framegen_target_fps', 'framegen_scene_detection', 'frame_generation_saved_mode'):
        value.pop(key, None)
    import native_effects
    if native_effects.has_stage(value, 'nr'):
        value['_selected_nr_runtime'] = __import__('dlssnr_runtime').selection_key()
    if any(native_effects.has_stage(value, kind) for kind in ('illustration', 'animesharp')):
        value['_selected_image_models'] = __import__('image_model_store').selection_key(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class ImageResult:
    id: str
    source: tuple
    settings_key: str
    original: np.ndarray
    pixels: np.ndarray
    hdr: bool
    diagnostics: dict
    created: float
    frame_index: int | None = None

    def info(self, diagnostics=True):
        result = dict(id=self.id, path=self.source[0], filename=os.path.basename(self.source[0]),
            width=self.pixels.shape[1], height=self.pixels.shape[0], hdr=self.hdr,
            processing_format='rgba16f/scRGB' if self.hdr else 'rgba8/sRGB',
            host_bytes=self.original.nbytes+self.pixels.nbytes, created_at=self.created)
        if diagnostics:result['diagnostics']=self.diagnostics
        if self.frame_index is not None:result['frame_index']=self.frame_index
        return result


class ImageResults:
    """At most current + previous, never GPU resources or unbounded history."""
    def __init__(self):
        self.lock = threading.RLock()
        self.entries = OrderedDict()

    def put(self, result):
        result.original.setflags(write=False)
        result.pixels.setflags(write=False)
        with self.lock:
            self.entries[result.id] = result
            while len(self.entries) > 2:
                self.entries.popitem(last=False)

    def get(self, result_id):
        with self.lock:
            result = self.entries.get(result_id)
            if result is None:
                raise ValueError('图片结果缓存已清理，请重新渲染')
            return result

    def previous(self, result_id):
        with self.lock:
            keys = list(self.entries)
            if result_id not in keys or keys.index(result_id) == 0:
                return None
            return self.entries[keys[keys.index(result_id)-1]]

    def status(self, path, settings, frame_index=None):
        normalized = os.path.normcase(os.path.realpath(path))
        with self.lock:
            entries = list(reversed(self.entries.values()))
            result = next((r for r in entries if r.source[0] == normalized and
                           (frame_index is None or r.frame_index == frame_index)), None)
            if result is None:
                return dict(available=False, results=[r.info(False) for r in entries])
            try:
                same_source = source_key(path) == result.source
            except OSError:
                same_source = False
            previous = self.previous(result.id)
            return dict(available=True, current=same_source and result.settings_key == signature(settings,result.frame_index is not None),
                source_unchanged=same_source, result=result.info(False),
                previous=previous.info(False) if previous else None, results=[r.info(False) for r in entries])

    def clear(self):
        with self.lock:
            self.entries.clear()


class PixelUpload:
    """Scoped CUDA upload of an exact host plane; supports a borrowed context."""
    def __init__(self, pixels, context=None):
        if pixels.dtype not in (np.uint8, np.float16) or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError('图片中间结果必须是 RGBA8 或线性 FP16 RGBA')
        self.context = C.c_void_p(context)
        self.owns_context = not context
        self.pointer = C.c_uint64()
        self.width, self.height = pixels.shape[1], pixels.shape[0]
        self.pitch = self.width*4*pixels.itemsize
        self.host_upload_bytes = pixels.nbytes
        self.cuda = C.WinDLL('nvcuda.dll')
        for name, args in {
            'cuInit':[C.c_uint], 'cuDeviceGet':[C.POINTER(C.c_int), C.c_int],
            'cuCtxCreate_v2':[C.POINTER(C.c_void_p), C.c_uint, C.c_int],
            'cuCtxDestroy_v2':[C.c_void_p], 'cuCtxPushCurrent_v2':[C.c_void_p],
            'cuCtxPopCurrent_v2':[C.POINTER(C.c_void_p)],
            'cuMemAlloc_v2':[C.POINTER(C.c_uint64), C.c_size_t],
            'cuMemFree_v2':[C.c_uint64], 'cuMemcpyHtoD_v2':[C.c_uint64,C.c_void_p,C.c_size_t],
        }.items():
            getattr(self.cuda, name).argtypes = args
        try:
            self.check(self.cuda.cuInit(0))
            if self.owns_context:
                device = C.c_int()
                self.check(self.cuda.cuDeviceGet(C.byref(device), 0))
                self.check(self.cuda.cuCtxCreate_v2(C.byref(self.context), 0, device))
                popped = C.c_void_p()
                self.check(self.cuda.cuCtxPopCurrent_v2(C.byref(popped)))
            with self.current():
                data = np.ascontiguousarray(pixels)
                self.check(self.cuda.cuMemAlloc_v2(C.byref(self.pointer), data.nbytes))
                self.check(self.cuda.cuMemcpyHtoD_v2(self.pointer, data.ctypes.data, data.nbytes))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'图片中间结果 CUDA 传输失败：{code}')

    @contextmanager
    def current(self):
        self.check(self.cuda.cuCtxPushCurrent_v2(self.context))
        try:
            yield
        finally:
            popped = C.c_void_p()
            self.check(self.cuda.cuCtxPopCurrent_v2(C.byref(popped)))

    def frame(self, frame):
        frame.width, frame.height = self.width, self.height
        frame.cuda_context = self.context.value
        frame.time_base_num = frame.time_base_den = 1
        frame.pts = 0
        frame.image_rgba = self.pointer.value, self.pitch
        return 1

    def descriptor(self, hdr=False):
        return dict(format='rgba16f_cuda' if hdr else 'rgba8_cuda', width=self.width, height=self.height,
            cuda_context=self.context.value, device_ptr=self.pointer.value, pitch_bytes=self.pitch)

    def free_pixels(self):
        if self.pointer.value:
            with self.current():
                self.check(self.cuda.cuMemFree_v2(self.pointer))
                self.pointer.value = 0

    def close(self):
        self.free_pixels()
        if self.context.value:
            if self.owns_context:
                self.check(self.cuda.cuCtxDestroy_v2(self.context))
            self.context.value = None


def download(candidate, frame):
    """Complete producer -> CUDA copy -> exact host plane, before releasing it."""
    from gpu_preview_engine import _Copy2D
    from native_effects import Exchange
    candidate = dict(candidate)
    candidate.setdefault('width', int(frame.width))
    candidate.setdefault('height', int(frame.height))
    hdr = bool(candidate.get('output_hdr'))
    wide = candidate.get('dxgi_format') == 10 or candidate.get('format') == 'rgba16f_cuda'
    exchange = None
    try:
        if 'device_ptr' not in candidate:
            exchange = Exchange(candidate['d3d12_device'], frame.cuda_context,
                (candidate['width'], candidate['height']), wide=wide)
            candidate = exchange.color(candidate)
        result = np.empty((candidate['height'],candidate['width'],4), np.float16 if wide else np.uint8)
        cuda = C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        PixelUpload.check(cuda.cuCtxPushCurrent_v2(candidate['cuda_context']))
        try:
            copy = _Copy2D(srcMemoryType=2, srcDevice=candidate['device_ptr'], srcPitch=candidate['pitch_bytes'],
                dstMemoryType=1, dstHost=result.ctypes.data, dstPitch=result.strides[0],
                WidthInBytes=result.strides[0], Height=result.shape[0])
            PixelUpload.check(cuda.cuMemcpy2D_v2(C.byref(copy)))
        finally:
            popped = C.c_void_p()
            PixelUpload.check(cuda.cuCtxPopCurrent_v2(C.byref(popped)))
        if hdr and result.dtype != np.float16:
            raise ValueError('HDR 中间结果不是 FP16，已停止处理')
        return result, hdr
    finally:
        if exchange:
            exchange.close()


def effect_pass(pixels, hdr, settings, stage):
    """One non-SDR-NR stage. All owners close before the next pass opens."""
    import native_effects as effects
    from gpu_transfer_export import _DeviceFrame, multi_scale_config
    from gpu_cleanup import finish_cleanup
    upload = live = entry = effect = tail = None
    cleanup = []
    started = time.perf_counter()
    try:
        upload = PixelUpload(pixels)
        size = pixels.shape[1], pixels.shape[0]
        frame = _DeviceFrame(struct_size=C.sizeof(_DeviceFrame))
        upload.frame(frame)
        kind = effects.stage_kind(stage)
        local = dict(effects.stage_settings(settings,stage), render_pipeline=[stage],
                     truehdr_enabled=kind=='truehdr', guidance_mode=0, frame_generation='off')
        local.pop('render_stage_enabled', None)
        if kind == 'nr':
            from dlss_gpu_engine import Live
            from truehdr_pipeline import HdrNRStage
            multi = multi_scale_config(effects.nr_effect_settings(local), *size)
            owner = Live(*multi['dims'][0], local, color_format='rgba16f')
            try:
                live = effects.Passthrough(*size,local,owner=owner,wide=True)
            except BaseException:
                owner.close()
                raise
            effect = HdrNRStage(live,local,size,None)
        else:
            local.update(working_scale=1, reconstruction_mode='single', layer_scales=[1],
                         layer_weights=[1], layer_settings=[], multi_pass=1, post_controls={})
            live = effects.make_live(*size,local,wide=hdr)
            if kind == 'truehdr':
                from truehdr_pipeline import Tail
                tail = Tail(live,local,ordered=True)
            effect = effects.EffectSeries(live,local,size,[stage],None,wide=hdr,hdr=hdr,truehdr_tail=tail)
        entry = effects.Exchange(live.d3d12_device(), frame.cuda_context, size, wide=hdr)
        candidate = entry.texture(upload.pointer.value,upload.pitch)
        candidate['output_hdr'] = hdr
        candidate = effect.process(candidate,frame,True)
        result, result_hdr = download(candidate,frame)
        report = effect.report()
    finally:
        for name, owner in (('effect',effect),('truehdr',tail),('input_exchange',entry),('live',live),('upload',upload)):
            if owner is not None:
                try: owner.close()
                except Exception as error: cleanup.append((name,error))
        finish_cleanup(cleanup)  # A failed release must block later passes.
    return result, result_hdr, dict(stage=stage, process_with_release_ms=(time.perf_counter()-started)*1000,
                                    resources_released=True, report=report)


class HdrImagePipeline:
    """Resident ordered FP16 image stages; no video decoder or encoder."""
    def __init__(self, original, settings):
        import native_effects as effects
        from gpu_transfer_export import _DeviceFrame, multi_scale_config
        self.upload = self.live = self.entry = self.effect = self.tail = None
        h,w = original.shape[:2]
        pixels = np.pad(original, ((0,(-h)%8),(0,(-w)%8),(0,0)), mode='edge')
        pixels[:, :, 3] = 1
        self.source = dict(image_width=w,image_height=h,width=pixels.shape[1],height=pixels.shape[0])
        self.settings = dict(settings)
        size = pixels.shape[1],pixels.shape[0]
        ordered = effects.stages(settings)
        try:
            self.upload = PixelUpload(pixels)
            self.frame = _DeviceFrame(struct_size=C.sizeof(_DeviceFrame))
            self.upload.frame(self.frame)
            if 'nr' in ordered:
                from dlss_gpu_engine import Live
                nr_size = effects.output_size(settings,size,ordered[:ordered.index('nr')])
                multi = multi_scale_config(effects.nr_effect_settings(settings),*nr_size)
                owner = Live(*multi['dims'][0],settings,color_format='rgba16f')
                try:
                    self.live = effects.Passthrough(*size,settings,owner=owner,wide=True)
                except BaseException:
                    owner.close()
                    raise
            else:
                self.live = effects.make_live(*size,settings,wide=True)
            if 'truehdr' in ordered:
                from truehdr_pipeline import Tail
                self.tail = Tail(self.live,settings,ordered=True)
            # Keep the partially constructed series reachable for cleanup.
            self.effect = effects.EffectSeries.__new__(effects.EffectSeries)
            self.effect.__init__(self.live,settings,size,ordered,None,wide=True,hdr=True,truehdr_tail=self.tail)
            self.entry = effects.Exchange(self.live.d3d12_device(),self.frame.cuda_context,size,wide=True)
        except BaseException:
            self.close()
            raise

    def render(self):
        from image_source import output_pixels
        with self.upload.current():
            candidate = self.entry.texture(self.upload.pointer.value,self.upload.pitch)
            candidate['output_hdr'] = True
            candidate = self.effect.process(candidate,self.frame,True)
            pixels,hdr = download(candidate,self.frame)
        return output_pixels(pixels,self.source,self.settings),hdr,self.effect.report()

    def close(self):
        from gpu_cleanup import release_owned,finish_cleanup
        errors=[]
        for name in ('effect','tail','entry','live','upload'):
            release_owned(self,name,errors)
        finish_cleanup(errors)


def render_passes(original, settings, nr_runner, stage_runner=effect_pass):
    from native_effects import stages, stage_kind
    from image_source import output_pixels
    h,w = original.shape[:2]
    pixels = np.pad(original, ((0,(-h)%8),(0,(-w)%8),(0,0)), mode='edge')
    pixels[:, :, 3] = 1 if original.dtype == np.float16 else 255
    source = dict(image_width=w,image_height=h,width=pixels.shape[1],height=pixels.shape[0])
    hdr, reports, transfer_bytes = original.dtype == np.float16, [], 0
    for stage in stages(settings):
        if stage_kind(stage) == 'vfg':
            continue
        from native_watchdog import phase
        phase(f'图片处理：{stage}', 600)
        transfer_bytes += pixels.nbytes
        pixels, hdr, report = (nr_runner(pixels,settings) if stage == 'nr' and not hdr
                                else stage_runner(pixels,hdr,settings,stage))
        if pixels.dtype != (np.float16 if hdr else np.uint8):
            raise ValueError('图片分步处理精度不符，已停止；不会自动降精度')
        if pixels.ndim != 3 or pixels.shape[2] != 4 or (hdr and not np.isfinite(pixels[:, :, :3]).all()):
            raise ValueError('图片中间结果的通道或亮度值无效，已停止处理')
        transfer_bytes += pixels.nbytes
        reports.append(report)
    return np.ascontiguousarray(output_pixels(pixels,source,settings)), hdr, reports, transfer_bytes


def sdr_pixels(pixels):
    """Linear SDR FP16 to the image export's sRGB RGBA8 contract."""
    if pixels.dtype == np.uint8:return pixels
    if not np.isfinite(pixels).all():raise ValueError('单帧结果包含非有限值')
    linear = np.clip(pixels[:,:,:3].astype(np.float32),0,1)
    encoded = np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
    rgba = np.empty(pixels.shape,np.uint8)
    rgba[:,:,:3] = np.rint(encoded*255).astype(np.uint8)
    rgba[:,:,3] = np.rint(np.clip(pixels[:,:,3],0,1)*255).astype(np.uint8)
    return rgba


def make_result(path, initial_key, settings, original, pixels, hdr, diagnostics):
    if source_key(path) != initial_key:
        raise ValueError('渲染期间原图片已改变，请重新渲染')
    from PIL import Image
    source_alpha = original[:, :, 3].astype(np.float32)
    if original.dtype == np.uint8:
        source_alpha /= 255
    alpha = np.asarray(Image.fromarray(source_alpha).resize(
        (pixels.shape[1],pixels.shape[0]),Image.Resampling.BILINEAR))
    pixels = np.ascontiguousarray(pixels).copy()
    pixels[:, :, 3] = alpha if hdr else np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)
    return ImageResult(uuid.uuid4().hex, initial_key, signature(settings), original,
                       pixels, hdr, diagnostics, time.time())


def display_bgr(pixels, hdr):
    if hdr:
        from gpu_preview_engine import display_linear
        return display_linear(pixels, True)
    return np.ascontiguousarray(pixels[:, :, 2::-1])


def preview_pair(result, reference=None):
    processed = display_bgr(result.pixels,result.hdr)
    source = display_bgr(reference.pixels,reference.hdr) if reference else display_bgr(result.original,result.original.dtype==np.float16)
    if source.shape != processed.shape:
        source = cv2.resize(source,(processed.shape[1],processed.shape[0]),interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode('.png', np.hstack((source,processed)))
    if not ok:
        raise ValueError('无法生成图片对比显示图')
    return encoded.tobytes()


def save_result(result, output_path, fmt, quality, preserve_alpha, overwrite, color_mode):
    """CPU format conversion only. Never opens an inference or display session."""
    alpha = (np.rint(result.pixels[:, :, 3].astype(np.float32)*255).astype(np.uint8)
             if result.hdr else result.pixels[:, :, 3])
    if color_mode == 'hdr10':
        if not result.hdr:
            raise ValueError('这份渲染结果是 SDR，不能直接标记为 HDR')
        from image_hdr_export import save_hdr_image
        saved = save_hdr_image(result.source[0],result.pixels[:, :, :3],output_path,fmt,
                               quality,preserve_alpha,overwrite,source_alpha=alpha)
    elif color_mode == 'sdr':
        from image_source import save_image
        if result.hdr:
            from hdr_display_mapping import hdr_to_sdr_linear
            linear = np.clip(hdr_to_sdr_linear(result.pixels[:, :, :3]),0,1)
            rgb = np.rint(np.where(linear<=.0031308,12.92*linear,
                1.055*np.power(linear,1/2.4)-.055)*255).astype(np.uint8)
        else:
            rgb = result.pixels[:, :, :3]
        saved = save_image(result.source[0],rgb,output_path,fmt,quality,preserve_alpha,
                           overwrite,source_alpha=alpha)
        saved['hdr_mapped_to_sdr'] = result.hdr
    else:
        raise ValueError('图片输出色彩模式无效')
    saved.update(rendered_result_id=result.id,diagnostics=result.diagnostics)
    return saved


class DisplayResources:
    """Feature-free presenter inputs. Pixel allocations free after GPU copy."""
    def __init__(self, result, reference=None):
        from native_effects import Passthrough
        self.live = self.source = self.output = None
        try:
            pixels = reference.pixels if reference else result.original
            self.source = PixelUpload(pixels)
            self.output = PixelUpload(result.pixels, self.source.context.value)
            self.live = Passthrough(result.pixels.shape[1],result.pixels.shape[0],{'render_pipeline':[]})
            self.source_hdr, self.result_hdr = pixels.dtype == np.float16, result.hdr
        except BaseException:
            self.close()
            raise

    def submit(self, presenter):
        presenter.frame(self.source.descriptor(self.source_hdr),self.output.descriptor(self.result_hdr),
                        self.source_hdr,self.result_hdr)
        self.output.free_pixels()
        self.source.free_pixels()

    def close(self):
        from gpu_cleanup import finish_cleanup
        errors=[]
        for name in ('output','source','live'):
            owner = getattr(self,name,None)
            setattr(self,name,None)
            if owner:
                try:owner.close()
                except Exception as error:errors.append((name,error))
        finish_cleanup(errors)
