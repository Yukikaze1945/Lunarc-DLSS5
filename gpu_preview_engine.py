"""Display adapter for the existing GPU export frame processors.

One dedicated thread owns decoder/CUDA/NGX lifetime. Only the finished composite
and source display image are read back, explicitly counted; no encoder or remux
is started. Processing precision is independent of the final 8-bit JPEG display.
"""
import ctypes as C
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import cv2
import numpy as np

import bundle_paths
import dlss_gpu_engine as dge
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx
import pipeline
import native_effects
from native_watchdog import phase as native_phase
from layer_settings import layer_parameters, resolve_layer_settings


def display_linear(rgba, hdr=False, display_edge=0):
    # Display resampling is independent from full-resolution neural processing.
    # Average linear radiance before the display transfer, never the HDR master.
    if display_edge and max(rgba.shape[:2]) > display_edge:
        scale = display_edge / max(rgba.shape[:2])
        rgba = cv2.resize(rgba.astype(np.float32),
            (max(1, round(rgba.shape[1]*scale)), max(1, round(rgba.shape[0]*scale))),
            interpolation=cv2.INTER_AREA)
    if not hdr and rgba.dtype == np.float16:
        # Exact FP16 code -> SDR display lookup. Avoid two full FP32 planes and
        # a power evaluation per pixel; this never modifies the HDR master.
        return np.ascontiguousarray(_SDR_HALF_DISPLAY[rgba[..., 2::-1].view(np.uint16)])
    rgb = np.asarray(rgba[..., :3], dtype=np.float32)
    if hdr:
        # Fixed display-only luminance mapping for every frame and both views.
        # scRGB is linear Rec.709, white=80 nits. Never feeds processing/export.
        from hdr_display_mapping import hdr_to_sdr_linear
        rgb = hdr_to_sdr_linear(rgb)
    rgb = np.clip(rgb, 0, 1)
    encoded = np.where(rgb < .018, 4.5 * rgb, 1.099 * rgb ** .45 - .099)
    return np.ascontiguousarray(np.rint(encoded[..., ::-1] * 255), dtype=np.uint8)


with np.errstate(invalid='ignore', over='ignore'):
    _half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    _half = np.nan_to_num(np.clip(_half, 0, 1), nan=0)
    _SDR_HALF_DISPLAY = np.rint(np.where(_half < .018, 4.5 * _half,
        1.099 * _half ** .45 - .099) * 255).astype(np.uint8)
del _half


class TextureReader:
    def __init__(self, device, width, height, fmt=28):
        self.lib = C.CDLL(bundle_paths.require_native('gpu_preview_readback.dll',
                                                       'display GPU preview'))
        self.lib.gpu_preview_open.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32,
                                             C.c_uint32, C.POINTER(C.c_void_p)]
        self.lib.gpu_preview_read.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                              C.c_uint64, C.c_void_p, C.c_size_t]
        self.lib.gpu_preview_close.argtypes = [C.c_void_p]
        self.lib.gpu_preview_last_error.restype = C.c_char_p
        self.handle = C.c_void_p()
        self.image = np.empty((height, width, 4), dtype=np.uint8 if fmt == 28 else np.float16)
        self._check(self.lib.gpu_preview_open(device, width, height, fmt, C.byref(self.handle)))
        self.bytes = 0

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_preview_last_error().decode('utf-8', 'replace'))

    def read(self, candidate):
        if candidate.get('state', 0x800) != 0x800:
            raise RuntimeError('preview consumer requires COPY_SOURCE')
        self._check(self.lib.gpu_preview_read(self.handle, candidate['d3d12_resource'],
            candidate.get('d3d12_fence'), candidate.get('fence_value', 0),
            self.image.ctypes.data, self.image.strides[0]))
        self.bytes += self.image.nbytes
        return self.image  # Borrowed until next read; cvtColor below owns its output.

    def close(self):
        if self.handle:
            self.lib.gpu_preview_close(self.handle)
            self.handle = None


class _Copy2D(C.Structure):
    _fields_ = [('srcXInBytes', C.c_size_t), ('srcY', C.c_size_t),
        ('srcMemoryType', C.c_uint), ('srcHost', C.c_void_p), ('srcDevice', C.c_uint64),
        ('srcArray', C.c_void_p), ('srcPitch', C.c_size_t), ('dstXInBytes', C.c_size_t),
        ('dstY', C.c_size_t), ('dstMemoryType', C.c_uint), ('dstHost', C.c_void_p),
        ('dstDevice', C.c_uint64), ('dstArray', C.c_void_p), ('dstPitch', C.c_size_t),
        ('WidthInBytes', C.c_size_t), ('Height', C.c_size_t)]


class ViewportReader:
    def __init__(self, device, context, width, height, edge, pixel_bytes):
        bundle_paths.add_dll_directory(bundle_paths.native_dir())
        self.lib = C.CDLL(bundle_paths.require_native('gpu_preview_display.dll', 'GPU preview display'))
        self.lib.gpu_preview_display_open.argtypes = [C.c_void_p, C.c_void_p] + [C.c_uint32]*5 + [C.POINTER(C.c_void_p)]
        self.lib.gpu_preview_display_read.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint32, C.c_void_p, C.c_size_t]
        self.lib.gpu_preview_display_close.argtypes = [C.c_void_p]
        self.lib.gpu_preview_display_error.restype = C.c_char_p
        scale = min(1., edge/max(width, height))
        w, h = max(1, round(width*scale)), max(1, round(height*scale))
        self.image = np.empty((h, w, 4), dtype=np.uint8)
        self.handle = C.c_void_p()
        self.bytes = 0
        self._check(self.lib.gpu_preview_display_open(device, context, width, height,
            w, h, pixel_bytes, C.byref(self.handle)))

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_preview_display_error().decode('utf8', 'replace'))

    def read(self, descriptor, hdr):
        self._check(self.lib.gpu_preview_display_read(self.handle, descriptor['device_ptr'],
            descriptor['pitch_bytes'], int(hdr), self.image.ctypes.data, self.image.strides[0]))
        self.bytes += self.image.nbytes
        return cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

    def close(self):
        if self.handle:
            self.lib.gpu_preview_display_close(self.handle)
            self.handle = None


class GpuPreview:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='GPUPreview')
        self.chain = None
        self.live = None
        self.provider = None
        self.bridges = None
        self.reader = None
        self.key = None
        self.last_index = None
        self.current_frame = None
        self.diag = {}
        self.display_bytes = 0
        self.generation = 0
        self.pending_layers = []
        self.display_buffers = {}
        self.cached_display = None
        self.options = {}
        self.active_settings = None
        self.display_edge = 0
        self.cached_edge = None
        self.viewport_readers = {}
        self.display_exchange = None
        # One bounded weight cache on the preview owner thread. Histories and
        # native sessions still reset independently when source/mode changes.
        self.raft_shared = {}
        from preview_cache import PostCache
        self.post_cache = PostCache()
        self.post_only_update = False
        self.native_lock = threading.RLock()
        self.native_config = None
        self.native_presenter = None
        self.frozen_presenter = None
        self.native_display = False
        self.lifecycle_phase = 'idle'
        self.memory_release_report=None
        self.cold_memory_check=False
        from image_render import ImageResults
        self.image_results = ImageResults()
        self.image_display_resources = None
        self.hdr_image_pipeline = None
        self.hdr_image_key = None
        self._capture_image_master = False
        self._image_checkpoint_pixels = None

    def _submit(self, function, *args):
        # Carry only observation context; the native call stays on its GPU owner.
        return self.executor.submit(copy_context().run, function, *args).result()

    def render_image(self, path, settings, memory_mode='memory'):
        return self._submit(self._render_image, path, dict(settings), memory_mode)

    def render_video_frame(self, path, settings, index):
        return self._submit(self._render_video_frame,path,dict(settings),index)

    def _render_video_frame(self, path, settings, index):
        from image_render import source_key, signature, make_result, sdr_pixels, video_frame_settings
        initial_key = source_key(path)
        options = video_frame_settings(settings)
        self._capture_image_master = True
        self._capture_frame_source = True
        self._captured_image_master = self._captured_frame_source = None
        try:
            # Force a full processing pass, not a resized/composited display cache.
            _,_,diagnostics = self._render(path,options,index,need_source=False,reuse_display=False,native=False)
            if self._captured_image_master is None or self._captured_frame_source is None:
                raise RuntimeError('单帧渲染未返回完整尺寸结果')
            pixels,hdr = self._captured_image_master
            original,source_hdr = self._captured_frame_source
            if not hdr:pixels = sdr_pixels(pixels)
            if not source_hdr:original = sdr_pixels(original)
            result = make_result(path,initial_key,settings,original,pixels,hdr,diagnostics)
            result.frame_index = index
            result.settings_key = signature(settings,video_frame=True)
            self.image_results.put(result)
            return result.info()
        finally:
            self._capture_image_master = self._capture_frame_source = False
            self._captured_image_master = self._captured_frame_source = None
            # Single-frame mode holds the host result, not the model sessions.
            self._release_memory('video_single_frame',preserve_display=True)

    def _render_image(self, path, settings, memory_mode='memory'):
        from image_render import image_settings, source_key, signature, render_passes, make_result
        from image_source import read_pixels, output_pixels, probe
        if memory_mode not in ('memory','fast'):
            raise ValueError('图片渲染模式只能选择省显存或快速模式')
        settings = image_settings(settings)
        initial_key, initial_signature = source_key(path), signature(settings)
        original = read_pixels(path)
        native_effects.validate(settings,probe(path))
        started = time.perf_counter()
        if memory_mode == 'memory':
            self._release_memory('image_memory_mode',preserve_display=True)
            pixels, hdr, stages, transfers = render_passes(original,settings,
                lambda pixels,options:self._image_nr_pass(path,pixels,options))
            nr=next((stage['report'] for stage in stages if stage['stage']=='nr'),{})
            summaries=[]
            for stage in stages:
                report=stage['report']
                summary=dict((report.get('stages') or [{}])[0]) if isinstance(report.get('stages'),list) else dict(report)
                summary.update(stage=stage['stage'],process_ms=stage['process_with_release_ms'])
                summaries.append(summary)
            diagnostics = dict(stages=nr.get('stages',{}), image_passes=stages,
                native_output_effect=dict(stages=summaries),render_pipeline=native_effects.stages(settings),
                layer_count=nr.get('layer_count',1),stage_count=len(stages),image_memory_mode='memory', inference_resources_released=True,
                image_checkpoint_host_bytes=transfers, image_checkpoint_format='RGBA8 / linear FP16; lossless')
        elif original.dtype == np.float16:
            from image_render import HdrImagePipeline
            key = initial_key, initial_signature
            if getattr(self,'hdr_image_key',None) != key or self.hdr_image_pipeline is None:
                self._close(preserve_display=True)
                self.hdr_image_pipeline = HdrImagePipeline(original,settings)
                self.hdr_image_key = key
            try:
                pixels,hdr,report = self.hdr_image_pipeline.render()
            except BaseException:
                self._close(preserve_display=True)
                raise
            diagnostics = dict(native_output_effect=report,render_pipeline=native_effects.stages(settings),
                image_memory_mode='fast',inference_resources_released=False,
                stage_count=len(report['stages']),image_master_readback_bytes=pixels.nbytes)
        else:
            if getattr(self,'hdr_image_pipeline',None) is not None:
                self._close(preserve_display=True)
            self._close_image_display(preserve_display=True)
            self._captured_image_master = None
            self._capture_image_master = True
            try:
                _,_,diagnostics = self._render(path,settings,0,need_source=False,reuse_display=False,native=False)
                if self._captured_image_master is None:
                    raise RuntimeError('渲染链未返回完整图片结果')
                pixels, hdr = self._captured_image_master
                pixels = output_pixels(pixels,self.source,settings)
                diagnostics.update(image_memory_mode='fast',inference_resources_released=False,
                    image_master_readback_bytes=pixels.nbytes)
            finally:
                self._capture_image_master=False
                self._captured_image_master=None
        diagnostics.update(processing_ms=(time.perf_counter()-started)*1000,
            preview_resolution=[pixels.shape[1],pixels.shape[0]],output_hdr=hdr,frame=0)
        diagnostics['processing_fps']=1000/max(diagnostics['processing_ms'],.001)
        diagnostics['timing_clock']='completed wall time; image passes include initialization, transfers and release'
        result = make_result(path,initial_key,settings,original,pixels,hdr,diagnostics)
        result.settings_key = initial_signature
        self.image_results.put(result)
        self.diag = diagnostics
        return result.info()

    def _image_nr_pass(self, path, pixels, settings):
        from image_render import image_settings
        local = image_settings(dict(settings,render_pipeline=['nr'],truehdr_enabled=False,
                                    render_stage_enabled={}))
        started = time.perf_counter()
        self._image_checkpoint_pixels = pixels
        self._capture_image_master = True
        self._captured_image_master = None
        try:
            self._render(path,local,0,need_source=False,reuse_display=False,native=False)
            if self._captured_image_master is None:
                raise RuntimeError('NR 未返回完整图片中间结果')
            result, hdr = self._captured_image_master
            report = dict(self.diag)
        finally:
            self._image_checkpoint_pixels = None
            self._capture_image_master = False
            self._captured_image_master = None
            self._close(preserve_display=True)  # Retire models, keep the frozen display only.
            self.result_bgr = None
        return result,hdr,dict(stage='nr',process_with_release_ms=(time.perf_counter()-started)*1000,
                               resources_released=True,report=report)

    def present_image_result(self, result_id, reference_id=None):
        return self._submit(self._present_image_result,result_id,reference_id)

    def _freeze_native_display(self):
        current=self.native_presenter
        if current is None:return
        if current.freeze():
            previous=getattr(self,'frozen_presenter',None)
            if previous:previous.close()
            self.frozen_presenter=current
            self.native_presenter=None
        else:
            current.close();self.native_presenter=None

    def _retire_frozen_display(self):
        previous=getattr(self,'frozen_presenter',None)
        self.frozen_presenter=None
        if previous:previous.close()

    def _close_image_display(self,preserve_display=False):
        from gpu_cleanup import release_owned,finish_cleanup
        errors=[]
        with self.native_lock:
            if preserve_display:self._freeze_native_display()
            else:self._retire_frozen_display()
            release_owned(self,'native_presenter',errors)
            release_owned(self,'image_display_resources',errors)
        finish_cleanup(errors)

    def _present_image_result(self, result_id, reference_id=None):
        from image_render import DisplayResources
        from native_preview import Presenter
        result = self.image_results.get(result_id)
        reference = self.image_results.get(reference_id) if reference_id else None
        self._close_image_display(preserve_display=True)
        with self.native_lock:
            if not self.native_config:
                return dict(native=False)
            try:
                self.image_display_resources=DisplayResources(result,reference)
                self.native_presenter=Presenter(self.image_display_resources.live.d3d12_device(),self.native_config)
                self.image_display_resources.submit(self.native_presenter)
                self._retire_frozen_display()
                return self.native_presenter.report()
            except BaseException:
                self._close_image_display()
                raise

    def configure_native(self, layout):
        from native_preview import validate_layout
        config = validate_layout(layout) if layout is not None else None
        with self.native_lock:
            if self.native_presenter and config and config['parent'] != self.native_presenter.parent:
                raise ValueError('请关闭原生预览后再更换窗口')
            self.native_config = config
            frozen=getattr(self,'frozen_presenter',None)
            if frozen:
                if config:frozen.configure(config)
                else:self._retire_frozen_display()
            if self.native_presenter:
                if config:self.native_presenter.configure(config)
                else:
                    self.native_presenter.close(); self.native_presenter = None
                    resources=getattr(self,'image_display_resources',None)
                    if resources:
                        resources.close();self.image_display_resources=None
                    self.cached_display = None
            display=self.native_presenter or getattr(self,'frozen_presenter',None)
            return display.report() if display else dict(native=False,pending=bool(config))

    def native_status(self):
        with self.native_lock:
            display=self.native_presenter or getattr(self,'frozen_presenter',None)
            return display.report() if display else dict(native=False,pending=bool(self.native_config))

    def render(self, video, settings, frame_index, *, need_source=True, reuse_display=False, display_edge=0, flow_view=False, native=False):
        return self._submit(self._render, video, dict(settings), int(frame_index),
                            need_source, reuse_display, display_edge, flow_view, native)

    def export_image(self, path, settings, output_path, fmt, quality, preserve_alpha, overwrite, color_mode='sdr'):
        return self._submit(self._export_image, path, dict(settings), output_path,
                            fmt, quality, preserve_alpha, overwrite, color_mode)

    def _export_image(self, path, settings, output_path, fmt, quality, preserve_alpha, overwrite, color_mode='sdr'):
        from image_source import is_image
        if not is_image(path):
            raise ValueError('图片导出只接受图片素材')
        if color_mode not in ('sdr', 'hdr10'):
            raise ValueError('图片输出色彩模式无效')
        if color_mode == 'hdr10':
            from image_hdr_export import validate_request
            validate_request(settings, fmt, path, preserve_alpha)
        from image_render import save_result
        info=self._render_image(path,settings,settings.get('image_memory_mode','memory'))
        native_phase('图片格式编码', None)
        return save_result(self.image_results.get(info['id']),output_path,fmt,quality,
                           preserve_alpha,overwrite,color_mode)

    def close(self):
        self._submit(self._close)

    def release_memory(self):
        return self._submit(self._release_memory)

    def _release_memory(self,reason='manual',before=None,preserve_display=False):
        import gc,sys
        from preview_memory import sample
        before=before if before is not None else sample(self.live)
        self.lifecycle_phase='release preview memory'
        if preserve_display:self._close(preserve_display=True)
        else:self._close()  # A failing native release prevents new models from opening.
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            from torch_cuda_context import torch_primary_context
            with torch_primary_context('cuda'):
                self.raft_shared.clear();gc.collect()
                torch.cuda.synchronize();torch.cuda.empty_cache()
        else:self.raft_shared.clear();gc.collect()
        self.cold_memory_check=True
        self.memory_release_report=dict(reason=reason,before=before,after=sample(),
            vendor_memory_scope='SDK allocation retirement cannot be measured completely')
        self.diag=dict(status='released',memory_release=self.memory_release_report)
        return self.memory_release_report

    def shutdown(self):
        self.close()
        self._submit(self.raft_shared.clear)
        self.image_results.clear()
        self.executor.shutdown(wait=True)

    def _close(self,preserve_display=False):
        from gpu_memory import inactive
        inactive('preview')
        errors = []
        def release(label, callback):
            try:
                callback()
            except Exception as error:
                errors.append((label, error))
        def detach(name, method='close'):
            owner = getattr(self, name, None)
            setattr(self, name, None)
            if owner is not None:
                release(name, getattr(owner, method))
        with self.native_lock:
            if preserve_display:release('freeze_display',self._freeze_native_display)
            else:release('frozen_display',self._retire_frozen_display)
            detach('native_presenter')
            detach('image_display_resources')
        detach('native_exchange')
        self.native_exchange_key = None
        self.source_display_cache = None
        from preview_cache import PostCache
        post_cache, self.post_cache = self.post_cache, PostCache()
        release('post_cache', post_cache.close)
        # Display reader borrows the processing device and closes before owners.
        readers = list(self.viewport_readers.values())
        self.viewport_readers.clear()
        for _, reader in readers:
            release('viewport_reader', reader.close)
        detach('display_exchange')
        detach('reader')
        detach('hdr_image_pipeline')
        self.hdr_image_key = None
        chain, live, layers = self.chain, self.live, self.pending_layers
        self.chain = self.live = None
        self.pending_layers = []
        try:
            if chain:
                release('chain', chain.close)
                # Chain cleanup can stop before its anchor, e.g. on a failing
                # output effect. Live.close detaches its handle before the DLL
                # call, so retrying an already attempted anchor is harmless.
                if live:
                    release('anchor', live.close)
            else:
                for layer in reversed(layers):
                    if layer.get('live'):
                        release('layer', layer['live'].close)
                    if layer.get('flow_provider'):
                        release('layer_flow', layer['flow_provider'].close_independent_layer)
                if live:
                    release('anchor', live.close)
        finally:
            detach('provider', 'close_independent_layer')
            bridges, self.bridges = self.bridges, None
            if bridges:
                release('loader_cookies', lambda: htx.release_loader_cookies(bridges))
            self.key = self.last_index = self.current_frame = None
            self.display_buffers.clear()
            self.cached_display = None
            self.active_settings = None
        if errors:
            raise RuntimeError('GPU preview cleanup failed: ' + '; '.join(
                f'{label}: {error}' for label, error in errors)) from errors[0][1]

    def _reopen_vsr_with_decoder(self, video, settings):
        # Source identity is checked by _render. Keep its context alive until
        # every old effect is closed and the new chain has adopted the decoder.
        reuse = dict(lib=self.bridges['nvdec'], handle=self.chain.decoder,
                     source=self.source, origin=self.source_origin)
        self.chain.decoder = None
        try:
            self._close(preserve_display=True)
            self._open(video, settings, decoder_reuse=reuse)
        finally:
            if reuse['handle'] is not None:
                reuse['lib'].nvdec_bridge_close(reuse['handle'])

    def _open(self, video, settings, decoder_reuse=None):
        # Frame generation is an export setting; it must not load VFG or reject
        # an otherwise valid preview when its target rate is being edited.
        settings = dict(settings, frame_generation='off')
        checkpoint = getattr(self,'_image_checkpoint_pixels',None)
        if checkpoint is not None:
            h,w = checkpoint.shape[:2]
            source = dict(media_type='image',width=w,height=h,image_width=w,image_height=h,
                pix_fmt='rgba',bit_depth=8,color_range='pc',color_space='gbr',
                color_transfer='iec61966-2-1',color_primaries='bt709',r_frame_rate='1/1',
                video_stream_count=1,audio_streams=0,subtitle_streams=0,nb_chapters=0,metadata=False)
        else:
            source = decoder_reuse['source'] if decoder_reuse else gtx.probe_source(video)
        import native_effects
        native_effects.validate(settings, source)
        width, height = source['width'], source['height']
        if getattr(self,'cold_memory_check',False):
            from preview_memory import sample,check_cold_start
            check_cold_start(sample(),settings,(width,height))
        process_w, process_h = native_effects.processing_size(settings, (width, height))
        self.multi = gtx.multi_scale_config(settings, process_w, process_h)
        uses_flow = gtx.guidance_requires_flow(settings)
        self.bridges = gtx.load_bridges(requires_flow=uses_flow,
            requires_multiscale=len(self.multi['scales']) > 1)
        self.hdr = source.get('color_transfer') == 'smpte2084'
        from prores_decode import is_prores
        prores_source = is_prores(source)
        # Precision selects the decoder path; colour admission belongs to that
        # path's gate. Never send a 10-bit source to the NV12-only gate.
        self.sdr10 = (source.get('pix_fmt') == 'yuv420p10le' or prores_source) and not self.hdr
        if source.get('color_transfer') == 'arib-std-b67':
            raise ValueError('HLG GPU preview is not validated; no SDR conversion fallback')
        wide = self.hdr or self.sdr10
        self.live = native_effects.make_live(*self.multi['dims'][0], settings, wide=wide)
        uses_flow = uses_flow and not isinstance(self.live, native_effects.Passthrough)
        self.provider = native_effects.make_flow_provider(settings, self.live,
            **({'raft_shared': self.raft_shared} if settings.get('flow_engine') in ('raft', 'raft_large') else {}))
        target = native_effects.output_size(settings, (width, height))
        dims = dict(src_w=width, src_h=height, live_w=self.multi['dims'][0][0],
            live_h=self.multi['dims'][0][1], dst_w=target[0], dst_h=target[1])
        local = dict(settings, dlss_backend='gpu_flow_d3d12', export_backend='gpu_transfer')
        tx = dict(codec='hevc_nvenc', container='mkv', color_preset='preserve')
        if wide:
            tx.update(profile='main10', pixel_format='p010le')
            caps = (htx.load_sdr10_capabilities if self.sdr10 else htx.load_hdr_capabilities)(
                self.bridges, width, height)
            local['experimental_hdr_scrgb'] = self.hdr
            local['hdr_paper_white_scrgb'] = settings.get('hdr_paper_white_scrgb', 1.0)
            negotiate = htx.negotiate_sdr10 if self.sdr10 else htx.negotiate_hdr
            negotiation = negotiate(dims, source, local, tx, 24, False,
                'preview-display.mkv', self.live, hdr_caps=caps,
                option_available=self.bridges['option_available'], flow_provider=self.provider)
        else:
            negotiation = gtx.negotiate(dims, source, local, tx, 24, False,
                'preview-display.mkv', self.live,
                option_available=self.bridges['option_available'], flow_provider=self.provider,
                capabilities=self.bridges['capabilities'])
        if not negotiation['supported']:
            raise ValueError('GPU preview refused: ' + '; '.join(negotiation['reasons']))
        strength = float(settings.get('residual_strength', 1))
        if wide:
            extra = htx._open_extra_layers(self.multi, self.live, self.provider,
                                            uses_flow, process_w, process_h, settings=settings)
            self.pending_layers = extra
            args = (self.bridges, caps, video, None, self.live, negotiation['plan'],
                    negotiation['fps'], negotiation['hdr_policy'], dims)
            kwargs = dict(flow_provider=self.provider, residual_strength=strength,
                          uses_flow=uses_flow, preview_consumer=self._consume_linear)
            if extra:
                self.chain = htx.HdrMultiScaleChain(*args, **kwargs, multi=self.multi,
                    extra_layers=extra, fusion_plans=[gtx.TapPlans(process_w, process_h, *d)
                                                       for d in self.multi['dims']])
            else:
                self.chain = htx.HdrChain(*args, **kwargs)
        else:
            extra = []
            self.pending_layers = extra
            for scale, d in zip(self.multi['scales'][1:], self.multi['dims'][1:]):
                layer = dict(dims=d, scale=scale, live=None,
                    flow_provider=self.provider.new_independent_layer() if uses_flow else None,
                    lanczos=None, low_residual=None, tap_plans=None)
                extra.append(layer)
                from layer_settings import resolve_layer_settings
                layer['live'] = dge.Live(*d, resolve_layer_settings(settings, len(extra)),
                                         shared_with=self.live)
                layer['tap_plans'] = gtx.TapPlans(process_w, process_h, *d)
            args = (self.bridges, video, None, self.live, negotiation['plan'],
                    negotiation['fps'], strength, self.multi['dims'][0], self.provider)
            kwargs = dict(downscaled=negotiation['downscaled'], uses_flow=uses_flow,
                tap_plans=gtx.TapPlans(process_w, process_h, *self.multi['dims'][0])
                    if negotiation['plan_driven'] else None,
                preview_consumer=self._consume_texture)
            if extra:
                self.chain = gtx.MultiScaleObservationChain(*args, **kwargs,
                    extra_layers=extra, config=self.multi,
                    fusion_plans=[gtx.TapPlans(process_w, process_h, *d) for d in self.multi['dims']])
            else:
                self.chain = gtx.Chain(*args, **kwargs)
        from post_control import GpuPostControl, controls_from_settings
        self.pending_layers = []  # The constructed chain now owns every layer.
        from temporal_control import GpuTemporalControl, temporal_from_settings
        self.chain.post_control = GpuPostControl(controls_from_settings(settings), hdr=wide, sdr10=self.sdr10)
        self.chain.temporal_control = GpuTemporalControl(temporal_from_settings(settings), hdr=wide)
        native_effects.install(self.chain, self.live, settings, (width, height), self.provider)
        if decoder_reuse:
            if self.bridges['nvdec']._handle != decoder_reuse['lib']._handle:
                raise RuntimeError('VSR preview decoder runtime changed; reuse refused')
            self.chain.decoder, decoder_reuse['handle'] = decoder_reuse['handle'], None
        elif checkpoint is not None:
            from image_render import PixelUpload
            self.chain.image_decoder=PixelUpload(checkpoint)
        else:
            self.chain.open_decoder()
        self.decoder_reused = bool(decoder_reuse)
        lib = self.bridges['nvdec']
        lib.nvdec_bridge_seek_us.argtypes = [C.c_void_p, C.c_int64]
        lib.nvdec_bridge_seek_us.restype = C.c_int32
        self.source = source
        # NVDEC reports absolute PTS; the UI clock and seek ABI are source-relative.
        import subprocess
        if decoder_reuse:
            self.source_origin = decoder_reuse['origin']
        else:
            try:
                raw = (b'{"frames": []}' if source.get('media_type') == 'image' else subprocess.check_output([pipeline._bundle_ffprobe(), '-v', 'error',
                    '-select_streams', 'v:0', '-read_intervals', '%+#1', '-show_frames',
                    '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', video], timeout=10))
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"ffprobe timed out after 10 seconds while reading source presentation timestamp: {video}") from exc
            frames = json.loads(raw).get('frames', [])
            self.source_origin = float(frames[0]['best_effort_timestamp_time']) if frames else 0.0
        self.fps = negotiation['fps'][0] / negotiation['fps'][1]
        self.generation += 1
        self.display_bytes = 0

    def _consume_texture(self, candidate, frame):
        if getattr(self,'_capture_image_master',False):
            return self._consume_image_master(candidate,frame)
        if self.native_display:
            return self._native_candidate(candidate,frame)
        mark = time.perf_counter()
        if candidate.get('output_hdr'):
            if 'device_ptr' not in candidate:
                self._native_candidate(candidate, frame)
                candidate = self.native_result
            self.result_bgr = self._viewport('processed',candidate,self.live.d3d12_device(),True,
                                             edge=self.display_edge or max(candidate['width'],candidate['height']))
            self.processed_display_ms = (time.perf_counter()-mark)*1000
            return
        if self.display_edge:
            import native_effects
            if self.display_exchange is None:
                self.display_exchange = native_effects.Exchange(candidate['d3d12_device'], frame.cuda_context,
                    (candidate['width'], candidate['height']))
            color = self.display_exchange.color(candidate)
            self.result_bgr = self._viewport('processed', color, candidate['d3d12_device'], False)
            self.processed_display_ms = (time.perf_counter()-mark)*1000
            return
        if self.reader is None:
            self.reader = TextureReader(candidate['d3d12_device'], candidate['width'], candidate['height'])
        raw = self.reader.read(candidate)
        self.frame_readback_ms += (time.perf_counter()-mark)*1000
        self.result_bgr = cv2.cvtColor(raw[..., :3], cv2.COLOR_RGB2BGR)
        self.processed_display_ms = (time.perf_counter()-mark)*1000

    def _update_runtime(self, settings):
        runtime = {'style', 'intensity', 'local_tone', 'local_struct', 'skin_struct',
                   'use_auto_mask', 'ui_correction', 'motion_scale_x', 'motion_scale_y',
                   'post_controls', 'temporal_controls', 'layer_settings',
                   'truehdr_contrast','truehdr_saturation','truehdr_middle_gray','truehdr_peak_nits','truehdr_debanding'}
        if self.active_settings is None:
            return False
        changed = {k for k in self.active_settings.keys() | settings.keys()
                   if self.active_settings.get(k) != settings.get(k)}
        import native_effects
        inline_nr = native_effects.nr_as_effect(settings)
        inline = native_effects.inline_nr_stage(self.chain) if inline_nr else None
        strength_hot = len(inline.multi['scales'] if inline else self.multi['scales']) == 1
        solver_keys = {'layer_weights', 'fusion_beta', 'fusion_max_iterations',
                       'fusion_relative_tolerance'} if not strength_hot else set()
        instance_hot=False
        if 'render_stage_options' in changed:
            before=self.active_settings.get('render_stage_options',{});after=settings.get('render_stage_options',{})
            changed_instances={key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
            hot_keys={'vsr_mode','vsr_quality','vsr_strength'}
            instance_hot=bool(changed_instances) and all(native_effects.stage_kind(key)=='vsr' and
                {name for name in before.get(key,{}).keys()|after.get(key,{}).keys()
                 if before.get(key,{}).get(name)!=after.get(key,{}).get(name)}<=hot_keys for key in changed_instances)
        allowed = runtime | {'residual_strength', 'dlaa_preset', 'vsr_mode', 'vsr_quality', 'vsr_strength'} | solver_keys
        if instance_hot:allowed.add('render_stage_options')
        if strength_hot and getattr(self.chain, 'sdr10', False):
            allowed.add('residual_pyramid')
        if not changed <= allowed:
            return False
        if 'dlaa_preset' in changed and not native_effects.update_dlaa_model(self.chain, settings):
            return False
        if (instance_hot or changed & {'vsr_mode', 'vsr_quality', 'vsr_strength'}) and not native_effects.update_vsr_options(self.chain, settings):
            return False
        self.runtime_changed_keys = changed
        from post_control import controls_from_settings
        from temporal_control import temporal_from_settings
        post, temporal = controls_from_settings(settings), temporal_from_settings(settings)
        owners = ([inline.owner] + [layer['live'] for layer in getattr(inline.core, 'extra_layers', [])] if inline else
                  [self.live] + [layer['live'] for layer in getattr(self.chain, 'extra_layers', [])])
        from layer_settings import resolve_layer_settings
        from dlss_gpu_engine import baked_option_key
        # Each Feature owns its parameters; changing L2 must not overwrite L1/L3.
        updates = [(owner, resolve_layer_settings(settings, index))
                   for index, owner in enumerate(owners)
                   if baked_option_key(resolve_layer_settings(self.active_settings, index)) !=
                      baked_option_key(resolve_layer_settings(settings, index))]
        # Preset is create-only. Preflight every layer before hot-updating any
        # owner, so a model change closes the old pipeline before opening a new one.
        if any(baked_option_key(resolve_layer_settings(self.active_settings, index))[0] !=
               baked_option_key(resolve_layer_settings(settings, index))[0]
               for index in range(len(owners))):
            return False
        self.runtime_updated_layers = {index for index, owner in enumerate(owners)
                                       if any(owner is item[0] for item in updates)}
        if updates:
            if any(not hasattr(owner, 'update_runtime_options') for owner, _ in updates):
                return False
            for owner, resolved in updates:
                if not owner.update_runtime_options(resolved):
                    return False
        if inline:
            inline.update(settings)
        if not inline and changed & (solver_keys | ({'residual_strength'} if not strength_hot else set())):
            # Keep solver work textures with current DLLs. Older DLLs recreate
            # only the solver, preserving the decoder and all model sessions.
            import native_effects
            process_w, process_h = native_effects.processing_size(
                settings, (self.source['width'], self.source['height']))
            updated = gtx.multi_scale_config(settings, process_w, process_h)
            strength = float(settings.get('residual_strength', 1.))
            from fusion_runtime import update_parameters
            if hasattr(self.chain, 'fusion_handle'):
                if self.chain.fusion_handle is not None and not update_parameters(
                        self.chain.b['fusion'], self.chain.fusion_handle, updated, strength):
                    self.chain.b['fusion'].gpu_fusion_close(self.chain.fusion_handle)
                    self.chain.fusion_handle = None
                self.chain.config = updated
            elif hasattr(self.chain, 'fusion'):
                if self.chain.fusion is not None and not self.chain.fusion.update_parameters(updated, strength):
                    self.chain.fusion.close()
                    self.chain.fusion = None
                self.chain.multi = updated
            self.multi = updated
        truehdr_keys = {'truehdr_contrast','truehdr_saturation','truehdr_middle_gray','truehdr_peak_nits','truehdr_debanding'}
        if changed & truehdr_keys and getattr(self.chain,'truehdr_tail',None):
            self.chain.truehdr_tail.update(settings)
        # A conversion change alters every downstream stage's input. Reusing
        # the pre-conversion NR/post cache would display an obsolete HDR frame.
        conversion_upstream = 'truehdr' in native_effects.stages(settings)[:-1]
        self.post_only_update = changed <= ({'post_controls'} | truehdr_keys) and not inline_nr and not (changed & truehdr_keys and conversion_upstream)
        self.sequential_strength_changed = 'residual_strength' in changed
        if not self.post_only_update:
            self.post_cache.invalidate()
        if hasattr(self.chain, 'post_control'):
            self.chain.post_control.update_controls(controls_from_settings({}) if inline_nr else post)
        if hasattr(self.chain, 'temporal_control'):
            self.chain.temporal_control.update_controls(temporal_from_settings({'temporal_controls': {'scene_detection': False}}) if inline_nr else temporal)
        if 'residual_strength' in changed:
            self.chain.strength = float(settings.get('residual_strength', 1.))
        if 'residual_pyramid' in changed and getattr(self.chain, 'sdr10', False):
            self.chain.policy['preserve_source_detail'] = bool(settings.get('residual_pyramid', True))
        self.active_settings = dict(settings)
        self.cached_display = None
        return True

    def _linear_plane(self, plane):
        bridge = (getattr(self.chain, 'encode_bridge', None) if plane == 7
                  else getattr(self.chain, 'decode_bridge', None)) or self.chain.bridge
        lib = bridge._lib
        lib.gpu_hdr_readback_scrgb.argtypes = [C.c_void_p, C.c_int32, C.c_void_p, C.c_size_t]
        size = getattr(self.chain, 'output_size', None) if plane == 7 else None
        shape = (size[1], size[0], 4) if size else (self.source['height'], self.source['width'], 4)
        if plane not in self.display_buffers or self.display_buffers[plane].shape != shape:
            self.display_buffers[plane] = np.empty(shape, dtype=np.float16)
        image = self.display_buffers[plane]
        mark = time.perf_counter()
        code = lib.gpu_hdr_readback_scrgb(bridge._handle, plane,
                                         image.ctypes.data, image.strides[0])
        if code:
            raise RuntimeError('FP16 display readback failed')
        self.display_bytes += image.nbytes
        self.frame_readback_ms += (time.perf_counter() - mark) * 1000
        return image

    def _consume_linear(self, candidate, frame):
        if getattr(self,'_capture_image_master',False):
            return self._consume_image_master(candidate,frame)
        if self.native_display:
            return self._native_candidate(candidate,frame)
        if getattr(self, '_image_export_hdr', False) and not candidate.get('output_hdr'):
            raise ValueError('HDR 图片保存未收到 HDR 渲染结果，未保存')
        mark = time.perf_counter()
        if candidate.get('output_hdr') and getattr(self, '_image_export', False):
            from hdr_display_mapping import hdr_to_sdr_linear
            image = np.empty((candidate['height'],candidate['width'],4),np.float16)
            cuda = C.WinDLL('nvcuda.dll')
            cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
            cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
            cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
            code = cuda.cuCtxPushCurrent_v2(candidate['cuda_context'])
            if code:
                raise RuntimeError(f'图片 HDR 回读上下文失败：{code}')
            try:
                copy = _Copy2D(srcMemoryType=2,srcDevice=candidate['device_ptr'],srcPitch=candidate['pitch_bytes'],
                    dstMemoryType=1,dstHost=image.ctypes.data,dstPitch=image.strides[0],
                    WidthInBytes=image.strides[0],Height=image.shape[0])
                code = cuda.cuMemcpy2D_v2(C.byref(copy))
                if code:
                    raise RuntimeError(f'图片 HDR 回读失败：{code}')
            finally:
                popped = C.c_void_p()
                cuda.cuCtxPopCurrent_v2(C.byref(popped))
            if getattr(self, '_image_export_hdr', False):
                self.result_bgr = np.ascontiguousarray(image[:, :, 2::-1])
                self._image_hdr_captured = True
            else:
                rgb = np.clip(hdr_to_sdr_linear(image[:,:,:3].astype(np.float32)),0,1)
                encoded = np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
                self.result_bgr = np.ascontiguousarray(np.rint(encoded[:,:,::-1]*255),dtype=np.uint8)
            self.display_bytes += image.nbytes
            self.frame_readback_ms += (time.perf_counter()-mark)*1000
            self.processed_display_ms = (time.perf_counter()-mark)*1000
            return
        if candidate.get('output_hdr'):
            self.result_bgr = self._viewport('processed',candidate,candidate['d3d12_device'],True,
                                            edge=self.display_edge or max(candidate['width'],candidate['height']))
            self.processed_display_ms = (time.perf_counter()-mark)*1000
            return
        self.result_bgr = (self._viewport('processed', candidate, candidate['d3d12_device'], self.hdr)
            if self.display_edge else display_linear(self._linear_plane(7), self.hdr))
        self.processed_display_ms = (time.perf_counter() - mark) * 1000

    def _consume_image_master(self,candidate,frame):
        from image_render import download,display_bgr
        started=time.perf_counter()
        if getattr(self,'_capture_frame_source',False):
            candidate=dict(candidate)
            candidate.setdefault('output_hdr',bool(self.hdr))
        pixels,hdr=download(candidate,frame)
        self._captured_image_master=pixels,hdr
        self.result_bgr=display_bgr(pixels,hdr)
        self.display_bytes+=pixels.nbytes
        self.processed_display_ms=(time.perf_counter()-started)*1000
        self.frame_readback_ms+=self.processed_display_ms

    def _native_candidate(self,candidate,frame):
        candidate = candidate.as_descriptor() if hasattr(candidate,'as_descriptor') else dict(candidate)
        # Neutral SDR post-processing preserves the host's borrowed texture
        # descriptor; that older descriptor doesn't include its picture extent.
        candidate.setdefault('width', int(frame.width))
        candidate.setdefault('height', int(frame.height))
        if 'device_ptr' not in candidate:
            from native_effects import Exchange
            wide = bool(self.hdr or self.sdr10 or candidate.get('output_hdr'))
            size = (candidate['width'],candidate['height'])
            key = (*size,wide)
            if getattr(self,'native_exchange_key',None) != key:
                if getattr(self,'native_exchange',None):self.native_exchange.close()
                self.native_exchange = Exchange(self.live.d3d12_device(),frame.cuda_context,size,wide=wide)
                self.native_exchange_key = key
            candidate = self.native_exchange.color(candidate)
            candidate['format'] = 'rgba16f_cuda' if wide else 'rgba8_cuda'
        self.native_result = candidate

    def _viewport(self, kind, descriptor, device, hdr, edge=None):
        key = (device, descriptor['cuda_context'], descriptor['width'], descriptor['height'],
               self.display_edge if edge is None else edge, 8 if descriptor['format'] == 'rgba16f_cuda' else 4)
        previous = self.viewport_readers.get(kind)
        if previous is None or previous[0] != key:
            if previous:
                previous[1].close()
            reader = ViewportReader(*key)
            self.viewport_readers[kind] = key, reader
        reader = self.viewport_readers[kind][1]
        mark = time.perf_counter()
        result = reader.read(descriptor, hdr)
        self.frame_readback_ms += (time.perf_counter()-mark)*1000
        self.display_bytes += reader.image.nbytes
        return result

    def _source_bgr(self, outcome, frame):
        if self.display_edge:
            if self.hdr or self.sdr10:
                descriptor = outcome['_preview_master']
            else:
                ptr, pitch = outcome['_full_color']
                descriptor = dict(format='rgba8_cuda', cuda_context=frame.cuda_context,
                    width=frame.width, height=frame.height, device_ptr=ptr, pitch_bytes=pitch)
            return self._viewport('source', descriptor, self.live.d3d12_device(), self.hdr)
        if self.hdr or self.sdr10:
            return display_linear(self._linear_plane(2), self.hdr, self.display_edge)
        ptr, pitch = outcome['_full_color']
        image = np.empty((frame.height, frame.width, 4), dtype=np.uint8)
        cuda = C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        if cuda.cuCtxPushCurrent_v2(frame.cuda_context):
            raise RuntimeError('preview source CUDA context push failed')
        try:
            copy = _Copy2D(srcMemoryType=2, srcDevice=ptr, srcPitch=pitch,
                dstMemoryType=1, dstHost=image.ctypes.data, dstPitch=image.strides[0],
                WidthInBytes=frame.width * 4, Height=frame.height)
            if cuda.cuMemcpy2D_v2(C.byref(copy)):
                raise RuntimeError('preview source display copy failed')
        finally:
            popped = C.c_void_p()
            cuda.cuCtxPopCurrent_v2(C.byref(popped))
        self.display_bytes += image.nbytes
        return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)

    def _flow_bgr(self, descriptor):
        """Diagnostic-only readback, on the native session owner thread.

        The float2 resource stays borrowed until this synchronous copy completes.
        Never create a second CPU-facing RAFT/NVOF engine for the flow view.
        """
        if not descriptor or descriptor.get('format') != 'float32x2':
            raise ValueError('Native preview did not return a float2 flow resource')
        width, height = int(descriptor['width']), int(descriptor['height'])
        image = np.empty((height, width, 2), dtype=np.float32)
        cuda = C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        if cuda.cuCtxPushCurrent_v2(descriptor['cuda_context']):
            raise RuntimeError('flow diagnostic context push failed')
        try:
            copy = _Copy2D(srcMemoryType=2, srcDevice=descriptor['device_ptr'],
                srcPitch=descriptor['pitch_bytes'], dstMemoryType=1,
                dstHost=image.ctypes.data, dstPitch=image.strides[0],
                WidthInBytes=width*8, Height=height)
            if cuda.cuMemcpy2D_v2(C.byref(copy)):
                raise RuntimeError('flow diagnostic readback failed')
        finally:
            popped = C.c_void_p()
            cuda.cuCtxPopCurrent_v2(C.byref(popped))
        if not np.isfinite(image).all():
            raise ValueError('flow diagnostic contains nonfinite motion')
        return pipeline.colorize_flow(image), image.nbytes

    def _render(self, video, settings, index, need_source=True, reuse_display=False, display_edge=0, flow_view=False, native=False):
        settings = native_effects.execution_settings(settings)
        from image_source import is_image
        if is_image(video):
            if flow_view:
                raise ValueError('静态图片没有帧间光流')
            index = 0
            settings = dict(settings, guidance_mode=0,
                temporal_controls={'scene_detection':False, 'mode':'off'})
        if index < 0:
            raise ValueError('preview frame must be nonnegative')
        settings = {key:value for key,value in settings.items() if key not in (
            'frame_generation','frame_multiplier','framegen_quality','framegen_target_fps','framegen_scene_detection')}
        ordered = settings.get('render_pipeline')
        if ordered and (is_image(video) or any(stage != 'vfg' for stage in ordered)):
            settings['render_pipeline'] = [stage for stage in ordered if stage != 'vfg']
            if not settings['render_pipeline']:
                settings['_pipeline_bypass'] = True
        # Preview always uses this device pipeline; export routing is not an image
        # parameter and must not rebuild its decoder or model sessions.
        settings['export_backend'] = 'gpu_transfer'
        # A diagnostic flow view requires motion even with enhancement guidance
        # disabled; this local snapshot does not change the user's setting.
        if flow_view:
            settings = dict(settings, guidance_mode=1)
        key = (os.path.realpath(video), os.stat(video).st_mtime_ns,
               json.dumps(settings, sort_keys=True, default=str),
               __import__('dlssnr_runtime').selection_key()
               if 'nr' in __import__('native_effects').stages(settings) else None,
               __import__('image_model_store').selection_key(settings))
        start = time.perf_counter()
        self.frame_readback_ms = self.processed_display_ms = 0.0
        self.display_edge = max(0, int(display_edge))
        self.native_display = bool(native and not flow_view)
        if self.native_display and not self.native_config:
            raise ValueError('原生显示面板尚未连接')
        try:
            setup = time.perf_counter()
            runtime_changed = False
            self.post_only_update = False
            if key != self.key:
                same_source = self.key is not None and self.key[:2] == key[:2] and self.key[3:] == key[3:]
                from preview_memory import sample,pressured,model_changed
                if self.live is not None and self.active_settings is not None and (
                        model_changed(self.active_settings,settings) or not same_source):
                    memory=sample(self.live)
                    if pressured(memory):
                        self._release_memory('memory_pressure_model_switch',memory,preserve_display=True)
                        same_source=False
                self.lifecycle_phase = 'update live options'
                runtime_changed = same_source and self._update_runtime(settings)
                if not runtime_changed:
                    changed = ({k for k in self.active_settings.keys() | settings.keys()
                                if self.active_settings.get(k) != settings.get(k)} if self.active_settings else set())
                    if (same_source and changed and changed <= {'vsr_mode','vsr_quality','vsr_strength'}
                            and self.chain.decoder and not getattr(self.chain, 'prores_input', False)):
                        self.lifecycle_phase = 'resize VSR pipeline, retain decoder'
                        self._reopen_vsr_with_decoder(video, settings)
                    else:
                        self.lifecycle_phase = 'close previous pipeline'
                        self._close(preserve_display=True)
                        self.lifecycle_phase = 'open new pipeline'
                        native_phase('初始化预览链路',
                                     600 if is_image(video) or self._capture_image_master else 300)
                        self._open(video, settings)
                self.key = key
                self.active_settings = dict(settings)
                self.cold_memory_check=False
            self.lifecycle_phase = 'render frame'
            native_phase('处理预览帧',
                         600 if is_image(video) or self._capture_image_master else 300)
            setup_ms = (time.perf_counter() - setup) * 1000
            if (reuse_display and self.cached_display is not None and self.last_index == index
                    and self.cached_edge == self.display_edge
                    and getattr(self, 'cached_flow_view', False) == flow_view
                    and getattr(self, 'cached_native', False) == self.native_display
                    and (not need_source or self.cached_display[0] is not None)):
                original, result, cached = self.cached_display
                if self.native_display:
                    with self.native_lock:cached = dict(cached,native_display=self.native_presenter.report())
                return original, result, dict(cached, cached_display=True,
                    request_ms=(time.perf_counter()-start)*1000)
            discontinuity = runtime_changed or self.last_index is None or index != self.last_index + 1
            # Repeat requests reuse the decode surface but re-evaluate with reset:
            # changed parameters rebuild, and no previous enhancement is returned.
            if self.last_index != index:
                decode_start = time.perf_counter()
                target = index / self.fps
                # Short forward gaps decode from the current surface. Only an
                # initial/backward/large jump needs GOP seek; temporal reset is
                # independent from decoder positioning.
                seek = (getattr(self, 'decoder_reused', False) or (self.last_index is None and index > 0) or
                        (self.last_index is not None and (index <= self.last_index or
                         index - self.last_index > max(8, int(self.fps * 2)))))
                if seek:
                    if getattr(self.chain, 'prores_input', False):
                        code = self.chain.seek_decoder(round(target * 1e6))
                    else:
                        code = self.bridges['nvdec'].nvdec_bridge_seek_us(self.chain.decoder,
                                                                         round(target * 1e6))
                    if code:
                        if getattr(self.chain, 'prores_input', False):
                            raise RuntimeError(f'ProRes seek failed ({code})')
                        gtx._bridge_error(self.bridges['nvdec'], 'nvdec_bridge_last_error', code)
                self.decoder_reused = False
                while True:
                    frame = gtx._DeviceFrame(struct_size=C.sizeof(gtx._DeviceFrame))
                    code = self.chain.next_frame(frame)
                    if code < 0:
                        if getattr(self.chain, 'prores_input', False):
                            raise RuntimeError(f'ProRes decode failed ({code})')
                        gtx._bridge_error(self.bridges['nvdec'], 'nvdec_bridge_last_error', code)
                    if not code:
                        raise ValueError('preview reached end of source')
                    timestamp = frame.pts * frame.time_base_num / frame.time_base_den - getattr(self, 'source_origin', 0.0)
                    if timestamp + .5 / self.fps >= target:
                        self.current_frame = frame
                        break
                decode_ms = (time.perf_counter() - decode_start) * 1000
            else:
                decode_ms, seek = 0.0, False
            frame = self.current_frame
            if not (self.hdr or self.sdr10) and self.chain.nv12 is None:
                self.chain._open_color_stages(frame)
            process_start = time.perf_counter()
            post_reused = ((self.post_only_update or not runtime_changed) and self.post_cache.valid and self.last_index == index
                           and not flow_view and reuse_display)
            if post_reused:
                discontinuity = False
                outcome = self.post_cache.reprocess(self.chain)
            else:
                self.post_cache.invalidate()
                same_source_frame = self.last_index == index and reuse_display and not flow_view
                prefix_safe = getattr(self,'runtime_changed_keys',set()) <= {
                    'style','intensity','local_tone','local_struct','skin_struct','use_auto_mask','ui_correction',
                    'layer_settings','layer_weights','fusion_beta','fusion_max_iterations',
                    'fusion_relative_tolerance','residual_strength','post_controls',
                    'truehdr_contrast','truehdr_saturation','truehdr_middle_gray','truehdr_peak_nits','truehdr_debanding'}
                if getattr(self, 'runtime_changed_keys', set()) & {'truehdr_contrast','truehdr_saturation','truehdr_middle_gray','truehdr_peak_nits','truehdr_debanding'}:
                    prefix_safe = False
                self.chain._preview_reuse_prefix = bool(same_source_frame and runtime_changed and prefix_safe
                    and getattr(self.chain,'_preview_prefix_color',None) is not None)
                if not same_source_frame:
                    self.chain._preview_prefix_color = None
                if len(self.multi['scales']) > 1:
                    if not hasattr(self.chain, '_preview_nr_cache'):
                        self.chain._preview_nr_cache = {}
                    # A temporal native effect before NR may produce a different
                    # input when reset, even at the same source frame. Do not reuse
                    # NR output unless its pristine input is guaranteed unchanged.
                    same_frame = (same_source_frame and (getattr(self.chain,'input_effect',None) is None
                                                        or self.chain._preview_reuse_prefix))
                    if not same_frame:
                        self.chain._preview_nr_cache.clear()
                    self.chain._preview_reuse_layers = (
                        set(range(len(self.multi['scales']))) - getattr(self, 'runtime_updated_layers', set())
                        if same_frame and runtime_changed else set())
                    invalid_layers = set(getattr(self, 'runtime_updated_layers', set()))
                    if runtime_changed and self.multi['fusion'] == 'sequential' and getattr(self, 'sequential_strength_changed', False):
                        invalid_layers.add(1)
                    if self.multi['fusion'] == 'sequential' and invalid_layers:
                        # Downstream layers depend on the changed layer's output.
                        self.chain._preview_reuse_layers &= set(range(min(invalid_layers)))
                self.chain._preview_post_capture = self.post_cache.capture if reuse_display and not flow_view else None
                try:
                    outcome = self.chain.process(frame, self.provider, index, discontinuity)
                finally:
                    self.chain._preview_post_capture = None
                    self.chain._preview_reuse_layers = set()
                    self.chain._preview_reuse_prefix = False
                if self.post_cache.valid:
                    self.post_cache.outcome = dict(outcome)
            process_ms = (time.perf_counter()-process_start)*1000
            display_start = time.perf_counter()
            source_key = (index, self.display_edge)
            source_cached = getattr(self, 'source_display_cache', None)
            source_bgr = None
            native_report = None
            if getattr(self,'_capture_frame_source',False):
                from image_render import download
                source = (outcome['_preview_master'] if self.hdr or self.sdr10
                          else self.chain._preview_original_color)
                self._captured_frame_source = download(dict(source,output_hdr=self.hdr),frame)
            if self.native_display:
                source = (outcome['_preview_master'] if self.hdr or self.sdr10
                          else self.chain._preview_original_color)
                result = self.native_result
                if self.source.get('media_type') == 'image':
                    from image_source import output_extent
                    source, result = dict(source), dict(result)
                    source.update(width=self.source['image_width'], height=self.source['image_height'])
                    width, height = output_extent(result['width'], result['height'], self.source, settings)
                    result.update(width=width, height=height)
                with self.native_lock:
                    if not self.native_presenter:
                        from native_preview import Presenter
                        self.native_presenter = Presenter(self.live.d3d12_device(),self.native_config)
                    self.native_presenter.frame(source,result,self.hdr,
                                                bool(self.hdr or settings.get('truehdr_enabled')))
                    self._retire_frozen_display()
                    native_report = self.native_presenter.report()
                self.result_bgr = None
            elif need_source:
                if source_cached is not None and source_cached[0] == source_key:
                    source_bgr = source_cached[1]
                else:
                    source_bgr = self._source_bgr(outcome, frame)
                    self.source_display_cache = source_key, source_bgr
            source_display_ms = (time.perf_counter()-display_start)*1000
            flow_readback_bytes = 0
            if flow_view:
                self.result_bgr, flow_readback_bytes = self._flow_bgr(outcome.get('flow'))
            if self.source.get('media_type') == 'image' and not self.native_display:
                from image_source import output_pixels
                self.result_bgr = output_pixels(self.result_bgr, self.source, settings)
                if source_bgr is not None:
                    source_bgr = output_pixels(source_bgr, self.source, {'render_pipeline':['nr']})
            self.last_index = index
            elapsed = time.perf_counter() - start
            layers = (self.chain.layer_diagnostics() if hasattr(self.chain, 'layer_diagnostics')
                      else dict(layer_count=1, layer_session_ids=[self.live.session_id],
                                layer_working_dims=[list(self.multi['dims'][0])]))
            reports = getattr(self.chain, 'frame_reports', [])
            report = reports[-1] if reports else {}
            solver = (self.chain.fusion_reports[-1] if getattr(self.chain, 'fusion_reports', []) else None)
            self.diag = dict(frame=index, pts=int(frame.pts),
                time_base=[int(frame.time_base_num), int(frame.time_base_den)],
                source_resolution=[frame.width, frame.height],
                preview_resolution=(native_report['display_resolution'] if native_report else
                                    [self.result_bgr.shape[1], self.result_bgr.shape[0]]), quality_mode='full_resolution',
                colour_mode='hdr10' if self.hdr else 'sdr10_bt709' if self.sdr10 else 'sdr8',
                processing_format='rgba16f' if self.hdr or self.sdr10 else 'rgba8',
                source_colour_resolution=self.source.get('colour_resolution'),
                display_mode='fixed_HDR_to_SDR_non_calibrated' if self.hdr else 'SDR8_display',
                processing_ms=elapsed * 1000, processing_fps=1 / max(elapsed, 1e-9),
                generation=self.generation, reset=discontinuity,
                cached_display=False, decoder_seek=seek, setup_ms=setup_ms,
                post_only_reprocess=post_reused, inference_reused=post_reused,
                layer_inference_reused=([True] * len(self.multi['scales']) if post_reused
                                        else outcome.get('layer_inference_reused', [])),
                post_cache=self.post_cache.report(),
                decode_ms=decode_ms, process_with_display_ms=process_ms,
                readback_ms=self.frame_readback_ms,
                processed_display_ms=self.processed_display_ms,
                display_readback_scope='GPU display conversion/resampling plus viewport readback' if self.display_edge else 'full plane readback',
                source_display_ms=source_display_ms,
                source_display_reused=bool(need_source and source_cached is not None and
                                           source_cached[0] == source_key),
                flow_diagnostic_readback_bytes=flow_readback_bytes,
                flow_visualization=bool(flow_view),
                gpu_pipeline_completed_wall_ms=process_ms-self.processed_display_ms,
                timing_clock='CPU wall scopes around completed native work; not kernel timestamps',
                layer_count=len(self.multi['scales']), layers=layers,
                layer_parameters=[layer_parameters(resolve_layer_settings(settings, i))
                    for i in range(len(self.multi['scales']))],
                fusion_mode_requested=self.multi['fusion'],
                fusion_mode_actual=report.get('fusion_mode_actual') or
                    ('sequential_independent_nr_sessions' if self.multi['fusion'] == 'sequential'
                     else 'gpu_data_consistency_solve' if solver else 'catmull_baseline_layer_0'),
                solver=report.get('solver') or solver,
                display_readback_bytes=self.display_bytes + (self.reader.bytes if self.reader else 0),
                frame_report=report, stages=outcome['stages'])
            self.diag['neural_mode'] = settings.get('neural_mode', 'nr')
            self.diag['truehdr'] = self.chain.truehdr_tail.report() if getattr(self.chain,'truehdr_tail',None) else None
            self.diag['output_hdr'] = bool(settings.get('truehdr_enabled') or self.hdr)
            if getattr(self, '_image_export_hdr', False):
                self.diag.update(display_mode='linear_scRGB_FP16_image_export',
                    image_export_color_mode='hdr10', display_readback_scope='full FP16 HDR plane; no display tone mapping')
            self.diag['native_display'] = native_report
            if native_report:
                self.diag.update(display_mode=native_report['output_mode'],display_readback_scope='none; GPU native panel',
                                 frame_display_host_image_bytes=0)
            self.diag['source_relative_time'] = frame.pts * frame.time_base_num / frame.time_base_den - getattr(self, 'source_origin', 0.0)
            self.diag['seek_clock'] = 'nominal frame index; actual decoded PTS reported'
            if self.diag['neural_mode'] in ('dlaa', 'vsr'):
                self.diag['fusion_mode_actual'] = 'native_single_effect'
            self.diag['motion'] = self.provider.timing() if self.provider else None
            if self.diag['neural_mode'] == 'dlaa':
                owner = getattr(self.live, 'owner', None) or self.live
                self.diag['dlaa_process_global_counters'] = owner.counters()
            self.diag['native_output_effect'] = (self.chain.output_effect.report()
                if getattr(self.chain, 'output_effect', None) else None)
            self.diag['native_input_effect'] = (self.chain.input_effect.report()
                if getattr(self.chain, 'input_effect', None) else None)
            inline = native_effects.inline_nr_stage(self.chain)
            if inline:
                report = inline.report()
                self.diag.update(layer_count=len(inline.multi['scales']), layers=report['layers'],
                    processing_format='rgba16f', frame_report=report['frame_report'],
                    layer_parameters=[layer_parameters(resolve_layer_settings(settings, i))
                                      for i in range(len(inline.multi['scales']))],
                    fusion_mode_requested=inline.multi['fusion'],
                    fusion_mode_actual=(report['frame_report'] or {}).get('fusion_mode_actual', 'hdr_nr_after_truehdr'))
            self.diag['render_pipeline'] = native_effects.stages(settings)
            if self.source.get('media_type') == 'image':
                self.diag['image_source'] = dict(format='rgba8_cuda',
                    host_upload_bytes=self.chain.image_decoder.host_upload_bytes,
                    frames=1, guidance='zero_motion', temporal='off')
            from gpu_memory import capture
            self.diag['gpu_memory'] = capture(self, self.live, phase='preview', preview_cache=self.post_cache)
            if getattr(self.chain, 'scene_observer', None):
                self.chain.scene_observer.observations = self.chain.scene_observer.observations[-16:]
            if 'nr' not in self.diag['render_pipeline']:
                self.diag['fusion_mode_actual'] = 'none; native model stages only'
            # Preview has no final export report. Keep recent diagnostics bounded
            # without changing per-layer counters or the metadata reference.
            for name in ('frame_reports', 'fusion_reports', 'metadata_snapshots'):
                values = getattr(self.chain, name, None)
                if values is not None and len(values) > 16:
                    del values[:-16]
            for values in getattr(self.chain, 'layer_reset_flags', []):
                if len(values) > 16:
                    del values[:-16]
            temporal = getattr(self.chain, 'temporal_control', None)
            for name in ('observations', 'process_reports'):
                values = getattr(temporal, name, None)
                if values is not None and len(values) > 16:
                    del values[:-16]
            if self.memory_release_report is not None:self.diag['memory_release']=self.memory_release_report
            self.cached_display = (None if getattr(self, '_image_export_hdr', False)
                                   else (source_bgr, self.result_bgr, dict(self.diag)))
            self.cached_edge = self.display_edge
            self.cached_flow_view = flow_view
            self.cached_native = self.native_display
            return source_bgr, self.result_bgr, dict(self.diag)
        except Exception as error:
            try:
                self._close()
            except Exception as cleanup_error:
                raise RuntimeError(f'{error}; {cleanup_error}') from error
            raise
