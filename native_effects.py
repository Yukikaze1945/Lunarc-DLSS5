"""Ordered native models with format-matched GPU transfers.

DLAA accepts RGBA8 or linear FP16. VFX 1.3 VideoSuperRes uses RGB10A2 for wide
pictures; only its GPU encoding boundary quantises the FP16 master.
"""
import ctypes as C
import os

import bundle_paths

MODES = ('nr', 'dlaa', 'vsr', 'nr_dlaa', 'nr_vsr')
LEGACY_STAGES = {'nr': ['nr'], 'dlaa': ['dlaa'], 'vsr': ['vsr'],
                 'nr_dlaa': ['nr', 'dlaa'], 'nr_vsr': ['nr', 'vsr']}
DLAA_PRESETS = (0, 5, 6, 10, 11, 12, 13)


def stage_kind(stage):
    if not isinstance(stage,str): return None
    head, separator, suffix = stage.partition(':')
    if head not in ('nr','dlaa','vsr','vfg','truehdr','illustration','animesharp'): return None
    if separator and (head != 'vsr' or not suffix or len(suffix)>64 or not all(c.isalnum() or c=='-' for c in suffix)): return None
    return head


def stage_settings(settings, stage):
    options=settings.get('render_stage_options',{})
    if not isinstance(options,dict): raise ValueError('Render stage options must be a map')
    result=dict(settings)
    patch=options.get(stage,{})
    if not isinstance(patch,dict): raise ValueError('Render stage parameters must be a map')
    allowed=('vsr_','image_model_')
    if any(not isinstance(key,str) or not key.startswith(allowed) for key in patch):
        raise ValueError('Stage parameters must contain only VSR or image model options')
    result.update(patch)
    return result


def has_stage(settings, kind):
    return any(stage_kind(stage)==kind for stage in stages(settings))


def configured_stages(settings):
    value = settings.get('render_pipeline')
    if value is None:
        selected = str(settings.get('neural_mode') or 'nr')
        if selected not in LEGACY_STAGES:
            raise ValueError('Unknown native processing mode: ' + selected)
        value = list(LEGACY_STAGES[selected])
    if not isinstance(value, (list, tuple)) or (not value and not (settings.get('truehdr_enabled') or settings.get('_truehdr_input_pipeline') or settings.get('_pipeline_bypass'))):
        raise ValueError('Render pipeline must contain at least one stage')
    if any(stage_kind(stage) is None for stage in value) or len(set(value)) != len(value):
        raise ValueError('Render pipeline contains unknown or duplicate stage IDs')
    ordered = list(value)
    if settings.get('frame_generation', 'off') != 'off' and 'vfg' not in ordered:
        ordered.append('vfg')
    if settings.get('truehdr_enabled') and 'truehdr' not in ordered:
        ordered.append('truehdr')
    return ordered


def stages(settings):
    ordered = configured_stages(settings)
    enabled = settings.get('render_stage_enabled', {})
    if not isinstance(enabled, dict) or any(stage_kind(key) is None or
                                            not isinstance(value, bool) for key,value in enabled.items()):
        raise ValueError('Render stage switches must contain known stages and boolean values')
    return [stage for stage in ordered if enabled.get(stage, True)]


def execution_settings(settings):
    """Materialize switches for processing without changing saved parameters/order."""
    if not settings.get('render_stage_enabled'):
        return dict(settings)
    active = stages(settings)
    result = dict(settings, render_pipeline=active, truehdr_enabled='truehdr' in active)
    result.pop('render_stage_enabled', None)
    result.pop('frame_generation_saved_mode', None)
    if not active:
        result['_pipeline_bypass'] = True
    if 'vfg' not in active:
        result['frame_generation'] = 'off'
    if 'nr' not in active:
        result.update(working_scale=1, reconstruction_mode='single', layer_scales=[1],
                      layer_weights=[1], layer_settings=[], multi_pass=1,
                      post_controls={}, temporal_controls={})
    if not any(stage in active for stage in ('nr','dlaa')):
        result['guidance_mode'] = 0
    return result


def nr_as_effect(settings):
    ordered = stages(settings)
    return 'nr' in ordered and 'truehdr' in ordered and ordered.index('truehdr') < ordered.index('nr')


def nr_effect_settings(settings):
    # The outer chain only decodes SDR. The NR stage after TrueHDR owns its
    # independent FP16 reconstruction, with the user's original layer settings.
    return dict(settings, _inline_hdr_nr=True)


def core_working_scale(settings):
    return 1.0 if nr_as_effect(settings) and not settings.get('_inline_hdr_nr') else float(settings.get('working_scale', 1.0))


def mode(settings):
    if settings.get('render_pipeline') is not None:
        ordered = stages(settings)
        return next((key for key, value in LEGACY_STAGES.items() if value == ordered), 'pipeline')
    value = str(settings.get('neural_mode') or 'nr')
    if value not in MODES:
        raise ValueError('Unknown native processing mode: ' + value)
    return value


def validate(settings, source):
    settings = execution_settings(settings)
    from vfx_framegen import config
    multiplier, _ = config(settings)
    if multiplier != 1:
        from vfx_framegen import output_fps
        from gpu_transfer_export import fps_rational
        output_fps(settings, fps_rational(source) or (int(source.get('fps', 30)), 1))
        if settings.get('export_backend') != 'gpu_transfer':
            raise ValueError('VFG requires GPU Transfer')
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'), framegen=True)
    ordered = stages(settings)
    if 'truehdr' in ordered:
        from truehdr_engine import parameters, runtime_directory
        parameters(settings)
        runtime_directory(settings)
    if ordered == ['nr']:
        return
    image_input = source.get('media_type') == 'image' and source.get('pix_fmt') in ('rgba','rgba16f')
    from prores_decode import is_prores
    prores_input = is_prores(source)
    wide = prores_input or source.get('pix_fmt') in ('yuv420p10le', 'p010le', 'p010') or (image_input and source.get('pix_fmt')=='rgba16f')
    if source.get('color_transfer') == 'arib-std-b67':
        raise ValueError('HLG is unsupported by the existing native HDR10/PQ chain')
    if source.get('color_transfer') == 'smpte2084' and not wide:
        raise ValueError('HDR10 native stages require a 10-bit P010 source')
    if any(stage_kind(stage)=='vsr' and (wide or 'truehdr' in ordered[:index]) for index,stage in enumerate(ordered)):
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'))
    if not image_input and not prores_input and source.get('pix_fmt') not in ('yuv420p', 'nv12', 'yuv420p10le', 'p010le', 'p010'):
        raise ValueError('Native stages require a supported 8-bit or 10-bit YUV source')
    if settings.get('export_backend') != 'gpu_transfer':
        raise ValueError('Native DLAA / RTX VSR requires GPU Transfer')
    if 'dlaa' in ordered and (isinstance(settings.get('dlaa_preset', 0), bool) or
                             settings.get('dlaa_preset', 0) not in DLAA_PRESETS):
        raise ValueError('DLAA preset must be Default, E, F, J, K, L or M')
    if int(settings.get('guidance_mode', 0)) and settings.get('flow_engine', 'nvof') not in ('nvof','fidelityfx','raft','raft_large'):
        raise ValueError('Native DLAA / NR guidance requires device optical flow')
    if 'nr' not in ordered and (
            settings.get('reconstruction_mode', 'single') != 'single' or
            float(settings.get('working_scale', 1)) != 1):
        raise ValueError('Standalone DLAA / RTX VSR requires Single at 100% input')
    if not any(s in ordered for s in ('nr', 'dlaa')) and int(settings.get('guidance_mode', 0)):
        raise ValueError('RTX VSR standalone does not consume optical-flow guidance')
    for index,stage in enumerate(ordered):
        local=stage_settings(settings,stage)
        if stage_kind(stage)=='vsr':
            from vsr_config import configuration
            configuration(local)
        elif stage_kind(stage) in ('illustration','animesharp'):
            if not image_input or wide or 'truehdr' in ordered[:index]:
                raise ValueError('图片放大模型需要 SDR 图片输入，请放在 TrueHDR 前面')
            from image_upscale import model_config
            model_config(local,stage_kind(stage))
    output_size(settings,(source['width'],source['height']))


def output_size(settings, size, ordered=None):
    size=tuple(size)
    for stage in stages(settings) if ordered is None else ordered:
        kind=stage_kind(stage);local=stage_settings(settings,stage)
        if kind=='vsr': size=vsr_output_size(local,size)
        elif kind in ('illustration','animesharp'):
            from image_upscale import model_config
            factor=model_config(local,kind)['scale']
            size=tuple(value*factor for value in size)
            if max(size)>16384: raise ValueError('图片放大链输出单轴不能超过 16384 像素')
    return size


def vsr_output_size(settings, size):
    import math
    from vsr_config import configuration
    if configuration(settings)['same_size']:
        return tuple(size)
    mode = settings.get('vsr_output_mode', 'resolution')
    choices = {'720p': (1280, 720), '1080p': (1920, 1080),
               '1440p': (2560, 1440), '4k': (3840, 2160), '8k': (7680, 4320)}
    if mode == 'scale':
        scale = settings.get('vsr_scale', 2)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 1 < scale <= 4:
            raise ValueError('RTX VSR scale must be greater than 1 and at most 4')
        result = tuple(int(math.floor(value * scale / 2 + .5)) * 2 for value in size)
    elif mode == 'resolution':
        target = str(settings.get('vsr_target', '1080p'))
        if target == 'custom':
            result = (settings.get('vsr_width', 1920), settings.get('vsr_height', 1080))
        elif target in choices:
            result = choices[target]
        else:
            raise ValueError('Unknown RTX VSR target')
    else:
        raise ValueError('RTX VSR output mode must be scale or resolution')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 64 or v > 16384 or v % 2 for v in result):
        raise ValueError('RTX VSR output dimensions must be even integers in 64..16384')
    if size[0] >= result[0] or size[1] >= result[1]:
        raise ValueError('RTX VSR target must enlarge both source dimensions')
    if result[0] > size[0] * 4 or result[1] > size[1] * 4:
        raise ValueError('RTX VSR supports at most 4x enlargement per axis')
    if abs(size[0] / size[1] - result[0] / result[1]) > .001:
        raise ValueError('RTX VSR currently requires matching source/output aspect ratios')
    return result


def processing_size(settings, size):
    """Extent entering NR (or passthrough when NR is absent)."""
    ordered = stages(settings)
    if nr_as_effect(settings) and not settings.get('_inline_hdr_nr'):
        return tuple(size)
    if 'nr' in ordered:
        return output_size(settings,size,ordered[:ordered.index('nr')])
    return tuple(size)


def frame_at_size(frame, size):
    clone = type(frame).from_buffer_copy(frame)
    clone.width, clone.height = size
    return clone


def make_live(width, height, settings, *, wide=False):
    settings = execution_settings(settings)
    ordered = stages(settings)
    if nr_as_effect(settings):
        from gpu_transfer_export import multi_scale_config
        from dlss_gpu_engine import Live
        nr_settings = nr_effect_settings(settings)
        size = output_size(settings,(width,height),ordered[:ordered.index('nr')])
        multi = multi_scale_config(nr_settings, *size)
        owner = Live(*multi['dims'][0], settings, color_format='rgba16f')
        try:
            return Passthrough(width, height, settings, owner=owner, wide=wide)
        except Exception:
            owner.close()
            raise
    if (settings.get('render_pipeline') is not None or 'vfg' in ordered or len(ordered) > 1) and 'nr' not in ordered:
        # Explicit ordered DLAA runs in OutputEffect, including FP16. The core
        # only forwards the decoded picture; evaluating its owner here as well
        # would bind one DLAA host to two independent NVOF contexts in one frame.
        owner = None
        if 'dlaa' in ordered:
            from dlaa_gpu_engine import Live
            size = output_size(settings,(width,height),ordered[:ordered.index('dlaa')])
            hdr_at_dlaa = 'truehdr' in ordered[:ordered.index('dlaa')]
            owner = Live(*size, settings, dll_path=settings.get('dlss_dll_path'), is_hdr=wide or hdr_at_dlaa)
        try:
            return Passthrough(width, height, settings, owner=owner, wide=wide)
        except Exception:
            if owner:
                owner.close()
            raise
    selected = mode(settings)
    if selected == 'vsr':
        return Passthrough(width, height, settings, wide=wide)
    if selected == 'dlaa':
        from dlaa_gpu_engine import Live
        return Live(width, height, settings, dll_path=settings.get('dlss_dll_path'), is_hdr=wide)
    from dlss_gpu_engine import Live
    return Live(width, height, settings, color_format='rgba16f' if wide else 'rgba8')


def make_flow_provider(settings, live, *, raft_shared=None):
    if int(settings.get('guidance_mode', 0)) == 0:
        return None
    if settings.get('flow_engine') == 'fidelityfx':
        from fidelityfx_flow_engine import FlowProvider
        return FlowProvider(live.d3d12_device(), settings.get('flow_quality', 'Quality'))
    if settings.get('flow_engine') in ('raft', 'raft_large'):
        from raft_gpu_flow import FlowProvider
        return FlowProvider(settings, raft_shared)
    from optical_flow import FrameGuidanceService
    from pipeline import GuidanceServiceFlowProvider
    service = FrameGuidanceService(settings.get('flow_engine', 'nvof'),
        settings.get('flow_quality', '1F'), dedicated=True)
    return GuidanceServiceFlowProvider(service, 1)


def _library():
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    lib = C.CDLL(bundle_paths.require_native('gpu_rgba_exchange.dll', 'native effect GPU exchange'))
    lib.gpu_rgba_exchange_last_error.restype = C.c_char_p
    lib.gpu_effect_device_open.argtypes = [C.POINTER(C.c_void_p)]
    lib.gpu_effect_device_close.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_open.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32,
                                          C.c_uint32, C.POINTER(C.c_void_p)]
    lib.gpu_rgba_exchange_close.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_import.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
        C.c_uint64, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
    lib.gpu_rgba_exchange_export.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t,
                                           C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    lib.gpu_rgba_exchange_pointer.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
    lib.gpu_rgba_exchange_fence.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_fence.restype = C.c_void_p
    return lib


def _check(lib, rc):
    if rc:
        raise RuntimeError(lib.gpu_rgba_exchange_last_error().decode('utf-8', 'replace'))


class Exchange:
    def __init__(self, device, context, size, *, wide=False):
        self.lib = _library()
        self.device, self.context, self.size = device, context, tuple(size)
        self.wide = wide
        self.handle = C.c_void_p()
        self.device_copy_bytes = 0
        if wide:
            if not hasattr(self.lib, 'gpu_rgba_exchange_open2'):
                raise RuntimeError('FP16 exchange requires the updated native bridge')
            self.lib.gpu_rgba_exchange_open2.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
            _check(self.lib, self.lib.gpu_rgba_exchange_open2(device, context, *self.size, 8, C.byref(self.handle)))
        else:
            _check(self.lib, self.lib.gpu_rgba_exchange_open(device, context, *self.size, C.byref(self.handle)))

    def pointer(self):
        ptr, pitch = C.c_uint64(), C.c_uint32()
        _check(self.lib, self.lib.gpu_rgba_exchange_pointer(self.handle, C.byref(ptr), C.byref(pitch)))
        return ptr.value, pitch.value

    def color(self, texture):
        ptr, pitch = C.c_uint64(), C.c_uint32()
        _check(self.lib, self.lib.gpu_rgba_exchange_import(self.handle, texture['d3d12_resource'],
            texture.get('d3d12_fence'), texture.get('fence_value', 0), C.byref(ptr), C.byref(pitch)))
        self.device_copy_bytes += self.size[0] * self.size[1] * (8 if self.wide else 4)
        return dict(format='rgba16f_cuda' if self.wide else 'rgba8_cuda', width=self.size[0], height=self.size[1],
                    cuda_context=self.context, device_ptr=ptr.value, pitch_bytes=pitch.value)

    def texture(self, ptr, pitch):
        texture, value = C.c_void_p(), C.c_uint64()
        _check(self.lib, self.lib.gpu_rgba_exchange_export(self.handle, ptr, pitch,
                                                        C.byref(texture), C.byref(value)))
        self.device_copy_bytes += self.size[0] * self.size[1] * (8 if self.wide else 4)
        return dict(d3d12_device=self.device, d3d12_resource=texture.value,
                    d3d12_fence=self.lib.gpu_rgba_exchange_fence(self.handle), fence_value=value.value,
                    width=self.size[0], height=self.size[1],
                    dxgi_format=10 if self.wide else 28, state=0x800, producer_completed=True)

    def close(self):
        if self.handle:
            self.lib.gpu_rgba_exchange_close(self.handle)
            self.handle = None


class Passthrough:
    """GPU resource owner for standalone VSR. Opens no NGX feature."""
    color_format, bytes_per_pixel, session_id = 'rgba8', 4, 0
    native_passthrough = True
    host_dll, host_abi, last_error = 'gpu_rgba_exchange.dll', 1, None
    def __init__(self, width, height, settings, owner=None, *, wide=False):
        self._w, self._h, self.settings = width, height, dict(settings)
        self.color_format, self.bytes_per_pixel = ('rgba16f', 8) if wide else ('rgba8', 4)
        self.lib, self.device, self.exchange = _library(), C.c_void_p(), None
        self.owner = owner
        self.stage_providers = []
        if owner:
            self.device = C.c_void_p(owner.d3d12_device())
        else:
            _check(self.lib, self.lib.gpu_effect_device_open(C.byref(self.device)))

    def d3d12_device(self):
        return self.device.value

    def is_alive(self):
        return bool(self.device)

    def supports_zero_motion(self):
        return True

    def process_device_color_zero_motion_resource(self, color, reset=False):
        if not self.exchange:
            self.exchange = Exchange(self.device.value, color['cuda_context'], (self._w, self._h), wide=self.bytes_per_pixel == 8)
        return self.exchange.texture(color['device_ptr'], color['pitch_bytes'])

    def transfer_counters(self):
        return dict(host_color_stage_count=0, host_color_stage_bytes=0,
                    output_readback_count=0, output_readback_bytes=0,
                    device_color_copy_count=None, device_color_copy_bytes=None,
                    vendor_feature='none; ordered model stages run separately')

    def close(self):
        from gpu_cleanup import release_owned, finish_cleanup
        errors = []
        release_owned(self, 'exchange', errors)
        if self.owner is not None:
            self.device = None  # Borrowed from owner, never independently freed.
            release_owned(self, 'owner', errors)
        elif self.device:
            release_owned(self, 'device', errors, self.lib.gpu_effect_device_close)
        # The DLAA host imports motion memory in each provider's CUDA context.
        # Keep these contexts alive until that host has released its imports.
        providers, self.stage_providers = self.stage_providers, []
        for provider in providers:
            try:
                provider.close_independent_layer()
            except Exception as error:
                errors.append(('stage_flow', error))
        finish_cleanup(errors)


class OutputEffect:
    """Runs only after the final NR composite; returns a borrowed GPU texture."""
    def __init__(self, live, settings, source_size, flow_provider=None, *, kind=None, wide=None, hdr=False):
        self.live, self.settings, self.source_size = live, dict(settings), tuple(source_size)
        self.selected = ('nr_dlaa' if kind == 'dlaa' else 'vsr') if kind else mode(settings)
        self.target = self.source_size if self.selected == 'nr_dlaa' else vsr_output_size(settings, source_size)
        self.flow_provider = flow_provider
        self.input = self.output = self.effect = self.guidance = None
        self.calls, self.resets = 0, []
        self.owns_effect = True
        self.guidance_deferred = False
        self.wide = getattr(live, 'bytes_per_pixel', 4) == 8 if wide is None else wide
        self.output_hdr = hdr
        self.guide_bridge = None
        self.input_codec = self.output_codec = None

    def process(self, candidate, frame, reset):
        if self.input is None:
            self.input = Exchange(candidate['d3d12_device'], frame.cuda_context, self.source_size, wide=self.wide)
            if self.selected == 'nr_dlaa':
                from dlaa_gpu_engine import Live
                owner = getattr(self.live, 'owner', None)
                primary = owner or self.live
                if getattr(primary, 'native_dlaa', False) and primary.bytes_per_pixel == (8 if self.wide else 4):
                    self.effect, self.owns_effect = primary, False
                else:
                    self.effect = Live(*self.source_size, self.settings,
                        dll_path=self.settings.get('dlss_dll_path'), shared_with=owner or self.live, is_hdr=self.wide)
                if int(self.settings.get('guidance_mode', 0)):
                    if not callable(getattr(self.flow_provider, 'new_independent_layer', None)):
                        raise ValueError('DLAA guidance requires an independent device-flow history')
                    self.guidance = self.flow_provider.new_independent_layer()
                    if not self.owns_effect:
                        self.live.stage_providers.append(self.guidance)
                        self.guidance_deferred = True
            else:
                from native_runtime_paths import vsr_runtime
                runtime = vsr_runtime(self.settings.get('vsr_runtime_path'))
                self.runtime = runtime
                self.output = Exchange(candidate['d3d12_device'], frame.cuda_context, self.target, wide=self.wide)
                if self.wide:
                    from native_runtime_paths import wide_vfx_runtime
                    from vfx_rgb10 import Codec
                    wide_vfx_runtime(runtime)
                    hdr_picture = self.output_hdr or int(getattr(frame, 'color_transfer', 0)) == 16
                    self.input_codec = Codec(frame.cuda_context, self.source_size, runtime, hdr=hdr_picture)
                    self.output_codec = Codec(frame.cuda_context, self.target, runtime, hdr=hdr_picture)
        if self.selected != 'nr_dlaa' and self.effect is None:
            from rtx_vsr_engine import RtxVsr
            from vsr_config import configuration
            options = configuration(self.settings)
            self.effect = RtxVsr(bundle_paths.require_native('rtx_vsr_bridge.dll', 'RTX VSR'),
                self.runtime, frame.cuda_context, self.source_size, self.target,
                options['sdk_level'], packed10=self.wide, strength=options['strength'])
        color = self.input.color(candidate)
        if self.selected == 'nr_dlaa':
            if self.guidance:
                guide_color = color
                if self.wide:
                    # Keep the optional wide ABI behind the existing bundle's
                    # explicit hidden import, as the primary HDR chain does.
                    hdr = __import__('hdr_gpu_engine')
                    if self.guide_bridge is None:
                        self.guide_bridge = hdr.GpuHdrBridge(d3d12_device=candidate['d3d12_device'],
                            cuda_context=frame.cuda_context, full_width=self.source_size[0], full_height=self.source_size[1],
                            working_width=self.source_size[0], working_height=self.source_size[1],
                            paper_white_scrgb=float(self.settings.get('hdr_paper_white_scrgb', 8)), purpose='guidance')
                    self.guide_bridge.import_master(color)
                    self.guide_bridge.downscale()
                    _, guide = self.guide_bridge.proxy()
                    guide_color = guide.as_descriptor()
                flow = self.guidance.get_flow_device_rgba(self.calls, guide_color, frame.cuda_context,
                                                          is_discontinuous=reset)
                result = self.effect.process_device_color_flow_resource(color, flow, reset=reset)
            else:
                result = self.effect.process_device_color_zero_motion_resource(color, reset=reset)
        else:
            ptr, pitch = self.output.pointer()
            if self.wide:
                packed, row = self.input_codec.pack(color['device_ptr'], color['pitch_bytes'])
                target, target_row = self.output_codec.pointer()
                self.effect.process(packed, row, target, target_row)
                self.output_codec.unpack(target, target_row, ptr, pitch)
            else:
                self.effect.process(color['device_ptr'], color['pitch_bytes'], ptr, pitch)
            result = self.output.texture(ptr, pitch)
        self.calls += 1
        self.resets.append(bool(reset))
        self.resets = self.resets[-16:]
        if self.output_hdr:
            result['output_hdr'] = True
        return result

    def update_vsr(self, settings):
        if self.selected == 'nr_dlaa': return False
        from vsr_config import configuration
        configuration(settings)
        if vsr_output_size(settings, self.source_size) != self.target: return False
        effect, self.effect = self.effect, None
        if effect: effect.close()
        self.settings = dict(settings)
        return True

    def report(self):
        from vsr_config import configuration
        return dict(vsr_options=configuration(self.settings) if self.selected != 'nr_dlaa' else None,
                    mode=self.selected, input_size=self.source_size, output_size=self.target,
                    stage='dlaa' if self.selected == 'nr_dlaa' else 'vsr',
                    preset_requested=self.settings.get('dlaa_preset', 0) if self.selected == 'nr_dlaa' else None,
                    preset_effective=None,
                    runtime=(getattr(self.effect, 'dlss_dll', None) if self.selected == 'nr_dlaa'
                             else getattr(self, 'runtime', None)),
                    calls=self.calls, recent_resets=self.resets,
                    counters=self.effect.counters() if self.effect else None,
                    host_image_bytes=0, scalar_readback_bytes=None,
                    processing_format='rgba16f' if self.wide else 'rgba8',
                    motion_proxy_only=bool(self.guide_bridge),
                    motion_proxy_counters=self.guide_bridge.transfer_counters() if self.guide_bridge else None,
                    vfx_encoding='rgb10a2' if self.input_codec else None,
                    vfx_colour_counters=self.input_codec.counters() if self.input_codec else None,
                    limitation=('VSR mode/quality and effect strength are separate SDK parameters'
                        if self.selected != 'nr_dlaa' else
                        'DLAA uses video flow and zero depth/jitter; requested model may be overridden'))

    def close(self):
        from gpu_cleanup import release_owned, finish_cleanup
        errors=[]
        if self.owns_effect:release_owned(self,'effect',errors)
        else:self.effect=None  # Borrowed from the shared anchor.
        if not self.guidance_deferred:
            release_owned(self,'guidance',errors,method='close_independent_layer')
        else:self.guidance=None  # The anchor releases deferred flow imports.
        for name in ('guide_bridge','output_codec','input_codec','output','input'):
            release_owned(self,name,errors)
        finish_cleanup(errors)


class TrueHDRStage:
    def __init__(self, live, settings, size, wide, tail, *, hdr=False):
        self.live, self.target, self.input_wide = live, tuple(size), wide
        self.wide, self.output_hdr = True, True
        self.tail = tail
        self.input_hdr = bool(hdr)

    def process(self, candidate, frame, reset):
        return self.tail.process(dict(candidate,output_hdr=self.input_hdr), frame, at_stage=True)

    def report(self):
        return self.tail.effect.report() if self.tail.effect else dict(stage='truehdr', calls=0)

    def close(self):
        pass  # The chain owns the conversion and final HDR encoder together.


class EffectSeries:
    """Ordered borrowed GPU textures. Owns stages, never the shared device owner."""
    def __init__(self, live, settings, source_size, ordered, provider, *, wide=None, hdr=False, truehdr_tail=None):
        self.effects, self.order = [], list(ordered)
        self.input_wide = getattr(live, 'bytes_per_pixel', 4) == 8 if wide is None else wide
        self.live = live
        wide = self.input_wide
        size = tuple(source_size)
        for stage in ordered:
            kind=stage_kind(stage)
            local=stage_settings(settings,stage)
            if kind == 'truehdr':
                effect = TrueHDRStage(live, settings, size, wide, truehdr_tail, hdr=hdr)
            elif kind == 'nr':
                if not hdr:
                    raise ValueError('Inline NR requires the preceding TrueHDR FP16 output')
                from truehdr_pipeline import HdrNRStage
                effect = HdrNRStage(live, settings, size, provider)
            else:
                if kind in ('illustration','animesharp'):
                    from image_upscale import ImageModelEffect
                    effect=ImageModelEffect(local,kind,size,wide=wide,hdr=hdr)
                else:
                    effect = OutputEffect(live, local, size, provider, kind=kind, wide=wide, hdr=hdr)
            effect.stage_id=stage
            self.effects.append(effect)
            size = effect.target
            wide, hdr = effect.wide, effect.output_hdr
        self.target = size
        self.wide, self.output_hdr = wide, hdr

    def process(self, candidate, frame, reset):
        import time
        for effect in self.effects:
            started = time.perf_counter()
            candidate = effect.process(candidate, frame, reset)
            effect.last_process_ms = (time.perf_counter() - started) * 1000
        return candidate

    def color(self, color, frame, reset):
        # Enter from NV12 conversion, and exit back to a CUDA RGBA plane for NR.
        if not hasattr(self, 'entry'):
            self.entry = Exchange(self.live.d3d12_device(), frame.cuda_context,
                                  (color['width'], color['height']), wide=self.input_wide)
        result = self.process(self.entry.texture(color['device_ptr'], color['pitch_bytes']), frame, reset)
        if not hasattr(self, 'exit'):
            self.exit = Exchange(result['d3d12_device'], frame.cuda_context, self.target, wide=self.effects[-1].wide)
        color = self.exit.color(result)
        if self.output_hdr:
            color['output_hdr'] = True
        return color

    def report(self):
        return dict(order=self.order, output_size=self.target,
                    stages=[dict(effect.report(), stage=kind, process_ms=getattr(effect, "last_process_ms", None))
                            for kind, effect in zip(self.order, self.effects)], host_image_bytes=0)

    def close(self):
        from gpu_cleanup import release_owned, finish_cleanup
        errors=[]
        release_owned(self,'exit',errors)
        effects,self.effects=self.effects,[]
        for index,effect in reversed(list(enumerate(effects))):
            try:effect.close()
            except Exception as error:errors.append((f'effect {index}',error))
        release_owned(self,'entry',errors)
        finish_cleanup(errors)


def update_vsr_options(chain, settings):
    """Replace only VSR effects when their output extent remains unchanged."""
    owners = [effect for name in ('input_effect', 'output_effect')
              for effect in getattr(getattr(chain, name, None), 'effects', [])
              if getattr(effect, 'selected', None) in ('vsr', 'nr_vsr')]
    from vsr_config import configuration
    configuration(settings)
    if any(vsr_output_size(stage_settings(settings,getattr(owner,'stage_id','vsr')), owner.source_size) != owner.target for owner in owners):
        return False
    for owner in owners:
        if not owner.update_vsr(stage_settings(settings,getattr(owner,'stage_id','vsr'))): return False
    return True


def update_dlaa_model(chain, settings):
    """Change the DLAA feature while retaining the decoder and other stages."""
    stages = []
    for name in ('input_effect', 'output_effect'):
        series = getattr(chain, name, None)
        stages.extend(effect for effect in getattr(series, 'effects', [])
                      if getattr(effect, 'selected', None) == 'nr_dlaa')
    primary = getattr(chain.live, 'owner', None) or chain.live
    owners = {id(stage.effect): stage.effect for stage in stages if stage.effect is not None}
    if getattr(primary, 'native_dlaa', False):
        owners[id(primary)] = primary
    if any(not hasattr(owner, 'recreate_preset') or
           not hasattr(owner._lib, 'dlss_gpu_host_recreate_dlaa') for owner in owners.values()):
        return False  # Older hosts require a complete rebuild.
    for owner in owners.values():
        if not owner.recreate_preset(settings):
            return False
    for stage in stages:
        stage.settings = dict(settings)
    return True


def install(chain, live, settings, source_size, provider):
    ordered = stages(settings)
    truehdr = 'truehdr' in ordered
    settings = dict(settings, truehdr_enabled=truehdr)
    if truehdr:
        from truehdr_pipeline import Tail
        chain.truehdr_tail = Tail(live, settings, ordered=True)
    from vfx_framegen import config
    multiplier, _ = config(settings)
    if multiplier != 1 and getattr(chain, 'preview_consumer', None) is None:
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'), framegen=True)
        chain.framegen_settings = dict(settings)
    active_vfg = multiplier != 1 and getattr(chain, 'preview_consumer', None) is None
    spatial = [s for s in ordered if s != 'vfg']
    if active_vfg:
        from ordered_vfg import install_ordered_vfg
        spatial = install_ordered_vfg(chain, live, settings, source_size, provider, ordered)
    if has_stage(settings,'vsr') and getattr(live, 'bytes_per_pixel', 4) == 8:
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'))
    # A legacy standalone DLAA remains the core itself, preserving saved sessions.
    if ordered == ['dlaa'] and settings.get('render_pipeline') is None:
        return
    core_nr = 'nr' in spatial and not nr_as_effect(settings)
    split = spatial.index('nr') if core_nr else 0
    before = spatial[:split] if core_nr else []
    after = spatial[split + 1:] if core_nr else spatial
    chain.processing_size = (getattr(chain, 'ordered_input_size', source_size)
        if active_vfg and not core_nr else processing_size(settings, source_size))
    chain.render_order = ordered
    if active_vfg and getattr(getattr(chain, 'ordered_framegen', None), 'hdr_direct', False):
        return  # The VFG driver executes its HDR suffix on every emitted frame.
    tail = getattr(chain, 'truehdr_tail', None)
    source_hdr = getattr(chain,'sdr10',None) is False
    if before:
        chain.input_effect = EffectSeries(live, settings, getattr(chain, 'ordered_input_size', source_size), before, provider, hdr=source_hdr, truehdr_tail=tail)
    if after:
        chain.output_effect = EffectSeries(live, settings, chain.processing_size, after, provider,
            hdr=source_hdr or bool(active_vfg and 'truehdr' in ordered[:ordered.index('vfg')]),
            wide=True if active_vfg and 'truehdr' in ordered[:ordered.index('vfg')] else None, truehdr_tail=tail)
    if nr_as_effect(settings):
        # Post-NR controls belong to the HDR NR stage, not the SDR passthrough.
        from post_control import controls_from_settings
        chain.post_control.update_controls(controls_from_settings({}))
        from temporal_control import temporal_from_settings
        chain.temporal_control.update_controls(temporal_from_settings({'temporal_controls': {'scene_detection': False}}))


def inline_nr_stage(chain):
    series = [getattr(chain, name, None) for name in ('input_effect', 'output_effect', 'framegen_tail')]
    driver = getattr(chain, 'ordered_framegen', None)
    if driver:
        series.append(driver.prefix)
        series.append(driver.suffix)
    return next((effect for group in series if group for effect in group.effects
                 if getattr(effect, 'stage', None) == 'nr'), None)
