"""Routing/extent and resource lifetime contract for user-ordered model stages."""
from itertools import permutations
from types import SimpleNamespace
import ctypes as C
import pytest
import native_effects as effects
import native_runtime_paths as paths


@pytest.mark.parametrize('order', list(permutations(('nr', 'dlaa', 'vsr'))))
def test_every_permutation_preserves_order_and_resolves_nr_input(order):
    settings = dict(render_pipeline=list(order), export_backend='gpu_transfer',
                    guidance_mode=0, working_scale=.5, reconstruction_mode='dual')
    effects.validate(settings, dict(width=1280, height=720, pix_fmt='yuv420p', color_transfer='bt709'))
    assert effects.stages(settings) == list(order)
    assert effects.output_size(settings, (1280, 720)) == (1920, 1080)
    expected = (1920, 1080) if order.index('vsr') < order.index('nr') else (1280, 720)
    assert effects.processing_size(settings, (1280, 720)) == expected


@pytest.mark.parametrize('order', [[], ['nr','nr'], ['dlaa','bogus'], 'dlaa,nr', [None]])
def test_invalid_orders_fail(order):
    with pytest.raises(ValueError):
        effects.stages(dict(render_pipeline=order))


def test_legacy_keeps_original_order():
    assert effects.stages(dict(neural_mode='nr_dlaa')) == ['nr','dlaa']
    assert effects.stages(dict(neural_mode='nr_vsr')) == ['nr','vsr']
    assert effects.stages(dict(neural_mode='nr_dlaa', render_pipeline=['dlaa','nr'])) == ['dlaa','nr']


@pytest.mark.parametrize('wide', [False, True])
def test_explicit_solo_dlaa_evaluates_only_in_ordered_stage(monkeypatch, wide):
    import dlaa_gpu_engine
    opened = []
    owner = SimpleNamespace(native_dlaa=True, bytes_per_pixel=8 if wide else 4)
    def open_dlaa(*args, **kwargs):
        opened.append((args, kwargs))
        return owner
    def passthrough(width, height, settings, owner=None, wide=False):
        return SimpleNamespace(native_passthrough=True, owner=owner,
                               bytes_per_pixel=8 if wide else 4)
    monkeypatch.setattr(dlaa_gpu_engine, 'Live', open_dlaa)
    monkeypatch.setattr(effects, 'Passthrough', passthrough)
    settings = dict(render_pipeline=['dlaa'], guidance_mode=1)
    live = effects.make_live(1280, 720, settings, wide=wide)
    assert live.native_passthrough and live.owner is owner
    assert len(opened) == 1 and opened[0][1]['is_hdr'] is wide
    chain = SimpleNamespace(preview_consumer=lambda **kw: None)
    effects.install(chain, live, settings, (1280, 720), None)
    assert chain.output_effect.order == ['dlaa']
    assert not hasattr(chain, 'input_effect')


def test_legacy_solo_wide_dlaa_remains_the_core(monkeypatch):
    import dlaa_gpu_engine
    owner = SimpleNamespace(native_dlaa=True, bytes_per_pixel=8)
    monkeypatch.setattr(dlaa_gpu_engine, 'Live', lambda *a, **kw: owner)
    settings = dict(neural_mode='dlaa', guidance_mode=1)
    live = effects.make_live(1280, 720, settings, wide=True)
    chain = SimpleNamespace()
    effects.install(chain, live, settings, (1280, 720), None)
    assert live is owner and not hasattr(chain, 'output_effect')


def test_borrowed_dlaa_keeps_motion_context_alive_until_host_close(monkeypatch):
    events = []
    owner = SimpleNamespace(native_dlaa=True, bytes_per_pixel=4,
        process_device_color_flow_resource=lambda color, flow, reset:
            events.append(('evaluate', flow, reset)) or color,
        close=lambda: events.append('host_close'))
    live = object.__new__(effects.Passthrough)
    live.owner, live.device, live.exchange = owner, 123, None
    live.bytes_per_pixel, live.stage_providers = 4, []
    guidance = SimpleNamespace(
        get_flow_device_rgba=lambda *a, **kw: {'cuda_context': 456},
        close_independent_layer=lambda: events.append('flow_close'))
    provider = SimpleNamespace(new_independent_layer=lambda: guidance)
    monkeypatch.setattr(effects, 'Exchange', lambda *a, **kw: SimpleNamespace(
        color=lambda candidate: candidate, close=lambda: events.append('exchange_close')))
    stage = effects.OutputEffect(live, {'guidance_mode': 1}, (1280, 720),
                                 provider, kind='dlaa')
    frame = SimpleNamespace(cuda_context=999)
    stage.process({'d3d12_device': 123}, frame, True)
    stage.process({'d3d12_device': 123}, frame, False)
    assert live.stage_providers == [guidance]
    assert events == [('evaluate', {'cuda_context': 456}, True),
                      ('evaluate', {'cuda_context': 456}, False)]
    stage.close()
    assert events[-1] == 'exchange_close'
    assert 'flow_close' not in events and 'host_close' not in events
    live.close()
    live.close()
    assert events[-2:] == ['host_close', 'flow_close']


@pytest.mark.parametrize('order', [['dlaa'], ['dlaa', 'nr'], ['nr', 'dlaa']])
@pytest.mark.parametrize('transfer', ['bt709', 'smpte2084'])
def test_wide_dlaa_admission_preserves_source_domain(order, transfer):
    settings = dict(render_pipeline=order, export_backend='gpu_transfer', working_scale=1.)
    effects.validate(settings, dict(width=1280, height=720,
        pix_fmt='yuv420p10le', color_transfer=transfer))


def test_vsr_wide_refusal_also_guards_direct_chain_call(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('VFX 1.3 missing')
    monkeypatch.setattr(paths, 'wide_vfx_runtime', missing)
    with pytest.raises(FileNotFoundError, match='VFX 1.3'):
        effects.install(SimpleNamespace(), SimpleNamespace(bytes_per_pixel=8),
            dict(render_pipeline=['nr', 'vsr']), (1280,720), None)


def test_hdr_source_cut_resets_prefix_before_it_consumes_the_picture():
    from gpu_hdr_transfer_export import HdrChain
    events = []
    plane = SimpleNamespace(as_descriptor=lambda: dict(format='rgba16f_cuda', device_ptr=7))
    chain = object.__new__(HdrChain)
    chain.prores_input = False
    chain.preview_consumer = None
    chain.bridge = SimpleNamespace(decode=lambda **kw: plane, source_report=lambda: {},
        import_master=lambda value: plane, downscale=lambda: plane,
        proxy=lambda: (plane, plane), proxy_report=lambda: {})
    def observe(master, first, discontinuous):
        events.append(('observe', first))
        chain._frame_reset = True
        return True
    chain._detect_scene = observe
    chain.input_effect = SimpleNamespace(color=lambda color, frame, reset:
        events.append(('prefix', color['format'], reset)) or color)
    chain.policy = dict(support_ceiling_scrgb=250)
    frame = SimpleNamespace(y_device_ptr=1, y_pitch_bytes=256, uv_device_ptr=2, uv_pitch_bytes=256)
    chain._primary_colour(frame, {}, {}, False)
    assert events == [('observe', False), ('prefix', 'rgba16f_cuda', True)]
    assert chain._prefix_discontinuous is True


def test_series_executes_exact_order_with_the_same_reset_and_releases_reverse(monkeypatch):
    calls = []
    class FakeEffect:
        def __init__(self, live, settings, size, provider, *, kind, wide=False, hdr=False):
            self.kind, self.size = kind, size
            self.wide, self.output_hdr = wide, hdr
            self.target = (1920, 1080) if kind == 'vsr' else size
        def process(self, candidate, frame, reset):
            calls.append((self.kind, self.size, reset, candidate['origin']))
            return dict(origin=self.kind)
        def close(self):
            calls.append(('close', self.kind))
    monkeypatch.setattr(effects, 'OutputEffect', FakeEffect)
    series = effects.EffectSeries(None, {}, (1280,720), ['vsr','dlaa'], None)
    assert series.process(dict(origin='source'), None, True) == dict(origin='dlaa')
    assert calls == [('vsr',(1280,720),True,'source'),('dlaa',(1920,1080),True,'vsr')]
    series.close()
    assert calls[-2:] == [('close','dlaa'),('close','vsr')]


def test_frame_extent_change_preserves_pts_and_does_not_mutate_decoder():
    class Frame(C.Structure):
        _fields_ = [('width', C.c_uint32), ('height', C.c_uint32), ('pts', C.c_int64)]
    frame = Frame(1280,720,123)
    result = effects.frame_at_size(frame, (1920,1080))
    assert (result.width,result.height,result.pts) == (1920,1080,123)
    assert (frame.width,frame.height,frame.pts) == (1280,720,123)


def test_source_cut_resets_prefix_before_independent_processed_extent_observation(monkeypatch):
    from gpu_transfer_export import Chain
    import temporal_control
    events = []

    class Observer:
        controls = SimpleNamespace(enabled=True)
        def __init__(self, label):
            self.label = label
        def observe(self, **kwargs):
            events.append((self.label, kwargs['width'], kwargs['height'], kwargs['reset']))
            return dict(scene_cut=self.label == 'source', classification='cut')

    def convert(*args):
        args[-2]._obj.value = 100
        args[-1]._obj.value = 5120
        return 0

    def prefix_color(color, frame, reset):
        events.append(('prefix', frame.width, frame.height, reset))
        return dict(device_ptr=200, pitch_bytes=7680)

    monkeypatch.setattr(temporal_control, 'GpuTemporalControl', lambda controls: Observer('source'))
    chain = object.__new__(Chain)
    chain.b = dict(nv12=SimpleNamespace(nv12_rgba_convert=convert), lanczos=None)
    chain.nv12 = None
    chain.temporal_control = Observer('processed')
    chain.scene_observer = None
    chain.input_effect = SimpleNamespace(color=prefix_color)
    chain.live = SimpleNamespace(d3d12_device=lambda: 1,
        process_device_color_zero_motion_resource=lambda color, reset: dict(color=color, reset=reset,
            d3d12_resource=10, d3d12_fence=0, fence_value=0))
    chain._processing_frame = lambda frame: SimpleNamespace(width=1920, height=1080, cuda_context=1, pts=42)
    chain.first = chain.downscaled = chain.uses_flow = False
    chain.defer_encode = True
    chain.working = (1920,1080)
    chain.zero_motion_calls = 0
    chain.settings_native_mode = lambda: 'nr'
    chain.preview_consumer = None
    frame = SimpleNamespace(width=1280, height=720, cuda_context=1, pts=42,
        y_device_ptr=1, y_pitch_bytes=1280, uv_device_ptr=2, uv_pitch_bytes=1280)
    result = chain.process(frame, None, 1, False)
    assert events == [('source',1280,720,False), ('prefix',1280,720,True),
                      ('processed',1920,1080,True)]
    assert result['_dlss_resource']['reset'] is True
    assert result['_working_color']['device_ptr'] == 200
    assert result['_processing_frame'].pts == 42


def test_explicit_runtime_overrides_never_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match='override'):
        paths.dlss_runtime(tmp_path / 'missing.dll')
    with pytest.raises(FileNotFoundError, match='incomplete'):
        paths.vsr_runtime(tmp_path)


def test_dlss_discovery_prefers_local_ngx_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('LUNARC_DLSS_DLL', raising=False)
    monkeypatch.setenv('PROGRAMDATA', str(tmp_path))
    dll = tmp_path / 'NVIDIA/NGX/models/dlss_override/versions/123/files/abc/nvngx_dlss.dll'
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b'fixture')
    assert paths.dlss_runtime() == str(dll.resolve())


@pytest.mark.parametrize('preset', [0,5,6,10,11,12,13])
def test_dlaa_hint_is_independent_of_nr_preset(monkeypatch, preset):
    import dlaa_gpu_engine as dlaa
    import dlss_gpu_engine as nr
    monkeypatch.setattr(nr.Live, '_options', lambda self: SimpleNamespace(preset=2))
    live = object.__new__(dlaa.Live)
    live.render_preset = preset
    assert live._options().preset == preset


@pytest.mark.parametrize('preset', [True,1,2,3,4,7,8,9,14,15,'K'])
def test_removed_and_reserved_dlaa_hints_fail(preset):
    settings = dict(render_pipeline=['dlaa','nr'], export_backend='gpu_transfer', dlaa_preset=preset)
    with pytest.raises(ValueError, match='DLAA preset'):
        effects.validate(settings, dict(pix_fmt='nv12', color_transfer='bt709'))
