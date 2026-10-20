"""Confirmed audit regressions, with no GPU allocation or model execution."""
from types import SimpleNamespace

import pytest

from dlss_service import DLSSService


def test_defaults_and_partial_jobs_keep_scene_detection_on_compatible_backend():
    service = DLSSService()
    assert service.settings['export_backend'] == 'gpu_transfer'
    assert service.settings['temporal_controls']['scene_detection'] is True
    job = service.add_job({'input_path': __file__, 'settings': {'intensity': .5}})
    assert job['settings']['export_backend'] == 'gpu_transfer'
    assert job['settings']['temporal_controls']['scene_detection'] is True
    assert job['settings']['intensity'] == .5
    assert service.settings['intensity'] != .5


def test_auto_controls_migrate_without_overriding_explicit_legacy():
    service = DLSSService()
    assert service.normalize_settings({'export_backend': 'auto'})['export_backend'] == 'gpu_transfer'
    assert service.normalize_settings({'export_backend': 'legacy'})['export_backend'] == 'legacy'
    assert service.normalize_settings({'export_backend': 'auto', 'temporal_controls': {
        'scene_detection': False, 'mode': 'off'}})['export_backend'] == 'auto'


def test_old_pending_queue_migrates_auto_and_preserves_legacy_without_new_controls(tmp_path):
    from queue_store import QueueStore
    path = tmp_path / 'queue.json'
    QueueStore(path).save([dict(id=backend, status='pending', settings={'export_backend': backend})
                           for backend in ('auto', 'legacy')])
    service = DLSSService()
    service.enable_queue_storage(path)
    assert service.jobs[0]['settings']['export_backend'] == 'gpu_transfer'
    assert service.jobs[1]['settings']['export_backend'] == 'legacy'
    assert service.jobs[1]['settings']['temporal_controls']['scene_detection'] is False


def test_legacy_dlaa_truehdr_keeps_guidance_under_the_anchor(monkeypatch):
    import native_effects as effects
    import dlaa_gpu_engine
    evaluated = []
    owner = SimpleNamespace(native_dlaa=True, bytes_per_pixel=4,
        process_device_color_flow_resource=lambda color, flow, reset:
            evaluated.append(flow) or color)
    monkeypatch.setattr(dlaa_gpu_engine, 'Live', lambda *args, **kw: owner)
    monkeypatch.setattr(effects, 'Passthrough', lambda *args, **kw: SimpleNamespace(
        owner=kw['owner'], bytes_per_pixel=4, stage_providers=[]))
    monkeypatch.setattr(effects, 'Exchange', lambda *args, **kw: SimpleNamespace(color=lambda color: color))
    options = {'neural_mode': 'dlaa', 'truehdr_enabled': True, 'guidance_mode': 1}
    live = effects.make_live(64, 64, options)
    guidance = SimpleNamespace(get_flow_device_rgba=lambda *args, **kw: {'cuda_context': 2})
    stage = effects.OutputEffect(live, options, (64, 64),
        SimpleNamespace(new_independent_layer=lambda: guidance), kind='dlaa')
    stage.process({'d3d12_device': 1}, SimpleNamespace(cuda_context=2), True)
    assert live.owner is owner and live.stage_providers == [guidance]
    assert evaluated == [{'cuda_context': 2}]
    assert stage.guidance_deferred and not stage.owns_effect


def test_backend_migration_keeps_invalid_jobs_available_for_removal(tmp_path):
    from queue_store import QueueStore
    path = tmp_path / 'queue.json'
    options = {'export_backend': 'auto', 'temporal_controls': 'invalid old setting'}
    QueueStore(path).save([{'id': 'invalid', 'status': 'pending', 'settings': options}])
    service = DLSSService()
    service.enable_queue_storage(path)
    assert service._queue_storage_error is None
    assert service.jobs[0]['settings'] == options
    service.remove_job('invalid')
    assert service.jobs == []


def test_nvof_available_without_output_grid_is_a_clear_refusal(monkeypatch):
    import optical_flow
    monkeypatch.setattr(optical_flow, '_load_native_bridge', lambda: object())
    monkeypatch.setattr(optical_flow, 'probe_nvof_capabilities', lambda: {
        'available': True, 'device': 'fixture', 'supported_grids': []})
    with pytest.raises(RuntimeError, match='输出网格'):
        optical_flow.NvOFHardwareEngine()
