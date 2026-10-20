"""Trust/option tests use inert PE fixtures; no model DLL is loaded."""
from types import SimpleNamespace

import pytest

import dlssnr_runtime as runtime
from dlss_service import DLSSService
from dlss_gpu_engine import baked_option_key
from layer_settings import resolve_layer_settings
from preview_memory import model_changed
from tests.unit.test_dlssnr_runtime import dll, identity


@pytest.fixture
def community(tmp_path, monkeypatch):
    path = dll(tmp_path / runtime.NAME, marker=b'C')
    digest = runtime.sha256(path)
    monkeypatch.setattr(runtime, 'COMMUNITY_VERSIONS', {digest: {
        'version': '310.9.16.0', 'bytes': path.stat().st_size,
        'publisher': 'ShortFuse / OpenNR'}})
    monkeypatch.setattr(runtime, 'MODEL_AB_HASHES', frozenset({digest}))
    def check(path):
        return dict(identity(path), version='310.9.16.0', status='NotSigned')
    return path, digest, check


def test_pinned_community_import_is_experimental_and_preserves_signature_state(community, tmp_path):
    path, digest, check = community
    manager = runtime.RuntimeManager(tmp_path / 'cache', identity_checker=check)
    result = manager.import_file(path)
    assert result['sha256'] == digest and result['trust'] == 'pinned_community'
    assert result['signature'] == 'NotSigned' and result['certificate'] is None
    assert result['publisher'] == 'ShortFuse / OpenNR'
    assert result['compatibility'] == 'uncharacterized' and result['model_presets'] == [1, 2]
    assert manager.load()['active'] is None
    manager.activate(digest)
    assert manager.load()['active'] == digest
    fresh = runtime.RuntimeManager(manager.root, identity_checker=check)
    assert fresh.verified_path(digest)[1]['trust'] == 'pinned_community'


@pytest.mark.parametrize('field,value', [('status', 'UnknownError'),
                                        ('product', 'NVIDIA DLSS')])
def test_pinned_community_identity_does_not_require_signature_or_product(community, field, value):
    path, digest, check = community
    result = runtime.inspect(path, digest, lambda p: {**check(p), field: value})
    assert result['trust'] == 'pinned_community'
    assert result['signature'] == (value if field == 'status' else 'NotSigned')


def test_metadata_mismatch_is_unrecognized_and_custom_new_build_offers_ab(community, monkeypatch):
    path, digest, check = community
    result = runtime.inspect(path, digest, lambda p: {**check(p), 'version': '310.8.0.0'})
    assert result['trust'] == 'unrecognized'
    monkeypatch.setitem(runtime.COMMUNITY_VERSIONS[digest], 'bytes', 512)
    assert runtime.inspect(path, identity_checker=check)['trust'] == 'unrecognized'
    path.write_bytes(path.read_bytes() + b'unknown')
    result = runtime.inspect(path, runtime.sha256(path), check)
    assert result['trust'] == 'unrecognized'
    assert result['model_presets'] == [1, 2]


def test_hash_mismatch_rejected_even_for_pinned_community(community):
    path, _, check = community
    with pytest.raises(runtime.RuntimeErrorDetail) as error:
        runtime.inspect(path, '0' * 64, check)
    assert error.value.code == 'hash_mismatch'


def test_model_hint_and_layer_snapshot_preserve_independent_choices():
    service = DLSSService()
    settings = service.normalize_settings({'preset': 1, 'layer_settings': [{}, {'preset': 2}]})
    assert [baked_option_key(resolve_layer_settings(settings, i))[0] for i in (0, 1)] == [1, 2]
    job = service.add_job({'input_path': __file__, 'settings': settings})
    service.update_settings({'preset': 2})
    assert job['settings']['preset'] == 1 and job['settings']['layer_settings'][1]['preset'] == 2
    assert service.normalize_settings({'preset': 3})['preset'] == 3  # old saved hint
    with pytest.raises(ValueError):
        service.normalize_settings({'layer_settings': [{}, {'preset': 1.5}]})


def test_layer_model_switch_requires_rebuild_before_any_hot_update():
    from gpu_preview_engine import GpuPreview
    preview = GpuPreview()
    calls = []
    owner = lambda: SimpleNamespace(update_runtime_options=lambda _: calls.append('updated') or True)
    old = dict(preset=1, reconstruction_mode='dual', layer_settings=[{}, {'preset': 1}])
    preview.active_settings = old
    preview.live = owner()
    preview.multi = {'scales': [1., 1.]}
    preview.chain = SimpleNamespace(extra_layers=[{'live': owner()}])
    changed = dict(old, intensity=.5, layer_settings=[{}, {'preset': 2}])
    try:
        assert model_changed(old, changed)
        assert not preview._update_runtime(changed)
        assert calls == []
    finally:
        preview.chain = preview.live = None
        preview.shutdown()
