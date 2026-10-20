"""Runtime trust/cache checks use inert PE fixtures; no NVIDIA code is loaded."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import dlssnr_runtime as runtime


def dll(path, marker=b'A'):
    data = bytearray(256)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<H', data, 68, 0x8664)
    struct.pack_into('<H', data, 86, 0x2000)
    struct.pack_into('<H', data, 88, 0x20b)
    data[120:121] = marker
    path.write_bytes(data)
    return path


def identity(path):
    return dict(status='Valid', signer='CN=NVIDIA Corporation, O=NVIDIA Corporation',
                product='NVIDIA DLSSNR', version='310,8,0,0', thumbprint='test')


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'KNOWN_VERSIONS', {**runtime.KNOWN_VERSIONS, runtime.sha256(dll(tmp_path / 'fixture.dll')): {'version': '310.8.0.0', 'label': 'fixture'}})
    return runtime.RuntimeManager(tmp_path / 'cache', identity_checker=identity)


def test_import_copies_verifies_deduplicates_and_activates(store, tmp_path):
    source = dll(tmp_path / runtime.NAME)
    record = store.import_file(source)
    assert record['version'] == '310.8.0.0'
    assert store.status()['error']['code'] == 'runtime_missing'
    assert Path(record['path']).read_bytes() == source.read_bytes()
    assert Path(record['path']).parent.name == runtime.sha256(source)
    store.import_file(source)
    assert len(store.status()['versions']) == 1
    store.activate(record['id'])
    assert store.status()['active']['path'] == record['path']
    with pytest.raises(runtime.RuntimeErrorDetail, match='不能删除'):
        store.delete(record['id'])
    store.activate(None)
    store.delete(record['id'])
    assert not Path(record['path']).exists()
    assert source.exists()


@pytest.mark.parametrize('field,value', [
    ('status', 'NotSigned'), ('status', 'HashMismatch'),
    ('signer', 'CN=Other Corporation'),
    ('signer', 'CN=NVIDIA Corporation Fake'),
    ('product', 'NVIDIA DLSS'), ('version', 'unknown')])
def test_unrecognized_identity_can_be_imported_and_selected(store, tmp_path, field, value):
    store.identity_checker = lambda path: {**identity(path), field: value}
    record = store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    assert record['trust'] == 'unrecognized'
    assert record['compatibility'] == 'uncharacterized'
    assert record['signature'] == (value if field == 'status' else 'Valid')
    assert store.load()['active'] == record['id']
    assert Path(record['path']).is_file()
    assert not list(store.root.glob('incoming-*'))
    assert not store.operation_lock.locked()


def test_unavailable_signature_is_informational_but_cache_tampering_still_fails(store, tmp_path):
    def unavailable(path):
        raise runtime.RuntimeErrorDetail('signature_unavailable', 'unavailable')
    store.identity_checker = unavailable
    record = store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    assert record['trust'] == 'unrecognized' and record['signature'] == 'Unavailable'
    assert record['version'] == 'unknown'
    Path(record['path']).write_bytes(b'tampered')
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.verified_path(record['id'])
    assert exc.value.code == 'cache_tampered'


def test_hash_and_architecture_rejected_before_signature(tmp_path):
    source = dll(tmp_path / runtime.NAME)
    def forbidden(path):
        pytest.fail('signature should not run')
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.inspect(source, '0' * 64, forbidden)
    assert exc.value.code == 'hash_mismatch'
    value = bytearray(source.read_bytes())
    struct.pack_into('<H', value, 68, 0x14c)
    source.write_bytes(value)
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.inspect(source, identity_checker=forbidden)
    assert exc.value.code == 'wrong_architecture'


def test_cache_tamper_and_missing_fail_without_fallback(store, tmp_path, monkeypatch):
    record = store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    monkeypatch.setattr(runtime, 'manager', lambda: store)
    Path(record['path']).write_bytes(b'tampered')
    assert store.status()['error']['code'] == 'cache_tampered'
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.resolve_runtime()
    assert exc.value.code == 'cache_tampered'
    Path(record['path']).unlink()
    assert store.status()['error']['code'] == 'cache_missing'


def test_fresh_process_rechecks_signature_and_verify_can_force(store, tmp_path):
    record = store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    calls = []
    fresh = runtime.RuntimeManager(store.root, identity_checker=lambda path: (calls.append(path), identity(path))[1])
    fresh.verified_path(record['id'])
    fresh.verified_path(record['id'])
    assert len(calls) == 1
    fresh.verified_path(record['id'], force_signature=True)
    assert len(calls) == 2


def test_unselect_does_not_automatically_reimport_legacy(store, tmp_path, monkeypatch):
    store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    store.activate(None)
    monkeypatch.setattr(runtime, 'manager', lambda: store)
    monkeypatch.setenv('LUNARC_DLSSNR_DLL', str(tmp_path / runtime.NAME))
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.resolve_runtime()
    assert exc.value.code == 'runtime_missing'


def test_directory_source_requires_consent_and_imports_only_named_file(store, tmp_path):
    dll(tmp_path / runtime.NAME)
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.add_source('directory', str(tmp_path), 'local')
    assert exc.value.code == 'source_not_approved'
    source = store.add_source('directory', str(tmp_path), 'local', confirmed=True)
    record = store.import_file(source_id=source['id'])
    assert record['id'] == runtime.sha256(tmp_path / runtime.NAME)
    store.remove_source(source['id'])
    assert (tmp_path / runtime.NAME).exists()


def test_download_requires_pinned_hash_and_partial_files_are_cleaned(store, tmp_path):
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.add_source('download', 'https://example.com/nvngx_dlssnr.dll', 'nvidia', confirmed=True)
    assert exc.value.code == 'hash_required'
    original = dll(tmp_path / runtime.NAME)
    source = store.add_source('download', 'https://example.com/nvngx_dlssnr.dll', 'user approved', runtime.sha256(original), True)
    def partial(url, target, progress):
        target.write_bytes(b'partial')
        raise OSError('lost connection')
    store.downloader = partial
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.import_file(source_id=source['id'])
    assert exc.value.code == 'cache_io'
    assert not list(store.root.glob('incoming-*'))
    store.downloader = lambda url, target, progress: target.write_bytes(original.read_bytes())
    assert store.import_file(source_id=source['id'])['id'] == source['sha256']


@pytest.mark.parametrize('url', ['http://example.com/a', 'https://a:b@example.com/a',
    'https://example.com:444/a', 'file:///a', 'https://example.com/a#fragment'])
def test_download_url_contract(url):
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.parsed_url(url)
    assert exc.value.code == 'invalid_url'


@pytest.mark.parametrize('ip', ['127.0.0.1', '192.168.1.2', '::1', '169.254.169.254'])
def test_download_blocks_private_dns(monkeypatch, ip):
    monkeypatch.setattr(runtime.socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', (ip, 443))])
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.public_address('untrusted.test')
    assert exc.value.code == 'unsafe_address'


def test_cross_host_redirect_rejected_and_tls_connection_closed(tmp_path, monkeypatch):
    closed = []
    response = SimpleNamespace(status=302, getheader=lambda name: 'https://other.test/a')
    connection = SimpleNamespace(request=lambda *a, **k: None, getresponse=lambda: response,
                                 close=lambda: closed.append(True))
    monkeypatch.setattr(runtime, 'public_address', lambda host: '1.1.1.1')
    monkeypatch.setattr(runtime.http.client, 'HTTPSConnection', lambda *a, **k: connection)
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        runtime.fetch_https('https://source.test/a', tmp_path / 'incoming', lambda *a: None)
    assert exc.value.code == 'unsafe_redirect'
    assert closed == [True]


def test_bad_registry_and_traversal_fail_explicitly(store):
    store.root.mkdir()
    store.config.write_text(json.dumps({'schema': 1, 'active': '../dll', 'versions': [], 'sources': []}))
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.load()
    assert exc.value.code == 'cache_metadata'
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.path('../dll')
    assert exc.value.code == 'invalid_id'


def test_host_receives_validated_path_and_layers_reuse_it(monkeypatch):
    import ctypes
    import dlss_gpu_engine as engine
    calls = []
    def root(w, h, path, options, handle):
        calls.append(('root', path))
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 1
        return 0
    def layer(parent, w, h, options, handle):
        calls.append(('layer', parent.value))
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 2
        return 0
    fake = SimpleNamespace(abi_version=3, multi_session_supported=True,
        dlss_gpu_host_open=root, dlss_gpu_host_open_layer=layer,
        dlss_gpu_host_session_id=lambda handle: handle.value,
        dlss_gpu_host_close=lambda handle: None)
    monkeypatch.setattr(engine, '_load', lambda: fake)
    resolutions = []
    monkeypatch.setattr(engine, 'resolve_runtime', lambda: (resolutions.append(1), 'cache/verified.dll')[1])
    first = engine.Live(16, 16)
    second = engine.Live(8, 8, shared_with=first)
    assert calls == [('root', 'cache/verified.dll'), ('layer', 1)]
    assert first.runtime_dll == second.runtime_dll == 'cache/verified.dll'
    assert len(resolutions) == 1
    first.close(); second.close()


def test_uncharacterized_signed_version_can_be_selected_after_identity_checks(store, tmp_path):
    record = store.import_file(dll(tmp_path / 'new.dll', b'B'))
    assert record['compatibility'] == 'uncharacterized'
    store.activate(record['id'])
    assert store.load()['active'] == record['id']


def test_local_zip_copies_only_named_dll_and_preserves_original(store, tmp_path):
    import zipfile
    source = dll(tmp_path / runtime.NAME)
    package = tmp_path / 'model.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.write(source, 'models/' + runtime.NAME)
        archive.writestr('unrelated.exe', b'never extracted')
    record = store.import_file(package)
    assert record['id'] == runtime.sha256(source)
    assert source.exists() and package.exists()
    assert not list(store.root.rglob('unrelated.exe'))


@pytest.mark.parametrize('names', [
    ['../nvngx_dlssnr.dll'], ['/nvngx_dlssnr.dll'], ['C:/nvngx_dlssnr.dll'],
    ['nvngx_dlssnr.dll', 'nested/nvngx_dlssnr.dll'], ['other.dll']])
def test_zip_rejects_unsafe_or_ambiguous_entries(store, tmp_path, names):
    import zipfile
    package = tmp_path / 'bad.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for name in names:
            archive.writestr(name, b'bad')
    with pytest.raises(runtime.RuntimeErrorDetail):
        store.import_file(package)
    assert not list(store.root.glob('incoming-*'))
    assert not store.load()['versions']


def test_archive_download_hash_checked_and_verified_cache_reused(store, tmp_path, monkeypatch):
    import zipfile
    source = dll(tmp_path / runtime.NAME)
    package = tmp_path / 'model.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.write(source, runtime.NAME)
    digest = runtime.sha256(source)
    monkeypatch.setitem(runtime.KNOWN_VERSIONS, digest, dict(version='310.8.0.0', label='fixture',
        source='test', url='https://example.com/test.zip', archive_sha256=runtime.sha256(package)))
    calls = []
    store.downloader = lambda url, target, progress: (calls.append(url), target.write_bytes(package.read_bytes()))
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.import_file(catalog_id=digest)
    assert exc.value.code == 'source_not_approved'
    assert store.import_file(catalog_id=digest, confirmed=True)['id'] == digest
    store.import_file(catalog_id=digest, confirmed=True)
    assert len(calls) == 1
    cached = next((store.root / 'downloads').glob('*.zip'))
    cached.write_bytes(b'tamper')
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.import_file(catalog_id=digest, confirmed=True)
    assert exc.value.code == 'cache_tampered'
    assert not list(store.root.glob('*.part'))


def test_cancelled_download_releases_lock_and_keeps_registry_empty(store):
    source = store.add_source('download', 'https://example.com/a.dll', 'test', '0' * 64, True)
    def cancel(url, target, progress):
        target.write_bytes(b'partial')
        store.cancel()
        progress('downloading', 7, 100)
    store.downloader = cancel
    with pytest.raises(runtime.RuntimeErrorDetail) as exc:
        store.import_file(source_id=source['id'])
    assert exc.value.code == 'cancelled'
    assert not store.operation_lock.locked()
    assert not list(store.root.glob('incoming-*'))
    assert not store.load()['versions']


def test_tun_dns_only_allowed_for_reviewed_catalog_hosts(monkeypatch):
    monkeypatch.setattr(runtime.socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('198.18.0.23', 443))])
    with pytest.raises(runtime.RuntimeErrorDetail):
        runtime.public_address('arbitrary.test')
    assert runtime.public_address('github.com', allow_tun=True) == '198.18.0.23'
    monkeypatch.setattr(runtime.socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('127.0.0.1', 443))])
    with pytest.raises(runtime.RuntimeErrorDetail):
        runtime.public_address('github.com', allow_tun=True)


def test_windows_signature_uses_only_builtin_modules_and_literal_path(monkeypatch, tmp_path):
    import base64
    import subprocess
    recorded = []
    def run(args, **kwargs):
        recorded.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity(None)).encode())
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    assert runtime.windows_identity(tmp_path / "quote'; dollar$.dll")['status'] == 'Valid'
    args, kwargs = recorded[0]
    script = base64.b64decode(args[-1]).decode('utf-16le')
    assert "quote'" not in script
    assert kwargs['env']['LUNARC_INSPECT_DLL'].endswith("quote'; dollar$.dll")
    assert kwargs['env']['PSModulePath'].endswith('Modules')
    assert 'Join-Path $PSHOME' in script


def test_dlaa_only_session_does_not_require_dlssnr(monkeypatch):
    import ctypes
    import dlss_gpu_engine as nr
    import dlaa_gpu_engine as dlaa
    import native_runtime_paths
    calls = []
    class Entry:
        def __call__(self, width, height, path, options, handle):
            calls.append(path)
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 1
            return 0
    fake = SimpleNamespace(abi_version=3, multi_session_supported=True,
        dlss_gpu_host_open_dlaa=Entry(), dlss_gpu_host_session_id=lambda handle: handle.value,
        dlss_gpu_host_close=lambda handle: None)
    monkeypatch.setattr(nr, '_load', lambda: fake)
    monkeypatch.setattr(nr, 'resolve_runtime', lambda: pytest.fail('DLAA-only tried to load NR'))
    monkeypatch.setattr(native_runtime_paths, 'dlss_runtime', lambda path: 'dlaa.dll')
    live = dlaa.Live(32, 32)
    assert calls == ['dlaa.dll'] and live.runtime_dll == 'dlaa.dll'
    live.close()


def test_sources_and_unused_versions_do_not_rebuild_active_preview(store, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'manager', lambda: store)
    record = store.import_file(dll(tmp_path / runtime.NAME), activate=True)
    selected = runtime.selection_key()
    store.add_source('directory', str(tmp_path), 'local', confirmed=True)
    store.import_file(dll(tmp_path / 'unused.dll', b'B'))
    assert runtime.selection_key() == selected
    Path(record['path']).write_bytes(b'tamper')
    assert runtime.selection_key() != selected


def test_dlaa_only_service_key_does_not_inspect_nr_cache(monkeypatch):
    from dlss_service import DLSSService
    service = DLSSService()
    service.settings['render_pipeline'] = ['dlaa']
    monkeypatch.setattr(runtime, 'selection_key', lambda: pytest.fail('DLAA consulted NR cache'))
    assert service._gpu_preview_options_key()[-1] is None
