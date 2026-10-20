from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor
import pytest
import dlssnr_runtime as runtime
from dlss_service import DLSSService
from test_dlssnr_runtime import dll, identity


@pytest.fixture
def api(monkeypatch, tmp_path):
    import server
    service = DLSSService()
    monkeypatch.setattr(runtime, 'KNOWN_VERSIONS', {**runtime.KNOWN_VERSIONS, runtime.sha256(dll(tmp_path / 'fixture.dll')): {'version': '310.8.0.0', 'label': 'fixture'}})
    store = runtime.RuntimeManager(tmp_path / 'cache', identity_checker=identity)
    monkeypatch.setattr(server, 'dlss_service', service)
    monkeypatch.setattr(runtime, 'manager', lambda: store)
    client = TestClient(server.app, base_url='http://127.0.0.1',
                        headers={'X-DLSS-Token': server.LOCAL_TOKEN})
    return service, store, client


def test_runtime_api_import_select_verify_and_delete(api, tmp_path):
    service, store, client = api
    source = dll(tmp_path / runtime.NAME)
    response = client.post('/api/runtime/dlssnr/import', json={'path': str(source), 'activate': False})
    assert response.status_code == 200
    version = response.json()['id']
    assert client.get('/api/runtime/dlssnr').json()['active'] is None
    assert client.post('/api/runtime/dlssnr/activate', json={'id': version}).status_code == 200
    assert client.post('/api/runtime/dlssnr/verify', json={'id': version}).json()['signature'] == 'Valid'
    assert client.delete('/api/runtime/dlssnr/versions/' + version).status_code == 409
    assert client.post('/api/runtime/dlssnr/activate', json={'id': None}).status_code == 200
    assert client.delete('/api/runtime/dlssnr/versions/' + version).status_code == 200


def test_runtime_api_uses_existing_auth_and_origin_gate(api):
    _, _, client = api
    assert client.get('/api/runtime/dlssnr', headers={'X-DLSS-Token': 'invalid'}).status_code == 401
    assert client.post('/api/runtime/dlssnr/import', json={'path': 'bad'},
                       headers={'Origin': 'https://evil.example'}).status_code == 403


def test_import_selects_verified_runtime_without_an_extra_confirmation(api, tmp_path):
    service, store, client = api
    closed = []
    service._close_gpu_preview = lambda: closed.append(True)
    source = dll(tmp_path / runtime.NAME)
    response = client.post('/api/runtime/dlssnr/import', json={'path': str(source)})
    assert response.status_code == 200
    assert store.load()['active'] == response.json()['id']
    assert closed == [True]


def test_unsigned_third_party_import_selects_without_extra_consent(api, tmp_path):
    service, store, client = api
    store.identity_checker = lambda path: dict(identity(path), status='NotSigned', signer='')
    closed = []
    service._close_gpu_preview = lambda: closed.append(True)
    source = dll(tmp_path / runtime.NAME, marker=b'U')
    response = client.post('/api/runtime/dlssnr/import', json={'path': str(source)})
    assert response.status_code == 200
    record = response.json()
    assert record['trust'] == 'unrecognized'
    assert store.load()['active'] == record['id']
    assert client.get('/api/runtime/dlssnr').json()['active']['trust'] == 'unrecognized'
    assert closed == [True]


def test_switch_during_export_refused_without_closing_sessions(api, tmp_path):
    service, store, client = api
    version = store.import_file(dll(tmp_path / runtime.NAME))['id']
    service.active_job_id = 'running'
    service._close_gpu_preview = lambda: pytest.fail('closed running session')
    result = client.post('/api/runtime/dlssnr/activate', json={'id': version})
    assert result.status_code == 409 and result.json()['detail']['code'] == 'export_busy'
    assert store.load()['active'] is None
    # Cache-only import remains available; automatic selection cannot interrupt an export.
    assert client.post('/api/runtime/dlssnr/import', json={'path': str(tmp_path / runtime.NAME)}).status_code == 409
    assert store.load()['active'] is None
    assert client.post('/api/runtime/dlssnr/import', json={'path': str(tmp_path / runtime.NAME), 'activate': False}).status_code == 200


def test_switch_closes_preview_and_missing_errors_are_structured(api):
    service, store, client = api
    calls = []
    service._close_gpu_preview = lambda: calls.append('owner close')
    assert client.post('/api/runtime/dlssnr/activate', json={'id': None}).status_code == 200
    assert calls == ['owner close']
    result = client.post('/api/runtime/dlssnr/import', json={'path': 'missing.dll'})
    assert result.status_code == 400 and result.json()['detail']['code'] == 'file_missing'
    def locks_available():
        for lock in (service._live_lock, service._queue_lock):
            if not lock.acquire(blocking=False):
                return False
            lock.release()
        return True
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(locks_available).result(timeout=2)


def test_sources_require_explicit_user_trust(api, tmp_path):
    _, _, client = api
    body = dict(kind='directory', location=str(tmp_path), label='test')
    assert client.post('/api/runtime/dlssnr/sources', json=body).status_code == 400
    source = client.post('/api/runtime/dlssnr/sources', json={**body, 'confirmed': True}).json()
    assert client.get('/api/runtime/dlssnr').json()['sources'][0]['id'] == source['id']
    assert client.delete('/api/runtime/dlssnr/sources/' + source['id']).status_code == 200
