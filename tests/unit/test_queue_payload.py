import asyncio
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import pipeline
from dlss_service import DLSSService
from queue_payload import diagnostic_json, queue_job


def test_native_encoder_options_and_binary_evidence_are_json_safe():
    raw = {'plan': {'codec': b'hevc_nvenc', 'preset': b'p7'},
           'packet': b'\x00\xff\x01', 'nested': (bytearray(b'hq'), memoryview(b'\x80'))}
    result = json.loads(json.dumps(diagnostic_json(raw)))
    assert result['plan'] == {'codec': 'hevc_nvenc', 'preset': 'p7'}
    assert base64.b64decode(result['packet']['data']) == raw['packet']
    assert result['packet']['byte_length'] == 3
    assert raw['plan']['codec'] == b'hevc_nvenc'


def test_queue_never_walks_large_frame_diagnostics():
    class Unreadable(dict):
        def items(self):
            raise AssertionError('queue walked frame diagnostics')
    job = {'id': 'done', 'status': 'completed',
           'residual_backend_diagnostics': Unreadable(frames=[{}] * 40000)}
    wire = queue_job(job)
    assert wire['diagnostics_available'] is True
    assert 'residual_backend_diagnostics' not in wire
    assert len(json.dumps(wire)) < 200


def test_summary_preserves_count_and_actual_solver_convergence():
    frames = [{'converged': True}] * 100
    source = {'fusion_solver_frames': frames, 'hdr_frame_reports': [{}] * 40000}
    summary = diagnostic_json(source, summary=True)
    assert summary['fusion_solver_frames']['item_count'] == 100
    assert summary['fusion_solver_frames']['all_converged'] is True
    assert summary['hdr_frame_reports']['item_count'] == 40000
    assert len(json.dumps(summary)) < 300
    frames[-1] = {'converged': False}
    assert diagnostic_json(source, summary=True)['fusion_solver_frames']['all_converged'] is False
    assert len(diagnostic_json(source)['hdr_frame_reports']) == 40000


@pytest.fixture
def api(monkeypatch):
    import server
    service = DLSSService()
    monkeypatch.setattr(server, 'dlss_service', service)
    monkeypatch.setattr(server, 'queue_ws_clients', set())
    monkeypatch.setattr(server, 'main_loop', None)
    monkeypatch.setattr(pipeline, 'video_info', lambda path: (2, 30., 256, 144))
    client = TestClient(server.app, base_url='http://127.0.0.1',
                        headers={'X-DLSS-Token': server.LOCAL_TOKEN})
    yield server, service, client
    client.close()
    if service._cap:
        service._cap.release()


def completed(service, name):
    job = service.add_job({'name': name, 'input_path': __file__,
                           'output_path': name + '.mkv'})
    job['status'] = 'completed'
    job['residual_backend_diagnostics'] = {
        'hdr_negotiation': {'plan': {'struct': {'codec': b'hevc_nvenc', 'preset': b'p7'}}},
        'hdr_frame_reports': [{}] * 40000, 'packet': b'\x00\xff'}
    return job


def test_ws_reconnect_delete_completed_add_and_load_next_material(api, tmp_path):
    server, service, client = api
    first = completed(service, 'first')
    second = completed(service, 'second')
    for _ in range(2):
        with client.websocket_connect('ws://127.0.0.1/ws/queue?token=' + server.LOCAL_TOKEN) as ws:
            state = ws.receive_json()
            assert state['type'] == 'init'
            assert len(state['jobs']) == 2
            assert len(json.dumps(state)) < 15000
    detail = client.get('/api/diagnostics/job/' + second['id']).json()
    assert detail['job']['residual_backend_diagnostics']['hdr_negotiation']['plan']['struct']['codec'] == 'hevc_nvenc'
    summary = client.get('/api/diagnostics/job/' + second['id'] + '?summary=true').json()
    assert summary['job']['residual_backend_diagnostics']['hdr_frame_reports']['item_count'] == 40000
    with client.websocket_connect('ws://127.0.0.1/ws/queue?token=' + server.LOCAL_TOKEN) as ws:
        ws.receive_json()
        removed = client.post('/api/queue/remove', json={'id': first['id']})
        assert removed.status_code == 200
        assert ws.receive_json()['type'] == 'job_removed'
        assert [j['id'] for j in client.get('/api/queue').json()['jobs']] == [second['id']]
        added = client.post('/api/queue/add', json={'input_path': __file__, 'output_path': 'next.mkv'})
        assert added.status_code == 200
        assert ws.receive_json()['type'] == 'job_added'
        video = tmp_path / 'next.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (256, 144))
        assert writer.isOpened()
        try:
            for value in (70, 110, 150):
                writer.write(np.full((144, 256, 3), value, np.uint8))
        finally:
            writer.release()
        loaded_video = client.post('/api/video/load', json={'path': str(video)})
        assert loaded_video.status_code == 200
        assert loaded_video.json()['video']['frames'] == 3
        assert service.get_frame(1).shape == (144, 256, 3)
        image = tmp_path / '新素材.png'
        pipeline.imwrite_u(str(image), np.full((144, 256, 3), 100, np.uint8))
        loaded = client.post('/api/video/load', json={'path': str(image)})
        assert loaded.status_code == 200
        assert loaded.json()['video']['filename'] == image.name
        preview = client.get('/api/source/frame', params={'path': str(image)})
        assert preview.status_code == 200
        assert cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_COLOR).shape == (144, 256, 3)


def test_thread_final_notification_does_not_send_binary_or_frame_arrays(api):
    server, service, _ = api
    job = completed(service, 'thread')
    class Socket:
        async def send_text(self, value):
            self.value = json.loads(value)
    socket = Socket()
    server.queue_ws_clients.add(socket)
    async def run():
        server.main_loop = asyncio.get_running_loop()
        await asyncio.to_thread(server.broadcast_queue_update_from_thread, job)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert socket.value['job']['status'] == 'completed'
    assert socket.value['job']['diagnostics_available'] is True
    assert 'residual_backend_diagnostics' not in socket.value['job']
