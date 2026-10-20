import asyncio
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
from preview_memory import pressured,model_changed,image_workspace_floor,check_cold_start,GIB
from gpu_preview_engine import GpuPreview
from dlss_service import DLSSService
from preview_player import PreviewPlayer,notify_memory,render_gate

def test_pressure_uses_live_budget_or_cuda_free_memory():
    assert pressured({'process':{'local_usage':14*GIB,'local_budget':16*GIB}})
    assert not pressured({'process':{'local_usage':8*GIB,'local_budget':16*GIB}})
    assert pressured({'cuda':{'free_bytes':GIB//2,'total_bytes':16*GIB}})
    assert not pressured({'process_error':'unavailable'})
    assert not model_changed({'post_controls':{}},{'post_controls':{'total_gain':2}})
    assert model_changed({'dlaa_preset':10},{'dlaa_preset':12})

def test_cold_start_rejects_known_image_workspace_shortfall_without_changing_settings():
    settings={'render_pipeline':['illustration'],'image_model_tile':128}
    assert image_workspace_floor(settings,(100,200))==100*200*16+400*800*19
    with pytest.raises(ValueError,match='可用显存'):
        check_cold_start({'cuda':{'free_bytes':GIB,'total_bytes':16*GIB}},settings,(1536,2752))
    assert settings['image_model_tile']==128
    check_cold_start({},settings,(1024,2048))

def test_release_runs_behind_current_owner_work_and_drops_shared_weights(monkeypatch):
    preview=GpuPreview();entered=threading.Event();finish=threading.Event();calls=[]
    preview.raft_shared['model']=object()
    monkeypatch.setattr('preview_memory.sample',lambda *args:{'cuda':None})
    monkeypatch.setattr(preview,'_close',lambda:calls.append(('close',threading.get_ident())))
    def native():entered.set();assert finish.wait(3)
    previous=preview.executor.submit(native);assert entered.wait(2)
    future=preview.executor.submit(preview._release_memory)
    try:
        assert not future.done() and not calls
        finish.set();previous.result(3)
        report=future.result(3)
        assert len(calls)==1 and calls[0][1]!=threading.get_ident()
        assert not preview.raft_shared and preview.cold_memory_check
        assert report['reason']=='manual'
    finally:finish.set();preview.executor.shutdown(wait=True)

def fake_service():
    service=DLSSService.__new__(DLSSService)
    service._live_lock=threading.Lock()
    service.active_job_id=service._live_session_owner=service._live_session=service._preview_backend=None
    service._ensure_native_healthy=lambda:None
    service._get_gpu_preview=lambda:SimpleNamespace(release_memory=lambda:{'released':True})
    return service

def test_service_rejects_release_during_export_and_keeps_manual_pause_until_resume():
    service=fake_service();service.active_job_id='export'
    with pytest.raises(ValueError,match='导出'):service.begin_preview_release()
    assert not service.preview_memory_state()['suspended']
    service.active_job_id=None;service.begin_preview_release()
    assert service.render_preview(0)==(None,'GPU_PREVIEW_RELEASED')
    assert service.release_preview_memory()=={'released':True}
    assert not service.preview_memory_state()['releasing']
    assert not service.resume_preview_memory()['suspended']


def test_failed_release_does_not_leave_the_service_permanently_busy():
    service = fake_service()
    def fail():raise RuntimeError('release failed')
    service._get_gpu_preview = lambda: SimpleNamespace(release_memory=fail)
    service.begin_preview_release()
    with pytest.raises(RuntimeError, match='release failed'):
        service.release_preview_memory()
    state = service.preview_memory_state()
    assert state['suspended'] and not state['releasing']
    assert state['error'] == 'release failed'

def test_suspension_invalidates_inflight_frame_and_rejects_refresh_until_resume():
    from tests.unit.test_preview_player import Socket,Service,until,close
    async def run():
        service=Service();service._preview_suspended=False
        socket=Socket();player=PreviewPlayer(socket,service)
        player.command(dict(action='play',frame=1,request_id=1))
        task=asyncio.create_task(player.run());await until(service.entered.is_set)
        service._preview_suspended=True;notify_memory(service,pause=True)
        player.command(dict(action='refresh',request_id=1))
        service.release.set()
        await until(lambda:any(m['type']=='preview_memory' for m in socket.messages))
        assert len(service.calls)==1 and not any(m['type']=='frame' for m in socket.messages)
        assert not player.playing and not player.dirty
        service._preview_suspended=False
        player.command(dict(action='seek',frame=2,request_id=2))
        await until(lambda:any(m['type']=='frame' for m in socket.messages))
        await close(player,task)
    asyncio.run(run())

def test_low_memory_model_switch_bypasses_hot_recreation_and_releases_cache(tmp_path,monkeypatch):
    from tests.unit.test_gpu_preview_lifecycle import _inject_open,_settings
    path=tmp_path/'source.mkv';path.write_bytes(b'fixture')
    preview=GpuPreview();opened=[];_inject_open(preview,opened)
    monkeypatch.setattr('preview_memory.sample',lambda *args:{'process':{'local_usage':15*GIB,'local_budget':16*GIB}})
    hot=[]
    try:
        settings=dict(_settings(),render_pipeline=['dlaa'],dlaa_preset=10)
        preview._render(str(path),settings,0)
        preview.raft_shared['model']=object()
        monkeypatch.setattr(preview,'_update_runtime',lambda settings:hot.append(settings) or True)
        preview._render(str(path),dict(settings,dlaa_preset=12),1)
        assert not hot and len(opened)==2 and opened[0]['chain'].closed
        assert not preview.raft_shared
        assert preview.memory_release_report['reason']=='memory_pressure_model_switch'
        assert opened[1]['settings']['dlaa_preset']==12
    finally:preview.shutdown()
