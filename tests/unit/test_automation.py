"""Automation contract tests. Native execution is replaced; no GPU or real media export."""
import copy
import threading
import time

import pytest

from automation import Automation,create_router
from dlss_service import DLSSService
from preset_store import PresetStore


@pytest.fixture
def fixture(tmp_path):
    service=DLSSService()
    automation=Automation(service,PresetStore(tmp_path/'presets.json'))
    source=tmp_path/'source.mp4';source.write_bytes(b'fake media; never decode')
    request=dict(input_path=str(source),output_path=str(tmp_path/'output.mkv'),request_id='stable-request',start=False)
    return service,automation,request


@pytest.mark.parametrize('patch',[[],False,'',0])
def test_empty_non_object_settings_are_rejected(fixture,patch):
    with pytest.raises(ValueError,match='对象'):
        fixture[1].settings(patch=patch)


def test_transient_block_clears_after_queue_finished_but_native_fault_remains(fixture):
    service, api, request = fixture
    job_id = api.submit(request, None)['job']['id']
    api.notify({'type': 'automation_blocked', 'error': 'transient'})
    assert api.state([job_id])['blocked'] == 'transient'
    api.notify({'type': 'queue_finished'})
    assert api.state([job_id])['blocked'] is None
    service._native_degraded = 'device removed'
    api.notify({'type': 'queue_finished'})
    assert api.state([job_id])['blocked'] == 'device removed'


def test_draft_settings_leave_ui_revision_and_queued_snapshots_unchanged(fixture):
    service,api,request=fixture
    before=copy.deepcopy(service.settings)
    revision=getattr(service,'_preview_revision',0)
    api.presets.save('Saved',dict(intensity=1.25,render_pipeline=['nr','vfg'],frame_generation='vfg',frame_multiplier=16))
    resolved=api.settings('Saved',dict(post_controls={'chroma_gain':.5}))
    assert resolved['intensity']==1.25 and resolved['frame_multiplier']==16
    assert service.settings==before and getattr(service,'_preview_revision',0)==revision
    first=api.submit(dict(request,preset='Saved'),None)['job']
    service.update_settings(dict(intensity=.2))
    job=next(j for j in service.jobs if j['id']==first['id'])
    assert job['settings']['intensity']==1.25
    resolved['render_pipeline'].clear()
    assert job['settings']['render_pipeline']==['nr','vfg']


def test_request_id_is_persistent_and_conflicting_retry_is_refused(fixture,tmp_path):
    service,api,request=fixture
    service.enable_queue_storage(str(tmp_path/'queue.json'))
    first=api.submit(request,None)
    assert api.submit(request,None)['reused'] and len(service.jobs)==1
    recovered=DLSSService();recovered.enable_queue_storage(str(tmp_path/'queue.json'))
    recovered_api=Automation(recovered,api.presets)
    assert recovered_api.submit(request,None)['job']['id']==first['job']['id']
    with pytest.raises(ValueError,match='不同'):
        api.submit(dict(request,settings={'intensity':.1}),None)


def test_mcp_queue_starts_only_automation_jobs_and_not_failed_or_manual_rows(fixture):
    service,api,request=fixture
    manual=service.add_job(dict(input_path=request['input_path'],output_path=request['output_path']+'.other'))
    target=api.submit(request,None)['job']
    failed=api.submit(dict(request,request_id='failed'),None)['job']
    service.jobs[-1]['status']='failed'
    executed=[]
    service._execute_job=lambda job,callback:executed.append(job['id'])
    service.start_automation_queue(api.notify)
    assert api.wait([target['id']],2)['done']
    owner=service._automation_kick_thread
    if owner:owner.join(3)
    assert executed==[target['id']]
    assert manual['status']=='pending' and service.jobs[-1]['status']=='failed'
    assert api.wait([target['id']],2)['done']


def test_wait_returns_terminal_or_timeout_without_restarting_anything(fixture):
    service,api,request=fixture
    job_id=api.submit(request,None)['job']['id']
    assert api.wait([job_id],0)['timed_out']
    ready=threading.Event();result=[]
    def wait():
        ready.set();result.append(api.wait([job_id],2))
    thread=threading.Thread(target=wait);thread.start();ready.wait(1)
    with service._queue_lock:
        job=service.jobs[0];job.update(status='failed',error='simulated encoder failure')
    api.notify(job);thread.join(3)
    assert result[0]['done'] and result[0]['jobs'][0]['error']=='simulated encoder failure'
    assert api.wait([job_id],2)['done']
    assert service._queue_thread is None


def test_cancelling_pending_job_does_not_cancel_another_active_job(fixture):
    service,api,request=fixture
    target=api.submit(request,None)['job']
    service.active_job_id='other-job'
    service.cancel_active_job=lambda:pytest.fail('cancelled a different running job')
    assert api.cancel(target['id'])['status']=='cancelled'
    assert service.active_job_id=='other-job'


def test_explicit_start_rearms_coordinator_waiting_behind_cancelled_owner(fixture):
    service,api,request=fixture
    owner_release=threading.Event();joining=threading.Event()
    class Owner:
        def is_alive(self):return not owner_release.is_set()
        def join(self):joining.set();assert owner_release.wait(3)
    service._queue_thread=Owner()
    target=api.submit(request,None)['job']
    executed=[]
    service._execute_job=lambda job,callback:executed.append(job['id'])
    service.start_automation_queue(api.notify)
    assert joining.wait(2)
    with service._queue_lock:service._automation_kick_cancelled=True
    service.start_automation_queue(api.notify)
    kicker=service._automation_kick_thread
    owner_release.set();kicker.join(3)
    assert executed==[target['id']]


def test_image_override_cannot_inherit_optional_stage_fields_from_ui(fixture,tmp_path):
    from types import SimpleNamespace
    service,api,_=fixture
    settings=api.settings()
    service.update_settings(dict(render_pipeline=['truehdr'],truehdr_enabled=True))
    seen=[]
    engine=SimpleNamespace(export_image=lambda path,options,*a:seen.append(options) or {'path':str(tmp_path/'out.png')},
                           release_memory=lambda:None)
    service._get_gpu_preview=lambda:engine
    source=tmp_path/'image.png';source.write_bytes(b'not decoded by fake engine')
    service.export_image(str(source),str(tmp_path/'out.png'),settings_override=settings)
    assert 'truehdr' not in seen[0].get('render_pipeline',[])
    assert not seen[0].get('truehdr_enabled')


def test_new_start_survives_old_cancelled_coordinator_exit_window(fixture):
    service,api,request=fixture
    joined=threading.Event();owner_release=threading.Event()
    paused=threading.Event();exit_release=threading.Event();finished=threading.Event()
    class Owner:
        def is_alive(self):return not owner_release.is_set()
        def join(self):joined.set();assert owner_release.wait(3)
    class ExitBarrier:
        def __init__(self):self.lock=threading.RLock();self.old_ident=None
        def __enter__(self):self.lock.acquire();return self
        def __exit__(self,*args):
            self.lock.release()
            if (threading.get_ident()==self.old_ident and service._automation_kick_cancelled and not paused.is_set()):
                paused.set();assert exit_release.wait(3)
    barrier=ExitBarrier();service._queue_lock=barrier
    service._queue_thread=Owner()
    target=api.submit(request,None)['job']
    service._execute_job=lambda job,callback:finished.set()
    service.start_automation_queue(api.notify)
    old=service._automation_kick_thread;barrier.old_ident=old.ident
    assert joined.wait(2)
    with service._queue_lock:service._automation_kick_cancelled=True
    owner_release.set();assert paused.wait(2)
    try:
        service.start_automation_queue(api.notify)
        assert finished.wait(2)
    finally:
        exit_release.set();old.join(3)
        current=service._automation_kick_thread
        if current:current.join(3)
    assert api.wait([target['id']],2)['done']


@pytest.mark.parametrize('patch',[{'typo_intensity':1},{'frame_generation':'vfg','frame_multiplier':17},{'intensity':float('nan')}])
def test_bad_parameters_are_refused_before_queue_or_gpu(fixture,patch):
    service,api,request=fixture
    with pytest.raises(ValueError):api.submit(dict(request,settings=patch),None)
    assert not service.jobs


def test_router_wait_and_preset_paths_share_one_store(fixture):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    service,api,_=fixture
    app=FastAPI();app.include_router(create_router(api,lambda _:None))
    with TestClient(app) as client:
        assert client.get('/api/presets').json()['presets']==[]
        assert client.post('/api/presets/save',json={'name':'Shared','settings':{'intensity':1.2}}).status_code==200
        assert api.settings('Shared')['intensity']==1.2
        assert client.post('/api/mcp',json={'enabled':True}).status_code==200
        assert client.post('/api/automation/jobs/wait',json={'job_ids':['missing'],'timeout':0}).status_code==404
        assert client.post('/api/automation/settings',json={'settings':{'unknown':1}}).status_code==400
