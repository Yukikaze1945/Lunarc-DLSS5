"""Authenticated automation over the existing service; no independent GPU owner."""
import copy
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from preset_store import PresetStore, _validate_json
from mcp_control import MCPControl

TERMINAL = {'completed','failed','cancelled','skipped'}
EXTRA_SETTINGS = {
    'render_pipeline','render_stage_enabled','render_stage_options','frame_generation_saved_mode',
    'dlaa_preset','truehdr_enabled','truehdr_contrast','truehdr_saturation','truehdr_middle_gray',
    'truehdr_peak_nits','truehdr_debanding','layer_settings','layer_execution','raft_analysis_edge',
    'raft_iterations','image_model_precision','image_model_tile','image_model_variant','image_preview_realtime',
}


def merge(base, patch):
    result=copy.deepcopy(base)
    for key,value in patch.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else copy.deepcopy(value)
    return result


def job_summary(job):
    return {key:copy.deepcopy(job.get(key)) for key in (
        'id','name','status','input_path','output_path','progress','current_frame','total_frames',
        'error','skipped_reason','interrupted','automation_request_id') if job.get(key) is not None}


class Automation:
    def __init__(self,service,presets=None):
        self.service=service
        self.presets=presets or PresetStore()
        self.control=MCPControl(self.presets.path.with_name('mcp.json'))
        self.defaults=copy.deepcopy(service.settings)
        self.defaults['export_backend']='gpu_transfer'
        self.condition=threading.Condition()
        self.revision=0
        self.blocked=None

    def notify(self,data):
        if not isinstance(data,dict):return
        if data.get('status') in TERMINAL or data.get('type') in ('queue_finished','job_removed','automation_blocked'):
            with self.condition:
                if data.get('type')=='automation_blocked':self.blocked=data.get('error')
                elif data.get('type')=='queue_finished':self.blocked=None
                self.revision+=1
                self.condition.notify_all()

    def settings(self,preset=None,patch=None):
        patch={} if patch is None else patch
        _validate_json(patch)
        if not isinstance(patch,dict):raise ValueError('settings 必须是对象')
        selected={}
        if preset:
            found=next((p for p in self.presets.snapshot()['presets'] if p['name']==preset),None)
            if found is None:raise ValueError(f'未找到预设：{preset}')
            selected=found['settings']
        unknown=set(patch)-set(self.defaults)-EXTRA_SETTINGS
        if unknown:raise ValueError('未知参数：'+', '.join(sorted(unknown)))
        from native_effects import execution_settings
        from vfx_framegen import config
        result=self.service.normalize_settings(merge(selected,patch),base=self.defaults)
        config(execution_settings(result))
        return result

    def submit(self,data,callback):
        from image_source import is_image
        source=Path(data['input_path']).expanduser()
        target=Path(data['output_path']).expanduser()
        if not source.is_absolute() or not source.is_file():raise ValueError('请输入存在的素材绝对路径')
        if is_image(str(source)):raise ValueError('图片请使用 render_image 工具')
        if not target.is_absolute() or target.suffix.lower()!='.mkv':raise ValueError('自动化视频导出需要 MKV 绝对路径')
        if not target.parent.is_dir():raise ValueError('输出目录不存在')
        settings=self.settings(data.get('preset'),data.get('settings'))
        encoding=copy.deepcopy(data.get('encoding') or {})
        _validate_json(encoding)
        if not isinstance(encoding,dict):raise ValueError('encoding 必须是对象')
        if encoding.get('extra_args'):raise ValueError('MCP 请使用结构化编码参数，不接受 extra_args')
        if encoding.get('finish_action','none')!='none':raise ValueError('MCP 渲染不执行睡眠或关机动作')
        encoding.update(finish_action='none',overwrite_output=data.get('overwrite',False),
                        skip_existing=not data.get('overwrite',False))
        codec=encoding.get('codec','av1_nvenc')
        if codec not in ('hevc_nvenc','av1_nvenc'):raise ValueError('自动化视频编码请选择 hevc_nvenc 或 av1_nvenc')
        encoding.setdefault('codec',codec)
        if encoding.get('container','mkv')!='mkv':raise ValueError('自动化视频导出需要 MKV')
        encoding['container']='mkv'
        request_id=data.get('request_id') or str(uuid.uuid4())
        if not isinstance(request_id,str) or not 1<=len(request_id)<=128:raise ValueError('request_id 长度应为 1–128')
        payload=dict(input_path=str(source.resolve()),output_path=str(target.resolve()),
            name=target.stem,settings=settings,codec=codec,container='mkv',transcode_settings=encoding,
            frame_range=data.get('frame_range'))
        fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
        with self.service._queue_lock:
            previous=next((j for j in self.service.jobs if j.get('automation_request_id')==request_id),None)
            if previous:
                if previous.get('automation_fingerprint')!=fingerprint:raise ValueError('request_id 已用于不同的渲染参数')
                return dict(job=job_summary(previous),reused=True)
            self.service._ensure_native_healthy()
            self.blocked=None
            payload.update(automation_job=True,automation_request_id=request_id,automation_fingerprint=fingerprint)
            job=self.service.add_job(copy.deepcopy(payload))
        if data.get('start',True):
            self.service.start_automation_queue(callback)
        return dict(job=job_summary(job),reused=False)

    def state(self,ids):
        if not isinstance(ids,list) or not 1<=len(ids)<=100 or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError('job_ids 需要包含 1–100 个任务 ID')
        jobs={j['id']:j for j in self.service.get_jobs()}
        missing=[i for i in ids if i not in jobs]
        if missing:raise KeyError('任务不存在或已删除：'+', '.join(missing))
        summaries=[job_summary(jobs[i]) for i in ids]
        blocked=self.blocked or getattr(self.service,'_native_degraded',None) or getattr(self.service,'_queue_storage_error',None)
        return dict(done=all(j['status'] in TERMINAL for j in summaries),jobs=summaries,blocked=blocked or None)

    def wait(self,ids,timeout=25):
        if not isinstance(timeout,(int,float)) or not 0<=timeout<=30:raise ValueError('单次等待需为 0–30 秒')
        deadline=time.monotonic()+timeout
        while True:
            with self.condition:revision=self.revision
            state=self.state(ids)
            if state['done'] or state['blocked']:return state
            remaining=deadline-time.monotonic()
            if remaining<=0:return dict(state,timed_out=True)
            with self.condition:
                if self.revision==revision:self.condition.wait(remaining)

    def cancel(self,job_id):
        with self.service._queue_lock:
            job=next((j for j in self.service.jobs if j['id']==job_id),None)
            if job is None:raise KeyError('任务不存在')
            if job['status']=='running':
                if self.service.active_job_id!=job_id:raise ValueError('任务状态已经变化')
                self.service.cancel_active_job()
            elif job['status']=='pending':
                job['status']='cancelled'
                if not self.service._persist_queue():
                    job['status']='pending'
                    raise RuntimeError('无法保存取消状态')
            self.notify(job)
            return job_summary(job)


def create_router(automation,callback):
    from fastapi import APIRouter,HTTPException,Depends,Request
    router=APIRouter()
    def require_enabled():
        if not automation.control.enabled:
            raise HTTPException(403,'MCP 访问已关闭，请在软件设置中开启「允许 AI 智能体控制 Lunarc」。')
    agent=APIRouter(dependencies=[Depends(require_enabled)])
    def call(function,*args,**kwargs):
        try:return function(*args,**kwargs)
        except KeyError as error:raise HTTPException(404,str(error)) from error
        except (ValueError,TypeError) as error:raise HTTPException(400,str(error)) from error
        except (RuntimeError,OSError) as error:raise HTTPException(409,str(error)) from error

    @router.get('/api/presets')
    def presets():return call(automation.presets.snapshot)
    @router.post('/api/presets/save')
    def save(data:dict):return call(automation.presets.save,data.get('name'),data.get('settings'),data.get('overwrite',False))
    @router.post('/api/presets/delete')
    def delete(data:dict):return call(automation.presets.delete,data.get('name'))
    @router.post('/api/presets/import')
    def migrate(data:dict):return call(automation.presets.import_missing,data.get('presets'))
    @router.get('/api/mcp')
    def connection(request:Request):return automation.control.info(request.url.port or 18923)
    @router.post('/api/mcp')
    def permission(data:dict,request:Request):
        call(automation.control.set_enabled,data.get('enabled'))
        return automation.control.info(request.url.port or 18923)
    @agent.get('/api/automation/status')
    def status():
        jobs=automation.service.get_jobs()
        return dict(active_id=automation.service.active_job_id,
            queue_running=bool(getattr(automation.service,'_queue_thread',None) and automation.service._queue_thread.is_alive()),
            blocked=automation.blocked or getattr(automation.service,'_native_degraded',None),
            jobs=[job_summary(job) for job in jobs[-100:]])
    agent.add_api_route('/api/automation/presets',presets,methods=['GET'])
    agent.add_api_route('/api/automation/presets/save',save,methods=['POST'])
    @agent.post('/api/automation/settings')
    def settings(data:dict):return dict(settings=call(automation.settings,data.get('preset'),data.get('settings')))
    @agent.get('/api/automation/parameters')
    def parameters():return dict(defaults=automation.defaults,optional_parameters=sorted(EXTRA_SETTINGS))
    @agent.post('/api/automation/probe')
    def probe(data:dict):
        import pipeline
        path=Path(data.get('path','')).expanduser()
        if not path.is_absolute() or not path.is_file():raise HTTPException(400,'请输入存在的素材绝对路径')
        return call(pipeline.probe_pixel_info,str(path))
    @agent.post('/api/automation/render')
    async def render(data:dict):
        import asyncio
        result=await asyncio.to_thread(call,automation.submit,data,callback)
        callback({'type':'job_added','job':result['job']})
        return result
    @agent.post('/api/automation/jobs/wait')
    def wait(data:dict):return call(automation.wait,data.get('job_ids'),data.get('timeout',25))
    @agent.post('/api/automation/jobs/cancel')
    def cancel(data:dict):
        result=call(automation.cancel,data.get('job_id'))
        callback({'type':'job_update','job':result})
        return result
    @agent.post('/api/automation/jobs/find')
    def find(data:dict):
        request_id=data.get('request_id')
        if not isinstance(request_id,str) or not request_id:raise HTTPException(400,'request_id 必填')
        job=next((j for j in automation.service.get_jobs() if j.get('automation_request_id')==request_id),None)
        if job is None:raise HTTPException(404,'未找到此 request_id')
        return job_summary(job)
    @agent.post('/api/automation/jobs/start')
    def start():
        call(automation.service.start_automation_queue,callback)
        return dict(status='scheduled')
    @agent.post('/api/automation/image')
    def image(data:dict):
        for field in ('input_path','output_path'):
            if not isinstance(data.get(field),str) or not Path(data[field]).is_absolute():
                raise HTTPException(400,f'{field} 需要绝对路径')
        settings=call(automation.settings,data.get('preset'),data.get('settings'))
        result=call(automation.service.export_image,data.get('input_path'),data.get('output_path'),
            fmt=data.get('format','png'),quality=data.get('quality',95),
            preserve_alpha=data.get('preserve_alpha',True),overwrite=data.get('overwrite',False),
            color_mode=data.get('color_mode','sdr'),settings_override=settings)
        return {key:result[key] for key in ('path','bytes','width','height','color_mode','rendered_result_id') if key in result}
    router.include_router(agent)
    return router
