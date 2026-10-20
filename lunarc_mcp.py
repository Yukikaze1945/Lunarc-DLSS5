"""Shared tools for the built-in HTTP MCP server and optional stdio adapter."""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any


class Backend:
    def __init__(self,port=18923,token_file=None,*,app=None,token=None):
        if not 1<=port<=65535:raise ValueError('Invalid localhost port')
        self.url=f'http://127.0.0.1:{port}'
        self.app=app
        self.token=token
        self.token_file=Path(token_file) if token_file else Path(
            os.environ.get('LUNARC_DATA_DIR') or
            Path(os.environ.get('APPDATA') or Path.home())/'DaVinci_DLSS5_Studio')/'.session_token'

    async def request(self,path,data=None,timeout=35):
        import httpx
        if self.app is not None:token=self.token
        else:
            try:token=self.token_file.read_text(encoding='utf-8').strip()
            except OSError:raise ValueError('Lunarc 尚未启动，或会话凭据不可读。请启动软件后再连接。') from None
        if not token:raise ValueError('Lunarc 会话凭据为空，请重新启动软件。')
        try:
            transport=httpx.ASGITransport(app=self.app) if self.app is not None else None
            async with httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=timeout,transport=transport) as client:
                response=await client.request('GET' if data is None else 'POST',self.url+path,
                    json=data,headers={'X-DLSS-Token':token})
            result=response.json()
        except (httpx.HTTPError,ValueError) as error:
            raise ValueError('无法连接 Lunarc 后端：'+type(error).__name__) from None
        if response.status_code==401:raise ValueError('Lunarc 会话已更新，请重试连接。')
        if response.status_code==404:raise ValueError('任务不存在，或后端尚未加载 MCP 更新。请确认任务 ID 和软件版本。')
        if not response.is_success:raise ValueError(str(result.get('detail','Lunarc 请求失败'))[:2000])
        return result

    async def wait(self,job_ids,timeout_seconds=600):
        if not 0<=timeout_seconds<=3600:raise ValueError('timeout_seconds 应为 0–3600')
        deadline=time.monotonic()+timeout_seconds
        while True:
            remaining=max(0,deadline-time.monotonic())
            result=await self.request('/api/automation/jobs/wait',
                dict(job_ids=job_ids,timeout=min(25,remaining)))
            if result['done'] or result.get('blocked'):return result
            if time.monotonic()>=deadline:return dict(result,timed_out=True)


def create_server(backend):
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
    from mcp.server.mcpserver.exceptions import ToolError
    from functools import wraps
    server=MCPServer('Lunarc',version='1.0.0',log_level='WARNING',instructions=(
        'Control the running local Lunarc workbench. Read presets/parameters before choosing settings. '
        'Render requests keep independent snapshots. Use render_video(wait=True) or wait_for_jobs '
        'to receive completion/errors without repeated model polling. A wait timeout or cancellation '
        'does not cancel rendering. Provide a stable request_id for each render; reuse it on retry. '
        'Only explicit cancel_job cancels work. Never retry failed GPU work automatically.'))
    def tool(*,annotations):
        def decorate(function):
            @wraps(function)
            async def guarded(*args,**kwargs):
                try:return await function(*args,**kwargs)
                except ValueError as error:raise ToolError(str(error)) from None
            return server.tool(structured_output=True,annotations=annotations)(guarded)
        return decorate
    read=ToolAnnotations(readOnlyHint=True,destructiveHint=False,openWorldHint=False)
    write=ToolAnnotations(readOnlyHint=False,destructiveHint=False,openWorldHint=False)

    @tool(annotations=read)
    async def get_status() -> dict[str,Any]:
        """Get backend health and compact queue state; does not start GPU work."""
        return await backend.request('/api/automation/status')

    @tool(annotations=read)
    async def probe_media(path:str) -> dict[str,Any]:
        """Inspect an absolute local media path without loading models or starting preview."""
        return await backend.request('/api/automation/probe',dict(path=path))

    @tool(annotations=read)
    async def list_presets() -> dict[str,Any]:
        """List names of the same saved presets shown in the workbench."""
        data=await backend.request('/api/automation/presets')
        return dict(presets=[p['name'] for p in data['presets']],revision=data['revision'])

    @tool(annotations=read)
    async def get_preset(name:str) -> dict[str,Any]:
        """Read one saved preset, including its complete settings."""
        data=await backend.request('/api/automation/presets')
        for preset in data['presets']:
            if preset['name']==name:return preset
        raise ValueError('未找到预设：'+name)

    @tool(annotations=read)
    async def get_parameters() -> dict[str,Any]:
        """Read processing defaults and optional setting names for custom parameter overrides."""
        return await backend.request('/api/automation/parameters')

    @tool(annotations=read)
    async def resolve_settings(settings:dict,preset:str|None=None) -> dict[str,Any]:
        """Validate overrides on a preset/defaults without changing workbench settings or rendering."""
        return await backend.request('/api/automation/settings',dict(preset=preset,settings=settings))

    @tool(annotations=write)
    async def save_preset(name:str,settings:dict,preset:str|None=None,overwrite:bool=False) -> dict[str,Any]:
        """Save validated settings for reuse in the UI and MCP; replacing a name needs overwrite=true."""
        resolved=await backend.request('/api/automation/settings',dict(preset=preset,settings=settings))
        result=await backend.request('/api/automation/presets/save',dict(name=name,settings=resolved['settings'],overwrite=overwrite))
        return dict(name=name,revision=result['revision'])

    @tool(annotations=write)
    async def render_video(input_path:str,output_path:str,request_id:str,preset:str|None=None,settings:dict|None=None,
                           encoding:dict|None=None,frame_range:dict|None=None,overwrite:bool=False,
                           wait:bool=True,timeout_seconds:float=600) -> dict[str,Any]:
        """Queue MKV video export, optionally wait once for completion/error. Same request_id is idempotent.

        settings overrides processing parameters. encoding accepts existing export settings such as
        codec (av1_nvenc/hevc_nvenc), preset, tune, rate_mode and cq. frame_range uses zero-based
        start_frame inclusive and end_frame exclusive. Work waits behind existing GPU jobs.
        """
        if not 0<=timeout_seconds<=3600:raise ValueError('timeout_seconds 应为 0–3600')
        if not request_id or len(request_id)>128:raise ValueError('request_id 长度应为 1–128')
        try:
            result=await backend.request('/api/automation/render',dict(input_path=input_path,output_path=output_path,
                preset=preset,settings=settings or {},encoding=encoding or {},frame_range=frame_range,
                overwrite=overwrite,request_id=request_id,start=True))
        except ValueError as error:
            raise ValueError(f'{error}；request_id={request_id}，重试时请保留此 ID。') from None
        result['request_id']=request_id
        if wait:
            try:result.update(await backend.wait([result['job']['id']],timeout_seconds))
            except ValueError as error:
                result.update(done=False,wait_error=str(error),recoverable=True)
        return result

    @tool(annotations=read)
    async def wait_for_jobs(job_ids:list[str],timeout_seconds:float=600) -> dict[str,Any]:
        """Wait for terminal jobs and return concise results; no repeated LLM polling needed."""
        return await backend.wait(job_ids,timeout_seconds)

    @tool(annotations=read)
    async def get_job(job_id:str|None=None,request_id:str|None=None) -> dict[str,Any]:
        """Find a render by job ID or request_id, including after reconnect or a lost submission response."""
        if job_id:return await backend.request('/api/automation/jobs/wait',dict(job_ids=[job_id],timeout=0))
        if not request_id:raise ValueError('请提供 job_id 或 request_id')
        return await backend.request('/api/automation/jobs/find',dict(request_id=request_id))

    @tool(annotations=ToolAnnotations(readOnlyHint=False,destructiveHint=True,openWorldHint=False))
    async def cancel_job(job_id:str) -> dict[str,Any]:
        """Explicitly cancel this pending/active job. Never cancels a different active job."""
        return await backend.request('/api/automation/jobs/cancel',dict(job_id=job_id))

    @tool(annotations=write)
    async def start_pending_renders() -> dict[str,Any]:
        """Resume pending MCP jobs only, without starting unrelated manual queue jobs."""
        return await backend.request('/api/automation/jobs/start',{})

    @tool(annotations=write)
    async def render_image(input_path:str,output_path:str,format:str='png',color_mode:str='sdr',
                           preset:str|None=None,settings:dict|None=None,quality:int=95,
                           preserve_alpha:bool=True,overwrite:bool=False) -> dict[str,Any]:
        """Render and save one image with a preset/overrides. Returns saved file or error; GPU-busy is explicit.

        format: png/jpeg/webp/avif/heic/heif/uhdr/jxl. color_mode: sdr/hdr10. No UI parameter changes.
        """
        return await backend.request('/api/automation/image',dict(input_path=input_path,output_path=output_path,
            format=format,color_mode=color_mode,preset=preset,settings=settings or {},quality=quality,
            preserve_alpha=preserve_alpha,overwrite=overwrite),timeout=3600)

    return server


def create_http_app(app,token):
    """Mount into the existing backend; dispatch tools through the same authenticated routes."""
    try:
        import httpx  # Both tool dispatch and the SDK must be available.
        server=create_server(Backend(app=app,token=token))
        return server.streamable_http_app(stateless_http=True,json_response=True)
    except ImportError as error:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        message=f'MCP 组件不可用：{error}。请更新软件基础依赖后重启。'
        async def unavailable(request):return JSONResponse({'detail':message},status_code=503)
        fallback=Starlette(routes=[Route('/mcp',unavailable,methods=['GET','POST','DELETE'])])
        fallback.state.mcp_error=message
        return fallback


def main():
    parser=argparse.ArgumentParser(description='Lunarc MCP stdio server')
    parser.add_argument('--port',type=int,default=18923)
    parser.add_argument('--token-file',type=Path)
    args=parser.parse_args()
    create_server(Backend(args.port,args.token_file)).run(transport='stdio')


if __name__=='__main__':main()
