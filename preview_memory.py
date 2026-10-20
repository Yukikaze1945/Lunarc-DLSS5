"""Preview-owner memory checks; budgets are not vendor peak predictions."""
import sys
GIB=1024**3

def cuda_memory():
    torch=sys.modules.get('torch')
    # Never create a fresh primary CUDA context just to measure a crowded GPU.
    if torch is None or not torch.cuda.is_initialized():return None
    from torch_cuda_context import torch_primary_context
    with torch_primary_context('cuda'):
        free,total=torch.cuda.mem_get_info()
    return dict(free_bytes=free,total_bytes=total,scope='Torch CUDA device; includes other applications')

def sample(live=None):
    report={}
    if live is not None:
        try:
            from gpu_memory import process_memory
            report['process']=process_memory(live.d3d12_device())
        except Exception as error:report['process_error']=str(error)
    try:report['cuda']=cuda_memory()
    except Exception as error:report['cuda_error']=str(error)
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        try:
            from torch_cuda_context import torch_primary_context
            with torch_primary_context('cuda'):
                report['torch']=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
        except Exception as error:report['torch_error']=str(error)
    return report

def pressured(report):
    process=report.get('process')or{}
    budget=process.get('local_budget',0);used=process.get('local_usage',0)
    if budget and budget-used<max(GIB//2,int(budget*.15)):return True
    cuda=report.get('cuda')or{}
    return bool(cuda.get('total_bytes') and cuda.get('free_bytes',GIB)<GIB)

def model_changed(previous,current):
    from layer_settings import resolve_layer_settings
    if any(resolve_layer_settings(previous, index).get('preset', 1) !=
           resolve_layer_settings(current, index).get('preset', 1) for index in range(3)):
        return True
    keys={'render_pipeline','render_stage_enabled','render_stage_options','dlaa_preset',
        'reconstruction_mode','working_scale','layer_scales','guidance_mode','dlss_dll_path'}
    return any(previous.get(key)!=current.get(key) and
        (key in keys or key.startswith(('vsr_','image_model_','flow_','raft_')))
        for key in previous.keys()|current.keys())

def image_workspace_floor(settings,size):
    from native_effects import stages,stage_kind,stage_settings,vsr_output_size
    required=0;w,h=size
    for stage in stages(settings):
        kind=stage_kind(stage);local=stage_settings(settings,stage)
        if kind=='vsr':w,h=vsr_output_size(local,(w,h))
        elif kind in ('illustration','animesharp'):
            scale=4 if kind=='illustration' else 2
            # Full FP32 RGB result + RGBA8 output + RGB8 quantization temporary,
            # input RGBA8 and normalized FP32 RGB. Excludes weights/activations.
            required=max(required,w*h*16+(w*scale)*(h*scale)*19)
            w,h=w*scale,h*scale
    return required

def check_cold_start(report,settings,size):
    cuda=report.get('cuda')or{}
    if not cuda.get('total_bytes'):return  # Report unknown; do not invent a budget.
    minimum=max(GIB//2,image_workspace_floor(settings,size))
    if cuda['free_bytes']<minimum:
        raise ValueError(f'预览已释放，但可用显存约 {cuda["free_bytes"]/GIB:.2f} GiB，低于启动检查所需的 {minimum/GIB:.2f} GiB。请减少渲染层数、手动选择分块或关闭其他占用显存的程序后再恢复预览。此检查不包含 SDK 内部峰值。')
