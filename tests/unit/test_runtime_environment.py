import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import runtime_environment as env


@pytest.fixture(autouse=True)
def isolated_runtime_selection(monkeypatch):
    # Unit checks must not inspect the developer's selected model/cache links.
    monkeypatch.setattr('dlssnr_runtime.selection_key', lambda: ('fixture-cache', None))


def test_bootstrap_explains_broken_native_import(monkeypatch):
    def load(name):
        if name == 'cv2':
            raise OSError('missing dependency DLL')
        return SimpleNamespace(__version__='test')
    monkeypatch.setattr(env.importlib, 'import_module', load)
    result = env.bootstrap()
    assert not result['ready']
    assert [x['id'] for x in result['checks'] if x['status'] != 'ok'] == ['cv2']
    assert 'missing dependency DLL' in result['checks'][1]['detail']


def test_bootstrap_refuses_environment_without_websocket_transport(monkeypatch):
    def load(name):
        if name == 'websockets':
            raise ModuleNotFoundError("No module named 'websockets'")
        return SimpleNamespace(__version__='test')
    monkeypatch.setattr(env.importlib, 'import_module', load)
    result = env.bootstrap()
    assert not result['ready']  # HTTP imports alone cannot play preview or connect queue.
    failed = [c for c in result['checks'] if c['status'] != 'ok']
    assert len(failed) == 1 and failed[0]['id'] == 'websockets'
    assert 'websockets' in failed[0]['detail']


@pytest.mark.parametrize('missing',['mcp','httpx'])
def test_missing_mcp_component_does_not_block_the_workbench(monkeypatch,missing):
    def load(name):
        if name==missing:raise ModuleNotFoundError(f"No module named '{name}'")
        return SimpleNamespace(__version__='test')
    monkeypatch.setattr(env.importlib,'import_module',load)
    result=env.bootstrap()
    assert result['ready']
    failed=[c for c in result['checks'] if c['status']!='ok']
    assert len(failed)==1 and failed[0]['id']==missing and failed[0]['required'] is False


def test_system_never_invents_gpu_when_driver_unavailable(monkeypatch):
    def fail(): raise OSError("driver unavailable")
    monkeypatch.setattr(env, "driver_device", fail)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    info = env.quick_system_info()
    assert info['gpu'] is None
    assert info['vram_total_gb'] is None
    assert info['vram_used_gb'] is None
    assert info['gpu_error']


def test_system_does_not_double_count_allocated_vram(monkeypatch):
    monkeypatch.setattr(env, "driver_device", lambda: {"name":"real GPU","bytes":16*1024**3})
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: 'real GPU',
        get_device_properties=lambda n: SimpleNamespace(total_memory=16 * 1024**3),
        memory_reserved=lambda n: 2 * 1024**3)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    info = env.quick_system_info()
    assert info['gpu'] == 'real GPU'
    assert info['vram_used_gb'] == 2
    assert info['vram_scope'] == 'process_reserved'


def test_failed_worker_is_an_error_not_a_pass_and_cached(monkeypatch):
    monkeypatch.setattr(env, '_cache', None)
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], 60)
    monkeypatch.setattr(env.subprocess, 'run', run)
    first = env.environment_report()
    assert first['checks'][0]['status'] == 'error'
    assert first['features'] == []
    assert env.environment_report() is first
    assert len(calls) == 1
    env.environment_report(refresh=True)
    assert len(calls) == 2


def test_successful_worker_handles_native_stdout_and_utf8(monkeypatch):
    monkeypatch.setattr(env, '_cache', None)
    data = {'checks': [{'label': '运行环境', 'status': 'ok'}], 'features': []}
    monkeypatch.setattr(env.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='native diagnostic\nLUNARC_ENV=' + json.dumps(data) + '\n', stderr=''))
    assert env.environment_report() == data


def test_custom_sdk_change_invalidates_cache_and_is_passed_to_worker(monkeypatch):
    monkeypatch.setattr(env, '_cache', None)
    paths = []
    def run(*args, **kwargs):
        paths.append(kwargs['env'].get('LUNARC_VSR_RUNTIME'))
        return SimpleNamespace(returncode=0, stdout='LUNARC_ENV={"checks": [], "features": []}', stderr='')
    monkeypatch.setattr(env.subprocess, 'run', run)
    env.environment_report(overrides={'LUNARC_VSR_RUNTIME': 'custom-one'})
    env.environment_report(overrides={'LUNARC_VSR_RUNTIME': 'custom-two'})
    assert paths == ['custom-one', 'custom-two']


def test_standalone_ffmpeg_never_borrows_path(monkeypatch, tmp_path):
    import pipeline
    monkeypatch.setenv('LUNARC_STANDALONE', '1')
    monkeypatch.setenv('LUNARC_RUNTIME_DIR', str(tmp_path))
    monkeypatch.setattr(pipeline.shutil, 'which', lambda *a: pytest.fail('borrowed PATH'))
    assert pipeline._bundle_ffmpeg() is None
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/ffmpeg.exe').write_bytes(b'test')
    assert pipeline._bundle_ffmpeg() == str(tmp_path / 'tools/ffmpeg.exe')


def test_builder_does_not_confuse_private_dll_with_bootstrap_dependency(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'packaging/build_runtime.py'
    spec = importlib.util.spec_from_file_location('build_runtime', path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out = tmp_path / 'out'
    private = out / 'python/Lib/site-packages/torch/lib'
    private.mkdir(parents=True)
    (private / 'zlib.dll').write_bytes(b'private')
    (out / 'python/python.exe').write_bytes(b'python')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'zlib.dll').write_bytes(b'bootstrap')
    native = tmp_path / 'native'
    native.mkdir()
    monkeypatch.setattr(builder, 'imports', lambda p: ['zlib.dll'] if p.name == 'python.exe' else [])
    builder.collect_dependencies(out, [source], native)
    assert (out / 'python/zlib.dll').read_bytes() == b'bootstrap'


def test_native_gpu_information_does_not_require_or_import_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    original=__import__
    def guarded(name,*args,**kwargs):
        if name=='torch':pytest.fail('Native GPU detection imported optional torch')
        return original(name,*args,**kwargs)
    monkeypatch.setattr('builtins.__import__',guarded)
    monkeypatch.setattr(env,'driver_device',lambda:{'name':'driver GPU','bytes':16*1024**3})
    info=env.quick_system_info()
    assert info['gpu']=='driver GPU'
    assert info['vram_total_gb']==16
    assert info['vram_used_gb'] is None
    assert info['vram_scope']=='unavailable'
