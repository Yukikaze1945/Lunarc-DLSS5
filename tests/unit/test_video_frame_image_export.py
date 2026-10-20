"""CPU-only contracts for full-resolution video stills and cached image exports."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import image_render as images
from dlss_service import DLSSService
from gpu_preview_engine import GpuPreview


def result(path, frame, settings, hdr=False):
    pixels = np.full((4,8,4),1 if hdr else 255,np.float16 if hdr else np.uint8)
    return images.ImageResult(str(frame),images.source_key(str(path)),
        images.signature(settings,True),pixels.copy(),pixels,hdr,{},0,frame)


def test_cache_selects_video_frame_and_tracks_temporal_changes(tmp_path):
    path=tmp_path/'source.mkv';path.write_bytes(b'source')
    settings=dict(render_pipeline=['vsr'],guidance_mode=1,temporal_controls={'mode':'off'})
    cache=images.ImageResults()
    cache.put(result(path,10,settings));cache.put(result(path,20,settings))
    assert cache.status(str(path),settings,10)['result']['frame_index']==10
    assert not cache.status(str(path),settings,11)['available']
    assert cache.status(str(path),settings,20)['current']
    assert not cache.status(str(path),dict(settings,guidance_mode=0),20)['current']
    assert not cache.status(str(path),dict(settings,temporal_controls={'mode':'static_low_frequency'}),20)['current']


def test_video_cached_export_needs_no_gpu_and_refuses_wrong_frame(tmp_path,monkeypatch):
    path=tmp_path/'source.mkv';path.write_bytes(b'source')
    settings=dict(render_pipeline=['vsr'])
    cache=images.ImageResults();cache.put(result(path,12,settings))
    service=DLSSService.__new__(DLSSService)
    service.settings=settings;service._settings_lock=threading.RLock()
    service._gpu_preview=SimpleNamespace(image_results=cache)
    service._ensure_native_healthy=lambda:pytest.fail('cached export opened GPU')
    monkeypatch.setattr(__import__('image_source'),'validate_destination',lambda *a,**k:None)
    monkeypatch.setattr(images,'save_result',lambda *a,**k:dict(path='saved.png'))
    saved=service.export_image(str(path),'saved.png',render_source='preview',result_id='12',frame_index=12)
    assert saved['reused_render']
    with pytest.raises(ValueError,match='缓存帧'):
        service.export_image(str(path),'saved.png',render_source='preview',result_id='12',frame_index=13)
    with pytest.raises(ValueError,match='指定单帧'):
        service.export_image(str(path),'saved.png',render_source='preview',result_id='12')


@pytest.mark.parametrize('hdr',[False,True])
def test_single_frame_caches_full_plane_and_releases_models(tmp_path,hdr):
    path=tmp_path/'source.mkv';path.write_bytes(b'source')
    engine=GpuPreview.__new__(GpuPreview);engine.image_results=images.ImageResults()
    events=[]
    def render(path,settings,index,**options):
        assert index==7 and options['reuse_display'] is False
        assert settings['render_pipeline']==['vsr'] and settings['frame_generation']=='off'
        pixels=np.full((4,8,4),1,np.float16 if hdr else np.uint8)
        if hdr:pixels[0,0,:3]=[-.25,12.5,2]
        engine._captured_image_master=pixels.copy(),hdr
        engine._captured_frame_source=pixels.copy(),hdr
        return None,None,{'frame':7,'processing_ms':1}
    engine._render=render
    engine._release_memory=lambda *a,**k:events.append('released')
    info=engine._render_video_frame(str(path),{'render_pipeline':['vsr','vfg'],'frame_generation':'vfg'},7)
    cached=engine.image_results.get(info['id'])
    assert info['width']==8 and info['height']==4 and info['frame_index']==7
    assert events==['released'] and not engine._capture_image_master
    if hdr:np.testing.assert_array_equal(cached.pixels[0,0,:3],[-.25,12.5,2])


def test_failed_single_frame_releases_without_caching(tmp_path):
    path=tmp_path/'source.mkv';path.write_bytes(b'source')
    engine=GpuPreview.__new__(GpuPreview);engine.image_results=images.ImageResults()
    engine._render=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('failed frame'))
    events=[];engine._release_memory=lambda *a,**k:events.append('released')
    with pytest.raises(RuntimeError,match='failed frame'):
        engine._render_video_frame(str(path),{'render_pipeline':['vsr']},0)
    assert events==['released'] and not engine.image_results.entries
    assert not engine._capture_frame_source and engine._captured_image_master is None


def test_sdr_linear_frame_converts_once_to_srgb():
    pixels=np.full((2,2,4),.5,np.float16);pixels[:,:,3]=1
    converted=images.sdr_pixels(pixels)
    assert converted.dtype==np.uint8 and converted[0,0,0]==188
    assert converted[0,0,3]==255
    assert images.sdr_pixels(converted) is converted


@pytest.mark.parametrize('frame',[None,-1,10,True])
def test_render_rejects_missing_or_out_of_bounds_video_frame(frame):
    service=DLSSService.__new__(DLSSService)
    service.video_path='source.mkv';service.video_info={'frames':10}
    with pytest.raises(ValueError,match='视频范围'):
        service.render_image('source.mkv',frame_index=frame)
