import ctypes as C

import cv2
import numpy as np
import pytest
from PIL import Image

import image_source as images
import native_effects
from gpu_transfer_export import Chain, _DeviceFrame


@pytest.fixture
def source(tmp_path):
    path = tmp_path/'中文图片.png'
    rgba = np.full((13,17,4),100,np.uint8)
    rgba[:,:,3] = 255
    rgba[0,0,3] = 0
    Image.fromarray(rgba).save(path)
    return str(path), rgba


def test_image_decodes_as_rgb_with_one_frame_and_padding_report(source):
    path, rgba = source
    assert np.array_equal(images.read_rgba(path),rgba)
    probe = images.probe(path)
    assert (probe['width'],probe['height']) == (24,16)
    assert (probe['image_width'],probe['image_height']) == (17,13)
    assert probe['media_type'] == 'image' and probe['pix_fmt'] == 'rgba'
    capture = images.Capture(path)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 1
    assert capture.read()[0]
    assert not capture.read()[0]
    capture.set(cv2.CAP_PROP_POS_FRAMES,0)
    assert capture.read()[0]
    capture.release()
    assert not capture.isOpened()


def test_high_depth_image_is_not_silently_reduced(tmp_path):
    path=tmp_path/'16bit.png'
    path.write_bytes(cv2.imencode('.png',np.ones((8,8,3),np.uint16)*40000)[1].tobytes())
    with pytest.raises(ValueError,match='16-bit'):
        images.read_rgba(str(path))


@pytest.mark.parametrize('pipeline',[['nr','dlaa'],['vsr'],['dlaa']])
def test_native_stages_accept_explicit_image_rgba(source,pipeline):
    native_effects.validate(dict(render_pipeline=pipeline,export_backend='gpu_transfer',
        guidance_mode=0,working_scale=1,reconstruction_mode='single',vsr_output_mode='scale',
        vsr_scale=4),images.probe(source[0]))
    with pytest.raises(ValueError,match='YUV source'):
        native_effects.validate(dict(render_pipeline=pipeline,export_backend='gpu_transfer'),
            dict(images.probe(source[0]),media_type='video'))


def test_chain_uses_image_owner_without_touching_nvdec(source,monkeypatch):
    class Owner:
        def __init__(self,path): self.path=path
        def frame(self,frame): frame.image_rgba=(123,96);return 1
    monkeypatch.setattr(images,'DeviceImage',Owner)
    chain=Chain.__new__(Chain)
    chain.video=source[0];chain.b={};chain.image_decoder=None
    chain.open_decoder()
    frame=_DeviceFrame(C.sizeof(_DeviceFrame))
    assert chain.next_frame(frame)==1 and frame.image_rgba==(123,96)


@pytest.mark.parametrize('fmt,ext',[('png','png'),('jpeg','jpg'),('webp','webp')])
def test_export_actual_image_formats_and_alpha(source,tmp_path,fmt,ext):
    path,rgba=source
    output=str(tmp_path/('导出.'+ext))
    result=images.save_image(path,rgba[:,:,:3],output,fmt,quality=100)
    with Image.open(output) as image:
        assert image.size==(17,13)
        assert image.info.get('icc_profile')
        actual=np.array(image.convert('RGBA'))
    if fmt!='jpeg':
        assert np.array_equal(actual,rgba)
    else:
        assert actual[0,0,:3].min()>245 and np.all(actual[:,:,3]==255)
    assert result['bytes']>0


def test_export_heif_when_encoder_is_available(source,tmp_path):
    if not images.export_formats()['heif']:
        pytest.skip('HEIF encoder is not installed')
    path,rgba=source
    output=str(tmp_path/'导出.heif')
    result=images.save_image(path,rgba[:,:,:3],output,'heif',quality=100)
    exported=images.heif_plugin().open_heif(output,convert_hdr_to_8bit=False)
    assert exported.size==(17,13)
    assert exported.info.get('bit_depth',8)==8
    assert result['format']=='heif' and result['path']==output


def test_heif_format_requires_matching_extension(source,tmp_path,monkeypatch):
    monkeypatch.setattr(images,'heif_plugin',lambda:object())
    with pytest.raises(ValueError,match='扩展名'):
        images.validate_destination(source[0],str(tmp_path/'wrong.heic'),'heif',95)


def test_export_never_overwrites_original_and_preserves_existing_target(source,tmp_path):
    path,rgba=source
    original=open(path,'rb').read()
    with pytest.raises(ValueError,match='原图片'):
        images.save_image(path,rgba[:,:,:3],path,'png',overwrite=True)
    assert open(path,'rb').read()==original
    target=tmp_path/'existing.png';target.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        images.save_image(path,rgba[:,:,:3],str(target),'png')
    assert target.read_bytes()==b'existing'
    assert not list(tmp_path.glob('.lunarc-image-*'))


def test_padding_is_trimmed_without_resizing_content(source):
    probe=images.probe(source[0])
    padded=np.zeros((16,24,3),np.uint8)
    assert images.output_pixels(padded,probe,{'render_pipeline':['nr']}).shape==(13,17,3)
