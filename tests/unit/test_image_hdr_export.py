"""HDR still regressions. Not executed while the user's GPU job is running."""
import numpy as np
import pytest

import image_hdr_export as hdr


def test_scrgb_white_keeps_absolute_luminance_without_normalizing_image_peak():
    values = np.array([[[0, 0, 0], [1, 1, 1], [12.5, 12.5, 12.5], [125, 125, 125]]], np.float16)
    encoded, report = hdr.scrgb_to_pq16(values)
    # ST2084: 80 nits ~= .485857, 1000 nits ~= .751827, 10000 nits = 1.
    assert encoded.dtype == np.dtype('<u2')
    assert np.allclose(encoded[0, 1] / 65535, .4858568, atol=2/65535)
    assert np.allclose(encoded[0, 2] / 65535, .7518271, atol=2/65535)
    assert np.all(encoded[0, 3] == 65535)
    assert np.all(encoded[0, 0] <= 1)
    assert report['tone_mapping'] is False
    assert report['scrgb_nits_per_unit'] == 80


def test_gamut_conversion_precedes_clipping():
    # Extended Rec709 contains valid Rec2020 colours with a negative Rec709 R.
    value = np.array([[[-.1, 1, .2]]], np.float32)
    encoded, report = hdr.scrgb_to_pq16(value)
    assert (value @ hdr.REC709_TO_REC2020.T).min() > 0
    assert report['negative_bt2020_components_clipped'] == 0
    assert np.all(encoded > 0)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_hdr_is_rejected(invalid):
    with pytest.raises(ValueError, match='非有限'):
        hdr.scrgb_to_pq16(np.full((2, 2, 3), invalid, np.float32))


def test_pq_limits_are_counted():
    _, report = hdr.scrgb_to_pq16(np.array([[[-1, -1, -1], [126, 126, 126]]], np.float32))
    assert report['negative_bt2020_components_clipped'] == 3
    assert report['above_10000_nits_components_clipped'] == 3


def test_avif_command_sets_primary_hdr_and_separate_linear_alpha():
    args = hdr.avif_command('ffmpeg.exe', 'c.yuv', 'a.yuv', (17, 13), 'result.avif', 95)
    def option(key):
        return args[args.index(key)+1]
    assert option('-c:v') == 'libaom-av1'
    assert option('-pix_fmt:v:0') == 'yuv444p10le'
    assert option('-profile:v:0') == '1'  # AV1 High profile for 4:4:4, 10-bit.
    assert option('-color_primaries:v:0') == 'bt2020'
    assert option('-color_trc:v:0') == 'smpte2084'
    assert option('-colorspace:v:0') == 'bt2020nc'
    assert option('-color_range:v:0') == 'pc'
    assert option('-filter:v:0') == 'setparams=range=full:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc'
    assert '-filter:v:1' not in args  # Alpha coverage is independent of PQ color.
    assert option('-pix_fmt:v:1') == 'gray10le'
    assert option('-crf:v:1') == '0'
    assert '-color_trc:v:1' not in args  # Alpha is coverage, never PQ luminance.
    assert option('-still-picture') == '1'
    assert not any('nvenc' in arg or 'cuda' in arg or '8le' in arg for arg in args)


def test_missing_hdr_tags_still_rejected_with_only_actionable_fields(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import pipeline
    monkeypatch.setattr(pipeline,'_bundle_ffmpeg',lambda:'ffmpeg')
    monkeypatch.setattr(pipeline,'_bundle_ffprobe',lambda:'ffprobe')
    actual=dict(codec_name='av1',pix_fmt='yuv444p10le',color_space='bt2020nc',
        color_range='pc',width=17,height=13,codec_long_name='irrelevant verbose details')
    def run(args,**kwargs):
        return SimpleNamespace(returncode=0,stderr=b'',stdout=json.dumps({'streams':[actual]}).encode())
    monkeypatch.setattr(hdr,'_run',run)
    with pytest.raises(ValueError) as error:
        hdr._save_avif(str(tmp_path/'result.avif'),np.zeros((13,17,3),'<u2'),None,95)
    message=str(error.value)
    assert '原色应为 bt2020' in message and '传递曲线应为 smpte2084' in message
    assert '未声明' in message and 'codec_long_name' not in message


def test_hdr_requires_enabled_truehdr_before_encoder_queries(monkeypatch):
    monkeypatch.setattr(hdr, 'capabilities', lambda: pytest.fail('encoder query should not run'))
    with pytest.raises(ValueError, match='开启 TrueHDR'):
        hdr.validate_request(dict(render_pipeline=['truehdr'],
            render_stage_enabled={'truehdr': False}), 'avif', 'unused.png', True)


def test_hdr_heif_alias_routes_through_heic_encoder(tmp_path,monkeypatch):
    source=tmp_path/'source.png';source.write_bytes(b'input')
    output=tmp_path/'result.heif'
    monkeypatch.setattr('image_source.validate_destination',lambda *args,**kwargs:None)
    calls=[]
    def save_heic(path,pq,alpha,quality):
        calls.append((pq.shape,alpha,quality))
        open(path,'wb').write(b'encoded')
    monkeypatch.setattr(hdr,'_save_heic',save_heic)
    result=hdr.save_hdr_image(str(source),np.ones((2,3,3),np.float32),str(output),
        'heif',quality=91,preserve_alpha=False)
    assert output.read_bytes()==b'encoded'
    assert calls==[((2,3,3),None,91)]
    assert result['format']=='heif' and result['color_mode']=='hdr10'


def test_fp16_image_export_never_populates_sdr_display_cache(monkeypatch):
    from gpu_preview_engine import GpuPreview
    import image_render
    engine = GpuPreview.__new__(GpuPreview)
    engine.cached_display=engine.result_bgr=None
    pixels = np.full((13,17,4),12.5,np.float16)
    original=np.full((13,17,4),255,np.uint8)
    cached=image_render.ImageResult('result',('source.png',1,2),'settings',original,pixels,True,{},0)
    engine.image_results=image_render.ImageResults();engine.image_results.put(cached)
    def render(*args):return cached.info()
    def save(result,*args):
        assert result.pixels.dtype == np.float16 and np.all(result.pixels == 12.5)
        return dict(path='result.avif', color_mode='hdr10')
    monkeypatch.setattr(engine, '_render_image', render)
    monkeypatch.setattr(hdr, 'validate_request', lambda *args: None)
    monkeypatch.setattr(image_render, 'save_result', save)
    result = engine._export_image('source.png', {}, 'result.avif', 'avif', 95, True, False, 'hdr10')
    assert result['color_mode'] == 'hdr10'
    assert engine.cached_display is None and engine.result_bgr is None
