from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import prores_export as prores


def test_plan_rejects_wrong_container_profile_and_depth():
    plan = prores.encoder_plan(dict(codec='prores_ks', container='.mkv',
                                   profile='main10', pixel_format='p010le'))
    assert len(plan['unsupported']) >= 3


def test_truehdr_tail_marks_software_output_as_hdr_and_reports_without_encoder():
    from truehdr_pipeline import Tail
    tail = Tail(None, {}, ordered=True)
    captured = []
    tail.process = lambda *args: dict(width=64, height=32, dxgi_format=10)
    sink = SimpleNamespace(encode=lambda color, frame, pts: captured.append(color) or (None, {}))
    chain = SimpleNamespace(prores_output=sink, output_size=(64, 32))
    tail.encode(chain, {}, SimpleNamespace(), 0)
    assert captured[0]['output_hdr'] is True
    assert 'ProRes 422 HQ' in tail.report()['output_contract']


def test_real_cpu_encoder_writes_hq_10bit_422_and_timestamps(tmp_path, monkeypatch):
    av = pytest.importorskip('av')
    import image_render
    pixels = np.full((32, 64, 4), 128, np.uint8)
    pixels[:, :, 3] = 255
    monkeypatch.setattr(image_render, 'download', lambda *args: (pixels, False))
    path = tmp_path / 'prores.mov'
    chain = SimpleNamespace(encoder_output=str(path), fps=(24000, 1001))
    sink = prores.ProResOutput(chain)
    frame = SimpleNamespace(width=64, height=32, cuda_context=0,
                            time_base_num=1001, time_base_den=24000)
    try:
        for pts in (40, 41, 42):
            frame.pts = pts
            sink.encode({'dxgi_format': 28}, frame)
        sink.finish()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.name == 'prores'
            assert stream.codec_context.profile == 'HQ'
            assert stream.codec_context.format.name == 'yuv422p10le'
            decoded = list(container.decode(stream))
            assert len(decoded) == 3
            times = [Fraction(f.pts) * f.time_base for f in decoded]
            assert times == [Fraction(i * 1001, 24000) for i in range(3)]
        assert sink.report()['frame_host_readback_bytes'] == pixels.nbytes * 3
        assert sink.report()['zero_copy'] is False
    finally:
        sink.close()


@pytest.mark.parametrize('hdr', [False, True])
def test_wide_codec_keeps_422_and_color_tags(tmp_path, monkeypatch, hdr):
    av = pytest.importorskip('av')
    path = tmp_path / ('hdr.mov' if hdr else 'sdr10.mov')
    sink = prores.ProResOutput(SimpleNamespace(encoder_output=str(path), fps=(24, 1)))
    def wide(*args):
        frame = av.VideoFrame(64, 32, 'yuv422p10le')
        for plane in frame.planes:
            prores.ProResOutput._put_plane(plane, np.full((plane.height, plane.width), 512, '<u2'))
        return frame
    monkeypatch.setattr(sink, '_wide_frame', wide)
    frame = SimpleNamespace(width=64, height=32, cuda_context=0, pts=0,
                            time_base_num=1, time_base_den=24)
    try:
        sink.encode(dict(dxgi_format=10, output_hdr=hdr), frame)
        sink.finish()
        with av.open(str(path)) as container:
            cc = container.streams.video[0].codec_context
            assert cc.format.name == 'yuv422p10le'
            assert cc.color_primaries == (9 if hdr else 1)
            assert cc.color_trc == (16 if hdr else 1)
            assert cc.colorspace == (9 if hdr else 1)
    finally:
        sink.close()
