"""CPU-only tests for the ProRes yuv422p10le to persistent P210 upload."""
import ctypes
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import prores_decode as prores


class Plane:
    def __init__(self, values, stride_words=None):
        values = np.asarray(values, dtype="<u2")
        stride_words = stride_words or values.shape[1]
        padded = np.zeros((values.shape[0], stride_words), dtype="<u2")
        padded[:, :values.shape[1]] = values
        self.line_size = stride_words * 2
        self._bytes = padded.tobytes()

    def __bytes__(self):
        return self._bytes


def video_frame(pts, y, u, v):
    h, w = y.shape
    return SimpleNamespace(
        width=w, height=h, pts=pts, time_base=Fraction(1, 24),
        format=SimpleNamespace(name="yuv422p10le"),
        planes=[Plane(y, w + 2), Plane(u, w // 2 + 1), Plane(v, w // 2 + 1)],
        color_range=1, color_primaries=9, color_trc=16, colorspace=9)


class Packet:
    def __init__(self, frames):
        self.frames = frames

    def decode(self):
        return self.frames


class Container:
    def __init__(self, frames):
        self.frames = frames
        self.streams = [SimpleNamespace(type="video", time_base=Fraction(1, 24), start_time=0)]
        self.seek_calls = []
        self.closed = False

    def demux(self, stream):
        yield Packet(self.frames)

    def seek(self, *args, **kwargs):
        self.seek_calls.append((args, kwargs))

    def close(self):
        self.closed = True


class FakeUpload:
    instances = []

    def __init__(self, pixels):
        self.width, self.height = pixels.shape[1], pixels.shape[0]
        self.pointer = ctypes.c_uint64(0x100000)
        self.context = ctypes.c_void_p(0xCAFE)
        self.updates = []
        self.cuda = SimpleNamespace(cuMemcpyHtoD_v2=self._copy)

        self.closed = False
        FakeUpload.instances.append(self)

    def _copy(self, pointer, host_ptr, size):
        self.updates.append((pointer.value, ctypes.string_at(host_ptr, size)))
        return 0

    @contextmanager
    def current(self):
        yield

    @staticmethod
    def check(code):
        assert code == 0

    def close(self):
        self.closed = True


@pytest.fixture
def decoder_env(monkeypatch):
    frames = [video_frame(i, np.array([[0, 1, 1022, 1023], [4, 8, 12, 16]]),
                          np.array([[10, 20], [30, 40]]),
                          np.array([[50, 60], [70, 80]])) for i in (12, 13)]
    container = Container(frames)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    fake_render = SimpleNamespace(PixelUpload=FakeUpload)
    monkeypatch.setitem(sys.modules, "image_render", fake_render)
    FakeUpload.instances.clear()
    return container, frames


def test_is_prores_requires_codec_and_exact_planar_format():
    assert prores.is_prores({"codec_name": "prores", "pix_fmt": "yuv422p10le"})
    assert not prores.is_prores({"codec_name": "prores", "pix_fmt": "yuv422p10be"})
    assert not prores.is_prores({"codec_name": "h264", "pix_fmt": "yuv422p10le"})


def test_pack_p210_msb_alignment_and_y_uv_plane_layout():
    y = np.array([[0, 1, 1022, 1023], [4, 8, 12, 16]], dtype=np.uint16)
    u = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    v = np.array([[50, 60], [70, 80]], dtype=np.uint16)
    packed = prores.pack_p210(video_frame(0, y, u, v))
    words = packed.view("<u2").reshape(4, 4)
    np.testing.assert_array_equal(words[:2], y << 6)
    np.testing.assert_array_equal(words[2:].reshape(2, 2, 2),
                                  np.stack((u << 6, v << 6), axis=-1))


def test_decoder_reuses_device_allocation_and_preserves_frame_clock(decoder_env):
    container, _ = decoder_env
    dec = prores.ProResDecoder("clip.mov", {"codec_name": "prores", "pix_fmt": "yuv422p10le"})
    out = SimpleNamespace()
    assert dec.next(out) == 1
    ptr = out.y_device_ptr
    assert out.software_format == 3
    assert out.uv_device_ptr == ptr + 2 * out.width * out.height
    assert out.y_pitch_bytes == out.uv_pitch_bytes == 8
    assert (out.pts, out.time_base_num, out.time_base_den) == (12, 1, 24)
    assert (out.color_range, out.color_primaries, out.color_transfer, out.color_space) == (1, 9, 16, 9)
    assert dec.next(out) == 1 and out.y_device_ptr == ptr
    assert dec.counters() == {"frames_decoded": 2, "host_upload_bytes": 2 * 2 * 4 * 4,
                              "host_upload_count": 2}
    assert dec.next(out) == 0
    assert len(FakeUpload.instances) == 1
    assert len(FakeUpload.instances[0].updates) == 1
    assert len(FakeUpload.instances[0].updates[0][1]) == 2 * 4 * 4
    dec.close()
    assert FakeUpload.instances[0].closed and container.closed


def test_seek_us_and_cancel(decoder_env):
    container, _ = decoder_env
    dec = prores.ProResDecoder("clip.mov", {"codec_name": "prores", "pix_fmt": "yuv422p10le"})
    dec.seek_us(500_000)
    assert container.seek_calls[0][0][0] == 12
    dec.cancel()
    assert dec.next(SimpleNamespace()) == 0
    dec.close()


def test_wrong_format_and_missing_pts_are_rejected(decoder_env):
    dec = prores.ProResDecoder("clip.mov", {"codec_name": "prores", "pix_fmt": "yuv422p10le"})
    bad = video_frame(None, np.ones((2, 4), np.uint16),
                      np.ones((2, 2), np.uint16), np.ones((2, 2), np.uint16))
    bad.format = SimpleNamespace(name="yuv420p10le")
    dec._iterator = iter([bad])
    with pytest.raises(ValueError, match="expected yuv422p10le"):
        dec.next(SimpleNamespace())
    dec.close()

