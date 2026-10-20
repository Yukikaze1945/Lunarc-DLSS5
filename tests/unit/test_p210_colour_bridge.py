"""CPU contract checks for the additive P210 bridge purposes."""
import pathlib

import hdr_gpu_engine as hdr
from test_hdr_gpu_engine import FakeLibrary, _bridge


def test_p210_purposes_are_additive_and_encoded_uv_is_full_height():
    fake = FakeLibrary()
    bridge = _bridge(fake, purpose="encode422")
    assert any(call[0] == "gpu_hdr_open_purpose" for call in fake.calls)
    assert next(call[1][-2] for call in fake.calls if call[0] == "gpu_hdr_open_purpose") == 5
    planes = bridge.encode(clip_ceiling_scrgb=125.0)
    assert planes.y.pixel_format == "P210LE luma"
    assert planes.uv.width == 256
    assert planes.uv.height == 512
    assert "4:2:2" in planes.uv.pixel_format


def test_decode422_uses_dedicated_purpose():
    fake = FakeLibrary()
    bridge = _bridge(fake, purpose="decode422")
    assert any(call[0] == "gpu_hdr_open_purpose" for call in fake.calls)
    assert next(call[1][-2] for call in fake.calls if call[0] == "gpu_hdr_open_purpose") == 4


def test_native_shader_keeps_p010_420_and_adds_p210_row_chroma():
    source = pathlib.Path("native/gpu_hdr_scrgb_bridge.cpp").read_text(encoding="utf-8")
    assert "SourceUV.Load(id.y * SourceUVPitch" in source
    assert "SourceUV.Load((id.y >> 1) * SourceUVPitch" in source
    assert "P210_OUTPUT ? Height : (Height >> 1)" in source
    assert "P210_OUTPUT ? 0.5f : 0.25f" in source

