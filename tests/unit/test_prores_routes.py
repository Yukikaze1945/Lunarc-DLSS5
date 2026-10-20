"""The ProRes decoder and encoder selections are independent routes."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_hdr_transfer_export_branch as fixtures
import gpu_hdr_transfer_export as htx


@pytest.mark.parametrize("prores_input,prores_output", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_prores_source_and_output_routes_are_independent(
        monkeypatch, prores_input, prores_output):
    src = dict(fixtures.HDR_SOURCE, color_range="tv", color_space="bt709",
        color_transfer="bt709", color_primaries="bt709", metadata=False,
        codec_name="prores" if prores_input else "hevc",
        pix_fmt="yuv422p10le" if prores_input else "yuv420p10le")
    output = dict(fixtures.NO_MUX, codec="prores_ks" if prores_output else "hevc_nvenc",
        container="mov" if prores_output else "mkv",
        profile="hq" if prores_output else "main10",
        pixel_format="yuv422p10le" if prores_output else "p010le")

    if prores_output:
        monkeypatch.setattr("prores_export.encoder_plan", lambda *_: dict(
            codec="prores_ks", profile="hq", pixel_format="yuv422p10le",
            unsupported=[], av_options=[], struct={"codec": b"prores_ks"}))

    result = htx.negotiate_sdr10(fixtures.HDR_DIMS, src,
        dict(fixtures.HDR_JOB, experimental_hdr_scrgb=False), output, 24,
        False, "out.mov" if prores_output else "out.mkv", fixtures.Fp16Live(),
        hdr_caps=fixtures.HDR_CAPS, flow_provider=fixtures.DeviceFlowProvider())

    assert result["supported"], result["reasons"]
    assert (result["plan"]["codec"] == "prores_ks") is prores_output
    assert result["plan"].get("_source", {}).get("codec_name") == (
        "prores" if prores_input else None)
