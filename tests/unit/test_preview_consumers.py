"""Preview-only sinks on the existing SDR and FP16 frame chains."""

from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx
from hdr_gpu_engine import DevicePlane, FORMAT_RGBA16F_CUDA, PLANE_COMPOSITED
from temporal_control import TemporalControls


class _Nv12:
    def nv12_rgba_convert(self, _handle, *_args):
        _args[-2]._obj.value = 0x101
        _args[-1]._obj.value = 64
        return 0


class _Nvenc:
    def __init__(self):
        self.encoded = []
        self.p010_encoded = []
        self.opened = 0

    def nvenc_open(self, *_args):
        self.opened += 1
        _args[-1]._obj.value = 0x44
        return 0

    def nvenc_encode_texture(self, *args):
        self.encoded.append(args)
        return 0

    def nvenc_encode_p010(self, *args):
        self.p010_encoded.append(args)
        return 0


class _Live:
    def d3d12_device(self):
        return 0xD3D

    def process_device_color_zero_motion_resource(self, *_args, **_kwargs):
        return {"d3d12_resource": 0x222, "d3d12_fence": 0x333,
                "fence_value": 4, "d3d12_device": 0xD3D,
                "width": 16, "height": 8, "dxgi_format": 28}


def _frame():
    return SimpleNamespace(width=16, height=8, cuda_context=0xC0,
                           y_device_ptr=0x11, y_pitch_bytes=32,
                           uv_device_ptr=0x12, uv_pitch_bytes=32, pts=7,
                           time_base_num=1, time_base_den=30)


def test_sdr_single_preview_consumes_final_candidate_without_opening_encoder():
    nvenc = _Nvenc()
    consumed = []
    chain = gtx.Chain(
        {"nv12": _Nv12(), "lanczos": object(), "nvenc": nvenc}, "source.mkv", "unused.mkv",
        _Live(), {}, (30, 1), 1.0, (16, 8), uses_flow=False,
        downscaled=False,
        preview_consumer=lambda descriptor, frame: consumed.append((descriptor, frame)) or "jpeg",
    )
    # These consumer tests use fake native pointers; scene detection is covered separately.
    chain.temporal_control.update_controls(TemporalControls(scene_detection=False,mode='off'))
    chain.nv12 = object()

    outcome = chain.process(_frame(), None, 0, False)

    assert chain.encoder is None
    assert nvenc.encoded == []
    assert outcome["preview_output"] == "jpeg"
    descriptor, frame = consumed[0]
    assert descriptor["d3d12_resource"] == 0x222
    assert (descriptor["width"], descriptor["height"]) == (16, 8)
    assert descriptor["cuda_context"] == 0xC0
    assert descriptor["d3d12_device"] == 0xD3D
    assert descriptor["format"] == "rgba8_unorm"
    assert descriptor["dxgi_format"] == 28
    assert frame.pts == 7


def test_sdr_single_export_without_consumer_still_submits_nvenc():
    nvenc = _Nvenc()
    chain = gtx.Chain(
        {"nv12": _Nv12(), "lanczos": object(), "nvenc": nvenc}, "source.mkv", "output.mkv",
        _Live(), {"struct": {"codec": b"hevc_nvenc", "preset": b"p7",
            "tune": b"hq", "multipass": 0, "rate_control": 0,
            "constant_quality": 24, "bitrate": 0, "gop": 0,
            "color_matrix": 0}, "av_options": {}}, (30, 1), 1.0, (16, 8), uses_flow=False,
        downscaled=False,
    )
    # These consumer tests use fake native pointers; scene detection is covered separately.
    chain.temporal_control.update_controls(TemporalControls(scene_detection=False,mode='off'))
    chain.nv12 = object()

    outcome = chain.process(_frame(), None, 0, False)

    assert nvenc.opened == 1
    assert chain.encoder.value == 0x44
    assert len(nvenc.encoded) == 1
    assert nvenc.encoded[0][0] == chain.encoder
    assert nvenc.encoded[0][1] == 0x222
    assert "nvenc_submit" in outcome["stages"]
    assert "preview_output" not in outcome


def test_multiscale_preview_consumes_postprocessed_fusion_output(monkeypatch):
    consumed = []
    post_inputs = []

    class Fusion:
        def gpu_fusion_process(self, _handle, *_args):
            out = C.cast(_args[-2], C.POINTER(C.c_void_p))
            out[0] = C.c_void_p(0x999)
            report = C.cast(_args[-1], C.POINTER(gtx._FusionReport)).contents
            report.iterations = 3
            report.initial_residual_norm = 2.0
            report.final_residual_norm = 0.01
            report.relative_residual = 0.005
            report.converged = 1
            return 0

    chain = gtx.MultiScaleObservationChain.__new__(gtx.MultiScaleObservationChain)
    chain.preview_consumer = lambda descriptor, frame: consumed.append((descriptor, frame)) or "preview"
    chain._defer_preview_consumer = True
    chain.output_effect = None
    chain.extra_layers = []
    chain.base_low_residual = None
    chain.layer_calls = [0]
    chain.layer_reset_flags = [[]]
    chain.uses_flow = False
    chain.defer_encode = True
    chain.config = {'fusion':'data_consistency_experimental'}
    chain.fusion_handle = object()
    chain.fusion_reports = []
    chain.b = {"fusion": Fusion()}
    chain._observe = lambda *args: (object(), (0x51, 0x52, 5))

    base_outcome = {
        "_dlss_reused": False,
        "_is_discontinuous": False,
        "_dlss_resource": {"d3d12_device": 0xD3D},
        "_working_color": {"width": 16, "height": 8},
        "_full_color": (0x101, 64),
        "stages": {},
    }
    monkeypatch.setattr(gtx.Chain, "process", lambda *_args: dict(base_outcome))

    def post_process(_self, _frame, _source, _pitch, candidate):
        post_inputs.append(candidate["d3d12_resource"])
        return {"d3d12_resource": 0xAAA, "d3d12_device": 0xD3D,
                "producer_completed": True}

    chain._post_process = post_process.__get__(chain)
    outcome = chain.process(_frame(), None, 0, False)

    assert post_inputs == [0x999]
    assert outcome["preview_output"] == "preview"
    descriptor, frame = consumed[0]
    assert descriptor["d3d12_resource"] == 0xAAA
    assert descriptor["format"] == "rgba8_unorm"
    assert descriptor["d3d12_device"] == 0xD3D
    assert frame.pts == 7


def test_hdr_preview_skips_p010_conversion_and_nvenc():
    class Bridge:
        def encode(self, *, clip_ceiling_scrgb):
            assert clip_ceiling_scrgb == 1.0
            return object()

        def encode_report(self):
            return {"encoded": True}

    nvenc = _Nvenc()
    chain = htx.HdrChain.__new__(htx.HdrChain)
    chain.prores_output = None
    chain.preview_consumer = lambda descriptor, frame: (descriptor, frame.pts)
    chain.bridge = Bridge()
    chain.encode_bridge = None
    chain.policy = {"clip_ceiling_scrgb": 1.0}
    chain.encoder = None
    stages = {}

    _, report = chain._encode(_frame(), stages)

    assert report is None
    assert "hdr_encode" not in stages
    assert "nvenc_submit" not in stages
    assert nvenc.encoded == []

    chain.src = (16, 8)
    chain.context = 0xC0
    chain.live = _Live()
    composite = DevicePlane(fmt=FORMAT_RGBA16F_CUDA,
        pixel_format="R16G16B16A16_FLOAT", width=16, height=8,
        cuda_context=0xC0, device_ptr=0xABCD, pitch_bytes=128,
        bytes_per_pixel=8, plane=PLANE_COMPOSITED)
    descriptor, pts = chain._consume_preview_output(_frame(), composite)
    assert descriptor["device_ptr"] == 0xABCD
    assert descriptor["width"] == 16 and descriptor["height"] == 8
    assert descriptor["d3d12_device"] == 0xD3D
    assert descriptor["cuda_context"] == 0xC0
    assert descriptor["dxgi_format"] == 10
    assert descriptor["format"] == FORMAT_RGBA16F_CUDA
    assert pts == 7


def test_hdr_without_preview_consumer_keeps_p010_nvenc_submission():
    class Bridge:
        def encode(self, *, clip_ceiling_scrgb):
            assert clip_ceiling_scrgb == 4.0
            return SimpleNamespace(
                y=SimpleNamespace(device_ptr=0x51, pitch_bytes=64),
                uv=SimpleNamespace(device_ptr=0x52, pitch_bytes=64))

        def encode_report(self):
            return {"encoded": True}

    nvenc = _Nvenc()
    chain = htx.HdrChain.__new__(htx.HdrChain)
    chain.preview_consumer = None
    chain.prores_output = None
    chain.bridge = Bridge()
    chain.encode_bridge = None
    chain.policy = {"clip_ceiling_scrgb": 4.0}
    chain.encoder = C.c_void_p(0x44)
    chain.b = {"nvenc": nvenc}
    stages = {}

    _, report = chain._encode(_frame(), stages)

    assert report == {"encoded": True}
    assert len(nvenc.p010_encoded) == 1
    assert nvenc.p010_encoded[0] == (chain.encoder, 0x51, 64, 0x52, 64, 7)
    assert "nvenc_submit" in stages
