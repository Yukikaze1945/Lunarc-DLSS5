"""Service integration contracts for the shared GPU preview session."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dlss_service  # noqa: E402


class _PreviewSession:
    def __init__(self, original, processed=None, diagnostics=None):
        self.original = original
        self.processed = processed if processed is not None else original.copy()
        self.diagnostics = diagnostics or {
            "layer_count": 2, "layers": {"layer_session_ids": [10, 11]},
            "fusion_mode_actual": "catmull_baseline_layer_0", "reset": True,
            "processing_ms": 1.25,
        }
        self.calls = []

    def render(self, video, settings, frame, **display_options):
        self.calls.append((video, dict(settings), frame))
        self.display_options = display_options
        return self.original.copy(), self.processed.copy(), dict(self.diagnostics)

    def close(self):
        pass


def _service(monkeypatch, *, guidance_mode=0, reconstruction_mode="single"):
    svc = dlss_service.DLSSService()
    svc.resume_preview_memory()
    svc.video_path = "synthetic-source.mkv"
    svc.video_info = {"frames": 8, "fps": 30.0, "width": 8, "height": 4}
    svc.settings.update({"guidance_mode": guidance_mode,
                         "reconstruction_mode": reconstruction_mode,
                         "layer_scales": [1.0, 0.5], "layer_weights": [1.0, 1.0]})
    source = np.zeros((4, 8, 3), np.uint8)
    enhanced = np.full((4, 8, 3), 100, np.uint8)
    session = _PreviewSession(source, enhanced)
    svc._gpu_preview = session
    svc.get_frame = lambda _index: (_ for _ in ()).throw(
        AssertionError("GPU preview must use the source video path, not CPU frame decoding"))
    return svc, session


def test_unguided_multiscale_preview_delegates_full_request_to_gpu_session(monkeypatch):
    svc, session = _service(monkeypatch, reconstruction_mode="dual")
    result, status = svc.render_preview(3, view="dlss")

    assert status == "ok" and result
    video, settings, frame = session.calls[0]
    assert (video, frame) == ("synthetic-source.mkv", 3)
    assert settings["guidance_mode"] == 0
    assert settings["reconstruction_mode"] == "dual"
    assert settings["layer_scales"] == [1.0, 0.5]
    assert svc.preview_diagnostics()["layer_count"] == 2


def test_preview_session_output_is_encoded_without_legacy_live_session(monkeypatch):
    svc, session = _service(monkeypatch)
    result, status = svc.render_preview(0, view="dlss")

    import cv2
    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    assert status == "ok" and decoded.shape == (4, 8, 3)
    assert np.all(decoded > 90)
    assert len(session.calls) == 1
    assert svc._live_session is None


def test_original_split_and_dual_views_use_the_preview_session_result(monkeypatch):
    svc, session = _service(monkeypatch)
    original, original_status = svc.render_preview(1, view="orig")
    split, split_status = svc.render_preview(1, view="split", split_x=0.5)
    dual, dual_status = svc.render_preview(1, view="dual")

    import cv2
    decode = lambda data: cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert original_status == split_status == dual_status == "ok"
    assert np.all(decode(original) < 10)
    assert decode(split).shape == (4, 8, 3)
    assert decode(dual).shape == (4, 16, 3)
    assert [call[2] for call in session.calls] == [1, 1, 1]


@pytest.mark.parametrize('engine', ['nvof', 'fidelityfx', 'raft', 'raft_large'])
def test_device_flow_view_uses_preview_owner_even_with_guidance_disabled(monkeypatch, engine):
    svc, session = _service(monkeypatch, guidance_mode=0)
    svc.settings['flow_engine'] = engine
    monkeypatch.setattr(dlss_service, 'get_optical_flow_engine',
                        lambda *a, **k: pytest.fail('legacy engine must not be initialized'))
    data, status = svc.render_preview(1, view='flow')
    assert data and status == 'ok'
    assert session.display_options['flow_view'] is True
    assert session.calls[0][1]['guidance_mode'] == svc.settings['guidance_mode'] == 0
