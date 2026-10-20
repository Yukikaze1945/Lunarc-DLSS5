"""Guided preview requests use the production GPU preview session boundary."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dlss_service  # noqa: E402


class _PreviewSession:
    def __init__(self):
        self.calls = []
        self.frame = np.zeros((4, 8, 3), np.uint8)
        self.diagnostics = {"layer_count": 1, "layers": {"layer_session_ids": [7]},
                            "fusion_mode_actual": "catmull_baseline_layer_0",
                            "reset": False, "processing_ms": 0.5}

    def render(self, video, settings, frame_index, **display_options):
        self.calls.append((video, dict(settings), frame_index))
        return self.frame, np.full_like(self.frame, 80), dict(self.diagnostics)

    def close(self):
        pass


@pytest.mark.parametrize("flow_engine", ["nvof", "fidelityfx", "dis", "raft"])
def test_guided_preview_passes_flow_and_multiscale_configuration_to_session(flow_engine):
    service = dlss_service.DLSSService()
    service.resume_preview_memory()
    service.video_path = "guided-source.mkv"
    service.video_info = {"frames": 12, "fps": 24, "width": 8, "height": 4}
    service.settings.update({"guidance_mode": 1, "flow_engine": flow_engine,
                             "reconstruction_mode": "triple",
                             "layer_scales": [1.0, 0.75, 0.5],
                             "layer_weights": [1.0, 1.0, 0.5]})
    session = _PreviewSession()
    service._gpu_preview = session
    service.get_frame = lambda _index: (_ for _ in ()).throw(
        AssertionError("guided GPU preview unexpectedly used CPU frame decoding"))

    result, status = service.render_preview(5, view="dlss")

    assert status == "ok" and result
    video, settings, frame_index = session.calls[0]
    assert (video, frame_index) == ("guided-source.mkv", 5)
    assert settings["guidance_mode"] == 1
    assert settings["flow_engine"] == flow_engine
    assert settings["reconstruction_mode"] == "triple"
    assert settings["layer_scales"] == [1.0, 0.75, 0.5]
    assert service.preview_diagnostics()["layer_count"] == 1


def test_preview_session_failure_is_reported_and_does_not_escape_render_preview():
    service = dlss_service.DLSSService()
    service.resume_preview_memory()
    service.video_path = "guided-source.mkv"
    service.video_info = {"frames": 12, "fps": 24, "width": 8, "height": 4}

    class FailedSession:
        def render(self, *_args, **display_options):
            raise RuntimeError("decoder initialization failed")

    service._gpu_preview = FailedSession()
    result, status = service.render_preview(0, view="dlss")

    assert result is None
    assert "decoder initialization failed" in status
    assert service.preview_diagnostics()["status"] == "failed"


def test_busy_preview_lock_reports_waiting_without_calling_session():
    service = dlss_service.DLSSService()
    service.resume_preview_memory()
    service.video_path = "guided-source.mkv"
    service.video_info = {"frames": 12, "fps": 24, "width": 8, "height": 4}
    session = _PreviewSession()
    service._gpu_preview = session
    assert service._live_lock.acquire(timeout=1)
    try:
        result, status = service.render_preview(2, view="dlss")
    finally:
        service._live_lock.release()

    assert result is None
    assert "GPU is busy" in status
    assert session.calls == []
