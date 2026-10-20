"""Preview uses the configured export frame core and its layer diagnostics."""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dlss_service  # noqa: E402


class _ExportCorePreview:
    """Stand-in for PreviewSession's export-chain frame consumer."""

    def __init__(self):
        self.requests = []
        self.source = np.zeros((16, 24, 3), np.uint8)
        self.output = np.full((16, 24, 3), 127, np.uint8)

    def render(self, path, settings, frame_index, **display_options):
        self.requests.append((path, dict(settings), frame_index))
        return self.source.copy(), self.output.copy(), {
            "layer_count": 3,
            "layers": {"layer_session_ids": [101, 102, 103],
                       "layer_working_dims": [[24, 16], [16, 16], [8, 8]]},
            "fusion_mode_actual": "gpu_data_consistency_solve",
            "reset": frame_index == 0,
            "processing_ms": 2.0,
        }

    def close(self):
        pass


def test_preview_forwards_export_multiscale_settings_and_reports_same_layers():
    service = dlss_service.DLSSService()
    service.resume_preview_memory()
    service.video_path = "shared-export-source.mkv"
    service.video_info = {"frames": 10, "fps": 30, "width": 24, "height": 16}
    service.settings.update({"export_backend": "gpu_transfer",
                             "reconstruction_mode": "triple",
                             "layer_scales": [1.0, 0.67, 0.33],
                             "layer_weights": [1.0, 0.8, 0.5],
                             "fusion_mode": "data_consistency_experimental"})
    session = _ExportCorePreview()
    service._gpu_preview = session

    jpeg, status = service.render_preview(4, view="dlss")

    assert status == "ok"
    _path, request_settings, index = session.requests[0]
    assert index == 4
    assert request_settings["export_backend"] == "gpu_transfer"
    assert request_settings["reconstruction_mode"] == "triple"
    assert request_settings["layer_scales"] == [1.0, 0.67, 0.33]
    assert request_settings["layer_weights"] == [1.0, 0.8, 0.5]
    diagnostics = service.preview_diagnostics()
    assert diagnostics["layer_count"] == 3
    assert diagnostics["layers"]["layer_session_ids"] == [101, 102, 103]
    assert diagnostics["fusion_mode_actual"] == "gpu_data_consistency_solve"

    displayed = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert displayed.shape == session.output.shape
    assert np.max(np.abs(displayed.astype(np.int16) - session.output.astype(np.int16))) <= 2


def test_changing_settings_is_forwarded_to_session_and_not_a_cached_legacy_backend():
    service = dlss_service.DLSSService()
    service.resume_preview_memory()
    service.video_path = "shared-export-source.mkv"
    service.video_info = {"frames": 10, "fps": 30, "width": 24, "height": 16}
    session = _ExportCorePreview()
    service._gpu_preview = session

    service.settings["layer_scales"] = [1.0, 0.5]
    service.render_preview(1, view="dlss")
    service.settings["layer_scales"] = [1.0, 0.75, 0.5]
    service.settings["reconstruction_mode"] = "triple"
    service.render_preview(2, view="dlss")

    assert [request[2] for request in session.requests] == [1, 2]
    assert session.requests[0][1]["layer_scales"] == [1.0, 0.5]
    assert session.requests[1][1]["layer_scales"] == [1.0, 0.75, 0.5]
    assert service._preview_backend is None
