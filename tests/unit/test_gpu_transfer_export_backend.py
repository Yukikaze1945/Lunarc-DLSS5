# -*- coding: utf-8 -*-
"""Gate: the GPU-transfer export backend is opt-in, and a request it cannot serve refuses.

Nothing in here needs a GPU. The whole point of separating the gate from the chain is that
the decision - which settings this backend can honour, which encoder options reach the
codec, which container clauses the copy pass has to re-apply - is made from facts the
caller already has: one ffprobe of the source, the resolved dimensions, and what the
loaded libraries advertise about themselves. So the decision is testable on a machine with
no display attached, and the real export stays a single run rather than a matrix.

Three claims are checked rather than asserted:

* an explicitly requested backend is never answered by running the old way, so a refusal
  has to name every unmet condition instead of picking the first one;
* an encoding setting the in-process encoder cannot be given is a refusal, not a silent
  drop, because a dropped -bf is indistinguishable from an ignored one in the finished file;
* a byte figure in the report is either a bridge's own counter or None. A buffer size the
  code allocated is labelled logical, and a logical figure never shares a name with a
  measured one, because the two added together would double-count the same pixels.
"""

import json
import os
import re
import sys

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UNIT_DIR)
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_transfer_export as gtx  # noqa: E402
import pipeline  # noqa: E402

# A 4K source at the only working scale the two exact-2x GPU stages accept.
DIMS = {"src_w": 3840, "src_h": 2160, "live_w": 1920, "live_h": 1080,
        "dst_w": 3840, "dst_h": 2160, "fps": 29.97}
# The other served shape: the full RGBA is the working colour, no residual stage.
DIMS_FULL = {"src_w": 3840, "src_h": 2160, "live_w": 3840, "live_h": 2160,
             "dst_w": 3840, "dst_h": 2160, "fps": 29.97}
SOURCE = {"video_stream_count": 1, "codec_name": "hevc", "pix_fmt": "yuv420p",
          "width": 3840, "height": 2160, "color_range": "tv",
          "color_space": "smpte170m", "color_transfer": "bt709",
          "color_primaries": "bt709", "r_frame_rate": "30000/1001",
          "audio_streams": 1, "subtitle_streams": 0, "nb_chapters": 0,
          "metadata": True}
GPU_JOB = {"dlss_backend": "gpu_flow_d3d12", "guidance_mode": 1,
           "flow_engine": "nvof", "working_scale": 0.5,
           "residual_pyramid": True, "multi_pass": 1,
           "export_backend": "gpu_transfer"}
GPU_JOB_FULL = dict(GPU_JOB, working_scale=1.0)


class DeviceFlowProvider:
    """A flow provider whose engine reads a CUDA surface, as the NVOF bridge does."""

    def __init__(self, supported=True):
        self._supported = supported
        self.calls = []

    def get_flow_device_rgba(self, frame_idx, color, cuda_context,
                             is_discontinuous=False):
        self.calls.append(frame_idx)
        return {"format": "float32x2"}

    def supports_device_rgba_flow(self):
        return self._supported

    def device_input_counters(self):
        return (0, 0)

    def timing(self):
        return {}


class DeviceLive:
    """A DLSS session that can hand back the texture it wrote instead of a copy of it."""

    def process_device_color_flow_resource(self, color, flow, reset=False):
        return {"d3d12_device": 1, "d3d12_resource": 2, "d3d12_fence": 3,
                "fence_value": 1, "width": 1920, "height": 1080}


class HostLive:
    """The shipping host-pointer session: it returns a numpy array and nothing else."""

    def process(self, rgba, flow, reset=False, out=None):
        return rgba


class ZeroMotionLive(DeviceLive):
    """An ABI-3 session: it binds its own zero motion texture and says so."""

    def supports_zero_motion(self):
        return True

    def process_device_color_zero_motion_resource(self, color, reset=False):
        return {"d3d12_device": 1, "d3d12_resource": 2, "d3d12_fence": 3,
                "fence_value": 1, "width": 1920, "height": 1080}


class BannedFlowProvider(DeviceFlowProvider):
    """A provider the unguided chain must never reach, not even to offer it work."""

    def get_flow_device_rgba(self, *args, **kwargs):
        raise AssertionError("an unguided frame asked a flow provider for motion")


class _Nv12Stub:
    def nv12_rgba_convert(self, handle, y_ptr, y_pitch, uv_ptr, uv_pitch,
                          out_ptr, out_pitch):
        return 0


class _NvencStub:
    def __init__(self):
        self.handed = []

    def nvenc_encode_texture(self, encoder, texture, fence, value, pts):
        self.handed.append((texture, fence, value, pts))
        return 0


class _Frame:
    cuda_context = 1
    width, height = 3840, 2160
    y_device_ptr = y_pitch_bytes = uv_device_ptr = uv_pitch_bytes = 2048
    pts = 1001


def _identity_chain(uses_flow, live):
    """The 1.0 chain with everything but the DLSS stage stubbed, GPU-free.

    ``first`` is cleared because opening the residual and the encoder is what
    needs a real device; the routing this chain exists to test happens after it.
    """
    nvenc = _NvencStub()
    chain = gtx.Chain({"nv12": _Nv12Stub(), "nvenc": nvenc, "lanczos": None,
                       "residual": None, "nvdec": None},
                      "in.mp4", "out.mkv", live, {"codec": "hevc_nvenc"},
                      (30000, 1001), 1.0, (3840, 2160), None,
                      downscaled=False, tap_plans=None, uses_flow=uses_flow)
    chain.first = False
    # This routing fixture has no scene-observation device; that stage is tested separately.
    from temporal_control import GpuTemporalControl, TemporalControls
    chain.temporal_control = GpuTemporalControl(TemporalControls(scene_detection=False))
    return chain, nvenc


def _negotiate(job=None, transcode=None, source=None, dims=None, live=None,
               provider="default", out_path="out.mkv", **kwargs):
    provider = (DeviceFlowProvider() if provider == "default" else provider)
    kwargs.setdefault("bridges_ok", True)
    return gtx.negotiate(dict(dims or DIMS), dict(source or SOURCE),
                         dict(job or GPU_JOB), dict(transcode or {}), 24, True,
                         out_path, live or DeviceLive(),
                         flow_provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def test_a_request_the_chain_can_serve_is_accepted():
    negotiation = _negotiate()
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["reasons"] == []
    assert negotiation["fps"] == (30000, 1001)
    assert negotiation["plan"]["struct"]["color_matrix"] == gtx.COLOR_MATRIX_BT601
    assert negotiation["plan"]["decoder_color_matrix"] == gtx.COLOR_MATRIX_BT601
    assert negotiation["declared_color_space"] == "smpte170m"


def test_bt709_source_selects_bt709_for_both_conversion_and_encoding():
    negotiation = _negotiate(source=dict(SOURCE, color_space="bt709"))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["plan"]["struct"]["color_matrix"] == gtx.COLOR_MATRIX_BT709
    assert negotiation["plan"]["decoder_color_matrix"] == gtx.NV12_COLOR_MATRIX_BT709
    assert negotiation["declared_color_space"] == "bt709"


def test_every_unmet_condition_is_named_rather_than_the_first_one_only():
    """One job with four problems has to report four, or the fix iterates blindly."""
    negotiation = _negotiate(
        job=dict(GPU_JOB, dlss_backend="host_pointer", flow_engine="dis",
                 working_scale=0.75),
        live=HostLive())
    assert not negotiation["supported"]
    named = " ".join(negotiation["reasons"])
    for fragment in ("dlss_backend must be gpu_flow_d3d12", "flow_engine=dis",
                     "working_scale=0.75",
                     "cannot hand back a D3D12 output texture"):
        assert fragment in named, fragment


def test_a_requested_backend_is_never_answered_by_running_the_old_way(monkeypatch):
    """The refusal has to reach the caller as a failure, with the reasons in it."""
    def fail_load(requires_flow=True):
        raise gtx.BridgeError("no nvdec_bridge.dll here")

    monkeypatch.setattr(gtx, "load_bridges", fail_load)
    monkeypatch.setattr(gtx, "probe_source", lambda video, ffprobe=None: dict(SOURCE))
    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (14, 29.97, 3840, 2160))
    monkeypatch.setattr(pipeline, "build_transcode_command",
                        lambda *a, **k: pytest.fail("the pipe path was reached"))
    diagnostics = {}
    with pytest.raises(RuntimeError) as raised:
        pipeline.export_gpu_transfer_stream(
            video="in.mp4", live=DeviceLive(), frames=14, fps=29.97, cq=24,
            with_audio=False, out_path="out.mkv", transcode_settings={},
            job_settings=GPU_JOB, flow_provider=DeviceFlowProvider(),
            diagnostics_out=diagnostics, module=gtx)
    assert "no nvdec_bridge.dll here" in str(raised.value)
    assert diagnostics["export_backend_requested"] == "gpu_transfer"
    assert diagnostics["export_backend_actual"] == "refused"
    assert "no nvdec_bridge.dll here" in diagnostics["export_backend_fallback_reason"]


def test_a_flow_provider_that_cannot_read_device_memory_is_a_refusal():
    class HostOnlyFlow(DeviceFlowProvider):
        get_flow_device_rgba = None

    negotiation = _negotiate(provider=HostOnlyFlow())
    assert not negotiation["supported"]
    assert any("CUDA surface" in reason for reason in negotiation["reasons"]), negotiation["reasons"]


def test_a_bridge_set_without_the_device_rgba_entry_point_is_a_refusal():
    negotiation = _negotiate(provider=DeviceFlowProvider(supported=False))
    assert not negotiation["supported"]
    assert any("device RGBA input" in reason for reason in negotiation["reasons"])


# ---------------------------------------------------------------------------
# guidance_mode 0: motion from the host's own resident zero texture
# ---------------------------------------------------------------------------

def test_guidance_mode_0_is_accepted_on_a_session_that_binds_the_zero_texture():
    negotiation = _negotiate(job=dict(GPU_JOB, guidance_mode=0),
                             live=ZeroMotionLive(), provider=None)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["guidance_mode"] == 0
    assert negotiation["uses_flow"] is False


def test_guidance_mode_0_refuses_a_session_that_cannot_answer_for_the_texture():
    """The guided-only ABI-2 host has no unguided entry, and must not be assumed into one.

    Without this refusal the job would still run, and nothing in the finished
    file would say that DLSS never bound the zero motion texture.
    """
    negotiation = _negotiate(job=dict(GPU_JOB, guidance_mode=0),
                             live=DeviceLive(), provider=None)
    assert not negotiation["supported"]
    assert any("resident zero motion texture" in reason
               for reason in negotiation["reasons"]), negotiation["reasons"]
    assert any("process_device_color_zero_motion_resource" in reason
               for reason in negotiation["reasons"]), negotiation["reasons"]


def test_an_entry_point_claimed_but_not_backed_by_the_host_is_still_a_refusal():
    """hasattr is not the contract; the loaded library answering for it is."""
    class ClaimingLive(ZeroMotionLive):
        def supports_zero_motion(self):
            return False

    negotiation = _negotiate(job=dict(GPU_JOB, guidance_mode=0),
                             live=ClaimingLive(), provider=None)
    assert not negotiation["supported"]
    assert any("resident zero motion texture" in reason
               for reason in negotiation["reasons"]), negotiation["reasons"]


def test_guidance_mode_0_never_looks_at_the_flow_engine_setting():
    """Nothing computes flow when guidance is off, so DIS is no reason to refuse it."""
    negotiation = _negotiate(job=dict(GPU_JOB, guidance_mode=0, flow_engine="dis"),
                             live=ZeroMotionLive(), provider=None)
    assert negotiation["supported"], negotiation["reasons"]


def test_a_guidance_mode_that_is_neither_off_nor_flow_is_refused():
    negotiation = _negotiate(job=dict(GPU_JOB, guidance_mode=2),
                             live=ZeroMotionLive(), provider=None)
    assert not negotiation["supported"]
    assert any("guidance_mode=2 is neither 0 (off) nor 1 (flow)" in reason
               for reason in negotiation["reasons"]), negotiation["reasons"]


def test_the_unguided_chain_drives_the_zero_motion_entry_and_no_flow_provider():
    chain, nvenc = _identity_chain(False, ZeroMotionLive())
    outcome = chain.process(_Frame(), BannedFlowProvider(), 0, False)
    assert (chain.flow_calls, chain.zero_motion_calls) == (0, 1)
    assert outcome["flow"] is None
    assert outcome["nvof_timing"] is None
    assert "nvof_flow" not in outcome["stages"]
    # The identity chain hands the encoder the DLSS texture and its fence, not a copy.
    assert nvenc.handed == [(2, 3, 1, 1001)]


def test_the_guided_chain_still_asks_the_provider_and_the_flow_entry():
    """The contrast that makes the claim above a routing test, not an accident."""
    provider = DeviceFlowProvider()
    chain, nvenc = _identity_chain(True, ZeroMotionLive())
    outcome = chain.process(_Frame(), provider, 0, False)
    assert (chain.flow_calls, chain.zero_motion_calls) == (1, 0)
    assert provider.calls == [0]
    assert outcome["flow"] == {"format": "float32x2"}
    assert nvenc.handed == [(2, 3, 1, 1001)]


@pytest.mark.parametrize("source_change,fragment", [
    ({"pix_fmt": "yuv444p"}, "8-bit 4:2:0"),
    ({"color_space": "bt2020nc"}, "BT.601 or BT.709"),
    ({"color_space": "smpte240m"}, "BT.601 or BT.709"),
    ({"color_range": "pc"}, "limited range"),
    ({"color_transfer": "smpte2084"}, "HDR"),
    ({"video_stream_count": 2}, "video streams"),
])
def test_a_source_the_colour_contract_does_not_cover_is_refused(source_change, fragment):
    negotiation = _negotiate(source=dict(SOURCE, **source_change))
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), fragment


def test_a_resized_output_is_accepted_for_final_gpu_scaling():
    negotiation = _negotiate(dims=dict(DIMS, dst_w=1920, dst_h=1080))
    assert negotiation["supported"], negotiation["reasons"]


# The panel's own DEFAULT_SETTINGS (frontend/src/pages/DeliverPage.tsx), verbatim: the
# keys are what the delivery page has always put in every job, so a gate that calls any
# of them an unhonoured encoding choice makes GPU Transfer unreachable from the UI.
PANEL_DEFAULT_TRANSCODE = {
    "codec": "av1_nvenc", "container": ".mkv", "suffix": "_AV1_P7_UHQ_Q16",
    "rate_mode": "cq", "cq": 16, "video_bitrate": "80M", "qp": 20,
    "peak_policy": "auto", "maxrate": "380M", "bufsize": "1520M",
    "preset": "p7", "tune": "uhq", "multipass": "fullres", "gpu": "-1",
    "decode_mode": "auto", "finish_action": "none", "gop_mode": "auto", "gop": 240,
    "b_frames": "auto", "b_ref_mode": "auto", "lookahead": "auto",
    "lookahead_level": "auto", "aq_mode": "spatial", "aq_strength": 8,
    "weighted_prediction": False, "temporal_filter": "auto",
    "split_encode": "auto", "profile": "auto", "level": "auto", "tier": "auto",
    "resolution_mode": "original", "custom_width": 1920, "custom_height": 1080,
    "custom_factor": 1.0, "keep_aspect": True, "scale_algorithm": "lanczos",
    "pixel_format": "auto", "color_preset": "preserve", "custom_color_space": "bt709",
    "custom_color_primaries": "bt709", "custom_color_transfer": "bt709",
    "custom_color_range": "tv", "audio_mode": "copy", "audio_bitrate": "320k",
    "subtitle_mode": "copy", "copy_attachments": True, "copy_metadata": True,
    "copy_chapters": True, "overwrite_output": False, "skip_existing": True,
    "extra_args": "",
}


def test_the_payload_the_delivery_panel_sends_by_default_names_no_unsupported_key():
    reasons = _negotiate(transcode=dict(PANEL_DEFAULT_TRANSCODE))["reasons"]
    assert [r for r in reasons if "has no equivalent" in r] == [], reasons


def test_an_encoder_setting_this_codec_cannot_be_given_is_still_a_refusal():
    negotiation = _negotiate(transcode={"two_pass": "on"})
    assert not negotiation["supported"]
    assert any("transcode_settings['two_pass']" in reason
               for reason in negotiation["reasons"])


def test_a_custom_factor_is_accepted_at_its_resolved_size():
    """The factor resolves the final encoder size instead of being ignored."""
    ts = {"resolution_mode": "custom_factor", "custom_factor": 2.0}
    dst_w, dst_h = pipeline.resolve_output_dims(3840, 2160, ts)
    assert (dst_w, dst_h) == (7680, 4320)
    negotiation = _negotiate(transcode=ts, dims=dict(DIMS, dst_w=dst_w, dst_h=dst_h))
    assert negotiation["supported"], negotiation["reasons"]
    assert not any("has no equivalent" in reason for reason in negotiation["reasons"])


def test_the_encoder_is_asked_what_it_accepts_of_the_installation_that_will_encode(tmp_path,
                                                                                   monkeypatch):
    """Two FFmpeg installs on one machine can disagree about one option's values.

    The encoder that has to honour a tune is the ``avcodec-*.dll`` the bridges bound
    to, so the CLI beside that one answers; ``ffmpeg.exe`` from PATH would be a
    different build and would approve a value the loaded library refuses.
    """
    libraries = tmp_path / "libs"
    libraries.mkdir()
    (libraries / "avcodec-61.dll").write_text("")
    (libraries / "ffmpeg.exe").write_text("")
    other = tmp_path / "other"
    other.mkdir()
    (other / "ffmpeg.exe").write_text("")

    monkeypatch.setattr(gtx, "ffmpeg_dll_directories", lambda: [str(libraries)])
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: str(other / "ffmpeg.exe"))
    assert gtx.encoder_cli() == str(libraries / "ffmpeg.exe")

    # A library directory with no command line tool beside it is the case the fallback
    # exists for: the question then goes to the tool this process would run.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "avcodec-61.dll").write_text("")
    monkeypatch.setattr(gtx, "ffmpeg_dll_directories", lambda: [str(bare)])
    assert gtx.encoder_cli() == str(other / "ffmpeg.exe")


def test_a_tune_the_loaded_encoder_does_not_list_is_downgraded_not_sent():
    """uhq is a preset on some builds and a tune on others; only the tune matters."""
    plan = gtx.plan_encoder(dict(PANEL_DEFAULT_TRANSCODE), 24, dict(SOURCE),
                            tune_available=lambda codec: False)
    assert not plan["unsupported"], plan["unsupported"]
    assert plan["struct"]["tune"] == b"hq"
    plan = gtx.plan_encoder(dict(PANEL_DEFAULT_TRANSCODE), 24, dict(SOURCE),
                            tune_available=lambda codec: True)
    assert plan["struct"]["tune"] == b"uhq"


# ---------------------------------------------------------------------------
# the two served scales: exact 2x, and identity at 1.0
# ---------------------------------------------------------------------------

def test_working_scale_one_is_accepted_as_the_identity_chain():
    """1.0 is the CPU plan's own no-residual shape, so the gate can serve it."""
    negotiation = _negotiate(job=GPU_JOB_FULL, dims=DIMS_FULL)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["reasons"] == []
    assert negotiation["downscaled"] is False


def test_working_scale_one_does_not_need_a_pyramid():
    """The CPU plan has no residual stage at 1.0, so pyramid off is inert there
    and refusing on its account would name a problem the run does not have."""
    negotiation = _negotiate(job=dict(GPU_JOB_FULL, residual_pyramid=False),
                            dims=DIMS_FULL)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["downscaled"] is False


def test_working_scale_half_still_requires_the_residual_pyramid():
    negotiation = _negotiate(job=dict(GPU_JOB, residual_pyramid=False))
    assert not negotiation["supported"]
    assert any("residual_pyramid" in reason and "CPU resize" in reason
               for reason in negotiation["reasons"]), negotiation["reasons"]


def test_the_identity_report_labels_the_full_plane_as_the_working_colour():
    """At 1.0 the working colour is the converter's own full RGBA, so the label
    names that plane; and because no bridge ran, no measured field may claim a
    working-color copy or a residual readback that never happened."""
    report = gtx.transfer_report({}, 1, 3840, 2160, downscaled=False)
    assert report["working_color_bytes_logical"] == 3840 * 2160 * 4
    assert report["nv12_to_rgba_bytes_logical"] == 3840 * 2160 * 4
    for key in ("working_color_device_bytes_measured",
                "working_color_process_count_measured",
                "working_color_source_device_copy_bytes_measured",
                "working_color_scratch_device_write_bytes_measured",
                "residual_full_device_copy_bytes_measured",
                "residual_working_device_copy_bytes_measured",
                "residual_output_readback_bytes_measured"):
        assert report[key] is None, key


def test_the_half_report_keeps_labelling_the_half_plane():
    report = gtx.transfer_report({}, 1, 3840, 2160)
    assert report["working_color_bytes_logical"] == 1920 * 1080 * 4


# Both plan-driven exports present and the loaded bridge answering for its own
# window: what a build that can serve a scale other than the half advertises, and
# what the native/ build really reports. The figure is deliberately part of the
# fixture rather than left out, because a fixture that omits it is a different
# build - the pre-widening one that exports the plan-driven open and answers
# nothing. That build is LEGACY_SCALED below, and the two must not be confused:
# one is measured against what it says, the other against the 16 its own kernel
# was compiled with.
SCALED = {"lanczos_scaled": True, "residual_scaled": True,
          "lanczos_max_taps": gtx.LANCZOS_MAX_TAPS}
# The ABI-2 build that has gpu_lanczos_open_scaled and no gpu_lanczos_max_taps,
# which is the one still shipped in dist/ - it walks 16 and says nothing.
LEGACY_SCALED = {"lanczos_scaled": True, "residual_scaled": True}


def _dims_for(scale, src_w=3840, src_h=2160):
    """The extents a request at this scale resolves to, from the one resolver."""
    live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
    return {"src_w": src_w, "src_h": src_h, "live_w": live_w, "live_h": live_h,
            "dst_w": src_w, "dst_h": src_h, "fps": 29.97}


def test_a_served_ratio_with_extents_it_did_not_resolve_to_is_refused():
    """0.63 is served, and 2419x1360 is not what 0.63 resolves to.

    The extents decide the chain, so a caller that computed its own working size
    - here one that is not even 8-aligned - is refused by name rather than handed
    to a bridge whose plan axes would disagree with the frame it was given.
    """
    negotiation = _negotiate(job=dict(GPU_JOB, working_scale=0.63),
                            dims=dict(DIMS, live_w=2419, live_h=1360))
    assert not negotiation["supported"]
    reason = " ".join(negotiation["reasons"])
    for fragment in ("working_scale=0.63", "resolves to 2419x1360",
                     "resolves to 2416x1360, not the size this job declared",
                     "every whole percentage from 25% to 100%", "no residual stage"):
        assert fragment in reason, fragment


def test_a_ratio_no_proven_stage_exists_for_is_served_by_the_same_arithmetic():
    """40% never had a byte-identical measurement and is served anyway.

    What used to refuse it was a list of ratios that had been measured, which made
    the gate's answer depend on what someone had run before rather than on the
    extents and the build. The two questions that remain are the ones a request
    can actually fail: whether the resolved size has a chain shape, and whether
    this bridge build walks the window that size needs.
    """
    live_w, live_h = pipeline.resolve_working_dims(3840, 2160, 0.4)
    assert (live_w, live_h) == (1536, 864)
    assert gtx.plan_tap_widths(3840, 2160, live_w, live_h)[0] < \
        gtx.LANCZOS_MAX_TAPS
    negotiation = _negotiate(job=dict(GPU_JOB, working_scale=0.4),
                             dims=_dims_for(0.4), capabilities=SCALED)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["downscaled"] and negotiation["plan_driven"]


def test_the_served_plan_driven_ratios_are_the_ui_buttons_this_chain_can_run():
    """Every ratio the DLSS page offers, and what the gate says about it.

    The frontend's fixed buttons and the gate have to agree in both directions: a
    button the backend refuses is a control that lies. The page offers six presets
    out of the seventy-six the range contains, so what is checked here is that each
    of them is served and classified the way its own extents say, including the
    half, which takes the constant-tap pair rather than a plan.
    """
    served, refused = {0.75, 0.67, 0.63, 0.5, 0.25, 1.0}, set()
    for scale in sorted(served | refused, reverse=True):
        negotiation = _negotiate(job=dict(GPU_JOB, working_scale=scale),
                                 dims=_dims_for(scale), capabilities=SCALED)
        assert negotiation["supported"] == (scale in served), (
            scale, negotiation["reasons"])
        assert negotiation["downscaled"] == (scale != 1.0), scale
        assert negotiation["plan_driven"] == (scale not in (0.5, 1.0)), scale


def test_each_served_plan_driven_ratio_is_classified_at_its_resolved_size():
    """0.67 resolves to 2576x1448, 0.63 to 2416x1360 and 0.25 to 960x544.

    The numbers the tap plans are built for, therefore the numbers the chain has
    to be opened at: 8-alignment and the CPU plan's own rounding make both, and
    no stage hard-codes them.
    """
    for scale, size in ((0.75, (2880, 1616)), (0.67, (2576, 1448)),
                        (0.63, (2416, 1360)), (0.25, (960, 544))):
        assert pipeline.resolve_working_dims(3840, 2160, scale) == size, scale
        negotiation = _negotiate(job=dict(GPU_JOB, working_scale=scale),
                                 dims=_dims_for(scale), capabilities=SCALED)
        assert negotiation["supported"], (scale, negotiation["reasons"])
        assert negotiation["downscaled"] and negotiation["plan_driven"]
        assert negotiation["working_scale"] == scale


ALL_PERCENTS = list(range(gtx.WORKING_PERCENT_MIN, gtx.WORKING_PERCENT_MAX + 1))


@pytest.mark.parametrize("percent", ALL_PERCENTS)
def test_every_whole_percentage_is_served_on_its_own_extents(percent):
    """The range the gate promises, walked one percentage at a time.

    Each request is asked only what the gate decides on: the size the ratio
    resolves to through the one resolver, which of the three chain shapes that
    size describes, and whether the tap window that shape needs is inside what the
    bridge kernel walks. Nothing here is a membership test - 26%, 40% and 99% are
    served by the same arithmetic as 50% and 75%, and the two shapes that are not
    plan-driven at 4K (the half, which halves this source exactly, and 100%) are
    named by the extents rather than by a list.
    """
    scale = percent / 100.0
    dims = _dims_for(scale)
    negotiation = _negotiate(job=dict(GPU_JOB, working_scale=scale), dims=dims,
                             capabilities=SCALED)
    assert negotiation["supported"], (percent, negotiation["reasons"])
    assert (dims["live_w"], dims["live_h"]) == pipeline.resolve_working_dims(
        dims["src_w"], dims["src_h"], scale)
    assert negotiation["working_scale"] == scale
    assert negotiation["downscaled"] == (percent != 100)
    # 4K halves exactly, so 50% is the constant-tap pair and 100% is the identity.
    assert negotiation["plan_driven"] == (percent not in (50, 100))
    if negotiation["plan_driven"]:
        working_color, upscale = gtx.plan_tap_widths(
            dims["src_w"], dims["src_h"], dims["live_w"], dims["live_h"])
        assert working_color <= gtx.LANCZOS_MAX_TAPS, (percent, working_color)
        assert upscale <= gtx.RESIDUAL_MAX_TAPS, (percent, upscale)


def test_no_resolved_size_at_any_whole_percentage_outgrows_the_kernel_window():
    """Why LANCZOS_MAX_TAPS is 24 rather than a figure measured per ratio.

    A Lanczos-2 resample of ``src`` source pixels onto ``dst`` destination pixels
    is ``floor(4*src/dst) + 1`` taps wide at its centre row, so the whole question
    is how small ``dst/src`` can get. The resolver's sixteen-pixel floor only
    reaches under src = 48, where the ratio is over a third; above it the axis is
    8*round(src*r/8) with r >= 1/4, and the rounding is only paid away at a half
    step - src = 32k + 16 rounds down to 8k, a ratio of k/(4k+2), whose least
    value past the floor is k = 2: 80 -> 16, exactly a fifth, 21 taps. That is the
    walk a 16-wide kernel could not do and this one can, with three columns spare.

    Every even axis the products use, and every percentage, is then asked of the
    plan builder's own window rather than of this argument.
    """
    axes = list(range(16, 258, 2)) + [480, 720, 1080, 1280, 1440, 1920, 2160,
                                      2576, 2880, 3840, 3842, 4092, 7680]
    worst = (0, None)
    for src in axes:
        for percent in range(gtx.WORKING_PERCENT_MIN, gtx.WORKING_PERCENT_MAX):
            dst = pipeline.resolve_working_dims(src, src, percent / 100.0)[0]
            taps = pipeline.lanczos2_axis_taps(src, dst)
            assert taps <= gtx.LANCZOS_MAX_TAPS, (src, dst, percent, taps)
            assert dst / src >= 0.2 - 1e-12, (src, dst, percent)
            if taps > worst[0]:
                worst = (taps, (src, dst, percent))
    assert worst == (21, (80, 16, 25)), worst
    # The residual side needs no such argument: a Catmull-Rom row is four taps.
    assert pipeline.CATMULL_ROM_TAPS == gtx.RESIDUAL_MAX_TAPS


def test_a_plan_wider_than_the_narrower_kernel_window_is_refused_before_it_runs():
    """What happens when a ratio's plan outgrows what a build's kernel walks.

    The window a bridge can read is a property of its shader, so the seam is the
    figure the gate compares against - which the gate takes from the loaded
    bridge. Report a narrower one and an otherwise accepted ratio has to be
    refused by name rather than truncated into a different filter or left to fail
    at the open after the job has started.
    """
    dims, job = _dims_for(0.75), dict(GPU_JOB, working_scale=0.75)
    assert _negotiate(job=job, dims=dims, capabilities=SCALED)["supported"]
    taps = gtx.plan_tap_widths(dims["src_w"], dims["src_h"], dims["live_w"],
                               dims["live_h"])[0]
    negotiation = _negotiate(job=job, dims=dims,
                             capabilities=dict(SCALED, lanczos_max_taps=taps - 1))
    assert not negotiation["supported"]
    reason = " ".join(negotiation["reasons"])
    assert f"is {taps} taps wide" in reason, reason
    assert f"past the {taps - 1}" in reason and "refused" in reason
    # The exact-2x and identity chains carry their taps in the shaders and build
    # no plan, so a window this ratio outgrows says nothing about them.
    assert _negotiate(job=GPU_JOB, dims=DIMS,
                      capabilities=SCALED)["supported"]


def test_the_window_a_bridge_reports_is_the_window_a_request_has_to_fit():
    """A build whose kernel is narrower than this module's mirror is asked, not assumed.

    3842x2162 at a quarter resolves to the same 960x544 as a clean 4K but needs 17
    taps, because 8-alignment rounds one axis away from a quarter and the support
    grows with it. That is inside what a 24-wide kernel walks and outside what a
    16-wide one does, so the figure the loaded DLL volunteers decides the answer -
    which is what keeps a stale gpu_lanczos_bridge.dll from being handed a table it
    would sum the tail of as garbage.
    """
    src_w, src_h = 3842, 2162
    dims = _dims_for(0.25, src_w=src_w, src_h=src_h)
    assert (dims["live_w"], dims["live_h"]) == (960, 544)
    taps = gtx.plan_tap_widths(src_w, src_h, dims["live_w"], dims["live_h"])[0]
    assert taps == 17, taps
    job = dict(GPU_JOB, working_scale=0.25)
    source = dict(SOURCE, width=src_w, height=src_h)
    assert _negotiate(job=job, dims=dims, source=source,
                      capabilities=SCALED)["supported"], "24 taps serve 17"
    narrow = dict(SCALED, lanczos_max_taps=16)
    negotiation = _negotiate(job=job, dims=dims, source=source,
                             capabilities=narrow)
    assert not negotiation["supported"], negotiation["reasons"]
    reason = " ".join(negotiation["reasons"])
    assert f"is {taps} taps wide" in reason, reason
    assert "past the 16" in reason and "refused" in reason
    # The figure a build reports outranks this module's mirror of the header: a
    # kernel wider than this checkout describes is still served, not refused
    # against a number the loaded binary never gave.
    wider = dict(SCALED, lanczos_max_taps=gtx.LANCZOS_MAX_TAPS + 1)
    assert _negotiate(job=job, dims=dims, source=source,
                      capabilities=wider)["supported"], wider


def test_a_build_that_reports_no_window_is_measured_at_the_16_it_walks():
    """The gap the getter closes: absence of the question is not an answer of 24.

    gpu_lanczos_bridge added its plan-driven open under ABI 2 with a 16-wide
    window and added gpu_lanczos_max_taps when it widened that window to 24, so a
    loaded bridge that has the open and answers nothing is that earlier build -
    and the only bound a request can be measured against is the width its own
    kernel was compiled with. This module's LANCZOS_MAX_TAPS describes this
    checkout, so answering a question the DLL never got would hand a 17..21 tap
    plan to a 16-wide kernel and let the job decode frames before the open refused
    it.
    """
    assert gtx.LEGACY_LANCZOS_MAX_TAPS == 16
    assert gtx.LEGACY_LANCZOS_MAX_TAPS < gtx.LANCZOS_MAX_TAPS
    src_w, src_h = 3842, 2162
    dims = _dims_for(0.25, src_w=src_w, src_h=src_h)
    taps = gtx.plan_tap_widths(src_w, src_h, dims["live_w"], dims["live_h"])[0]
    assert taps == 17, taps
    job = dict(GPU_JOB, working_scale=0.25)
    source = dict(SOURCE, width=src_w, height=src_h)
    assert "lanczos_max_taps" not in LEGACY_SCALED
    negotiation = _negotiate(job=job, dims=dims, source=source,
                             capabilities=LEGACY_SCALED)
    assert not negotiation["supported"], negotiation["reasons"]
    reason = " ".join(negotiation["reasons"])
    assert f"is {taps} taps wide" in reason, reason
    assert "past the 16" in reason and "refused" in reason, reason
    assert "no gpu_lanczos_max_taps" in reason, reason
    # A clean 4K at a quarter is 16 taps, and a whole percentage below it is
    # fewer: the plan-driven range a 16-wide kernel can read is still served by
    # the old build rather than refused for the window it never reported.
    for scale, expected in ((0.25, 16), (0.26, 16), (0.30, 14), (0.75, 6)):
        clean = _dims_for(scale)
        width = gtx.plan_tap_widths(clean["src_w"], clean["src_h"],
                                    clean["live_w"], clean["live_h"])[0]
        assert width == expected, (scale, width)
        served = _negotiate(job=dict(GPU_JOB, working_scale=scale), dims=clean,
                            capabilities=LEGACY_SCALED)
        assert served["supported"], (scale, width, served["reasons"])
    # And the bound is the binary's, not a pair of hard-coded cases: a build that
    # answers 17 serves the same 17-tap request this fixture refuses.
    reported_17 = dict(LEGACY_SCALED, lanczos_max_taps=17)
    assert _negotiate(job=dict(GPU_JOB, working_scale=0.25),
                      dims=_dims_for(0.25, src_w=3842, src_h=2162),
                      source=dict(SOURCE, width=3842, height=2162),
                      capabilities=reported_17)["supported"]


# 3840x2160 at 0.75 through the one function that resolves working sizes. This is
# 2880x1616, not the 2880x1620 the ratio alone suggests: the rule that rounds to
# eight is the CPU plan's, and the GPU chain follows it rather than inventing a
# second answer to the same question.
DIMS_75 = {"src_w": 3840, "src_h": 2160, "live_w": 2880, "live_h": 1616,
           "dst_w": 3840, "dst_h": 2160, "fps": 29.97}


def test_the_quarter_resolution_scale_is_served_by_the_plan_driven_stages():
    live_w, live_h = pipeline.resolve_working_dims(3840, 2160, 0.75)
    assert (live_w, live_h) == (2880, 1616)
    negotiation = _negotiate(job=dict(GPU_JOB, working_scale=0.75), dims=DIMS_75,
                             capabilities=SCALED)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["downscaled"] and negotiation["plan_driven"]
    assert negotiation["working_scale"] == 0.75


# Percentages sampled across the plan-driven part of the range: the deepest
# window, the shallowest, the three that used to be the whole served set, and one
# that was refused by the whitelist this replaced. 50 and 100 are the other two
# chain shapes and are asked about separately.
PLAN_DRIVEN_SAMPLE = (0.25, 0.40, 0.63, 0.67, 0.75, 0.99)


@pytest.mark.parametrize("scale", PLAN_DRIVEN_SAMPLE)
def test_a_scale_the_constant_tables_cannot_describe_needs_the_plan_exports(scale):
    """An older bridge set is a valid partner for 0.5 and 1.0 and not for a ratio
    whose taps it cannot be given.

    The refusal has to name what is missing and the scale it is missing for, and
    say that the other chains still work, because the caller's own fix is a
    rebuild of one DLL, not a change to the job. Every sampled percentage is asked
    the same question, so a ratio added to the range without its refusal path is
    not an option.
    """
    dims = _dims_for(scale)
    for capabilities, missing in (({}, "gpu_lanczos_bridge and gpu_residual_bridge"),
                                  ({"lanczos_scaled": True}, "gpu_residual_bridge"),
                                  ({"residual_scaled": True}, "gpu_lanczos_bridge")):
        negotiation = _negotiate(job=dict(GPU_JOB, working_scale=scale),
                                 dims=dims, capabilities=capabilities)
        assert not negotiation["supported"], capabilities
        reason = " ".join(negotiation["reasons"])
        assert missing in reason, (capabilities, reason)
        assert f"working_scale={scale:g}" in reason, reason
        assert "plan-driven open" in reason and "rebuild" in reason
        # The other two shapes are not held hostage to a scale this build cannot
        # open: the same libraries are still a valid pair for them.
        for other_job, other_dims in ((GPU_JOB, DIMS), (GPU_JOB_FULL, DIMS_FULL)):
            assert _negotiate(job=other_job, dims=other_dims,
                              capabilities=capabilities)["supported"]


def test_the_quarter_resolution_scale_follows_the_resolved_size_not_the_ratio():
    """2880x1620 is what 0.75 looks like, and is not what 0.75 resolves to.

    A caller that computed its own working size would be handed a plan whose axes
    disagree with the frame the decoder reports, so the gate refuses the pair
    rather than opening a bridge that would sample outside the working plane.
    """
    negotiation = _negotiate(job=dict(GPU_JOB, working_scale=0.75),
                             dims=dict(DIMS_75, live_h=1620),
                             capabilities=SCALED)
    assert not negotiation["supported"]
    assert any("resolves to 2880x1620" in reason
               for reason in negotiation["reasons"])


def _serialized_axis(view, axis):
    """One axis of a serialized tap table, as the rows the bridge uploads them in."""
    rows = (view.vertical_dst_len if axis == "vertical"
            else view.horizontal_dst_len)
    indices, weights = ((view.vertical_indices, view.vertical_weights)
                        if axis == "vertical"
                        else (view.horizontal_indices, view.horizontal_weights))
    return (np.ctypeslib.as_array(indices, shape=(rows, view.taps)),
            np.ctypeslib.as_array(weights, shape=(rows, view.taps)))


@pytest.mark.parametrize("scale", PLAN_DRIVEN_SAMPLE)
def test_the_serialized_tables_are_the_cpu_plans_for_those_extents(scale):
    """What the plan-driven opens hand a bridge, across the sampled range.

    The claim behind a plan-driven stage is that there is one resampler rather
    than a second one tuned to agree with the first, so the comparison is against
    ``pipeline``'s own AxisPlans for the same extents: same indices, same
    weights, and the columns the shared window adds carrying weight 0. The widths
    the gate compared against the kernel limits are the widths of these tables,
    which is what keeps the gate's arithmetic and the serializer's one answer.
    """
    dims = _dims_for(scale)
    src_w, src_h = dims["src_w"], dims["src_h"]
    live_w, live_h = dims["live_w"], dims["live_h"]
    widths = gtx.plan_tap_widths(src_w, src_h, live_w, live_h)
    plans = gtx.TapPlans(src_w, src_h, live_w, live_h)
    assert (plans.working_color.taps, plans.upscale.taps) == widths

    for view, axis, cpu in (
            (plans.working_color, "vertical",
             pipeline._build_lanczos2_axis_plan(src_h, live_h)),
            (plans.working_color, "horizontal",
             pipeline._build_lanczos2_axis_plan(src_w, live_w)),
            (plans.upscale, "vertical",
             pipeline._build_catmull_rom_axis_plan(live_h, src_h)),
            (plans.upscale, "horizontal",
             pipeline._build_catmull_rom_axis_plan(live_w, src_w))):
        indices, weights = _serialized_axis(view, axis)
        assert (view.vertical_src_len if axis == "vertical"
                else view.horizontal_src_len) == cpu.src_len
        assert (view.vertical_dst_len if axis == "vertical"
                else view.horizontal_dst_len) == cpu.dst_len
        assert np.array_equal(indices[:, :cpu.max_taps], cpu.indices), (scale, axis)
        assert np.array_equal(weights[:, :cpu.max_taps], cpu.weights), (scale, axis)
        assert not np.any(weights[:, cpu.max_taps:]), (scale, axis)


def test_the_kernel_window_the_gate_compares_against_is_the_window_the_shader_walks():
    """GPU_*_MAX_TAPS in the headers and TAPS in the plan kernels are one number.

    The negotiation-time check refuses a ratio whose plan is wider than the
    kernel walks, and this module's LANCZOS_MAX_TAPS is the mirror of the kernel
    this checkout compiles - the figure a request is measured against when the
    loaded bridge answers for itself. The mirror is only a description of a kernel
    until the kernel itself is read; a widened header over an unmodified shader
    would hand a bridge a table it sums the tail of as garbage, and a build that
    reports nothing is measured at LEGACY_LANCZOS_MAX_TAPS instead of at this one.
    """
    def kernel_window(name, marker):
        source = open(os.path.join(os.path.dirname(_ROOT), "native", name),
                      encoding="utf-8").read()
        scaled = source.split(marker, 1)[1]
        declared = scaled.split("#define TAPS ", 1)[1].split("u", 1)[0]
        return int(declared)

    assert kernel_window("gpu_lanczos_bridge.cpp",
                         "static constexpr char scaled_shader[]") == \
        gtx.LANCZOS_MAX_TAPS
    assert kernel_window("gpu_residual_bridge.cpp",
                         "static constexpr char plan_shader[]") == \
        gtx.RESIDUAL_MAX_TAPS


def test_a_plan_wider_than_the_bridge_kernel_is_refused_not_truncated():
    """The window the 0.75 chain builds is well inside what the kernels walk.

    A plan that needed more is a different filter, so the serializer that hands
    one to a bridge has to say so rather than hand over the first 24 taps.
    """
    view, buffers = gtx.working_color_plan(3840, 2160, 2880, 1616)
    assert view.taps == pipeline._build_lanczos2_axis_plan(
        2160, 1616).max_taps
    assert view.taps <= gtx.LANCZOS_MAX_TAPS
    # The deepest resample a served percentage can ask for on a real source is the
    # 8-alignment accident at 80 -> 16, and one step past the kernel's window is
    # refused rather than silently truncated.
    assert pipeline.lanczos2_axis_taps(80, 16) == 21
    assert 21 <= gtx.LANCZOS_MAX_TAPS
    with pytest.raises(gtx.BridgeError) as raised:
        gtx.working_color_plan(2160, 2160, 216, 216)
    assert f"wider than the {gtx.LANCZOS_MAX_TAPS} taps" in str(raised.value)


def test_the_half_plan_the_serializer_builds_is_the_half_table_in_the_kernel():
    """0.5 must stay one filter, not two that happen to agree.

    The exact-2x kernel walks eight taps of a symmetric table it hard-codes as
    bit patterns. Read those patterns back and compare them with the plan the
    serializer would hand a plan-driven kernel for the same extents: equality
    here is what lets the two opens claim the same resampler.
    """
    source = os.path.join(os.path.dirname(os.path.abspath(gtx.__file__)),
                          "native", "gpu_lanczos_bridge.cpp")
    with open(source, encoding="utf-8") as handle:
        source = handle.read()
    helper = source.split("float wt(uint i) {", 1)[1].split("}", 1)[0]
    paired = re.findall(r"return asfloat\(0x([0-9a-fA-F]{8})\)", helper)
    assert len(paired) == 4, paired
    outer, next_, inner, centre = (np.uint32(int(h, 16)).view(np.float32)
                                   for h in paired)
    table = np.array([outer, next_, inner, centre,
                      centre, inner, next_, outer], dtype=np.float32)
    plan = pipeline._build_lanczos2_axis_plan(8, 4)
    assert np.array_equal(plan.weights, np.tile(table, (4, 1)))
    view, buffers = gtx.working_color_plan(8, 8, 4, 4)
    assert view.taps == 8
    vi, vw, hi, hw = buffers
    rows = np.ctypeslib.as_array(
        view.vertical_weights, shape=(4, view.taps))
    assert np.array_equal(rows, np.tile(table, (4, 1)))
    assert np.array_equal(vw, rows) and np.array_equal(hw, rows)
    cols = np.ctypeslib.as_array(
        view.vertical_indices, shape=(4, view.taps))
    assert np.array_equal(cols, plan.indices) and np.array_equal(vi, cols)
    assert np.array_equal(hi, cols)


def test_the_half_residual_table_in_the_kernel_is_the_catmull_rom_plan():
    """The other half of the same claim, on the stage that reconstructs.

    The exact-2x residual kernel hard-codes two rows of four weights and picks one
    by the output pixel's parity. Read them back and compare against the plan the
    serializer would hand the plan-driven kernel for the same 4->8 axis: same
    weights, and the window the kernel starts its four taps at is the plan's
    ``center - 1`` shifted by the same amount the parity trick uses. This is what
    lets 50% keep its constant taps without becoming a second resample.
    """
    source = os.path.join(os.path.dirname(os.path.abspath(gtx.__file__)),
                          "native", "gpu_residual_bridge.cpp")
    with open(source, encoding="utf-8") as handle:
        source = handle.read()
    kernel = source.split("static constexpr char shader[]", 1)[1]
    helper = kernel.split("float weight(uint phase, uint tap) {", 1)[1]
    helper = helper.split("float3 rgb(", 1)[0]
    values = [float(value) for value in re.findall(r"return (-?\d+\.\d+);", helper)]
    assert len(values) == 8, values
    tables = [values[:4], values[4:]]
    # The kernel's own window start, per parity: base = (x >> 1) - (px == 0 ? a : b),
    # written the same way on both axes.
    windows = re.findall(r"p[xy] == 0 \? (\d+) : (\d+)", kernel)
    assert windows == [("2", "1"), ("2", "1")], windows
    offsets = tuple(int(value) for value in windows[0])

    src_len, dst_len = 4, 8
    plan = pipeline._build_catmull_rom_axis_plan(src_len, dst_len)
    assert plan.max_taps == gtx.RESIDUAL_MAX_TAPS
    expected_weights = np.array([tables[d % 2] for d in range(dst_len)],
                                dtype=np.float32)
    assert np.array_equal(plan.weights, expected_weights)

    base = np.array([d // 2 - offsets[d % 2] for d in range(dst_len)])
    expected_indices = np.clip(base[:, None] + np.arange(4)[None, :],
                               0, src_len - 1).astype(np.int32)
    assert np.array_equal(plan.indices, expected_indices)


def test_dims_that_disagree_with_the_stated_scale_are_refused():
    """The resolved extents decide, so a scale text that lies about them is caught."""
    for job, dims in ((GPU_JOB_FULL, DIMS), (GPU_JOB, DIMS_FULL)):
        negotiation = _negotiate(job=job, dims=dims)
        assert not negotiation["supported"], (job["working_scale"], dims)
        assert any("resolves to" in reason for reason in negotiation["reasons"])


# ---------------------------------------------------------------------------
# encoder options: applied under the name the command line used, or refused
# ---------------------------------------------------------------------------

def plan(transcode, source=None, option_available=None, tune_available=None):
    return gtx.plan_encoder(transcode, 24, dict(source or SOURCE),
                            option_available, tune_available)


def test_the_default_plan_is_the_default_command():
    """No transcode_settings means the same nine choices build_transcode_command makes."""
    made = plan({})
    assert made["unsupported"] == []
    struct = made["struct"]
    assert struct["codec"] == b"av1_nvenc" and struct["preset"] == b"p7"
    assert struct["tune"] == b"hq" and struct["multipass"] == b"qres"
    assert struct["rate_control"] == b"vbr" and struct["constant_quality"] == 24
    assert struct["bitrate"] == 0 and struct["gop"] == 60
    assert struct["color_matrix"] == gtx.COLOR_MATRIX_BT601


def test_settings_the_encoder_has_are_passed_through_under_their_own_names():
    made = plan({"b_frames": "2", "lookahead": "8", "weighted_prediction": True,
                 "temporal_filter": "2", "profile": "main"})
    options = dict(made["av_options"])
    assert options["bf"] == "2" and options["rc-lookahead"] == "8"
    assert options["tf_level"] == "2" and options["profile"] == "main"
    assert options["weighted_pred"] == "1"


def test_an_option_the_encoder_does_not_have_is_refused_not_dropped():
    def only_some(codec, name):
        return name != "bf"

    made = plan({"b_frames": "2"}, option_available=only_some)
    assert ("bf", "2") not in made["av_options"]
    assert any("has no option bf" in reason for reason in made["unsupported"]), made["unsupported"]


def test_a_setting_with_no_encoder_equivalent_at_all_is_refused():
    made = plan({"made_up_key": "1"})
    assert any("made_up_key" in reason and "no equivalent" in reason
               for reason in made["unsupported"]), made["unsupported"]


def test_container_only_settings_are_not_mistaken_for_refusals():
    """Audio, subtitle and metadata clauses belong to the copy pass, not to NVENC."""
    made = plan({"audio_mode": "aac", "subtitle_mode": "copy", "container": ".mp4",
                 "extra_args": "-movflags +faststart", "resolution_mode": "auto"})
    assert made["unsupported"] == [], made["unsupported"]


def test_the_rate_modes_take_the_values_the_command_line_takes():
    cbr = plan({"rate_mode": "cbr", "video_bitrate": "50M"})
    assert cbr["struct"]["rate_control"] == b"cbr"
    assert dict(cbr["av_options"]) == {"b": "50M", "maxrate": "50M",
                                       "bufsize": "50M", "spatial-aq": "1",
                                       "temporal-aq": "0", "aq-strength": "8"}
    qp = plan({"rate_mode": "constqp", "qp": 18})
    assert qp["struct"]["rate_control"] == b"constqp"
    assert ("qp", "18") in qp["av_options"]
    assert plan({"rate_mode": "crf"})["unsupported"] == ["rate_mode=crf"]


def test_uhq_is_downgraded_only_where_the_encoder_cannot_have_it():
    """The legacy command asks FFmpeg; this asks the same question through a callback."""
    assert plan({"tune": "uhq"}, tune_available=lambda codec: True)["struct"]["tune"] == b"uhq"
    assert plan({"tune": "uhq"}, tune_available=lambda codec: False)["struct"]["tune"] == b"hq"
    assert plan({"tune": "uhq", "codec": "h264_nvenc"},
                tune_available=lambda codec: True)["struct"]["tune"] == b"hq"


def test_an_option_value_cannot_smuggle_a_second_option_into_the_blob():
    with pytest.raises(gtx.BridgeError):
        gtx.extra_options_blob([("b", "8M\nqp=1")])
    assert gtx.extra_options_blob([("bf", "2"), ("b", "8M")]) == b"bf=2\nb=8M"


# ---------------------------------------------------------------------------
# container, audio and mapping
# ---------------------------------------------------------------------------

def _in_order(command, clauses):
    """Whether each clause appears at or after the last one's position."""
    position = 0
    for clause in clauses:
        found = -1
        for start in range(position, len(command) - len(clause) + 1):
            if command[start:start + len(clause)] == clause:
                found = start
                break
        assert found >= 0, f"{clause!r} missing from {command}"
        position = found + len(clause)
    return True


def test_the_copy_pass_reapplies_the_mapping_clauses_without_reencoding():
    cmd = gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mp4",
        {"codec": "hevc_nvenc", "audio_mode": "opus", "audio_bitrate": "160k",
         "subtitle_mode": "convert", "extra_args": "-movflags +faststart"},
        "out.mp4", True)
    assert cmd[0] == "ffmpeg" and "-y" in cmd and cmd[-1] == "out.mp4"
    head = cmd[:cmd.index("out.mp4")]
    _in_order(head, [["-i", "enc.mkv"], ["-i", "in.mp4"],
                     ["-map", "0:v:0"], ["-c:v", "copy"],
                     ["-map", "1:a?"], ["-map", "1:s?"],
                     ["-map_metadata", "1"], ["-map_chapters", "1"],
                     ["-c:a", "libopus", "-b:a", "160k", "-shortest"],
                     ["-c:s", "mov_text"], ["-tag:v", "hvc1"],
                     ["-movflags", "+faststart"]])
    # Input 0 is the encode, so the video is never decoded and encoded again.
    assert head[head.index("-c:v") + 1] == "copy"
    assert head.count("-movflags") == 2, "the request's own flag survives next to ours"


def test_video_authoritative_copy_pass_is_bounded_by_encoded_video_span():
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", {"audio_mode": "copy"}, "out.mkv",
        True, video_authoritative=True, video_duration="0.12")
    assert "-shortest" not in command
    assert command[command.index("-c:a") + 1] == "copy"
    assert command[-3:] == ["-t", "0.12", "out.mkv"]
    assert ["-map", "0:v:0"] in [command[i:i + 2]
                                  for i in range(len(command) - 1)]
    assert "-map_metadata" in command and "-map_chapters" in command


def test_video_authoritative_rejects_extra_shortest_without_changing_legacy_defaults():
    with pytest.raises(gtx.BridgeError, match="timeline"):
        gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv",
            {"extra_args": "-shortest"}, "out.mkv", True,
            video_authoritative=True, video_duration="0.12")
    assert "-shortest" in gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv",
                                              {}, "out.mkv", True)


@pytest.mark.parametrize("failure", [RuntimeError("missing muxer"), pipeline.JobCancelled("cancelled")])
def test_sdr_export_owns_intermediate_on_post_encode_failure(monkeypatch, tmp_path, failure):
    intermediate = tmp_path / "temporary-encode.mkv"
    def reserve(**kwargs):
        descriptor = os.open(intermediate, os.O_CREAT | os.O_RDWR)
        return descriptor, str(intermediate)
    def fail_after_encode(*args, **kwargs):
        assert kwargs["intermediate"] == str(intermediate)
        intermediate.write_bytes(b"finished encode")
        raise failure
    monkeypatch.setattr(gtx.tempfile, "mkstemp", reserve)
    monkeypatch.setattr(gtx, "_export_gpu_transfer", fail_after_encode)
    with pytest.raises(type(failure)):
        gtx.export_gpu_transfer("source.mkv", None, None, "final.mkv", {}, 3,
            18, True, {}, {}, {"copy_pass": True})
    assert not intermediate.exists()


def test_video_authoritative_pass_supports_no_audio_or_subtitle_streams():
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "silent.mkv", {"subtitle_mode": "drop"},
        "out.mkv", with_audio=False, video_authoritative=True,
        video_duration="0.12")
    assert "-an" in command and "-sn" in command
    assert "1:a?" not in command and "1:s?" not in command
    assert "-shortest" not in command
    assert command[-3:] == ["-t", "0.12", "out.mkv"]


@pytest.mark.parametrize("mode", gtx.MUX_AUDIO_MODES)
def test_video_authoritative_audio_modes_never_add_shortest(mode):
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", {"audio_mode": mode}, "out.mkv",
        True, video_authoritative=True, video_duration="1.25")
    assert "-shortest" not in command
    assert command[-3:] == ["-t", "1.25", "out.mkv"]


@pytest.mark.parametrize("duration", [None, "0", "-1", "nan", "inf", "bad"])
def test_video_authoritative_requires_a_positive_finite_video_duration(duration):
    with pytest.raises(gtx.BridgeError):
        gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv", {}, "out.mkv",
                              video_authoritative=True, video_duration=duration)


def test_encoded_video_span_uses_packet_pts_duration_and_rational_timebase(
        monkeypatch):
    answer = {"streams": [{"time_base": "1/1000"}], "packets": [
        {"pts": 5080, "duration": 40}, {"pts": 5000, "duration": 40},
        {"pts": 5040, "duration": 40}]}
    monkeypatch.setattr(gtx.subprocess, "check_output",
                        lambda *args, **kwargs: json.dumps(answer).encode())
    assert gtx.encoded_video_span("ffprobe", "encoded.mkv") == "0.12"


def test_encoded_video_span_refuses_an_unbounded_last_packet(monkeypatch):
    answer = {"streams": [{"time_base": "1/1000"}], "packets": [
        {"pts": 0, "duration": 40}, {"pts": 40, "duration": 0}]}
    monkeypatch.setattr(gtx.subprocess, "check_output",
                        lambda *args, **kwargs: json.dumps(answer).encode())
    with pytest.raises(gtx.BridgeError, match="timestamps or durations"):
        gtx.encoded_video_span("ffprobe", "encoded.mkv")


def test_a_removed_audio_track_is_an_an_flag_rather_than_an_empty_map():
    cmd = gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mp4",
                                {"audio_mode": "remove"}, "out.mkv", True)
    assert "-an" in cmd and "1:a?" not in cmd


def test_the_copy_pass_is_skipped_only_when_the_source_has_nothing_to_carry():
    quiet = dict(SOURCE, audio_streams=0, subtitle_streams=0, metadata=False)
    assert not gtx.needs_copy_pass(quiet, {}, True, "out.mkv")
    assert gtx.needs_copy_pass(SOURCE, {}, True, "out.mkv"), "source has audio"
    assert gtx.needs_copy_pass(quiet, {}, True, "out.mp4"), "container is not the encoder's"
    assert gtx.needs_copy_pass(quiet, {"copy_attachments": False,
                                       "audio_mode": "remove"}, True, "out.mkv") is False


@pytest.mark.parametrize("mode", gtx.MUX_AUDIO_MODES)
def test_the_audio_modes_a_gate_is_told_to_admit_are_the_modes_the_pass_expresses(
        mode):
    """The shared vocabulary and ``build_mux_command`` cannot drift apart.

    A gate elsewhere in the project admits or refuses a mode by this name list, so the
    list is only honest if the builder really expresses every one of them - and refuses
    the ones outside it instead of quietly encoding something else.
    """
    cmd = gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv",
                                {"audio_mode": mode}, "out.mkv", True)
    if mode == "remove":
        assert "-an" in cmd and "-c:a" not in cmd and "1:a?" not in cmd
    else:
        assert cmd[cmd.index("-c:a") + 1] == {
            "copy": "copy", "aac": "aac", "opus": "libopus", "flac": "flac",
            "pcm": "pcm_s24le"}[mode]
    with pytest.raises(gtx.BridgeError):
        gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv",
                              {"audio_mode": "mp3"}, "out.mkv", True)


@pytest.mark.parametrize("mode", gtx.MUX_SUBTITLE_MODES)
def test_the_subtitle_modes_a_gate_is_told_to_admit_are_the_modes_the_pass_expresses(
        mode):
    cmd = gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv",
                                {"subtitle_mode": mode}, "out.mkv", True)
    if mode == "drop":
        assert "-sn" in cmd and "-c:s" not in cmd
    else:
        assert cmd[cmd.index("-c:s") + 1] == ("copy" if mode == "copy" else "srt")


# --- the probe, and what it is willing to say about chapters -------------------

# A Matroska file with two real chapters and no ``nb_chapters`` in its format section:
# what ffprobe actually answers for this project's HDR fixture, not a shape invented
# here. Reading the format field as the count is how a chapter-only request ended up
# with no copy pass and no chapters.
PROBE_ANSWER = {
    "streams": [{
        "codec_type": "video", "codec_name": "hevc", "pix_fmt": "yuv420p10le",
        "width": 1920, "height": 1080, "color_range": "tv",
        "color_space": "bt2020nc", "color_transfer": "smpte2084",
        "color_primaries": "bt2020", "r_frame_rate": "25/1",
        "tags": {"ENCODER": "Lavc61.19.101 libx265",
                 "DURATION": "00:00:00.240000000", "title": "Master shot",
                 "language": "jpn"},
        "disposition": {"default": 1, "dub": 0, "original": 1, "comment": 0},
    }],
    "format": {"tags": {"title": "HDR remux fixture"}},
    "chapters": [{"id": 1, "tags": {"title": "chapter one"}},
                 {"id": 2, "tags": {"title": "chapter two"}}],
}


@pytest.fixture
def fake_ffprobe(monkeypatch):
    """Answer one probe and record the argv it was asked with."""
    seen = []

    def install(answer):
        def check_output(argv, **kwargs):
            seen.append(list(argv))
            return json.dumps(answer).encode("utf-8")

        monkeypatch.setattr(gtx.subprocess, "check_output", check_output)
        monkeypatch.setattr(pipeline, "_bundle_ffprobe", lambda: "ffprobe")
        return seen
    return install


def test_the_default_probe_still_asks_ffprobe_for_streams_and_format_only(
        fake_ffprobe):
    """The SDR chain's call is the call it has always made, and answers the same.

    ``nb_chapters`` stays the format field for a caller that did not ask for the
    chapter list: the point of the opt-in below is not to change what the SDR gate
    admits, and a job that runs on it today has to admit the same thing tomorrow.
    """
    seen = fake_ffprobe(PROBE_ANSWER)
    source = gtx.probe_source("in.mkv")
    assert seen == [["ffprobe", "-v", "error", "-print_format", "json",
                     "-show_streams", "-show_format", "in.mkv"]]
    assert source["nb_chapters"] == 0, "the format field is absent, as it is for real MKV"
    assert "video_tags" not in source and "video_disposition" not in source
    assert source["metadata"] is True and source["video_stream_count"] == 1


def test_a_probe_that_asks_for_chapters_counts_the_list_it_was_answered(fake_ffprobe):
    """Two chapters is two, whatever the format section says about it."""
    seen = fake_ffprobe(PROBE_ANSWER)
    source = gtx.probe_source("in.mkv", include_chapters=True,
                              include_video_metadata=True)
    assert seen[0][-2:] == ["-show_chapters", "in.mkv"]
    assert "-show_streams" in seen[0] and "-show_format" in seen[0]
    assert source["nb_chapters"] == 2
    assert source["video_tags"]["ENCODER"] == "Lavc61.19.101 libx265"
    assert source["video_disposition"] == {"default": 1, "original": 1}


def test_a_chapter_only_request_is_admitted_onto_the_copy_pass_by_the_true_count():
    """The admission that loses chapters, and the one that does not.

    Audio removed, subtitles dropped, attachments and format metadata off and only
    chapters kept: every clause of ``needs_copy_pass`` but the chapter one says no, so
    the whole request rests on the count. With the count the default probe reports it is
    a deliverable with no chapters and no word about it; with the count the HDR route
    asks for it is a copy pass.
    """
    chapters_only = {"audio_mode": "remove", "subtitle_mode": "drop",
                     "copy_attachments": False, "copy_metadata": False,
                     "copy_chapters": True}
    probed_without_chapters = {"nb_chapters": 0}
    probed_with_them = {"nb_chapters": 2}
    assert gtx.needs_copy_pass(probed_without_chapters, chapters_only,
                               False, "out.mkv") is False
    assert gtx.needs_copy_pass(probed_with_them, chapters_only,
                               False, "out.mkv") is True


def test_the_chapter_list_is_the_truth_and_an_absent_list_is_no_answer(fake_ffprobe):
    """An empty list is zero chapters; a missing key is not a claim of any number.

    ``-show_chapters`` makes FFmpeg answer the list, so when it is asked for the list
    wins - including when it is empty beside a format field that says otherwise. When
    the key is not there at all, the probe reports the format's own number rather than
    inventing a zero the file never stated.
    """
    answer = {"streams": PROBE_ANSWER["streams"],
              "format": {"nb_chapters": 3, "tags": {}}, "chapters": []}
    seen = fake_ffprobe(answer)
    assert gtx.probe_source("in.mkv", include_chapters=True)["nb_chapters"] == 0
    assert "-show_chapters" in seen[0]
    # The same file without the list asked for: the format field is all there is.
    fake_ffprobe(answer)
    assert gtx.probe_source("in.mkv")["nb_chapters"] == 3
    no_list = {"streams": PROBE_ANSWER["streams"],
               "format": {"nb_chapters": 3, "tags": {}}}
    fake_ffprobe(no_list)
    assert gtx.probe_source("in.mkv", include_chapters=True)["nb_chapters"] == 3


# --- the video stream's own descriptive fields ---------------------------------

def test_only_the_descriptive_fields_of_the_source_video_are_read_as_descriptive():
    """A title and a language are what the track says it is; the rest is a measurement.

    ``ENCODER``, ``BPS``, ``DURATION`` and the frame counts were written by the encoder
    the *source* was made with. Carrying them onto a stream this chain encoded would be
    reporting somebody else's numbers as this run's, so the narrowing drops them.
    """
    source = {"video_tags": {"ENCODER": "Lavc61.19.101 libx265",
                             "DURATION": "00:00:00.240000000",
                             "BPS": "18750000", "title": "Master shot",
                             "language": "jpn"},
              "video_disposition": {"default": 1, "original": 1, "dub": 0}}
    assert gtx.source_video_descriptive(source) == {
        "title": "Master shot", "language": "jpn",
        "disposition": ["default", "original"]}
    # A track with nothing descriptive to carry is not a request to rewrap for.
    assert gtx.source_video_descriptive({"video_tags": {"ENCODER": "x"},
                                         "video_disposition": {}}) is None
    # And a probe that never answered is not a claim that the track has no title.
    assert gtx.source_video_descriptive({}) is None
    # A track that is named but states no flags answers with an empty flag list, which
    # is a claim the pass has to honour rather than leave to the muxer's own default.
    assert gtx.source_video_descriptive({"video_tags": {"title": "Master shot"},
                                         "video_disposition": {}}) == {
        "title": "Master shot", "disposition": []}


def test_the_descriptive_fields_are_written_onto_the_encoded_stream_not_mapped():
    """The clause changes which tags the output carries, never which packets.

    The clauses name the first *output* video stream, which is the encode's, and the
    source's video stays unmapped: ``-map 1:v`` would deliver the original pictures and
    is not in this command.
    """
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", {}, "out.mkv", True,
        video_descriptive_metadata={"title": "Master shot", "language": "jpn",
                                   "disposition": ["default", "original"]})
    assert ["-metadata:s:v:0", "title=Master shot"] in [
        command[i:i + 2] for i in range(len(command) - 1)]
    assert ["-metadata:s:v:0", "language=jpn"] in [
        command[i:i + 2] for i in range(len(command) - 1)]
    assert ["-disposition:v:0", "+default+original"] in [
        command[i:i + 2] for i in range(len(command) - 1)]
    assert _in_order(command, [["-map", "0:v:0"], ["-c:v", "copy"],
                               ["-metadata:s:v:0", "title=Master shot"]])
    assert not any(arg.startswith("-map") and "1:v" in arg for arg in command)
    assert "1:v" not in command


def test_a_source_that_states_no_flags_clears_the_one_the_muxer_would_assume():
    """An all-zero video track is a fact the deliverable has to repeat, not omit.

    The clause is ``-default`` because that is the spelling FFmpeg's disposition option
    takes: ``default=0`` is rejected by the option parser as an invalid expression, and a
    pass whose command FFmpeg refuses to start is a failed export, not a clean one.
    """
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", {}, "out.mkv", True,
        video_descriptive_metadata={"title": "Master shot", "disposition": []})
    assert ["-disposition:v:0", "-default"] in [
        command[i:i + 2] for i in range(len(command) - 1)]
    assert "default=0" not in command


def test_a_job_that_did_not_ask_gets_the_command_it_has_always_gotten():
    """The default is the old command, byte for byte, with the parameter left out.

    Naming ``copy_video_stream_metadata`` in the plan changes nothing here on purpose:
    the shared builder writes the clause only from the metadata argument, so a caller
    that set the key without computing the fields cannot rewrap a file for a clause it
    never wrote. That is the SDR chain's command, unchanged.
    """
    asked = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", {"copy_video_stream_metadata": False},
        "out.mkv", True)
    base = {"audio_mode": "copy", "subtitle_mode": "copy", "copy_attachments": True,
            "copy_metadata": True, "copy_chapters": True}
    keyed = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "in.mkv", dict(base, copy_video_stream_metadata=True),
        "out.mkv", True)
    plain = gtx.build_mux_command("ffmpeg", "enc.mkv", "in.mkv", base, "out.mkv", True)
    assert "-metadata:s:v:0" not in asked and "-disposition:v:0" not in asked
    assert "-metadata:s:v:0" not in plain and "-disposition:v:0" not in plain
    assert keyed == plain


def test_the_descriptive_fields_are_only_carried_when_the_job_asks_and_source_has_them():
    """``needs_copy_pass`` for a request nothing would satisfy is still no request.

    The fields reach the decision as the narrowed argument, never as a plan key, because
    the argument is the same object the pass is handed: a gate that rewrapped for a
    clause its builder did not write would be inventing a failure, and one that wrote a
    clause its gate never accounted for would be delivering a file nobody asked to have
    rewrapped.
    """
    wanted = {"audio_mode": "remove", "subtitle_mode": "drop",
              "copy_attachments": False, "copy_metadata": False,
              "copy_chapters": False}
    with_title = {"video_tags": {"title": "Master shot"},
                  "video_disposition": {}, "nb_chapters": 0}
    titled = gtx.source_video_descriptive(with_title)
    assert gtx.needs_copy_pass(with_title, wanted, False, "out.mkv",
                               video_descriptive_metadata=titled) is True
    empty = dict(with_title, video_tags={"ENCODER": "x"})
    assert gtx.needs_copy_pass(empty, wanted, False, "out.mkv") is False
    assert gtx.needs_copy_pass(empty, wanted, False, "out.mkv",
                               video_descriptive_metadata=None) is False
    # A caller that asks for nothing decides exactly what it always decided.
    for source in (with_title, empty):
        assert gtx.needs_copy_pass(source, wanted, False, "out.mkv") is False
    assert gtx.needs_copy_pass(with_title, {}, False, "out.mkv") is False


# ---------------------------------------------------------------------------
# what the run reports
# ---------------------------------------------------------------------------

def test_a_report_field_is_a_bridges_measurement_or_nothing():
    """The names say which it is, so a 0 cannot be read as a missing counter."""
    measured = gtx.transfer_report(
        {"encoder": {"host_pixel_bytes": 0, "staging_gpu_copy_bytes": 10,
                     "conversion_output_bytes": 11, "frames_submitted": 1,
                     "packets_written": 1},
         "residual": {"full_device_copy_bytes": 20, "working_device_copy_bytes": 21,
                      "output_readback_bytes": 0},
         "dlss": {"host_color_stage_bytes": 0, "host_color_stage_count": 0,
                  "device_color_copy_bytes": 30, "output_readback_bytes": 0}},
        1, 3840, 2160)
    assert measured["encoder_host_pixel_bytes_measured"] == 0
    assert measured["dlss_output_readback_bytes_measured"] == 0
    assert measured["residual_full_device_copy_bytes_measured"] == 20
    # Nothing ran, so nothing may be claimed.
    empty = gtx.transfer_report({}, 0, 3840, 2160)
    for key, value in empty.items():
        if key.endswith("_measured"):
            assert value is None, key
        else:
            assert key.endswith("_logical") or key.startswith("gpu_transfer"), key
    assert empty["decode_nv12_bytes_logical"] == 3840 * 2160 * 3 // 2


def test_the_report_never_calls_a_buffer_size_a_transfer():
    measured = [k for k in gtx.transfer_report({}, 1, 64, 64) if k.endswith("_measured")]
    logical = [k for k in gtx.transfer_report({}, 1, 64, 64) if k.endswith("_logical")]
    assert set(measured).isdisjoint(logical)
    assert any("logical" in key for key in logical)


# ---------------------------------------------------------------------------
# the chain's three shapes: exact 2x at 0.5, plan-driven at 0.75, identity at 1.0
# ---------------------------------------------------------------------------

class RecordingBridge:
    """Stub bridge library: every symbol records its call, returns success.

    ``fills`` maps a symbol to the argument positions that must be filled
    through the byref out-parameters the real bridge writes.
    """

    def __init__(self, name, calls, fills=None):
        self._name = name
        self._calls = calls
        self._fills = fills or {}

    def __getattr__(self, symbol):
        def call(*args):
            self._calls.append((self._name, symbol, args))
            for position, value in self._fills.get(symbol, {}).items():
                args[position]._obj.value = value
            return 0
        return call


class WorkingRecorder(DeviceFlowProvider):
    """The device-flow stub that also remembers the colour it was handed."""

    def __init__(self):
        super().__init__()
        self.working = []

    def get_flow_device_rgba(self, frame_idx, color, cuda_context,
                             is_discontinuous=False):
        self.working.append(color)
        return {"format": "float32x2"}


class FixedDeviceLive:
    """A DLSS session stub with recognisable borrowed pointers."""

    def process_device_color_flow_resource(self, color, flow, reset=False):
        return {"d3d12_device": 0x10, "d3d12_resource": 0x20,
                "d3d12_fence": 0x30, "fence_value": 0x42,
                "width": color["width"], "height": color["height"]}


def _run_chain(downscaled, src=(3840, 2160), plan_driven=False):
    """Drive Chain.process over one stubbed frame and return what it touched."""
    if plan_driven:
        live = pipeline.resolve_working_dims(src[0], src[1], 0.75)
    else:
        live = (src[0] // 2, src[1] // 2) if downscaled else src
    calls = []
    bridges = {
        "nv12": RecordingBridge("nv12", calls,
            fills={"nv12_rgba_convert": {5: 4321, 6: 15360}}),
        "lanczos": RecordingBridge("lanczos", calls,
            fills={"gpu_lanczos_process": {3: 8888, 4: 7744}}),
        "residual": RecordingBridge("residual", calls,
            fills={"gpu_residual_process": {11: 777}}),
        "nvenc": RecordingBridge("nvenc", calls),
    }
    flow = WorkingRecorder()
    plan = gtx.plan_encoder({}, 24, dict(SOURCE))
    plan["decoder_color_matrix"] = gtx.COLOR_MATRIX_BT601  # negotiate adds this
    chain = gtx.Chain(bridges, "in.mp4", "out.mkv", FixedDeviceLive(), plan,
                     (30000, 1001), 1.0, live, flow, downscaled=downscaled,
                     tap_plans=(gtx.TapPlans(src[0], src[1], live[0], live[1])
                                if plan_driven else None))
    frame = gtx._DeviceFrame()
    frame.struct_size = gtx.C.sizeof(gtx._DeviceFrame)
    frame.width, frame.height = src
    frame.y_device_ptr, frame.uv_device_ptr = 1, 2
    frame.y_pitch_bytes, frame.uv_pitch_bytes = 7680, 7680
    frame.cuda_context, frame.pts = 0x5, 7
    frame.time_base_num, frame.time_base_den = 1, 1001
    from temporal_control import GpuTemporalControl, TemporalControls
    chain.temporal_control = GpuTemporalControl(TemporalControls(scene_detection=False))
    chain._open_color_stages(frame)
    outcome = chain.process(frame, flow, 0, True)
    return chain, calls, flow, outcome


def test_the_identity_chain_neither_resamples_nor_composites():
    """At 1.0 the full RGBA is the working colour and the DLSS output is the
    encoder input: neither exact-2x stage may so much as be opened."""
    chain, calls, flow, outcome = _run_chain(downscaled=False)
    symbols = {(name, symbol) for name, symbol, _ in calls}
    assert ("lanczos", "gpu_lanczos_open") not in symbols
    assert ("lanczos", "gpu_lanczos_process") not in symbols
    assert ("residual", "gpu_residual_open") not in symbols
    assert ("residual", "gpu_residual_process") not in symbols
    assert ("nv12", "nv12_rgba_convert") in symbols
    # The working colour is the converter's own plane, borrowed, not a copy.
    working = flow.working[0]
    assert (working["width"], working["height"]) == (3840, 2160)
    assert working["device_ptr"] == 4321 and working["pitch_bytes"] == 15360
    # The encoder takes the DLSS texture with its producer fence, so the frame
    # crosses to NVENC without a readback or an intermediate copy.
    encode = [args for name, symbol, args in calls
              if (name, symbol) == ("nvenc", "nvenc_encode_texture")]
    assert len(encode) == 1
    assert encode[0][1] == 0x20 and encode[0][2] == 0x30 and encode[0][3] == 0x42
    assert encode[0][4] == 7, "the decoder's PTS is the encoder's"
    assert "working_color" not in outcome["stages"]
    assert "residual" not in outcome["stages"]
    assert "nv12_to_rgba" in outcome["stages"]


def test_the_half_chain_still_runs_both_exact_2x_stages():
    """0.5 keeps its shape: Lanczos down, residual up, encoder behind them."""
    chain, calls, flow, outcome = _run_chain(downscaled=True)
    symbols = {(name, symbol) for name, symbol, _ in calls}
    assert ("lanczos", "gpu_lanczos_open") in symbols
    assert ("lanczos", "gpu_lanczos_process") in symbols
    assert ("residual", "gpu_residual_open") in symbols
    assert ("residual", "gpu_residual_process") in symbols
    working = flow.working[0]
    assert (working["width"], working["height"]) == (1920, 1080)
    assert working["device_ptr"] == 8888 and working["pitch_bytes"] == 7744
    encode = [args for name, symbol, args in calls
              if (name, symbol) == ("nvenc", "nvenc_encode_texture")]
    assert len(encode) == 1
    # The residual bridge waits on its own fence before returning, so the
    # encoder is handed the finished texture with no producer fence to wait on.
    assert encode[0][1] == 777 and encode[0][2] is None and encode[0][3] == 0
    assert "working_color" in outcome["stages"]
    assert "residual" in outcome["stages"]


def test_the_half_chain_never_hands_a_plan_to_the_constant_table_stages():
    """0.5 keeps the opens it was validated with.

    A plan-driven open on this path would put a second resampler where every
    fixture and counter figure was measured, so the chain must not build one.
    """
    _, calls, _, _ = _run_chain(downscaled=True)
    symbols = {(name, symbol) for name, symbol, _ in calls}
    assert ("lanczos", "gpu_lanczos_open_scaled") not in symbols
    assert ("residual", "gpu_residual_open_scaled") not in symbols


def test_the_plan_driven_chain_opens_both_resamplers_with_the_serialized_plan():
    """0.75 is the one shape whose stage extents come from the resolved size.

    Both resamplers have to be reached through the plan-driven exports, with the
    extents the decoder reports and the table the CPU plan builds for them. The
    constant-table opens must not appear: their 2x arithmetic would sample a
    1920x1080 window out of a 2880x1616 plane and still return success.
    """
    chain, calls, flow, outcome = _run_chain(downscaled=True, plan_driven=True)
    symbols = {(name, symbol) for name, symbol, _ in calls}
    assert ("lanczos", "gpu_lanczos_open") not in symbols
    assert ("residual", "gpu_residual_open") not in symbols
    opened = {name: args for name, symbol, args in calls
              if symbol.endswith("_open_scaled")}
    assert set(opened) == {"lanczos", "residual"}
    for name, args in opened.items():
        # ctx/device, full extents, working extents, plan, out-handle.
        assert args[1:5] == (3840, 2160, 2880, 1616), name
    color = opened["lanczos"][5]._obj
    assert color.struct_size == gtx.C.sizeof(gtx._TapTables)
    assert (color.taps, color.vertical_src_len, color.vertical_dst_len,
            color.horizontal_src_len,
            color.horizontal_dst_len) == (6, 2160, 1616, 3840, 2880)
    upscale = opened["residual"][5]._obj
    assert (upscale.taps, upscale.vertical_src_len, upscale.vertical_dst_len,
            upscale.horizontal_src_len,
            upscale.horizontal_dst_len) == (4, 1616, 2160, 2880, 3840)
    # Both plans stay alive on the chain, so a bridge that reads them later, on
    # a second frame, cannot be reading freed arrays.
    assert chain.tap_plans.working_color is color
    assert chain.tap_plans.upscale is upscale
    working = flow.working[0]
    assert (working["width"], working["height"]) == (2880, 1616)
    assert "working_color" in outcome["stages"] and "residual" in outcome["stages"]


def test_the_plan_driven_report_labels_the_resolved_working_plane():
    """The logical size of the working colour is the resolved plane, not the half.

    It is a buffer this code sized, so it stays in the ``_logical`` name space
    whatever the scale; the measured fields beside it come from the bridges.
    """
    report = gtx.transfer_report({}, 1, 3840, 2160, downscaled=True,
                                 working=(2880, 1616))
    assert report["working_color_bytes_logical"] == 2880 * 1616 * 4
    assert report["nv12_to_rgba_bytes_logical"] == 3840 * 2160 * 4


class CounterFiller:
    """A bridge whose counter getter writes the named fields and succeeds."""

    def __init__(self, values):
        self.values = values

    def __getattr__(self, name):
        def fill(_handle, block):
            for field, value in self.values.items():
                setattr(block._obj, field, value)
            return 0
        return fill


class _CountingLive:
    """A DLSS session that answers for its own boundaries and no others."""

    def transfer_counters(self):
        return {"host_color_stage_count": 0, "host_color_stage_bytes": 0,
                "device_color_copy_count": 8, "device_color_copy_bytes": 8 * 33177600,
                "output_readback_count": 0, "output_readback_bytes": 0}


def test_both_chains_encode_the_source_extent_and_only_the_half_one_resamples():
    """The resource contract the two shapes differ on: the encoder always sees a
    full-size plane, while the exact-2x bridges - which are the only stages whose
    extents follow the working scale - exist on the 0.5 chain alone. An encoder
    opened at the working size would refuse the DLSS texture on hardware, so this
    is checked as the bridges' own arguments rather than inferred from the file."""
    for downscaled, residual_extent in ((False, None), (True, (1920, 1080))):
        _, calls, _, _ = _run_chain(downscaled=downscaled)
        opens = [args for name, symbol, args in calls if symbol == "nvenc_open"]
        assert len(opens) == 1, (downscaled, opens)
        assert opens[0][2:4] == (3840, 2160), (
            f"downscaled={downscaled}: the encoder is opened at "
            f"{opens[0][2:4]}, not at the source extent")
        assert opens[0][4:6] == (30000, 1001), "the rational time base is the probe's"
        residual_opens = [args for name, symbol, args in calls
                          if symbol == "gpu_residual_open"]
        lanczos_opens = [args for name, symbol, args in calls
                         if symbol == "gpu_lanczos_open"]
        if residual_extent is None:
            assert not residual_opens and not lanczos_opens
        else:
            assert lanczos_opens[0][1:3] == (3840, 2160), (
                "the working-color bridge owns a full-size source plane")
            assert residual_opens[0][1:3] == residual_extent, (
                "the residual bridge is opened for the working colour it composites")


def test_the_identity_run_counts_only_the_bridges_it_opened():
    """A skipped stage has no counter to read, so its fields stay None while the
    stages that did run answer with their own figures. The 1.0 chain therefore
    reports a stage that does not exist as uninstrumented, never as a boundary
    crossed zero times and never as the byte figure of a buffer it did not copy."""
    decoder = {"frames_decoded": 8, "device_frame_count": 8,
               "bitstream_bytes_submitted": 400000, "host_frame_copy_count": 0,
               "host_frame_copy_bytes": 0, "scale_device_output_bytes": 0}
    chain = gtx.Chain(
        {"nvdec": CounterFiller(decoder),
         "nv12": CounterFiller({"convert_count": 8,
                                "output_device_write_bytes": 8 * 33177600}),
         "nvenc": CounterFiller({"frames_submitted": 8, "packets_written": 8,
                                 "staging_gpu_copy_bytes": 8 * 33177600,
                                 "conversion_output_bytes": 8 * 24883200,
                                 "host_pixel_bytes": 0})},
        "in.mp4", "out.mkv", _CountingLive(), {}, (30000, 1001), 1.0, (3840, 2160),
        flow_provider=None, downscaled=False)
    chain.decoder, chain.nv12, chain.encoder = 1, 1, 1
    counters = chain.counters()
    assert "lanczos" not in counters and "residual" not in counters, (
        "the two bridges never opened, so nothing exists to ask for a count")

    report = gtx.transfer_report(counters, 8, 3840, 2160, downscaled=False)
    for key in ("working_color_device_bytes_measured",
                "working_color_process_count_measured",
                "working_color_source_device_copy_bytes_measured",
                "working_color_scratch_device_write_bytes_measured",
                "residual_full_device_copy_bytes_measured",
                "residual_working_device_copy_bytes_measured",
                "residual_output_readback_bytes_measured"):
        assert report[key] is None, f"{key} claims a number for a stage that did not run"
    # The boundaries that did run keep their own measurements, so the Nones above
    # cannot be read as a run that counted nothing.
    assert report["decode_frames_decoded_measured"] == 8
    assert report["nv12_to_rgba_convert_count_measured"] == 8
    assert report["encoder_frames_submitted_measured"] == 8
    assert report["encoder_staging_device_copy_bytes_measured"] == 8 * 33177600
    assert report["dlss_device_color_copy_bytes_measured"] == 8 * 33177600
    assert report["working_color_bytes_logical"] == 3840 * 2160 * 4
    for key in ("decode_host_copy_bytes_measured", "decode_host_copy_count_measured",
                "encoder_host_pixel_bytes_measured",
                "dlss_host_color_stage_bytes_measured", "dlss_output_readback_bytes_measured"):
        assert report[key] == 0, f"{key} is a measured zero, not an absent counter"


def test_the_identity_chain_hands_the_flow_engine_a_full_size_descriptor():
    """The flow engine is the other consumer of the working plane, and at 1.0 it
    is asked for 4M-class flow over the whole frame - the descriptor it would
    reject as a half plane if the chain mislabelled its own extent."""
    _, _, flow, _ = _run_chain(downscaled=False)
    working = flow.working[0]
    assert (working["width"], working["height"]) == (3840, 2160)
    assert working["format"] == "rgba8_cuda"
    assert working["pitch_bytes"] >= 3840 * 4, (
        "the borrowed pitch has to cover the row the converter wrote")
    _, _, half_flow, _ = _run_chain(downscaled=True)
    assert (half_flow.working[0]["width"], half_flow.working[0]["height"]) == (1920, 1080)


def test_the_export_runs_and_reports_the_shape_the_gate_classified():
    """The classification from negotiate() is what export_gpu_transfer runs:
    the chain is built from it, the per-frame record names what ran, and the
    report says which of the two scales it was, so a finished file cannot be
    mistaken for the other chain."""
    import inspect

    source = inspect.getsource(gtx._export_gpu_transfer)
    assert 'negotiation["downscaled"]' in source
    assert "downscaled=downscaled" in source
    assert '"residual_backend": residual_backend' in source
    assert '"gpu_transfer_working_scale"' in source
    assert '"gpu_transfer_residual_backend"' in source
    # A caller that hands over a collector has to get its own rows in it:
    # TransferProfiler defines __len__, so a fresh one is falsy and `or` would
    # silently swap in a private ledger the caller then reads as empty.
    assert "transfer_collector or" not in source, (
        "the caller's collector must be tested against None, not for truthiness")
    assert "transfer_collector is None" in source


# ---------------------------------------------------------------------------
# the decode boundary
# ---------------------------------------------------------------------------

DECODER_COUNTERS = {"frames_decoded": 14, "device_frame_count": 14,
                    "bitstream_bytes_submitted": 700000, "host_frame_copy_count": 0,
                    "host_frame_copy_bytes": 0, "scale_device_output_bytes": 0}


def test_the_decode_boundary_reports_the_counters_the_bridge_kept():
    """A measured zero only means something next to the frames it counted over."""
    report = gtx.transfer_report({"decoder": DECODER_COUNTERS}, 14, 3840, 2160)
    assert report["decode_frames_decoded_measured"] == 14, (
        "a zero host figure is unreadable unless the same run says frames were decoded")
    assert report["decode_device_frame_count_measured"] == 14
    assert report["decode_host_copy_count_measured"] == 0
    assert report["decode_host_copy_bytes_measured"] == 0
    assert report["decode_bitstream_submitted_bytes_measured"] == 700000
    assert report["decode_scale_device_output_bytes_measured"] == 0
    # The logical frame size is a separate field, so it is never mistaken for the measured one.
    assert report["decode_nv12_bytes_logical"] == 3840 * 2160 * 3 // 2
    assert report["decode_host_copy_bytes_measured"] != report["decode_nv12_bytes_logical"]


def test_a_decode_boundary_the_bridge_did_not_answer_for_stays_unmeasured():
    """An older bridge, or a counter read that failed, reads None and never 0."""
    report = gtx.transfer_report({"encoder": {"frames_submitted": 1}}, 1, 3840, 2160)
    for key in ("decode_host_copy_bytes_measured", "decode_host_copy_count_measured",
                "decode_frames_decoded_measured", "decode_device_frame_count_measured",
                "decode_bitstream_submitted_bytes_measured",
                "decode_scale_device_output_bytes_measured"):
        assert report[key] is None, key
        assert report[key] != 0, key


def test_the_python_counter_mirror_matches_the_c_header():
    """The bridge fills this struct by offset, so the mirror's order is the ABI."""
    header = open(os.path.join(os.path.dirname(_ROOT), "native", "nvdec_bridge.h"),
                  encoding="utf-8").read()
    body = header.split("typedef struct NvdecTransferCounters {", 1)[1].split("}", 1)[0]
    declarations = [line.split("/*", 1)[0].strip() for line in body.splitlines()]
    fields = [line.split()[-1].rstrip(";") for line in declarations
              if line.startswith("uint")]
    assert fields[0] == "struct_size"
    assert fields[1:] == [name for name, _ in gtx._NvdecCounters._fields_[1:]], (
        "the struct the bridge fills and the one this module reads have drifted apart")
    assert f"NVDEC_BRIDGE_ABI_VERSION {gtx.NVDEC_ABI}u" in header, (
        "the module must read the ABI the header declares")


COUNTER_HEADERS = [
    ("nv12_rgba_bridge.h", "Nv12RgbaTransferCounters", gtx._Nv12Counters,
     "NV12_RGBA_BRIDGE_ABI_VERSION"),
    ("gpu_lanczos_bridge.h", "GpuLanczosTransferCounters", gtx._LanczosCounters,
     "GPU_LANCZOS_ABI_VERSION"),
]


@pytest.mark.parametrize("header_name,struct_name,mirror,abi_macro", COUNTER_HEADERS)
def test_the_conversion_and_working_color_mirrors_match_their_c_headers(
        header_name, struct_name, mirror, abi_macro):
    """Both bridges fill their struct by offset, so the mirror's order is the ABI."""
    header = open(os.path.join(os.path.dirname(_ROOT), "native", header_name),
                  encoding="utf-8").read()
    body = header.split(f"typedef struct {struct_name} {{", 1)[1].split("}", 1)[0]
    declarations = [line.split("/*", 1)[0].strip() for line in body.splitlines()]
    fields = [line.split()[-1].rstrip(";") for line in declarations
              if line.startswith("uint")]
    assert fields[0] == "struct_size"
    assert fields[1:] == [name for name, _ in mirror._fields_[1:]], (
        f"the struct {header_name} fills and the one this module reads have drifted apart")
    abi = header.split(f"#define {abi_macro} ", 1)[1].split("u", 1)[0]
    assert f"{abi_macro} {abi}u" in header
    assert int(abi) == (gtx.NV12_ABI if "nv12" in header_name else gtx.LANCZOS_ABI), (
        "the module must read the ABI the header declares")


PLAN_HEADERS = [
    ("gpu_lanczos_bridge.h", "GpuLanczosTapTables", "GPU_LANCZOS_MAX_TAPS",
     gtx.LANCZOS_MAX_TAPS, "GPU_LANCZOS_ABI_VERSION", gtx.LANCZOS_ABI),
    ("gpu_residual_bridge.h", "GpuResidualTapTable", "GPU_RESIDUAL_MAX_TAPS",
     gtx.RESIDUAL_MAX_TAPS, "GPU_RESIDUAL_ABI_VERSION", gtx.RESIDUAL_ABI),
]


@pytest.mark.parametrize(
        "header_name,struct_name,tap_macro,taps,abi_macro,abi", PLAN_HEADERS)
def test_the_plan_mirror_matches_both_c_headers_and_neither_abi_moves(
        header_name, struct_name, tap_macro, taps, abi_macro, abi):
    """One ctypes struct serves both resamplers, so both headers are checked.

    The tap window is the kernel's own bound, and the ABI number is what an
    existing caller pins: a plan-driven scale arrives as a new export under the
    number already published, so neither header may have had its ABI edited to
    accommodate it.
    """
    header = open(os.path.join(os.path.dirname(_ROOT), "native", header_name),
                  encoding="utf-8").read()
    body = header.split(f"typedef struct {struct_name} {{", 1)[1].split("}", 1)[0]
    declarations = [line.split("/*", 1)[0].strip() for line in body.splitlines()]
    fields = [line.split()[-1].rstrip(";") for line in declarations if line]
    assert fields == [name for name, _ in gtx._TapTables._fields_], (
        f"{header_name}'s plan and the struct this module hands it have drifted apart")
    declared = header.split(f"#define {tap_macro} ", 1)[1].split("u", 1)[0]
    assert int(declared) == taps, f"{tap_macro} and this module's window disagree"
    version = header.split(f"#define {abi_macro} ", 1)[1].split("u", 1)[0]
    assert int(version) == abi, (
        "the plan-driven open is added under the ABI existing callers pin")


def test_a_bridge_that_still_reports_the_old_abi_is_refused_not_read():
    """A counter getter is read by offset from a struct that only exists in the new
    ABI, so an older converter has to be refused rather than driven."""
    with pytest.raises(gtx.BridgeError) as raised:
        gtx._require_abi(2, gtx.NV12_ABI, "nv12_rgba_bridge")
    assert "reports ABI 2" in str(raised.value)
    with pytest.raises(gtx.BridgeError):
        gtx._require_abi(1, gtx.LANCZOS_ABI, "gpu_lanczos_bridge")


CONVERSION_COUNTERS = {"convert_count": 14,
                       "output_device_write_bytes": 14 * 3840 * 2160 * 4}
WORKING_COLOR_COUNTERS = {"process_count": 14,
                          "source_device_copy_bytes": 14 * 3840 * 2160 * 4,
                          "scratch_device_write_bytes": 14 * 3840 * 1080 * 16,
                          "output_device_write_bytes": 14 * 1920 * 1080 * 4}


def test_the_conversion_and_working_color_boundaries_report_their_counters():
    report = gtx.transfer_report({"nv12": CONVERSION_COUNTERS,
                                  "lanczos": WORKING_COLOR_COUNTERS}, 14, 3840, 2160)
    assert report["nv12_to_rgba_convert_count_measured"] == 14
    assert report["nv12_to_rgba_device_bytes_measured"] == 14 * 3840 * 2160 * 4
    assert report["working_color_process_count_measured"] == 14
    assert report["working_color_device_bytes_measured"] == 14 * 1920 * 1080 * 4
    assert report["working_color_source_device_copy_bytes_measured"] == 14 * 3840 * 2160 * 4
    assert report["working_color_scratch_device_write_bytes_measured"] == 14 * 3840 * 1080 * 16
    # The logical field is one frame's buffer size and stays separate from the
    # measured total, so the two can never be added as if they were different bytes.
    assert report["nv12_to_rgba_bytes_logical"] == 3840 * 2160 * 4
    assert report["working_color_bytes_logical"] == 1920 * 1080 * 4


def test_the_conversion_and_working_color_boundaries_stay_unmeasured_without_a_counter():
    report = gtx.transfer_report({"decoder": DECODER_COUNTERS}, 14, 3840, 2160)
    for key in ("nv12_to_rgba_device_bytes_measured", "nv12_to_rgba_convert_count_measured",
                "working_color_device_bytes_measured", "working_color_process_count_measured",
                "working_color_source_device_copy_bytes_measured",
                "working_color_scratch_device_write_bytes_measured"):
        assert report[key] is None, key
        assert report[key] != 0, key


def test_chain_counters_read_the_new_bridge_counters():
    """The Chain asks each opened bridge for its own block, under the stage's name."""
    class Live:
        def transfer_counters(self):
            return {"host_color_stage_bytes": 0}

    chain = gtx.Chain(
        {"nv12": CounterFiller({"convert_count": 3, "output_device_write_bytes": 7}),
         "lanczos": CounterFiller(WORKING_COLOR_COUNTERS)}, "in.mp4", "out.mkv", Live(),
        {}, (30000, 1001), 1.0, (1920, 1080), flow_provider=None)
    chain.nv12, chain.lanczos = 1, 1
    counters = chain.counters()
    assert counters["nv12"] == {"convert_count": 3, "output_device_write_bytes": 7}
    assert counters["lanczos"] == WORKING_COLOR_COUNTERS
    # Nothing opened the decoder, residual stage or encoder in this stub, so the
    # report for them reads uninstrumented rather than borrowing the new numbers.
    report = gtx.transfer_report(counters, 1, 3840, 2160)
    assert report["decode_frames_decoded_measured"] is None
    assert report["nv12_to_rgba_convert_count_measured"] == 3


def test_production_export_does_not_download_decoded_frames():
    """The readback entry point is the only call that can raise host_frame_copy_bytes.

    Naming it nowhere in the production module is what makes the zero in the report a
    measurement of the device path rather than a call that was merely not reached.
    """
    import inspect

    source = inspect.getsource(gtx)
    assert "nvdec_bridge_readback_last_frame" not in source
    assert "nvdec_bridge_get_transfer_counters" in source, (
        "the counters have to be read, or the boundary is uninstrumented again")


# ---------------------------------------------------------------------------
# the shared output verification
# ---------------------------------------------------------------------------

def test_verification_accepts_a_file_whose_probe_agrees(tmp_path):
    produced = tmp_path / "out.mkv"
    produced.write_bytes(b"\0" * 4096)
    assert pipeline.verify_export_output(str(produced), 10,
                                         probe=lambda path: (10, 30.0, 64, 64)) == 10


def test_verification_refuses_a_truncated_file(tmp_path):
    produced = tmp_path / "out.mkv"
    produced.write_bytes(b"\0" * 4096)
    with pytest.raises(RuntimeError) as raised:
        pipeline.verify_export_output(str(produced), 100,
                                      probe=lambda path: (10, 30.0, 64, 64))
    assert "帧数强校验不通过" in str(raised.value)


def test_verification_refuses_a_file_it_cannot_count_frames_in(tmp_path):
    produced = tmp_path / "out.mkv"
    produced.write_bytes(b"\0" * 4096)

    def unreadable(path):
        raise RuntimeError("ffprobe said nothing")

    with pytest.raises(RuntimeError) as raised:
        pipeline.verify_export_output(str(produced), 10, probe=unreadable,
                                      cv2_probe=lambda path: 0)
    assert "拒绝交付" in str(raised.value)


def test_a_file_that_is_absent_or_too_small_to_hold_a_frame_is_refused(tmp_path):
    for size in (0, 512):
        tiny = tmp_path / f"out{size}.mkv"
        tiny.write_bytes(b"\0" * size)
        with pytest.raises(RuntimeError):
            pipeline.verify_export_output(str(tiny), 10,
                                         probe=lambda path: (10, 30.0, 64, 64))
    with pytest.raises(RuntimeError):
        pipeline.verify_export_output(str(tmp_path / "never-written.mkv"), 10)


def test_both_backends_hand_their_file_to_the_same_check():
    """A GPU-resident export is not deliverable on looser terms than a piped one."""
    import inspect

    pipe = inspect.getsource(pipeline.export_dlss_stream)
    gpu = inspect.getsource(pipeline.export_gpu_transfer_stream)
    assert "verify_export_output(" in pipe, "the legacy path must still check its output"
    assert "export_gpu_transfer(" in gpu
    import gpu_transfer_export

    assert "_export_gpu_transfer(" in inspect.getsource(gpu_transfer_export.export_gpu_transfer)
    assert "verify_export_output(" in inspect.getsource(gpu_transfer_export._export_gpu_transfer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
