"""Phase A 1C: the cancellation error contract, tested against real pipes.

While a cancel is in progress, the child processes are expected to die and their pipes
to close. Those outcomes are part of cancelling and must surface as JobCancelled.
Without a cancel the very same EOF, broken pipe or non-zero exit is a real FFmpeg
failure and must stay a RuntimeError -- converting it would hide genuine corruption.

Cases, both directions, for decoder-side reads and encoder-side writes:
    cancel + decoder pipe closes  -> JobCancelled
    cancel + encoder pipe closes  -> JobCancelled
    no cancel + decoder EOF       -> RuntimeError
    no cancel + encoder failure   -> RuntimeError
    cancel_active_job, nothing else running -> flag set, every owned child dead
"""

import os
import io
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline  # noqa: E402

HANG_CHILD = "import time; time.sleep(30)"
FAIL_CHILD = "import os; os._exit(3)"


def _spawn(source, with_stdin=False):
    return subprocess.Popen([sys.executable, "-c", source],
                            stdin=subprocess.PIPE if with_stdin else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _read_after_cancel(kill_after_set):
    """Block a reader in _read_exactly, then cancel it while it is inside read()."""
    proc = _spawn(HANG_CHILD)
    stop_event = threading.Event()
    box = {}

    def _reader():
        try:
            pipeline._read_exactly(proc.stdout, 4096, proc=proc, stop_event=stop_event,
                                   what="decoder pipe")
        except BaseException as exc:      # recorded, then asserted by the test body
            box["error"] = exc

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    time.sleep(0.15)                      # the reader is now parked inside read()
    stop_event.set()
    if kill_after_set:
        proc.kill()
    proc.wait(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive(), "reader never returned after cancel"
    proc.stdout.close()
    proc.stderr.close()
    return box.get("error")


def test_cancel_with_decoder_pipe_close_raises_job_cancelled():
    error = _read_after_cancel(kill_after_set=True)
    assert isinstance(error, pipeline.JobCancelled), f"expected JobCancelled, got {error!r}"


def test_unexpected_decoder_exit_without_cancel_raises_runtime_error():
    proc = _spawn(FAIL_CHILD)
    proc.wait(timeout=10)                 # dies before anyone asks to cancel
    with pytest.raises(RuntimeError) as info:
        pipeline._read_exactly(proc.stdout, 4096, proc=proc,
                               stop_event=threading.Event(), what="decoder pipe")
    assert not isinstance(info.value, pipeline.JobCancelled), \
        "a real decoder failure must not be relabelled as a cancellation"
    proc.stdout.close()
    proc.stderr.close()


def test_decoder_timeout_after_partial_pipe_data_kills_and_reaps_child():
    proc = _spawn("import sys,time; sys.stdout.buffer.write(b'partial'); sys.stdout.flush(); time.sleep(10)")
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="超时"):
            pipeline._read_exactly(proc.stdout, 128, proc=proc, timeout=0.2,
                                   what="stalled integration child")
        assert time.monotonic() - started < 1.0
        assert proc.poll() is not None, "timed-out child must be reaped"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
        proc.stdout.close()
        proc.stderr.close()


def test_decoder_pipe_reads_final_payload_after_fast_normal_exit():
    payload = b"final-pipe-payload"
    source = "import os; os.write(1, %r)" % payload
    proc = _spawn(source)
    try:
        assert pipeline._read_exactly(proc.stdout, len(payload), proc=proc,
                                      timeout=2, what="fast-exit payload") == payload
        assert proc.wait(timeout=2) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
        proc.stdout.close()
        proc.stderr.close()


def test_decoder_reuses_one_reader_across_frame_boundaries_and_releases_it():
    payload = bytes(range(128))
    proc = _spawn("import os; os.write(1, %r)" % payload)
    try:
        first = pipeline._read_exactly(proc.stdout, 17, proc=proc, timeout=2)
        reader = proc._lunarc_decoder_pipe_reader
        second = pipeline._read_exactly(proc.stdout, 111, proc=proc, timeout=2)
        assert first + second == payload
        assert proc._lunarc_decoder_pipe_reader is reader
        assert proc.wait(timeout=2) == 0
    finally:
        pipeline._close_decoder_process(proc)
    assert not reader.thread.is_alive()


@pytest.mark.parametrize('complete', [False, True])
def test_input_decode_probe_reaps_child_and_closes_pipes(monkeypatch, complete):
    # Exercise lifecycle only; no CUDA or FFmpeg is used by this test.
    proc = _spawn(HANG_CHILD)
    monkeypatch.setattr(pipeline, 'probe_cuda_decode_support', lambda _: (True, 'fixture'))
    monkeypatch.setattr(pipeline, 'video_info', lambda _: (1, 30, 1, 1))
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *args, **kwargs: proc)
    monkeypatch.setattr(pipeline, '_popen_read_first',
                        lambda *args, **kwargs: b'123' if complete else b'')
    stderr = proc.stderr

    class CheckedStderr:
        def read(self):
            assert proc.poll() is not None, 'stderr must not be read from a live probe'
            return stderr.read()

        def close(self):
            stderr.close()

    proc.stderr = CheckedStderr()
    try:
        result = pipeline.probe_cuda_decode_for_input('fixture', ffmpeg='fixture')
        assert result[0] is complete
        assert proc.poll() is not None
        assert proc.stdout.closed and stderr.closed
    finally:
        pipeline._close_decoder_process(proc)


def test_decoder_pipe_cancel_while_child_stalls_is_bounded_and_reaped():
    proc = _spawn(HANG_CHILD)
    stop_event = threading.Event()
    result = {}

    def read():
        try:
            pipeline._read_exactly(proc.stdout, 64, proc=proc, stop_event=stop_event,
                                   timeout=5, what="cancelled stalled child")
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    time.sleep(0.1)
    started = time.monotonic()
    stop_event.set()
    thread.join(timeout=1)
    try:
        assert not thread.is_alive(), "cancelled pipe reader did not return"
        assert isinstance(result.get("error"), pipeline.JobCancelled)
        assert time.monotonic() - started < 1
        assert proc.poll() is not None, "cancelled child must be reaped"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
        proc.stdout.close()
        proc.stderr.close()


def test_read_exactly_supports_memory_stream_and_graceful_eof():
    assert pipeline._read_exactly(io.BytesIO(b"abcd"), 4) == b"abcd"
    assert pipeline._read_exactly(io.BytesIO(b"xy"), 4) == b"xy"


def _write_to_dead_encoder(cancelled):
    proc = _spawn(FAIL_CHILD, with_stdin=True)
    proc.wait(timeout=10)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError) as info:
        pipeline._write_frame_bytes(proc.stdin, frame, cancel_fn=lambda: cancelled,
                                    proc=proc, frame_idx=7)
    proc.stdin.close()
    proc.stderr.close()
    return info.value


def test_cancel_with_encoder_pipe_close_raises_job_cancelled():
    error = _write_to_dead_encoder(cancelled=True)
    assert isinstance(error, pipeline.JobCancelled), f"expected JobCancelled, got {error!r}"


def test_encoder_failure_without_cancel_raises_runtime_error():
    error = _write_to_dead_encoder(cancelled=False)
    assert not isinstance(error, pipeline.JobCancelled), \
        "a real encoder exit must not be relabelled as a cancellation"


def test_cancel_or_relabels_only_while_cancelling():
    """The shared rule behind every cancel path, both directions."""
    real_failure = RuntimeError("FFmpeg 解码进程异常退出 (等待 frame data，退出码 1)")

    clear = threading.Event()
    assert pipeline._cancel_or(real_failure, clear, None) is real_failure, \
        "an error with no cancel in progress must keep its own type and message"

    set_event = threading.Event()
    set_event.set()
    relabelled = pipeline._cancel_or(real_failure, set_event, None)
    assert isinstance(relabelled, pipeline.JobCancelled), "a cancel-time IO error must become JobCancelled"
    assert isinstance(relabelled, RuntimeError), "JobCancelled must stay catchable as RuntimeError"
    assert "退出码 1" in str(relabelled), "the relabelled error must still carry the original cause"

    assert isinstance(pipeline._cancel_or(real_failure, None, lambda: True), pipeline.JobCancelled)
    assert pipeline._cancel_or(real_failure, None, lambda: False) is real_failure


def test_cancel_or_does_not_double_wrap_a_cancellation():
    set_event = threading.Event()
    set_event.set()
    already = pipeline.JobCancelled("用户已取消任务")
    assert pipeline._cancel_or(already, set_event, None) is already


def test_read_error_during_cancel_becomes_job_cancelled():
    """A read() that fails because the cancel is landing right then is a cancellation.

    The stream raises the flag itself, so the failure happens strictly after the
    loop's own cancel check -- only the error handler can turn it into JobCancelled.
    """
    class ExplodingStream:
        def __init__(self, set_on_read=None):
            self.set_on_read = set_on_read

        def read(self, _n):
            if self.set_on_read is not None:
                self.set_on_read.set()
            raise OSError("Handle is closed")

    late_cancel = threading.Event()
    with pytest.raises(pipeline.JobCancelled):
        pipeline._read_exactly(ExplodingStream(late_cancel), 16,
                               stop_event=late_cancel, what="decoder pipe")

    with pytest.raises(RuntimeError) as info:
        pipeline._read_exactly(ExplodingStream(), 16, stop_event=threading.Event(),
                               what="decoder pipe")
    assert not isinstance(info.value, pipeline.JobCancelled)


def test_broken_pipe_during_cancel_becomes_job_cancelled():
    """A write that fails because the reader is gone, while the cancel lands in the
    same instant, is a cancellation. The flag flips after the loop's own check so only
    the error handler can decide."""
    r, w = os.pipe()
    os.close(r)                       # no reader left: every write now fails
    dead_pipe = os.fdopen(w, "wb", buffering=0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    calls = []

    def cancel_landing_mid_write():
        calls.append(1)
        return len(calls) > 1         # False at the loop check, True in the handler

    with pytest.raises(pipeline.JobCancelled):
        pipeline._write_frame_bytes(dead_pipe, frame, cancel_fn=cancel_landing_mid_write,
                                    proc=None, frame_idx=0)

    with pytest.raises(RuntimeError) as info:
        pipeline._write_frame_bytes(dead_pipe, frame, cancel_fn=lambda: False,
                                    proc=None, frame_idx=1)
    assert not isinstance(info.value, pipeline.JobCancelled), \
        "a broken pipe with no cancel in progress is a real encoder failure"


def test_cancel_latency_stays_short():
    start = time.perf_counter()
    error = _read_after_cancel(kill_after_set=True)
    latency_ms = (time.perf_counter() - start) * 1000.0
    assert isinstance(error, pipeline.JobCancelled)
    assert latency_ms < 1000.0, f"cancel path took {latency_ms:.1f} ms"


def _bare_service():
    """A DLSSService with only the state the cancel path touches.

    The end-to-end cancellation test cannot prove that cancel_active_job itself kills:
    the exporter's own teardown reaps its children too, so removing the kill there goes
    unnoticed. Here nothing else can clean up behind it.
    """
    import dlss_service
    svc = object.__new__(dlss_service.DLSSService)
    svc._cancel_requested = False
    svc._active_procs = []
    svc._procs_lock = threading.Lock()
    return svc


def test_cancel_active_job_sets_the_flag_and_kills_its_children():
    svc = _bare_service()
    procs = [_spawn(HANG_CHILD) for _ in range(2)]
    for proc in procs:
        svc._register_proc(proc)

    svc.cancel_active_job()

    assert svc._cancel_requested is True, "cancel must set the flag, not only kill"
    for proc in procs:
        proc.wait(timeout=5)
        assert proc.poll() is not None, "child survived cancel_active_job"
        for stream in (proc.stdout, proc.stderr):
            stream.close()


def test_cancel_active_job_is_repeatable_over_already_dead_children():
    """A worker that died on its own must not turn the cancel into an exception."""
    svc = _bare_service()
    proc = _spawn(FAIL_CHILD)
    svc._register_proc(proc)
    proc.wait(timeout=10)

    svc.cancel_active_job()
    svc.cancel_active_job()

    assert proc.poll() is not None
    proc.stdout.close()
    proc.stderr.close()
