"""Native stalls simulated with events/clocks, without allocating any GPU resources."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dlss_service import DLSSService
from native_watchdog import NativeWatchdog, NativeOperationTimeout, phase


def test_independent_observer_fuses_stalled_owner_and_preserves_queue(tmp_path):
    service = DLSSService(queue_path=tmp_path / 'queue.json')
    job = service.add_job({'input_path': __file__})
    job['status'] = 'running'
    now = [0.]
    watchdog = NativeWatchdog(service._mark_native_degraded, clock=lambda: now[0], threaded=False)
    service._native_watchdog = watchdog
    entered, release = threading.Event(), threading.Event()

    def hung_owner():
        with watchdog.operation('模拟 GPU 等待', 10, kind='export'):
            entered.set()
            assert release.wait(3)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(hung_owner)
        try:
            assert entered.wait(2)
            now[0] = 11
            watchdog.check()  # observer thread never acquires the GPU ownership lock
            health = service.backend_health()
            assert not health['healthy'] and health['recovery_checkpoint_ready']
            assert health['native_operation']['timed_out']
            restored = DLSSService(queue_path=tmp_path / 'queue.json')
            assert restored.jobs[0]['interrupted']
        finally:
            release.set()
        with pytest.raises(NativeOperationTimeout):
            result.result(2)  # A late native return must not be reported as success.
    assert watchdog.snapshot() is None


def test_progress_renews_deadline_but_cancel_cannot_be_extended():
    now, faults = [0.], []
    watchdog = NativeWatchdog(faults.append, clock=lambda: now[0], threaded=False)
    with pytest.raises(NativeOperationTimeout):
        with watchdog.operation('setup', 300, kind='export'):
            now[0] = 250
            phase('frame 1', 300)
            watchdog.check()
            assert not faults
            watchdog.cancel(grace=30)
            now[0] = 279
            phase('frame 2', 300)
            watchdog.check()
            assert not faults
            now[0] = 281
            watchdog.check()
            watchdog.check()
            assert len(faults) == 1
            with pytest.raises(NativeOperationTimeout):
                phase('must not start another native call', 300)


def test_watchdog_thread_runs_without_owner_polling_and_stops_when_idle():
    fired = threading.Event()
    watchdog = NativeWatchdog(lambda _: fired.set())
    with pytest.raises(NativeOperationTimeout):
        with watchdog.operation('blocked native call', .01):
            observer = watchdog.thread
            assert fired.wait(2), 'observer did not detect the blocked owner'
    observer.join(timeout=2)
    assert not observer.is_alive()


def test_completed_idle_and_cpu_mux_work_do_not_trigger_gpu_restart():
    now, faults = [0.], []
    watchdog = NativeWatchdog(faults.append, clock=lambda: now[0], threaded=False)
    with watchdog.operation('frame', 10, kind='export'):
        phase('CPU remux', None)
        watchdog.cancel(grace=1)
        now[0] = 1000
        watchdog.check()
        assert watchdog.snapshot() is None
    watchdog.check()
    assert not faults


def test_cancelling_export_does_not_cancel_preview_deadline():
    now, faults = [0.], []
    watchdog = NativeWatchdog(faults.append, clock=lambda: now[0], threaded=False)
    with watchdog.operation('preview', 300):
        watchdog.cancel(grace=1)
        now[0] = 2
        watchdog.check()
        assert not faults


def test_executor_propagates_observation_without_moving_owner_thread():
    from gpu_preview_engine import GpuPreview
    faults = []
    watchdog = NativeWatchdog(faults.append, threaded=False)
    engine = GpuPreview.__new__(GpuPreview)
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine.executor = executor
        caller = threading.get_ident()

        def native_work():
            phase('owner-thread frame', 40)
            return threading.get_ident()

        with watchdog.operation('request', 300):
            assert engine._submit(native_work) != caller
            assert watchdog.snapshot()['phase'] == 'owner-thread frame'
    assert not faults


def test_checkpoint_not_ready_until_fault_save_has_finished(tmp_path, monkeypatch):
    service = DLSSService(queue_path=tmp_path / 'queue.json')
    entered, release = threading.Event(), threading.Event()
    save = service._queue_store.save

    def slow_save(jobs):
        entered.set()
        assert release.wait(3)
        save(jobs)

    monkeypatch.setattr(service._queue_store, 'save', slow_save)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(service._mark_native_degraded, 'GPU stalled')
        try:
            assert entered.wait(2)
            health = service.backend_health()
            assert not health['healthy'] and not health['recovery_checkpoint_ready']
        finally:
            release.set()
        result.result(2)
    assert service.backend_health()['recovery_checkpoint_ready']
