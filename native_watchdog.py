"""Observe native work without moving calls or destroying another thread's resources.

The only recovery action here is a fault notification. The Electron parent owns
process termination/restart; a timed-out native stack stays in its owner thread.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
import time
import uuid


_current = ContextVar('native_operation', default=None)


class NativeOperationTimeout(RuntimeError):
    pass


class NativeWatchdog:
    def __init__(self, on_timeout, *, clock=time.monotonic, threaded=True):
        self.on_timeout = on_timeout
        self.clock = clock
        self.threaded = threaded
        self.lock = threading.Lock()
        self.operations = {}
        self.thread = None

    @contextmanager
    def operation(self, label, timeout=300, *, kind='preview'):
        previous = _current.get()
        if previous is not None and previous[0] is self:
            # Nested service calls share one deadline, also across the GPU executor.
            if previous[1]['expired']:
                raise NativeOperationTimeout('GPU 操作已超时，正在等待后端恢复')
            yield
            return
        operation = dict(id=str(uuid.uuid4()), phase=label, kind=kind,
                         started=self.clock(), timeout=timeout, cancelled_at=None,
                         expired=False)
        operation['deadline'] = operation['started'] + timeout
        with self.lock:
            self.operations[operation['id']] = operation
            if self.threaded and self.thread is None:
                self.thread = threading.Thread(target=self._run, name='NativeWatchdog', daemon=True)
                self.thread.start()
        token = _current.set((self, operation))
        try:
            yield
            if operation['expired']:
                raise NativeOperationTimeout('GPU 操作已超时，正在等待后端恢复')
        finally:
            _current.reset(token)
            with self.lock:
                self.operations.pop(operation['id'], None)

    def phase(self, operation, label, timeout):
        with self.lock:
            if operation['expired']:
                raise NativeOperationTimeout('GPU 操作已超时，正在等待后端恢复')
            operation.update(phase=label, started=self.clock(), timeout=timeout,
                             deadline=None if timeout is None else self.clock() + timeout)

    def cancel(self, *, kind='export', grace=30):
        with self.lock:
            for operation in self.operations.values():
                if operation['kind'] == kind and operation['cancelled_at'] is None:
                    operation['cancelled_at'] = self.clock() + grace

    @staticmethod
    def _deadline(operation):
        # CPU encoding/remux has its own child-process cancellation/timeout rules.
        if operation['deadline'] is None:
            return None
        cancel = operation['cancelled_at']
        return min(operation['deadline'], cancel) if cancel is not None else operation['deadline']

    def snapshot(self):
        with self.lock:
            timed = [op for op in self.operations.values() if self._deadline(op) is not None]
            if not timed:
                return None
            op = min(timed, key=self._deadline)
            return dict(id=op['id'], phase=op['phase'], kind=op['kind'],
                        elapsed_ms=max(0, (self.clock() - op['started']) * 1000),
                        remaining_ms=max(0, (self._deadline(op) - self.clock()) * 1000),
                        timeout_seconds=op['timeout'], cancel_requested=op['cancelled_at'] is not None,
                        timed_out=op['expired'])

    def check(self):
        faults = []
        with self.lock:
            for operation in self.operations.values():
                deadline = self._deadline(operation)
                if not operation['expired'] and deadline is not None and self.clock() >= deadline:
                    operation['expired'] = True
                    faults.append(dict(operation))
        for operation in faults:
            self.on_timeout(f"GPU 操作无响应：{operation['phase']}")

    def _run(self):
        while True:
            time.sleep(.5)
            with self.lock:
                if not self.operations:
                    self.thread = None
                    return
            self.check()


def phase(label, timeout=300):
    current = _current.get()
    if current is not None:
        current[0].phase(current[1], label, timeout)


def guarded(label, timeout=300, *, kind='preview'):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            try:
                watchdog = self._get_native_watchdog()
                with watchdog.operation(label, timeout, kind=kind):
                    if kind == 'export' and getattr(self, '_cancel_requested', False):
                        watchdog.cancel()
                    return function(self, *args, **kwargs)
            except NativeOperationTimeout as error:
                # Also latch on the owner if it returned just as the observer
                # expired, before the observer's checkpoint callback finished.
                self._mark_native_degraded(str(error))
                raise
        return wrapped
    return decorate
