"""One job at a time: a first-come, first-served gate around a whole request.

The NPU has its own lock, but a request is more than the NPU step: decoding, voice activity detection, beam search on the CPU
and speaker diarization all use the same few cores and the same memory. Two requests running side by side slow each other
down badly (a 30 s clip took 377 s instead of 27 s) and can run the container out of memory. So a request holds a slot for
its whole life; the others wait their turn in order.
"""
from __future__ import annotations

import collections
import contextlib
import threading
import time


class JobGate:
    def __init__(self, slots: int = 1):
        self.slots = max(1, int(slots))
        self._cv = threading.Condition()
        self._queue: collections.deque = collections.deque()
        self.running = 0

    @property
    def waiting(self) -> int:
        with self._cv:
            return len(self._queue)

    @contextlib.contextmanager
    def slot(self, timeout: float):
        """Wait (in order) for a free slot, hold it for the block. TimeoutError when `timeout` seconds pass first."""
        ticket = object()
        deadline = time.monotonic() + timeout
        with self._cv:
            self._queue.append(ticket)
            try:
                while not (self._queue[0] is ticket and self.running < self.slots):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("busy: queue wait exceeded")
                    self._cv.wait(remaining)
                self._queue.popleft()
                self.running += 1
                self._cv.notify_all()                    # the next in line may fit as well (more than one slot)
            except BaseException:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                    self._cv.notify_all()
                raise
        try:
            yield
        finally:
            with self._cv:
                self.running -= 1
                self._cv.notify_all()
