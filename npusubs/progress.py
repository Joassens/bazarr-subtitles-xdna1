"""Live progress of a transcription: what Whisper has written so far.

openai-whisper prints every segment it finishes as `[mm:ss.mmm --> mm:ss.mmm]  text` when `verbose=True`. `SegmentTap` catches
those lines, so a dashboard can show them while the job runs (like Lingarr shows its translation line by line). The times are
on the timeline Whisper worked on (silence cut out); `to_orig` maps them back to the video's timeline.
"""
from __future__ import annotations

import io
import re
import threading
from typing import Callable

LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d+(?:\.\d+)?) --> (?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")


def _secs(h, m, s) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + float(s)


class SegmentTap(io.TextIOBase):
    """A write-only text stream that turns Whisper's verbose lines into (start, end, text) callbacks."""

    def __init__(self, on_segment: Callable[[float, float, str], None]):
        self.on_segment = on_segment
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = LINE.match(line.strip())
            if m:
                self.on_segment(_secs(m[1], m[2], m[3]), _secs(m[4], m[5], m[6]), m[7].strip())
        return len(s)


class Progress:
    """Thread-safe record of a running transcription (what /health shows)."""

    RECENT = 12

    def __init__(self, total_s: float, to_orig: Callable[[float], float] | None = None):
        self.total_s = max(total_s, 0.001)
        self.to_orig = to_orig or (lambda t: t)
        self.done_s = 0.0
        self.count = 0
        self.recent: list[dict] = []
        self._lock = threading.Lock()

    def add(self, start: float, end: float, text: str) -> None:
        if not text:
            return
        with self._lock:
            self.done_s = max(self.done_s, end)
            self.count += 1
            self.recent.append({"start": round(self.to_orig(start), 1), "text": text})
            del self.recent[:-self.RECENT]

    def snapshot(self) -> dict:
        with self._lock:
            return {"percent": min(100, round(100 * self.done_s / self.total_s)), "segments": self.count,
                    "recent": list(self.recent)}
