"""Post-filters for typical Whisper hallucinations. Each one is independently switchable."""
from __future__ import annotations

import re


def _key(seg: dict) -> str:
    return re.sub(r"\s+", "", seg["text"]).strip().lower()


def drop_empty(segments: list[dict]) -> list[dict]:
    """Segments without a single word character (a lone '.', '♪' ...)."""
    return [s for s in segments if re.search(r"\w", s["text"])]


def collapse_loops(segments: list[dict], min_cycles: int = 4, max_period: int = 3) -> list[dict]:
    """A stuck decoder repeats the same line (or alternates between a couple of lines).

    Runs of >= min_cycles repeats of a 1..max_period line pattern are collapsed to one cycle;
    the last kept segment is stretched over the whole run so the timeline has no hole.
    """
    keys = [_key(s) for s in segments]
    n, out, i = len(segments), [], 0
    while i < n:
        best_period, best_cycles = 0, 1
        for period in range(1, max_period + 1):
            if i + period > n:
                break
            pattern = keys[i:i + period]
            if not any(pattern):
                continue
            cycles, pos = 1, i + period
            while pos + period <= n and keys[pos:pos + period] == pattern:
                cycles += 1
                pos += period
            if cycles >= min_cycles and cycles * period > best_cycles * best_period:
                best_period, best_cycles = period, cycles
        if best_period:
            run = segments[i:i + best_period * best_cycles]
            kept = [dict(s) for s in run[:best_period]]
            kept[-1]["end"] = run[-1]["end"]
            out.extend(kept)
            i += len(run)
        else:
            out.append(segments[i])
            i += 1
    return out


def drop_recurring(segments: list[dict], min_count: int = 3) -> list[dict]:
    """A full line that recurs min_count+ times in one transcript, not necessarily back to back.

    Real dialogue almost never repeats an identical long line that often; a hallucination
    dropped in separate bursts (e.g. over a musical intro) does. Very short lines are kept.
    """
    counts: dict[str, int] = {}
    for s in segments:
        k = _key(s)
        counts[k] = counts.get(k, 0) + 1
    return [s for s in segments if len(_key(s)) < 6 or counts[_key(s)] < min_count]


def drop_phantom_start(segments: list[dict], max_start: float = 1.0, max_len: float = 2.0) -> list[dict]:
    """Short segment right at t≈0 whose text reappears later (a known artefact)."""
    later = {_key(s) for s in segments[1:]}
    return [s for s in segments
            if not (s["start"] < max_start and s["end"] - s["start"] < max_len and _key(s) in later)]


def apply_all(segments: list[dict], loops: bool = True, recurring: bool = True) -> list[dict]:
    segments = drop_empty(segments)
    segments = drop_phantom_start(segments)
    if loops:
        segments = collapse_loops(segments)
    if recurring:
        segments = drop_recurring(segments)
    for i, s in enumerate(segments):
        s["id"] = i
    return segments
