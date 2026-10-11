"""Voice activity detection (Silero) and speech packing.

Whisper's encoder always processes a full 30 s window, so running it on many short
speech fragments is wasteful. Instead the speech regions are concatenated (separated by
a short stretch of silence) into one condensed signal, transcribed in one pass, and the
timestamps are mapped back to the original timeline.
"""
from __future__ import annotations

import bisect
import os
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class Piece:
    orig_start: float   # seconds in the original audio
    orig_end: float
    cond_start: float   # seconds in the condensed audio

    @property
    def length(self) -> float:
        return self.orig_end - self.orig_start


class TimeMap:
    """Maps timestamps on the condensed timeline back to the original timeline."""

    def __init__(self, pieces: list[Piece]):
        self.pieces = pieces
        self._starts = [p.cond_start for p in pieces]

    def to_orig(self, t: float, end: bool = False) -> float:
        """Condensed -> original time. A time that falls in a removed gap snaps to the nearest
        real speech: starts move forward to the next piece, ends move back to the previous one."""
        if not self.pieces:
            return t
        i = max(0, bisect.bisect_right(self._starts, t) - 1)
        p = self.pieces[i]
        off = t - p.cond_start
        if off <= p.length:
            return p.orig_start + max(off, 0.0)
        if end or i + 1 >= len(self.pieces):
            return p.orig_end
        return self.pieces[i + 1].orig_start


def condense_regions(audio: np.ndarray, regions: list[tuple[float, float]], join_silence_s: float = 0.3
                     ) -> tuple[np.ndarray, TimeMap]:
    """Concatenate the given regions (separated by join_silence_s of silence) and return the time map."""
    gap = np.zeros(int(join_silence_s * SR), dtype=np.float32)
    chunks, pieces, cond = [], [], 0.0
    for s, e in regions:
        seg = audio[int(s * SR): int(e * SR)]
        chunks.append(seg)
        chunks.append(gap)
        pieces.append(Piece(s, e, cond))
        cond += len(seg) / SR + join_silence_s
    return (np.concatenate(chunks) if chunks else audio), TimeMap(pieces)


class Vad:
    """Silero VAD plus music regions. Parameters come from the live configuration (defaults favour accuracy:
    generous thresholds and padding so no speech is cut off) unless given explicitly, for example in tests."""

    _CFG = {"threshold": "vad.threshold", "min_silence_ms": "vad.min_silence_ms", "speech_pad_ms": "vad.speech_pad_ms",
            "min_speech_ms": "vad.min_speech_ms", "join_gap_s": "vad.join_gap_s", "music": "vad.music.enabled",
            "music_db": "vad.music.db", "min_music_s": "vad.music.min_s"}

    def __init__(self, join_silence_s: float = 0.4, **overrides):
        self._ov = {k: v for k, v in overrides.items() if v is not None and k in self._CFG}
        self.join_silence_s = join_silence_s
        self._model = None

    def __getattr__(self, name):                     # only reached for attributes that are not set on the instance
        key = Vad._CFG.get(name)
        if key is None or name.startswith("_"):
            raise AttributeError(name)
        if name in self._ov:
            return self._ov[name]
        from . import config
        return config.get(key)

    def _load(self):
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        return self._model

    def speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(
            torch.from_numpy(audio), self._load(), sampling_rate=SR, threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms, speech_pad_ms=self.speech_pad_ms,
            min_speech_duration_ms=self.min_speech_ms, return_seconds=True)
        regions = [(float(t["start"]), float(t["end"])) for t in ts]
        merged: list[list[float]] = []
        for s, e in regions:                       # merge regions separated by very short pauses
            if merged and s - merged[-1][1] <= self.join_gap_s:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def music_regions(self, audio: np.ndarray, speech: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Sustained loud non-speech (opening/ending songs, long musical passages).

        A speech VAD cannot see singing under music, yet that is where lyrics subtitles live. A stretch
        counts as music when it lies between speech regions, lasts >= min_music_s and stays above
        music_db (short dips tolerated). Pauses between dialogue are short, so they stay cut.
        """
        if not self.music:
            return []
        frame = SR // 4                                    # 0.25 s
        n = len(audio) // frame
        if n == 0:
            return []
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
        loud = 20 * np.log10(rms + 1e-9) > self.music_db
        total = len(audio) / SR
        bounds = [0.0] + [x for r in speech for x in r] + [total]
        out: list[tuple[float, float]] = []
        for gs, ge in zip(bounds[0::2], bounds[1::2]):     # the gaps between speech regions
            if ge - gs < self.min_music_s:
                continue
            i0, i1 = int(gs / 0.25), min(int(ge / 0.25), n)
            run_start, last_loud = None, None
            for i in range(i0, i1):
                if loud[i]:
                    if run_start is None:
                        run_start = i
                    last_loud = i
                elif run_start is not None and (i - last_loud) * 0.25 > 1.5:       # a dip longer than 1.5 s ends the run
                    if (last_loud + 1 - run_start) * 0.25 >= self.min_music_s:
                        out.append((run_start * 0.25, (last_loud + 1) * 0.25))
                    run_start = None
            if run_start is not None and (last_loud + 1 - run_start) * 0.25 >= self.min_music_s:
                out.append((run_start * 0.25, (last_loud + 1) * 0.25))
        return out

    def regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        speech = self.speech_regions(audio)
        merged = sorted(speech + self.music_regions(audio, speech))
        out: list[list[float]] = []
        for s, e in merged:
            if out and s - out[-1][1] <= self.join_gap_s:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    def condense(self, audio: np.ndarray) -> tuple[np.ndarray, TimeMap, list[tuple[float, float]]]:
        """Return (condensed audio, time map, speech regions). Falls back to the full signal."""
        total = len(audio) / SR
        regions = self.regions(audio)
        if not regions:
            return audio, TimeMap([]), []
        speech = sum(e - s for s, e in regions)
        if total - speech < 5.0:                    # hardly any silence to cut: don't bother
            return audio, TimeMap([]), regions
        condensed, tmap = condense_regions(audio, regions, self.join_silence_s)
        return condensed, tmap, regions
