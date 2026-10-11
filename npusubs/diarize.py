"""Speaker diarization ("who speaks when") with pyannote, assigned to Whisper words.

Optional. Needs `pip install "pyannote.audio>=4"`, a Hugging Face token (HF_TOKEN) and a one-time
acceptance of the gated model terms on huggingface.co (pyannote/speaker-diarization-community-1).
Runs on the CPU, so it can overlap with the NPU working on the next request.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("npusubs")
SR = 16000


class Diarizer:
    def __init__(self, model: str | None = None, token: str | None = None):
        from . import config
        self.model = model or config.get("diarization.model")
        self._token = token
        self._pipe = None
        self._lock = threading.Lock()
        self._fail_until = 0.0        # after a failed model load (e.g. gated terms not accepted) do not hammer the hub
        self._fail_reason = ""

    @property
    def token(self) -> str:
        """Hugging Face token: explicit, else HF_TOKEN from the environment or the config folder's secrets.env."""
        from . import config
        return self._token or config.secret("HF_TOKEN") or config.secret("HUGGINGFACE_TOKEN")

    @property
    def available(self) -> bool:
        try:
            import pyannote.audio  # noqa: F401
        except Exception:
            return False
        return bool(self.token) and time.monotonic() >= self._fail_until

    @property
    def status(self) -> str:
        if not self.token:
            return "no HF_TOKEN"
        if time.monotonic() < self._fail_until:
            return f"model unavailable: {self._fail_reason}"
        return "ready"

    def _load(self):
        if self._pipe is None:
            import torch
            from pyannote.audio import Pipeline
            try:
                try:
                    pipe = Pipeline.from_pretrained(self.model, token=self.token)
                except TypeError:                          # pyannote.audio < 4 uses use_auth_token
                    pipe = Pipeline.from_pretrained(self.model, use_auth_token=self.token)
                if pipe is None:
                    raise RuntimeError("pipeline could not be loaded")
            except Exception as exc:                       # gated terms not accepted, no network, bad token ...
                self._fail_until = time.monotonic() + 600.0
                self._fail_reason = (f"{type(exc).__name__}: accept the terms at "
                                     f"https://huggingface.co/{self.model} and check HF_TOKEN")
                log.warning("diarization model %s unavailable (%s); retrying in 10 min", self.model, type(exc).__name__)
                raise
            pipe.to(torch.device("cpu"))
            self._pipe = pipe
        return self._pipe

    def run(self, audio: np.ndarray, min_speakers: int | None = None, max_speakers: int | None = None
            ) -> list[tuple[float, float, str]]:
        import torch
        with self._lock:                                   # one diarization at a time
            pipe = self._load()
            kw = {k: v for k, v in (("min_speakers", min_speakers), ("max_speakers", max_speakers)) if v}
            out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, **kw)
        ann = getattr(out, "speaker_diarization", out)      # pyannote 4 wraps the Annotation
        return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


def run_on_regions(diarizer: "Diarizer", audio: np.ndarray, regions: list[tuple[float, float]],
                   min_speakers: int | None = None, max_speakers: int | None = None
                   ) -> list[tuple[float, float, str]]:
    """Diarize only the given (speech) regions and map the turns back to the original timeline.

    Skipping silence and music makes it several times cheaper on the CPU, and speaker labels on a
    song are meaningless anyway.
    """
    from .vad import condense_regions
    if not regions or sum(e - s for s, e in regions) < 5.0:
        return diarizer.run(audio, min_speakers, max_speakers)
    cond, tmap = condense_regions(audio, regions, 0.5)
    turns = diarizer.run(cond, min_speakers, max_speakers)
    out = []
    for s, e, label in turns:
        so, eo = tmap.to_orig(s), tmap.to_orig(e, end=True)
        if eo > so:
            out.append((so, eo, label))
    return out


def assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Give every word (and segment) the speaker whose turn overlaps it the most."""
    if not turns:
        return segments
    order: dict[str, str] = {}

    def pretty(label: str) -> str:                         # SPEAKER_00 -> SPEAKER_00 in first-seen order
        if label not in order:
            order[label] = f"SPEAKER_{len(order):02d}"
        return order[label]

    def speaker_for(s: float, e: float) -> str | None:
        best, best_ov = None, 0.0
        for ts, te, lab in turns:
            ov = min(e, te) - max(s, ts)
            if ov > best_ov:
                best, best_ov = lab, ov
        if best is None:                                   # word in a gap: nearest turn within 1 s
            mid = (s + e) / 2
            near = min(turns, key=lambda t: 0 if t[0] <= mid <= t[1] else min(abs(mid - t[0]), abs(mid - t[1])))
            if min(abs(mid - near[0]), abs(mid - near[1])) <= 1.0 or near[0] <= mid <= near[1]:
                best = near[2]
        return pretty(best) if best else None

    for seg in segments:
        tally: dict[str, float] = {}
        for w in seg.get("words", []):
            w["speaker"] = speaker_for(w["start"], w["end"])
            if w["speaker"]:
                tally[w["speaker"]] = tally.get(w["speaker"], 0.0) + (w["end"] - w["start"])
        if not seg.get("words"):
            sp = speaker_for(seg["start"], seg["end"])
            if sp:
                tally[sp] = 1.0
        if tally:
            seg["speaker"] = max(tally, key=tally.get)
    return segments
