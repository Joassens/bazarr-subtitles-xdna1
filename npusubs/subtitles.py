"""Turn word-timed segments into subtitle cues and write SRT / WebVTT / plain text.

Rules follow common broadcast guidelines: at most N characters per line, at most 2 lines,
a new cue after a pause, after a speaker change, or after a sentence end once the cue has
some length, balanced two-line cues, minimum/maximum cue duration. CJK characters count
as double width so Japanese/Chinese lines do not overflow.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field

SENTENCE_END = re.compile(r"[.!?。！？…][\"'”’)\]」』）]*$")
SOFT_BREAK = re.compile(r"[,;:、，；：]$")
CONJ = frozenset({"and", "but", "or", "so", "yet", "for", "nor", "as", "if", "when", "then", "because", "although"})


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


@dataclass
class Options:
    max_line: int = 42                 # in display columns
    max_lines: int = 2
    gap_split: float = 0.5
    min_duration: float = 0.8
    max_duration: float = 7.0
    speaker_style: str = "dash"        # none | dash | label
    dash_merge_gap: float = 0.8        # merge short consecutive turns of 2 speakers into one dashed cue

    @classmethod
    def from_config(cls, context: dict | None = None) -> "Options":
        """Options from the live configuration (config file, environment, matching profile)."""
        from . import config
        return cls(max_line=config.get("subtitles.max_line", cls.max_line, context),
                   gap_split=config.get("subtitles.gap_split_s", cls.gap_split, context),
                   min_duration=config.get("subtitles.min_duration_s", cls.min_duration, context),
                   max_duration=config.get("subtitles.max_duration_s", cls.max_duration, context),
                   speaker_style=config.get("subtitles.speaker_style", cls.speaker_style, context))


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] = field(default_factory=list)


def _words_of(seg: dict) -> list[dict]:
    if seg.get("words"):
        return [{**w, "speaker": w.get("speaker", seg.get("speaker"))} for w in seg["words"]]
    return [{"word": seg["text"], "start": seg["start"], "end": seg["end"], "speaker": seg.get("speaker")}]


def _text(words: list[dict]) -> str:
    return "".join(w["word"] for w in words).strip()


def _balanced(words: list[dict], opt: Options) -> str:
    text = _text(words)
    if width(text) <= opt.max_line or opt.max_lines < 2:
        return text
    best, best_score = None, 1e9
    for i in range(1, len(words)):
        a, b = _text(words[:i]), _text(words[i:])
        if width(a) > opt.max_line or width(b) > opt.max_line:
            continue
        score = abs(width(a) - width(b))
        score += -8 if SENTENCE_END.search(words[i - 1]["word"].strip()) else -4 if SOFT_BREAK.search(words[i - 1]["word"].strip()) else 0
        score += 6 if words[i]["word"].strip().lower().rstrip(".,!?;:") in CONJ else 0
        if score < best_score:
            best, best_score = (a, b), score
    return f"{best[0]}\n{best[1]}" if best else text


def build_cues(segments: list[dict], opt: Options | None = None) -> list[Cue]:
    opt = opt or Options()
    words = [w for s in segments for w in _words_of(s) if w["word"].strip()]
    words.sort(key=lambda w: w["start"])
    cues: list[Cue] = []
    cur: list[dict] = []
    cap = opt.max_line * opt.max_lines

    def flush():
        if cur:
            cues.append(Cue(cur[0]["start"], cur[-1]["end"], _balanced(cur, opt), cur[0].get("speaker"), list(cur)))
            cur.clear()

    for w in words:
        if cur:
            if w["start"] - cur[-1]["end"] >= opt.gap_split:
                flush()
            elif opt.speaker_style != "none" and w.get("speaker") != cur[-1].get("speaker") and w.get("speaker"):
                flush()
            elif width(_text(cur) + w["word"]) > cap or w["end"] - cur[0]["start"] > opt.max_duration:
                flush()
        cur.append(w)
        if SENTENCE_END.search(w["word"].strip()) and width(_text(cur)) >= opt.max_line // 2:
            flush()
    flush()
    return _polish(cues, opt)


def _polish(cues: list[Cue], opt: Options) -> list[Cue]:
    # minimum duration (do not run into the next cue), monotonic non-overlapping times
    for i, c in enumerate(cues):
        limit = cues[i + 1].start - 0.04 if i + 1 < len(cues) else c.end + opt.min_duration
        if c.end - c.start < opt.min_duration:
            c.end = min(c.start + opt.min_duration, max(limit, c.end))
        if i + 1 < len(cues) and c.end > cues[i + 1].start:
            c.end = max(c.start + 0.2, cues[i + 1].start - 0.02)
    if opt.speaker_style == "dash":
        merged: list[Cue] = []
        for c in cues:
            p = merged[-1] if merged else None
            if (p and p.speaker and c.speaker and p.speaker != c.speaker and c.start - p.end <= opt.dash_merge_gap
                    and "\n" not in p.text and "\n" not in c.text
                    and width(p.text) <= opt.max_line - 2 and width(c.text) <= opt.max_line - 2):
                p.text = f"- {p.text}\n- {c.text}"
                p.end, p.speaker = c.end, p.speaker
                continue
            merged.append(c)
        cues = merged
    elif opt.speaker_style == "label":
        for c in cues:
            if c.speaker:
                c.text = f"[{c.speaker}] {c.text}"
    return cues


# ----------------------------------------------------------------------- writers
def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "\n".join(f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n{c.text}\n" for i, c in enumerate(cues, 1))


def to_vtt(cues: list[Cue]) -> str:
    return "WEBVTT\n\n" + "\n".join(f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n{c.text}\n" for c in cues)


def to_txt(segments: list[dict]) -> str:
    return "\n".join(s["text"] for s in segments) + "\n"


def to_tsv(segments: list[dict]) -> str:
    return "start\tend\ttext\n" + "\n".join(f"{int(s['start']*1000)}\t{int(s['end']*1000)}\t{s['text']}" for s in segments) + "\n"
