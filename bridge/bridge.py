# This file is a modified version of McCloudS/bazarr-openai-whisperbridge (Apache License 2.0),
# https://github.com/McCloudS/bazarr-openai-whisperbridge
#
# Changes made in this fork (see bridge/NOTICE-CHANGES.md for the list):
#   * backends: comma separated list of whisper-asr-webservice compatible services, first reachable wins,
#     offline ones are skipped for a while; OpenAI-compatible fallback; optional dedicated translation LLM
#   * word-level speaker labels from diarization split subtitles on a speaker change
#   * accuracy passes for translation: scene "bible" + conservative review (only real corrections)
#   * language aware translation prompt, repetition/phantom/recurring-line filters, OP/ED filename skip
#   * long timeouts for slow backends (NPU)

version = '1.0'

import os
import io
import math
import re
import traceback
import json
import ast
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union
from openai import OpenAI, APIStatusError
import ffmpeg
import time
import uvicorn
import httpx
import subprocess

in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')
whisper_translate_model = os.getenv('WHISPER_TRANSLATE_MODEL', whisper_model)
# Optional: point at an onerahmet/openai-whisper-asr-webservice instance
# instead of an OpenAI-compatible /audio/transcriptions endpoint. That
# service exposes its own /asr?output=json&word_timestamps=true API rather
# than the OpenAI schema, so when this is set we bypass the `client` SDK
# call entirely and hit it directly with httpx (already a dependency).
# WHISPER_ASR_URL may be a comma-separated list in order of preference (e.g. a fast GPU box first,
# an NPU box second); the first one that answers is used, the others are failover.
WHISPER_ASR_URLS = [u.strip().rstrip('/') for u in os.getenv('WHISPER_ASR_URL', '').split(',') if u.strip()]
WHISPER_ASR_URL = WHISPER_ASR_URLS[0] if WHISPER_ASR_URLS else ''

# ---------------------------------------------------------------------------
# Two-machine backend selection (whisper + LLM translation both run on
# either of two AI boxes). Before each transcription/translation we probe
# both with a short, cheap HTTP request and use whichever answers fastest —
# a saturated GPU tends to be slow to respond even to a throwaway request,
# so elapsed time is a reasonable proxy for "which one is free right now".
# If the picked backend then fails mid-task, we retry once on the other.
# Only takes effect when the *_FALLBACK env vars are set; otherwise behaves
# exactly as before (single configured backend, no probing).
# ---------------------------------------------------------------------------
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')
OPENAI_BASE_URL_FALLBACK = os.getenv('OPENAI_BASE_URL_FALLBACK', '').rstrip('/') or None
BACKEND_PROBE_TIMEOUT_SECS = float(os.getenv('BACKEND_PROBE_TIMEOUT_SECS', '3'))
fallback_client = OpenAI(base_url=OPENAI_BASE_URL_FALLBACK) if OPENAI_BASE_URL_FALLBACK else None


def _probe_latency(url: str) -> float | None:
    if not url:
        return None
    try:
        start = time.monotonic()
        httpx.get(url, timeout=BACKEND_PROBE_TIMEOUT_SECS)
        return time.monotonic() - start
    except Exception:
        return None


# A dedicated translation LLM (for example a stronger but slower model on its own port). When set and
# reachable it is used for every LLM call; the primary/fallback pair then only acts as a backup. Whisper
# (transcription) keeps using its own backends.
TRANSLATE_LLM_BASE_URL = os.getenv('TRANSLATE_LLM_BASE_URL', '').rstrip('/') or None
TRANSLATE_LLM_MODEL_DEDICATED = os.getenv('TRANSLATE_LLM_MODEL_DEDICATED', 'default')
translate_client = OpenAI(base_url=TRANSLATE_LLM_BASE_URL) if TRANSLATE_LLM_BASE_URL else None


def _pick_llm_client():
    """Whichever of the primary/fallback OpenAI-compatible endpoints answers
    fastest right now. Sticks to primary if both are unreachable. A dedicated
    translation LLM (TRANSLATE_LLM_BASE_URL) wins whenever it is reachable."""
    if translate_client is not None:
        health = TRANSLATE_LLM_BASE_URL[:-3] + '/health' if TRANSLATE_LLM_BASE_URL.endswith('/v1') else TRANSLATE_LLM_BASE_URL
        if _probe_latency(health) is not None:
            return translate_client
    if fallback_client is None:
        return client
    primary_latency = _probe_latency(OPENAI_BASE_URL)
    fallback_latency = _probe_latency(OPENAI_BASE_URL_FALLBACK)
    if fallback_latency is not None and (primary_latency is None or fallback_latency < primary_latency):
        return fallback_client
    return client
# Optional static vocabulary hint (character names, terminology), prepended to
# whatever dynamic *arr context is found below. Empty by default.
whisper_initial_prompt = os.getenv('WHISPER_INITIAL_PROMPT', '') or None

# ---------------------------------------------------------------------------
# *arr metadata lookup (Sonarr/Radarr) — used to auto-build a per-file Whisper
# initial_prompt from the show/movie's own title + overview, instead of
# hardcoding show-specific vocabulary. Works for anything already indexed by
# Sonarr/Radarr, no per-show configuration needed.
# ---------------------------------------------------------------------------
SONARR_URL     = os.getenv('SONARR_URL', '').rstrip('/')
SONARR_API_KEY = os.getenv('SONARR_API_KEY', '')
RADARR_URL     = os.getenv('RADARR_URL', '').rstrip('/')
RADARR_API_KEY = os.getenv('RADARR_API_KEY', '')
MEDIA_CACHE_TTL_SECS = float(os.getenv('MEDIA_CACHE_TTL_SECS', '1800'))

# Filename tokens that mark a file as pure OP/ED theme music with no dialogue
# (NCOP/NCED = "non-credit" clean opening/ending, plain OP1/ED2 etc.). Whisper
# reliably hallucinates (repeats a phantom phrase) on long musical passages
# with no speech, so these are skipped entirely rather than transcribed.
SKIP_FILENAME_PATTERN = re.compile(
    os.getenv('SKIP_FILENAME_PATTERN', r'(?i)\b(NC)?(OP|ED)\d{0,2}\b')
)


def should_skip_file(video_file: str | None) -> bool:
    if not video_file:
        return False
    return bool(SKIP_FILENAME_PATTERN.search(os.path.basename(video_file)))

_media_cache: list[tuple[str, str]] = []  # (root_path, context_text)
_media_cache_fetched_at = 0.0


def _fetch_arr_entries(base_url: str, api_key: str, list_endpoint: str) -> list[tuple[str, str]]:
    if not base_url or not api_key:
        return []
    entries: list[tuple[str, str]] = []
    try:
        resp = httpx.get(
            f"{base_url}{list_endpoint}",
            headers={"X-Api-Key": api_key},
            timeout=10.0,
        )
        resp.raise_for_status()
        for item in resp.json():
            path = item.get("path")
            title = item.get("title")
            overview = (item.get("overview") or "").strip()
            year = item.get("year")
            if not path or not title:
                continue
            context = f"{title}" + (f" ({year})" if year else "")
            if overview:
                # Keep it short — Whisper's initial_prompt is most effective as
                # a brief style/vocabulary primer, not a full summary.
                context += ": " + overview[:400]
            entries.append((path, context))
    except Exception as exc:
        print(f"*arr metadata lookup failed for {base_url}: {exc}")
    return entries


def _refresh_media_cache() -> None:
    global _media_cache, _media_cache_fetched_at
    now = time.time()
    if _media_cache and (now - _media_cache_fetched_at) < MEDIA_CACHE_TTL_SECS:
        return
    entries = _fetch_arr_entries(SONARR_URL, SONARR_API_KEY, "/api/v3/series")
    entries += _fetch_arr_entries(RADARR_URL, RADARR_API_KEY, "/api/v3/movie")
    # Longest path first so a nested match (if any) wins over a shorter parent.
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    _media_cache = entries
    _media_cache_fetched_at = now
    print(f"*arr metadata cache refreshed: {len(entries)} series/movies")


def get_media_context(video_file: str | None) -> str | None:
    if not video_file:
        return None
    try:
        _refresh_media_cache()
        for root_path, context in _media_cache:
            if video_file.startswith(root_path):
                return context
    except Exception as exc:
        print(f"*arr metadata match failed: {exc}")
    return None


# Short, natural-sounding subtitle-style primer per target language.
# Whisper's initial_prompt biases the decoder's language model toward
# whatever text precedes the audio — priming it with English prose right
# before non-English audio measurably nudges low-confidence/hallucinated
# passages (e.g. music with no real speech) toward English output instead of
# the target language. Keyed by language code; falls back to English.
SUBTITLE_STYLE_PRIMER = {
    'ja': 'これは日本語アニメの音声から書き起こした、自然な会話の字幕です。',
}
DEFAULT_SUBTITLE_STYLE_PRIMER = 'This is a natural conversation transcribed for subtitles.'


def build_initial_prompt(video_file: str | None, language: str | None = None) -> str | None:
    parts = [p for p in (whisper_initial_prompt,) if p]
    context = get_media_context(video_file)
    if context:
        if language and language != 'en':
            # Keep only the title (names/proper nouns are language-agnostic
            # vocabulary hints) — drop the English overview prose so it can't
            # bias hallucinated passages toward English text.
            parts.append(context.split(':', 1)[0])
        else:
            parts.append(context)
    parts.append(SUBTITLE_STYLE_PRIMER.get(language or 'en', DEFAULT_SUBTITLE_STYLE_PRIMER))
    return "\n".join(parts) or None

MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024
OPUS_BITRATE_BPS = int(os.getenv('OPUS_BITRATE_KBPS', '24')) * 1000
OPUS_APPLICATION = "audio"

# Netflix subtitle guidelines
MAX_LINE_LENGTH = int(os.getenv('MAX_LINE_LENGTH', '42'))  # chars per line
MAX_LINES       = 2                                         # lines per subtitle
# Start a new subtitle after a silence gap of this many seconds.
# Handles silence suppression — subtitle won't bleed into pauses.
GAP_SPLIT_SECS  = float(os.getenv('GAP_SPLIT_SECS', '0.4'))

# Phantom-segment filter (issue #3 — Groq/Whisper hallucination at t≈0)
# Drop any segment whose start is within this many seconds of 0, whose
# duration is shorter than the duration threshold, and whose text is a
# duplicate of a later entry.  Set PHANTOM_START_MAX_SECS=0 to disable.
PHANTOM_START_MAX_SECS    = float(os.getenv('PHANTOM_START_MAX_SECS',    '1.0'))
PHANTOM_DURATION_MAX_SECS = float(os.getenv('PHANTOM_DURATION_MAX_SECS', '2.0'))

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# Punctuation patterns for split decisions
_SENTENCE_END = re.compile(r'[.!?][\'")\]]*$')   # strong break after
_SOFT_BREAK   = re.compile(r'[,;:]$')             # weak break after
_CONJUNCTIONS = frozenset({
    'and', 'but', 'or', 'so', 'yet', 'for', 'nor',
    'as', 'if', 'when', 'then', 'because', 'although',
})


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis   = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis    = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n"
            f"{seconds_to_srt_timestamp(seg['start'])} --> {seconds_to_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(lines)


def filter_phantom_segments(segments: list[dict]) -> list[dict]:
    """
    Strip Whisper hallucination entries near timestamp 0 (issue #3).

    Groq (and occasionally OpenAI) emits two kinds of spurious near-zero
    segments:
      1. Duplicate text — same text appears again in a real later entry.
      2. Content-free — text contains no word characters (e.g. a lone ".").

    A segment is dropped when ALL of these hold:
      - start < PHANTOM_START_MAX_SECS
      - duration < PHANTOM_DURATION_MAX_SECS
      - text is a duplicate of a later entry OR contains no word characters

    A legitimate opening subtitle that is short but has unique, readable
    content is kept.  Set PHANTOM_START_MAX_SECS=0 to disable entirely.
    """
    if not segments or PHANTOM_START_MAX_SECS <= 0:
        return segments

    later_texts = {s["text"].strip().lower() for s in segments[1:]}
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text_stripped = seg["text"].strip()
        is_phantom_text = (
            text_stripped.lower() in later_texts          # duplicate of later entry
            or not re.search(r'\w', text_stripped)        # no real word chars (e.g. ".")
        )
        if (
            seg["start"] < PHANTOM_START_MAX_SECS
            and duration < PHANTOM_DURATION_MAX_SECS
            and is_phantom_text
        ):
            print(
                f"Filtered phantom segment at "
                f"{seconds_to_srt_timestamp(seg['start'])}: "
                f'"{text_stripped}"'
            )
            continue
        result.append(seg)
    return result


REPETITION_MIN_RUN_LENGTH = int(os.getenv('REPETITION_MIN_RUN_LENGTH', '4'))
REPETITION_MAX_PERIOD = int(os.getenv('REPETITION_MAX_PERIOD', '3'))


def filter_repetition_loops(segments: list[dict]) -> list[dict]:
    """
    Collapse runs of many consecutive segments that cycle through a short
    repeating pattern of 1..REPETITION_MAX_PERIOD distinct lines.

    A Whisper decoder that gets stuck (triggered by music, a chunk-boundary
    artifact, or a long low-confidence stretch) repeats the same line, or
    alternates between a couple of lines, across many consecutive segments —
    something real dialogue essentially never does. Starting at each
    position, the longest short-period cycle (period 1..REPETITION_MAX_PERIOD)
    that repeats at least REPETITION_MIN_RUN_LENGTH times is collapsed down
    to its first cycle; the rest of the run is dropped and the last kept
    segment's end time is extended to cover the whole collapsed run so the
    subtitle timeline has no gap.
    """
    if not segments or REPETITION_MIN_RUN_LENGTH <= 0:
        return segments

    def norm(seg: dict) -> str:
        return re.sub(r'\s+', '', seg['text']).strip().lower()

    keys = [norm(s) for s in segments]
    n = len(segments)
    result = []
    i = 0
    while i < n:
        best_period, best_cycles = 0, 1
        for period in range(1, REPETITION_MAX_PERIOD + 1):
            if i + period > n:
                break
            pattern = keys[i:i + period]
            if not any(pattern):
                continue
            cycles = 1
            pos = i + period
            while pos + period <= n and keys[pos:pos + period] == pattern:
                cycles += 1
                pos += period
            if cycles >= REPETITION_MIN_RUN_LENGTH and cycles * period > best_cycles * best_period:
                best_period, best_cycles = period, cycles
        if best_period:
            run_len = best_period * best_cycles
            run = segments[i:i + run_len]
            kept = [dict(s) for s in run[:best_period]]
            kept[-1]['end'] = run[-1]['end']
            print(
                f"Filtered repetition loop at {seconds_to_srt_timestamp(run[0]['start'])}: "
                f"pattern of {best_period} line(s) repeated {best_cycles}x, dropped {run_len - best_period}."
            )
            result.extend(kept)
            i += run_len
        else:
            result.append(segments[i])
            i += 1
    return result


GLOBAL_DUPLICATE_MIN_COUNT = int(os.getenv('GLOBAL_DUPLICATE_MIN_COUNT', '3'))


def filter_global_duplicate_lines(segments: list[dict]) -> list[dict]:
    """
    Drop every segment whose normalized text recurs GLOBAL_DUPLICATE_MIN_COUNT
    or more times anywhere in the transcription, even when the occurrences
    aren't consecutive.

    filter_repetition_loops only catches a hallucinated line while the
    decoder is stuck repeating it back-to-back. A decoder can also drop the
    same hallucinated line into separate, non-adjacent bursts (e.g. once per
    short blip of song/silence across a long musical intro) — real dialogue
    essentially never repeats an identical full line 3+ separate times in
    one episode, so any text that does is almost certainly a recurring
    hallucination and is removed outright rather than just collapsed.
    """
    if not segments or GLOBAL_DUPLICATE_MIN_COUNT <= 0:
        return segments

    def norm(seg: dict) -> str:
        return re.sub(r'\s+', '', seg['text']).strip().lower()

    counts: dict[str, int] = {}
    for seg in segments:
        key = norm(seg)
        if key:
            counts[key] = counts.get(key, 0) + 1

    reported: set[str] = set()
    result = []
    for seg in segments:
        key = norm(seg)
        if key and counts[key] >= GLOBAL_DUPLICATE_MIN_COUNT:
            if key not in reported:
                print(
                    f"Filtered recurring hallucinated line "
                    f'"{seg["text"].strip()}" ({counts[key]}x non-consecutive occurrences across episode).'
                )
                reported.add(key)
            continue
        result.append(seg)
    return result


# ---------------------------------------------------------------------------
# Netflix-style subtitle segmentation
# ---------------------------------------------------------------------------

def split_segments(words: list[dict]) -> list[dict]:
    """
    Convert a flat word list into subtitle segments following Netflix guidelines:
      - Max MAX_LINE_LENGTH characters per line
      - Max MAX_LINES lines per subtitle (2)
      - New subtitle after GAP_SPLIT_SECS silence (handles silence suppression)
      - New subtitle after sentence-ending punctuation
      - 2-line subtitles split at the most natural word boundary
      - Lines balanced in length; prefer breaking after punctuation,
        avoid breaking before conjunctions
    """
    if not words:
        return []

    segments: list[dict]       = []
    current: list[dict]        = []
    max_chars = MAX_LINE_LENGTH * MAX_LINES

    def flush():
        if current:
            segments.append(_format_subtitle(current))
            current.clear()

    for i, word in enumerate(words):
        # --- gap-based split (silence suppression) ---
        if current:
            gap = word["start"] - current[-1]["end"]
            if gap >= GAP_SPLIT_SECS:
                flush()
            # --- speaker change (diarization): never put two speakers in one subtitle ---
            elif (word.get("speaker") and current[-1].get("speaker")
                  and word["speaker"] != current[-1]["speaker"]):
                flush()

        # --- would exceed max subtitle length? ---
        # Each word token already carries its own natural leading space (or
        # none, for languages/scripts like Japanese that don't use spaces
        # between words) — concatenate directly rather than inserting one.
        candidate = "".join(w["word"] for w in current) + word["word"]
        if current and len(candidate) > max_chars:
            flush()

        current.append(word)

        # --- sentence-end split (only if subtitle is long enough to be worth it) ---
        text_so_far = "".join(w["word"] for w in current)
        if _SENTENCE_END.search(word["word"]) and len(text_so_far) >= MAX_LINE_LENGTH // 2:
            flush()

    flush()
    return segments


def _format_subtitle(words: list[dict]) -> dict:
    """
    Format a word group as a 1- or 2-line subtitle dict.
    For 2-line subtitles, find the most natural split point:
      - Prefer after punctuation near the middle
      - Avoid breaking before conjunctions
      - Aim for balanced line lengths
    """
    text = "".join(w["word"] for w in words).strip()
    start = words[0]["start"]
    end   = words[-1]["end"]

    if len(text) <= MAX_LINE_LENGTH:
        return {"start": start, "end": end, "text": text}

    # Find best split point for 2 lines
    best_idx   = _find_line_split(words)
    line1 = "".join(w["word"] for w in words[:best_idx]).strip()
    line2 = "".join(w["word"] for w in words[best_idx:]).strip()

    # Safety: if either line still exceeds the limit, fall back to single block
    if len(line1) > MAX_LINE_LENGTH or len(line2) > MAX_LINE_LENGTH:
        return {"start": start, "end": end, "text": text}

    return {"start": start, "end": end, "text": f"{line1}\n{line2}"}


def _find_line_split(words: list[dict]) -> int:
    """
    Return the word index at which to split words into two balanced lines.

    Scoring (lower = better split):
      - Penalise imbalance between line lengths
      - Reward splitting after sentence-end or soft punctuation
      - Penalise splitting before a conjunction
    """
    texts = [w["word"] for w in words]
    n     = len(texts)

    best_idx   = max(1, n // 2)
    best_score = float("inf")

    for i in range(1, n):
        line1 = "".join(texts[:i])
        line2 = "".join(texts[i:])

        if len(line1) > MAX_LINE_LENGTH or len(line2) > MAX_LINE_LENGTH:
            continue

        balance      = abs(len(line1) - len(line2))
        punct_bonus  = -8 if _SENTENCE_END.search(texts[i - 1]) else \
                       -4 if _SOFT_BREAK.search(texts[i - 1]) else 0
        conj_penalty =  6 if texts[i].lower().rstrip(".,!?;:") in _CONJUNCTIONS else 0

        score = balance + punct_bonus + conj_penalty
        if score < best_score:
            best_score = score
            best_idx   = i

    return best_idx


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _field(obj, key: str, default=None):
    """
    Read `key` from `obj` whether it's a Pydantic-style object (attribute
    access) or a plain dict (some whisper-server responses deserialize
    nested word lists as dicts rather than typed objects, depending on the
    SDK's schema matching — getattr() alone silently returns the default for
    those and made every word look timestamp-less, so this checks both).
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clean_word(word: str) -> str:
    """
    Drop stray U+FFFD replacement characters.

    whisper.cpp's tokenizer occasionally splits a single multi-byte CJK
    character's UTF-8 bytes across two word tokens; each half is invalid
    UTF-8 on its own and gets replaced with U+FFFD before we ever see it, so
    there is no way to recover the original character here — dropping the
    placeholder is less misleading than displaying it.
    """
    return word.replace('�', '')


def verbose_json_to_segments(response) -> list[dict]:
    """
    Extract and segment the API response into subtitle-ready dicts.

    Groq returns words at the top level; OpenAI nests them in segments.
    Either way we pass the word list through split_segments() so line
    length, gap splitting, and punctuation-aware breaks are applied uniformly.
    """
    # --- Groq: top-level words ---
    # Word text is intentionally NOT stripped here: each token carries its
    # own natural leading space (or none, for scripts like Japanese that
    # don't use spaces between words), which split_segments()/_format_subtitle()
    # rely on when concatenating words back into subtitle text.
    top_words = [
        {
            "word":  _clean_word(_field(w, "word", "") or ""),
            "start": _field(w, "start", None),
            "end":   _field(w, "end",   None),
            "speaker": _field(w, "speaker", None),
        }
        for w in (_field(response, "words", None) or [])
        if _field(w, "start", None) is not None
        and _field(w, "end",   None) is not None
    ]
    if top_words:
        top_words.sort(key=lambda w: w["start"])
        print(f"Using {len(top_words)} top-level words.")
        return split_segments(top_words)

    # --- OpenAI: segment-nested words ---
    raw_segments = _field(response, "segments", None) or []
    segments = []
    for seg in raw_segments:
        start = _field(seg, "start", None)
        end   = _field(seg, "end",   None)
        text  = (_field(seg, "text", "") or "").strip()
        if start is None or end is None:
            continue
        words = [
            {
                "word":  _clean_word(_field(w, "word", "") or ""),
                "start": _field(w, "start", None),
                "end":   _field(w, "end",   None),
                "speaker": _field(w, "speaker", None) or _field(seg, "speaker", None),
            }
            for w in (_field(seg, "words", None) or [])
            if _field(w, "start", None) is not None
            and _field(w, "end",   None) is not None
        ]
        if words:
            segments.extend(split_segments(words))
        else:
            segments.append({"start": start, "end": end, "text": text})
    return segments


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def is_too_large_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code == 413:
        return True
    msg = str(exc).lower()
    return "request_too_large" in msg or "request entity too large" in msg


# ---------------------------------------------------------------------------
# Opus encoding
# ---------------------------------------------------------------------------

def encode_pcm_to_opus(pcm_bytes: bytes) -> io.BytesIO:
    try:
        out, err = (
            ffmpeg.input("pipe:0", format="s16le", ar=PCM_SAMPLE_RATE, ac=PCM_NUM_CHANNELS)
            .output(
                "pipe:1",
                format="wav",
                acodec="pcm_s16le",
                ac=PCM_NUM_CHANNELS,
                ar=PCM_SAMPLE_RATE,
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, input=pcm_bytes)
        )
        if not out:
            stderr_msg = err.decode(errors="replace") if err else "(no stderr)"
            raise RuntimeError(f"FFmpeg produced no output. stderr: {stderr_msg}")
        opus = io.BytesIO(out)
        opus.name = "audio.wav"
        opus.seek(0)
        return opus
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode(errors="replace") if e.stderr else "(no stderr)"
        raise RuntimeError(f"FFmpeg error: {stderr_msg}") from e


# ---------------------------------------------------------------------------
# PCM chunking  (sample-accurate offsets)
# ---------------------------------------------------------------------------

SILENCE_SEARCH_WINDOW_SECS = float(os.getenv('SILENCE_SEARCH_WINDOW_SECS', '20'))
SILENCE_NOISE_DB           = os.getenv('SILENCE_NOISE_DB', '-35dB')
SILENCE_MIN_DURATION_SECS  = float(os.getenv('SILENCE_MIN_DURATION_SECS', '0.3'))


def find_silence_midpoints(pcm_bytes: bytes) -> list[float]:
    """Run ffmpeg's silencedetect over the raw PCM and return the midpoint
    (seconds) of every detected silence interval, so chunk boundaries can be
    snapped to an actual gap instead of an arbitrary byte offset — cutting
    mid-sentence can leave the audio on either side of the cut too short/
    context-less for VAD or Whisper to recognize reliably."""
    try:
        proc = subprocess.run(
            [
                'ffmpeg', '-f', 's16le', '-ar', str(PCM_SAMPLE_RATE), '-ac', str(PCM_NUM_CHANNELS),
                '-i', 'pipe:0', '-af', f'silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_DURATION_SECS}',
                '-f', 'null', '-',
            ],
            input=pcm_bytes, capture_output=True, timeout=120,
        )
        stderr = proc.stderr.decode(errors='ignore')
        starts = [float(m) for m in re.findall(r'silence_start:\s*([\d.]+)', stderr)]
        ends   = [float(m) for m in re.findall(r'silence_end:\s*([\d.]+)', stderr)]
        midpoints = [(s + e) / 2 for s, e in zip(starts, ends)]
        return sorted(midpoints)
    except Exception as exc:
        print(f"Silence detection failed, falling back to fixed-size splits: {exc}")
        return []


def split_pcm_into_chunks(pcm_bytes: bytes, num_chunks: int) -> list[tuple[bytes, float]]:
    total_samples  = len(pcm_bytes) // PCM_BYTES_PER_SAMPLE
    total_duration = total_samples / PCM_SAMPLE_RATE

    print(
        f"Splitting {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM / {total_duration:.1f}s "
        f"→ {num_chunks} chunks of ~{total_duration / num_chunks:.1f}s each (silence-aware)."
    )

    silence_midpoints = find_silence_midpoints(pcm_bytes)

    # Pick one boundary time per internal split point, snapped to the nearest
    # detected silence within the search window when one exists.
    boundary_times = [0.0]
    for i in range(1, num_chunks):
        target = total_duration * i / num_chunks
        candidates = [t for t in silence_midpoints if abs(t - target) <= SILENCE_SEARCH_WINDOW_SECS]
        chosen = min(candidates, key=lambda t: abs(t - target)) if candidates else target
        # Keep boundaries monotonically increasing even if two targets snap
        # to the same nearby silence gap.
        chosen = max(chosen, boundary_times[-1] + 1.0)
        boundary_times.append(min(chosen, total_duration))
        print(f"  Split {i}: target={target:.1f}s -> {chosen:.1f}s ({'silence-snapped' if candidates else 'fixed'})")
    boundary_times.append(total_duration)

    chunks = []
    for i in range(len(boundary_times) - 1):
        start_offset = boundary_times[i]
        end_offset   = boundary_times[i + 1]
        start_byte   = int(start_offset * PCM_SAMPLE_RATE) * PCM_BYTES_PER_SAMPLE
        end_byte     = min(int(end_offset * PCM_SAMPLE_RATE) * PCM_BYTES_PER_SAMPLE, len(pcm_bytes))
        chunk_pcm    = pcm_bytes[start_byte:end_byte]
        if not chunk_pcm:
            continue
        print(f"  Chunk {i + 1}/{len(boundary_times) - 1}: offset={start_offset:.3f}s  pcm={len(chunk_pcm) / 1024 / 1024:.1f} MB")
        chunks.append((chunk_pcm, start_offset))

    return chunks


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

TRANSLATE_LLM_MODEL = os.getenv("TRANSLATE_LLM_MODEL", "Qwen3-4B-Instruct-2507-GGUF-Q4_K_M")
# The two backends may expose the same underlying model under different
# names (e.g. Ollama's "hf.co/..." tag vs Lemonade's plain model name), so
# the fallback backend gets its own model name, defaulting to the primary's.
TRANSLATE_LLM_MODEL_FALLBACK = os.getenv("TRANSLATE_LLM_MODEL_FALLBACK", TRANSLATE_LLM_MODEL)


def _llm_model_for(attempt_client) -> str:
    if translate_client is not None and attempt_client is translate_client:
        return TRANSLATE_LLM_MODEL_DEDICATED
    return TRANSLATE_LLM_MODEL_FALLBACK if attempt_client is fallback_client else TRANSLATE_LLM_MODEL
# Lines are translated in small batches so the LLM gets surrounding-dialogue
# context instead of a single isolated line, without risking the JSON-shape
# mismatches and slow responses of translating an entire file in one call.
TRANSLATE_BATCH_SIZE = int(os.getenv("TRANSLATE_BATCH_SIZE", "5"))


TRANSLATE_TEMPERATURE = float(os.getenv("TRANSLATE_TEMPERATURE", "0.2"))
TRANSLATE_KEEP_HONORIFICS = os.getenv("TRANSLATE_KEEP_HONORIFICS", "true").lower() == "true"
# How many already-translated lines are shown to the LLM as preceding context.
TRANSLATE_CONTEXT_LINES = int(os.getenv("TRANSLATE_CONTEXT_LINES", "6"))

_LANGUAGE_NAMES = {"ja": "Japanese", "en": "English", "nl": "Dutch", "ko": "Korean", "zh": "Chinese",
                   "de": "German", "fr": "French", "es": "Spanish"}

_JAPANESE_RULES = (
    "- Japanese drops subjects and pronouns: infer who is speaking and to whom from the surrounding lines "
    "and supply the right pronoun in English; never write 'it' or 'that person' when a name or he/she fits.\n"
    "- Render sentence-final particles, hedges and politeness levels as English tone (e.g. casual, blunt, polite, "
    "playful), not as words. Do not translate set phrases literally (e.g. 'otsukaresama', 'yoroshiku onegaishimasu', "
    "'itadakimasu' -> what an English speaker would say in that situation, or keep itadakimasu when it is a meal ritual).\n"
    "- Onomatopoeia and exclamations: use natural English equivalents. Wordplay: keep the intent, not the sound.\n"
)


def _translation_system_prompt(source_language: str | None, context: str | None) -> str:
    source = _LANGUAGE_NAMES.get((source_language or "").lower(), "the source language")
    parts = [
        f"You are a professional subtitle translator. Translate subtitles from {source} into natural, "
        "idiomatic English, as a native English speaker would say it on screen.",
        "Rules:",
        "- Translate the meaning and intent, not the words. Never translate word-for-word or idiom-for-idiom. "
        "If a line sounds odd or ungrammatical in English, you translated too literally: rephrase it.",
        "- Keep it short and speakable like subtitles: natural spoken register, no stiff or bookish phrasing. "
        "Match each speaker's tone (casual, rude, formal, comedic, childlike).",
    ]
    if (source_language or "").lower() == "ja":
        parts.append(_JAPANESE_RULES.rstrip("\n"))
        if TRANSLATE_KEEP_HONORIFICS:
            parts.append("- Keep common honorifics (-san, -kun, -chan, -sensei, -senpai) attached to the name, "
                         "as in standard anime fansubs; do not invent honorifics that are not spoken.")
        else:
            parts.append("- Drop Japanese honorifics; address people by name or with natural English forms.")
    parts += [
        "- Keep character names and place names consistent; do not translate brand names or song titles.",
        "- The transcript comes from automatic speech recognition and may contain recognition errors: "
        "if a word makes no sense, pick the likeliest intended word from context.",
        "- Do not censor profanity; use an equally strong natural equivalent.",
        "- Reply ONLY in the requested format, with no notes. /no_think",
    ]
    if context:
        parts.insert(1, f"Show/movie context: {context}")
    return "\n".join(parts)


def _translate_batch(texts: list[str], system_prompt: str | None = None,
                     previous: list[str] | None = None) -> list[str] | None:
    """Translate a small batch of subtitle lines together for context.
    Returns the translated list, or None if the batch call failed/was malformed."""
    prior = ""
    if previous:
        prior = ("Already translated lines just before this batch (for context only, do NOT repeat them): "
                 + json.dumps(previous, ensure_ascii=False) + "\n\n")
    prompt = (
        prior
        + "Translate this JSON array of subtitle lines into natural English. "
        "The lines are consecutive dialogue from the same scene — use them as "
        "context for each other. "
        "Return ONLY a JSON array of the same length, in the same order, "
        "with the translated strings and nothing else: " + json.dumps(texts, ensure_ascii=False)
    )
    system_prompt = (system_prompt or "You are a precise subtitle translator.") + "\nReply with a JSON array only."
    picked = _pick_llm_client()
    candidates = [picked] + [c for c in (client, fallback_client) if c is not None and c is not picked]
    for attempt_client in candidates:
        try:
            response = attempt_client.chat.completions.create(
                model=_llm_model_for(attempt_client),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=TRANSLATE_TEMPERATURE,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()
            start = content.find("[")
            end = content.rfind("]")
            translated = json.loads(content[start:end + 1])
            if isinstance(translated, list) and len(translated) == len(texts):
                return [_sanitize_translation(str(t), texts[i]) for i, t in enumerate(translated)]
            got = len(translated) if isinstance(translated, list) else type(translated).__name__
            print(f"Translate LLM returned unexpected shape ({got} items for {len(texts)} lines).")
            return None  # malformed reply won't be fixed by switching backend
        except Exception as exc:
            suffix = " — trying next backend." if attempt_client is not candidates[-1] else ""
            print(f"Batch LLM translation failed ({exc}){suffix}")
    return None


def _sanitize_translation(text: str, fallback: str) -> str:
    """
    Defensively unwrap a translation that came back dict/JSON-shaped instead
    of as a plain string.

    The translation LLM occasionally echoes something like
    "{'text': 'actual translation'}" instead of just the translation —
    likely bleed-through from the batch prompt's JSON-array format. Detect
    that shape and pull the real text out rather than writing the literal
    dict repr into the subtitle.
    """
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}") and "text" in stripped:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                    return parsed["text"].strip() or fallback
            except Exception:
                pass
    return stripped or fallback


def _translate_single(text: str, system_prompt: str | None = None) -> str:
    picked = _pick_llm_client()
    candidates = [picked] + [c for c in (client, fallback_client) if c is not None and c is not picked]
    system_prompt = (system_prompt or "Translate the following subtitle line into natural English.") \
        + "\nReply with ONLY the translation of the single line you are given."
    for attempt_client in candidates:
        try:
            response = attempt_client.chat.completions.create(
                model=_llm_model_for(attempt_client),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=TRANSLATE_TEMPERATURE,
                max_tokens=2000,
            )
            return _sanitize_translation(response.choices[0].message.content, text)
        except Exception as exc:
            suffix = " — trying next backend." if attempt_client is not candidates[-1] else ""
            print(f"Per-segment translation failed ({exc}){suffix}")
    return text


# --- accuracy passes (slow on purpose: this runs as a background job) --------------------------------
# 1. "bible": the model reads the whole script once and writes down names, terms, gender/relationships and
#    likely speech-recognition errors, which then guides every batch (consistent names and pronouns).
# 2. "review": after the draft translation an editor pass compares source and draft in context and fixes
#    mistranslations, literal phrasing, wrong pronouns and inconsistencies.
TRANSLATE_BIBLE = os.getenv("TRANSLATE_BIBLE", "true").lower() == "true"
TRANSLATE_REVIEW = os.getenv("TRANSLATE_REVIEW", "true").lower() == "true"
TRANSLATE_REVIEW_BATCH = int(os.getenv("TRANSLATE_REVIEW_BATCH", "10"))
TRANSLATE_BIBLE_MAX_CHARS = int(os.getenv("TRANSLATE_BIBLE_MAX_CHARS", "24000"))
TRANSLATE_DEBUG = os.getenv("TRANSLATE_DEBUG", "false").lower() == "true"   # log the bible and every review edit


def _chat(system: str, user: str, max_tokens: int = 1500) -> str | None:
    """One LLM call on whichever backend answers; None on failure."""
    picked = _pick_llm_client()
    for attempt_client in [picked] + [c for c in (client, fallback_client) if c is not None and c is not picked]:
        try:
            response = attempt_client.chat.completions.create(
                model=_llm_model_for(attempt_client),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0, max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"LLM call failed ({exc}); trying next backend.")
    return None


def _build_scene_bible(texts: list[str], context: str | None, source_language: str | None) -> str | None:
    source = _LANGUAGE_NAMES.get((source_language or "").lower(), "the source language")
    script = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))[:TRANSLATE_BIBLE_MAX_CHARS]
    system = ("You are a translator's research assistant. You read a whole subtitle script and prepare a compact "
              "translation bible so that every line is translated consistently. /no_think")
    user = (f"The script below is {source} dialogue from automatic speech recognition (it can contain errors).\n"
            + (f"Show/movie context: {context}\n" if context else "")
            + "\nSCRIPT:\n" + script +
            "\n\nWrite the bible in English, at most 200 words, no preamble, with these parts:\n"
            "1) setting and tone;\n"
            "2) characters that are actually named or clearly present in the script: the English name as a subtitle "
            "would spell it, gender, relationships, speech style. Do not invent characters or facts;\n"
            "3) recurring terms, places, dishes, titles, each with ONE fixed English rendering. Only list terms that "
            "really recur.")
    return _chat(system, user, max_tokens=800)


def _review_translation(src: list[str], draft: list[str], system_prompt: str, reasons: dict | None = None) -> list[str]:
    """Editor pass over the draft. Never makes things worse: any malformed reply keeps the draft."""
    final = list(draft)
    review_system = (
        system_prompt + "\n\nYou are now acting as a senior subtitle EDITOR. For each numbered pair you get the source "
        "line and an English draft. The draft is usually already good: be CONSERVATIVE and only fix a line when it is "
        "clearly wrong, i.e. a mistranslation, a wrong pronoun or gender (use the context and the bible), a name or "
        "term that contradicts the bible, a literal rendering of an idiom that no native speaker would say, or "
        "when the source is a probable speech-recognition mishearing and the context shows what was meant. Do NOT "
        "rephrase lines that are merely different from how you would write them, and never invent words that are "
        "not in the source. Reply ONLY with a JSON array with one object per line, in order: "
        '{"n": <number>, "verdict": "keep"} or {"n": <number>, "verdict": "fix", "english": "<the corrected line>", '
        '"why": "<exactly one of: mistranslation | pronoun-gender | name | grammar | misheard | literal-idiom>"}. '
        "Style or flow preferences are NOT a reason to fix a line.")
    n = len(src)
    for start in range(0, n, TRANSLATE_REVIEW_BATCH):
        end = min(start + TRANSLATE_REVIEW_BATCH, n)
        items = [{"n": i + 1, "source": src[i], "draft": final[i]} for i in range(start, end)]
        before = [{"source": src[i], "english": final[i]} for i in range(max(0, start - 4), start)]
        after = [src[i] for i in range(end, min(n, end + 3))]
        user = ("Context before (already final, do not output): " + json.dumps(before, ensure_ascii=False) + "\n"
                "Context after (source only, do not output): " + json.dumps(after, ensure_ascii=False) + "\n\n"
                "Lines to review: " + json.dumps(items, ensure_ascii=False))
        reply = _chat(review_system, user, max_tokens=2500)
        if not reply:
            continue
        try:
            verdicts = json.loads(reply[reply.find("["): reply.rfind("]") + 1])
            if not isinstance(verdicts, list):
                raise ValueError("not a list")
            for v in verdicts:
                if not isinstance(v, dict):
                    continue
                num, verdict = v.get("n"), str(v.get("verdict", "")).lower()
                fix = v.get("english") or v.get("final") or v.get("fix")
                if verdict.startswith("fix") and isinstance(fix, str) and isinstance(num, int) and start < num <= end:
                    cleaned = _sanitize_translation(fix, final[num - 1])
                    old = final[num - 1]
                    # ignore edits that only change punctuation, quote style or capitalisation
                    same = re.sub(r"[\W_]+", "", cleaned.lower()) == re.sub(r"[\W_]+", "", old.lower())
                    # only accept real corrections, never style preferences
                    reason = str(v.get("why", "")).lower()
                    real = any(k in reason for k in ("mistransl", "pronoun", "gender", "name", "grammar", "misheard",
                                                      "mishear", "literal", "wrong", "incorrect", "error"))
                    if not real and TRANSLATE_DEBUG:
                        print(f"[review-skipped] #{num} '{old}' -> '{cleaned}' ({v.get('why', '')})")
                    if real and cleaned and not cleaned.startswith("{") and cleaned != old and not same:
                        if TRANSLATE_DEBUG:
                            print(f"[review] #{num} '{final[num - 1]}' -> '{cleaned}' ({v.get('why', '')})")
                        final[num - 1] = cleaned
                        if reasons is not None:
                            reasons[num - 1] = reason
        except Exception as exc:
            print(f"Review of lines {start}-{end - 1} unparseable ({exc}); keeping the draft.")
    return final


def translate_segments_via_llm(segments, video_file: str | None = None, source_language: str | None = None):
    if not segments:
        return segments

    texts = [seg["text"].strip() for seg in segments]
    context = get_media_context(video_file)
    bible = None
    if TRANSLATE_BIBLE:
        bible = _build_scene_bible(texts, context, source_language)
        if bible:
            print(f"Scene bible ({len(bible)} chars) prepared.")
            if TRANSLATE_DEBUG:
                print("[bible]\n" + bible)
            context = (context + "\n\n" if context else "") + "Translation bible (follow it for names, terms, gender):\n" + bible
    system_prompt = _translation_system_prompt(source_language, context)
    translated_so_far: list[str] = []

    for start in range(0, len(texts), TRANSLATE_BATCH_SIZE):
        chunk = texts[start:start + TRANSLATE_BATCH_SIZE]
        previous = translated_so_far[-TRANSLATE_CONTEXT_LINES:] if TRANSLATE_CONTEXT_LINES > 0 else None
        translated_chunk = _translate_batch(chunk, system_prompt, previous)

        if translated_chunk is None:
            print(
                f"Falling back to per-line translation for lines "
                f"{start}-{start + len(chunk) - 1}."
            )
            translated_chunk = [_translate_single(t, system_prompt) for t in chunk]

        translated_so_far.extend(translated_chunk)
        for i, new_text in enumerate(translated_chunk):
            segments[start + i]["text"] = new_text

    draft = [s["text"] for s in segments]
    reasons: dict = {}
    if TRANSLATE_REVIEW:
        reviewed = _review_translation(texts, draft, system_prompt, reasons)
        changed = sum(1 for a, b in zip(draft, reviewed) if a != b)
        print(f"Review pass changed {changed} of {len(reviewed)} lines.")
        for seg, text in zip(segments, reviewed):
            seg["text"] = text

    job = getattr(_job_local, 'job', None)           # what the Jobs view shows: source, draft, final, why
    if job is not None:
        d = job.setdefault('detail', {'source': [], 'draft': [], 'final': [], 'why': [], 'bible': ''})
        d['source'] += texts
        d['draft'] += draft
        d['final'] += [s["text"] for s in segments]
        d['why'] += [reasons.get(i, '') for i in range(len(texts))]
        if bible:
            d['bible'] = (d['bible'] + '\n\n' if d['bible'] else '') + bible
        d['review'] = bool(TRANSLATE_REVIEW)
    return segments


# Speaker diarization (whisperx engine + pyannote, needs HF_TOKEN set on the
# whisper-asr-webservice container itself). Adds a per-word "speaker" field.
WHISPER_DIARIZE = os.getenv('WHISPER_DIARIZE', 'true').lower() == 'true'
WHISPER_MIN_SPEAKERS = os.getenv('WHISPER_MIN_SPEAKERS', '')
WHISPER_DIARIZE_ENGINE = os.getenv('WHISPER_DIARIZE_ENGINE', 'auto')    # auto | pyannote | embedding (token-free)
WHISPER_MAX_SPEAKERS = os.getenv('WHISPER_MAX_SPEAKERS', '')
# Use the native backend whenever it is reachable, instead of whichever of the two answers fastest.
# For a dedicated NPU/GPU transcription service that should always win over the shared LLM host.
WHISPER_PREFER_NATIVE = os.getenv('WHISPER_PREFER_NATIVE', 'false').lower() == 'true'
# A whole film in one request takes a while on an NPU; the old 600 s limit was too short.
WHISPER_ASR_TIMEOUT = float(os.getenv('WHISPER_ASR_TIMEOUT', '600'))


def _same_host(a: str, b: str) -> bool:
    try:
        from urllib.parse import urlparse
        return bool(a) and bool(b) and urlparse(a).hostname == urlparse(b).hostname
    except Exception:
        return False


def _unload_primary_llm() -> None:
    """
    Best-effort: free the primary LLM's VRAM before a native-whisper call.
    Both the LLM (Ollama) and whisperx's alignment/diarization models live on
    the same GPU here, and together they don't reliably fit in 16GB at once —
    so we explicitly evict the LLM first rather than race it on memory. It
    reloads on its own the moment translate_segments_via_llm needs it again;
    that one-time reload cost is small next to the minutes a transcription
    job already takes. Ollama's native unload endpoint lives at the same
    host:port as OPENAI_BASE_URL, just without the '/v1' suffix.
    """
    if not OPENAI_BASE_URL:
        return
    native_base = OPENAI_BASE_URL[:-3] if OPENAI_BASE_URL.endswith('/v1') else OPENAI_BASE_URL
    try:
        httpx.post(f"{native_base}/api/generate", json={"model": TRANSLATE_LLM_MODEL, "keep_alive": 0}, timeout=5.0)
    except Exception:
        pass


def _call_whisper_asr_native(opus: io.BytesIO, language: str | None, prompt: str | None,
                             url: str | None = None) -> dict:
    """
    Transcribe via a native whisper-asr-webservice-compatible instance (onerahmet, npusubs, ...).
    Always requests task=transcribe (source-language output) — task=translate
    is still handled by translate_segments_via_llm afterwards, matching the
    OpenAI-client branch rather than this service's own (non-LLM) translate mode.
    """
    url = url or WHISPER_ASR_URL
    # Only needed when whisper and the LLM share one GPU (same host); otherwise it just adds a timeout.
    if WHISPER_DIARIZE and _same_host(url, OPENAI_BASE_URL):
        _unload_primary_llm()
    opus.seek(0)
    params = {"output": "json", "task": "transcribe", "word_timestamps": "true"}
    ctx = getattr(_job_local, 'ctx', None)
    if ctx and ctx.get('path'):
        params["video_file"] = ctx['path']        # lets npusubs apply the profile that matches this file
    if language:
        params["language"] = language
    if prompt:
        params["initial_prompt"] = prompt
    if WHISPER_DIARIZE:
        params["diarize"] = "true"
        if WHISPER_DIARIZE_ENGINE and WHISPER_DIARIZE_ENGINE != "auto":
            params["diarization_engine"] = WHISPER_DIARIZE_ENGINE
        if WHISPER_MIN_SPEAKERS:
            params["min_speakers"] = WHISPER_MIN_SPEAKERS
        if WHISPER_MAX_SPEAKERS:
            params["max_speakers"] = WHISPER_MAX_SPEAKERS
    files = {"audio_file": ("audio.opus", opus, "audio/ogg")}
    resp = httpx.post(f"{url}/asr", params=params, files=files, timeout=WHISPER_ASR_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# A native backend that did not answer is skipped for this many seconds, so an offline machine does not
# add a probe timeout to every request.
NATIVE_RETRY_SECS = float(os.getenv('NATIVE_RETRY_SECS', '60'))
_native_down_until: dict[str, float] = {}


def _reachable_natives() -> list[tuple[str, float]]:
    """Reachable native backends (url, latency) in order of preference."""
    now = time.monotonic()
    candidates = [u for u in WHISPER_ASR_URLS if _native_down_until.get(u, 0.0) <= now] or list(WHISPER_ASR_URLS)
    out = []
    for u in candidates:
        latency = _probe_latency(u)
        if latency is None:
            _native_down_until[u] = now + NATIVE_RETRY_SECS
        else:
            _native_down_until.pop(u, None)
            out.append((u, latency))
    return out


def _transcribe(opus: io.BytesIO, language: str | None, prompt: str | None, model_name: str) -> list[dict]:
    """
    Transcribe (source-language). Tries, in order: the native whisper-asr-webservice-compatible backends
    (WHISPER_ASR_URL, comma separated, first reachable wins) and the OpenAI-compatible endpoint
    (the primary `client`, or `fallback_client` when OPENAI_BASE_URL_FALLBACK is set), retrying on the next
    one if the picked one fails. Unless WHISPER_PREFER_NATIVE is set, a faster-answering OpenAI endpoint
    can be tried before the native ones.
    """
    attempts: list = []
    for u, _lat in (_reachable_natives() if WHISPER_ASR_URLS else []):
        attempts.append(("native", u))
    if fallback_client is not None:
        attempts.append(("openai_fallback", None))
    if not attempts:
        attempts.append(("openai_primary", None))
    if WHISPER_ASR_URLS and not any(k == "native" for k, _ in attempts) and WHISPER_ASR_URLS:
        # nothing native is reachable right now: still keep one as a last resort after the fallback
        attempts.append(("native", WHISPER_ASR_URLS[0]))

    if not WHISPER_PREFER_NATIVE and WHISPER_ASR_URLS and fallback_client is not None:
        natives = _reachable_natives()
        fallback_latency = _probe_latency(OPENAI_BASE_URL_FALLBACK)
        if fallback_latency is not None and natives and fallback_latency < natives[0][1]:
            attempts = [a for a in attempts if a[0] != "openai_fallback"]
            attempts.insert(0, ("openai_fallback", None))

    last_exc = None
    for kind, url in attempts:
        try:
            opus.seek(0)
            if kind == "native":
                try:
                    segments = verbose_json_to_segments(_call_whisper_asr_native(opus, language, prompt, url))
                    _note_backend(f"speech {url}")
                    return segments
                except Exception:
                    _native_down_until[url] = time.monotonic() + NATIVE_RETRY_SECS
                    raise
            attempt_client = fallback_client if kind == "openai_fallback" else client
            response = attempt_client.audio.transcriptions.create(
                model=model_name,
                file=opus,
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["word", "segment"],
                prompt=prompt,
            )
            _note_backend("openai-compatible fallback")
            return verbose_json_to_segments(response)
        except Exception as exc:
            last_exc = exc
            suffix = " — trying next backend." if (kind, url) != attempts[-1] else ""
            print(f"Transcription via {kind}{' ' + url if url else ''} failed ({exc}){suffix}")
    raise last_exc


def _call_api(opus: io.BytesIO, task: str, language: str | None, video_file: str | None) -> list[dict]:
    opus.seek(0)
    prompt = build_initial_prompt(video_file, language)
    model_name = whisper_model if task == "transcribe" else whisper_translate_model
    segments = _transcribe(opus, language, prompt, model_name)
    if task == "transcribe":
        return segments
    return translate_segments_via_llm(segments, video_file, language)


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
    video_file: str | None,
) -> str:
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        # Filter per-chunk before adding the offset so that a hallucinated
        # t=0 phantom on chunk N (which would shift to the chunk boundary
        # timestamp) is caught while it still looks like t=0.
        segs = filter_phantom_segments(_call_api(encode_pcm_to_opus(pcm_chunk), task, language, video_file))
        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)
    return segments_to_srt(filter_repetition_loops(filter_phantom_segments(all_segments)))


def call_transcription(pcm_bytes: bytes, task: str, language: str | None, video_file: str | None = None) -> str:
    opus      = encode_pcm_to_opus(pcm_bytes)
    opus_size = opus.getbuffer().nbytes
    print(
        f"Encoded to Opus: {opus_size / 1024 / 1024:.1f} MB "
        f"(from {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM, "
        f"{len(pcm_bytes) / opus_size:.0f}:1 compression)"
    )

    if opus_size <= MAX_UPLOAD_BYTES:
        try:
            return segments_to_srt(filter_repetition_loops(filter_phantom_segments(_call_api(opus, task, language, video_file))))
        except Exception as exc:
            if not is_too_large_error(exc):
                raise
            print(
                f"Provider returned 413 despite {opus_size / 1024 / 1024:.1f} MB Opus file. "
                f"Consider lowering MAX_UPLOAD_MB (currently {MAX_UPLOAD_BYTES // 1024 // 1024}). "
                f"Splitting into chunks and retrying."
            )
            num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES) + 1)
            return _transcribe_pcm_chunks(
                split_pcm_into_chunks(pcm_bytes, num_chunks), task, language, video_file
            )

    num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES))
    print(f"Opus file ({opus_size / 1024 / 1024:.1f} MB) exceeds limit — splitting into {num_chunks} chunks.")
    return _transcribe_pcm_chunks(
        split_pcm_into_chunks(pcm_bytes, num_chunks), task, language, video_file
    )


# ---------------------------------------------------------------------------
# Shared configuration (subcfg): /config/bridge.json, defaults and validation from the schema.
# Environment variables still work and override the file. Changed files are applied between requests.
# Set SUBCFG_DISABLE=1 to ignore the config folder completely.
# ---------------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
UI_ENABLED = os.getenv('WEBUI_ENABLED', 'true').lower() not in ('0', 'false', 'no', 'off')
UI_TOKEN = os.getenv('WEBUI_TOKEN', '')
UI_TRUSTED_NETWORKS = [n.strip() for n in os.getenv('WEBUI_TRUSTED_NETWORKS', '').split(',') if n.strip()]
_CFG = None
if not os.getenv('SUBCFG_DISABLE'):
    try:
        import subcfg as _subcfg
        _CFG = _subcfg.ConfigStore('bridge', schema_dirs=(os.getenv('SUBCFG_SCHEMA_DIR', '/opt/subcfg/schemas'),))
        if not _CFG.schema.get('properties'):
            print("subcfg: no bridge schema found; using environment variables only")
            _CFG = None
    except Exception as _exc:
        print(f"subcfg unavailable ({_exc}); using environment variables only")
        _CFG = None

# config path -> module level setting used by the rest of this file
_CFG_SIMPLE = {
    'backends.timeout_s': 'WHISPER_ASR_TIMEOUT', 'backends.prefer_native': 'WHISPER_PREFER_NATIVE',
    'backends.retry_offline_s': 'NATIVE_RETRY_SECS', 'backends.probe_timeout_s': 'BACKEND_PROBE_TIMEOUT_SECS',
    'diarization.enabled': 'WHISPER_DIARIZE', 'diarization.min_speakers': 'WHISPER_MIN_SPEAKERS',
    'diarization.max_speakers': 'WHISPER_MAX_SPEAKERS', 'diarization.engine': 'WHISPER_DIARIZE_ENGINE',
    'translation.batch_size': 'TRANSLATE_BATCH_SIZE', 'translation.context_lines': 'TRANSLATE_CONTEXT_LINES',
    'translation.temperature': 'TRANSLATE_TEMPERATURE', 'translation.keep_honorifics': 'TRANSLATE_KEEP_HONORIFICS',
    'translation.bible': 'TRANSLATE_BIBLE', 'translation.review': 'TRANSLATE_REVIEW',
    'translation.review_batch': 'TRANSLATE_REVIEW_BATCH', 'translation.bible_max_chars': 'TRANSLATE_BIBLE_MAX_CHARS',
    'translation.debug': 'TRANSLATE_DEBUG',
    'subtitles.max_line_length': 'MAX_LINE_LENGTH', 'subtitles.gap_split_s': 'GAP_SPLIT_SECS',
    'filters.phantom_start_max_s': 'PHANTOM_START_MAX_SECS', 'filters.phantom_duration_max_s': 'PHANTOM_DURATION_MAX_SECS',
    'filters.repetition_min_run': 'REPETITION_MIN_RUN_LENGTH', 'filters.repetition_max_period': 'REPETITION_MAX_PERIOD',
    'filters.global_duplicate_min': 'GLOBAL_DUPLICATE_MIN_COUNT',
    'audio.silence_window_s': 'SILENCE_SEARCH_WINDOW_SECS', 'audio.silence_noise_db': 'SILENCE_NOISE_DB',
    'audio.silence_min_s': 'SILENCE_MIN_DURATION_SECS', 'arr.cache_ttl_s': 'MEDIA_CACHE_TTL_SECS',
    'llm.model': 'TRANSLATE_LLM_MODEL', 'llm.dedicated_model': 'TRANSLATE_LLM_MODEL_DEDICATED',
    'whisper.model': 'whisper_model', 'whisper.forced_detect_language': 'force_detected_language_to',
    'ui.enabled': 'UI_ENABLED', 'ui.token': 'UI_TOKEN', 'ui.trusted_networks': 'UI_TRUSTED_NETWORKS',
    'guard.heavy_window': 'GUARD_WINDOW', 'guard.min_free_host_ram_gb': 'GUARD_MIN_FREE_RAM_GB',
    'guard.host_ram_url': 'GUARD_HOST_RAM_URL', 'guard.max_llm_models': 'GUARD_MAX_LLM_MODELS',
    'guard.max_wait_s': 'GUARD_MAX_WAIT_S',
}
# resource guard defaults (used when the config layer is off)
GUARD_WINDOW = os.getenv('GUARD_HEAVY_WINDOW', '')
GUARD_MIN_FREE_RAM_GB = float(os.getenv('GUARD_MIN_FREE_RAM_GB', '0') or 0)
GUARD_HOST_RAM_URL = os.getenv('GUARD_HOST_RAM_URL', '')
GUARD_MAX_LLM_MODELS = int(os.getenv('GUARD_MAX_LLM_MODELS', '1') or 1)
GUARD_MAX_WAIT_S = float(os.getenv('GUARD_MAX_WAIT_S', '21600') or 21600)


_applied_generation = [-1]
_applied_profile = [None]


def _cfg_get(eff: dict, path: str):
    for key in path.split('.'):
        eff = eff[key]
    return eff


def _apply_config(force: bool = False, context: dict | None = None, overrides: dict | None = None) -> bool:
    """Copy the effective configuration (with the profile matching `context`, if any) into the module level
    settings. Cheap when nothing changed. `overrides` ({'bridge': {...nested...}}) is for playground runs."""
    global client, fallback_client, translate_client, OPENAI_BASE_URL, OPENAI_BASE_URL_FALLBACK, TRANSLATE_LLM_BASE_URL
    global WHISPER_ASR_URLS, WHISPER_ASR_URL, SKIP_FILENAME_PATTERN, MAX_UPLOAD_BYTES, OPUS_BITRATE_BPS
    global whisper_initial_prompt, whisper_translate_model, TRANSLATE_LLM_MODEL_FALLBACK
    global SONARR_URL, SONARR_API_KEY, RADARR_URL, RADARR_API_KEY
    if _CFG is None:
        return False
    _CFG.refresh(force=force)
    prof = _CFG.match_profile(context)
    prof_name = prof.get('name') if prof else None
    if overrides:
        prof_name = '(playground)'
    elif not force and _CFG.generation == _applied_generation[0] and prof_name == _applied_profile[0]:
        return False                                                  # also true after a save made through the UI
    _applied_generation[0], _applied_profile[0] = _CFG.generation, prof_name
    eff = _CFG.effective(context)
    if overrides:
        eff = _subcfg.deep_merge(eff, overrides.get('bridge', {}))
    g = globals()
    for path, name in _CFG_SIMPLE.items():
        g[name] = _cfg_get(eff, path)
    WHISPER_ASR_URLS = [u.strip().rstrip('/') for u in eff['backends']['speech_urls'] if u.strip()]
    WHISPER_ASR_URL = WHISPER_ASR_URLS[0] if WHISPER_ASR_URLS else ''
    TRANSLATE_LLM_MODEL_FALLBACK = eff['llm']['model_fallback'] or eff['llm']['model']
    whisper_translate_model = eff['whisper']['translate_model'] or eff['whisper']['model']
    whisper_initial_prompt = eff['whisper']['initial_prompt'] or None
    MAX_UPLOAD_BYTES = int(eff['audio']['max_upload_mb']) * 1024 * 1024
    OPUS_BITRATE_BPS = int(eff['audio']['opus_kbps']) * 1000
    SONARR_URL, SONARR_API_KEY = eff['arr']['sonarr_url'].rstrip('/'), eff['arr']['sonarr_api_key']
    RADARR_URL, RADARR_API_KEY = eff['arr']['radarr_url'].rstrip('/'), eff['arr']['radarr_api_key']
    try:
        SKIP_FILENAME_PATTERN = re.compile(eff['filters']['skip_filename_pattern'])
    except re.error as exc:
        print(f"subcfg: invalid skip_filename_pattern ({exc}); keeping the previous one")
    # endpoints: only rebuild a client when its URL really changed
    primary = eff['llm']['primary_url']
    if primary and primary != OPENAI_BASE_URL:
        OPENAI_BASE_URL, client = primary, OpenAI(base_url=primary)
    fallback = eff['llm']['fallback_url'].rstrip('/') or None
    if fallback != OPENAI_BASE_URL_FALLBACK:
        OPENAI_BASE_URL_FALLBACK = fallback
        fallback_client = OpenAI(base_url=fallback) if fallback else None
    dedicated = eff['llm']['dedicated_url'].rstrip('/') or None
    if dedicated != TRANSLATE_LLM_BASE_URL:
        TRANSLATE_LLM_BASE_URL = dedicated
        translate_client = OpenAI(base_url=dedicated) if dedicated else None
    if _CFG.errors:
        print("subcfg: " + "; ".join(_CFG.errors))
    return True


_apply_config(force=True)

# ---------------------------------------------------------------------------
# Recent jobs (shown in the web UI) and a state snapshot for the UI
# ---------------------------------------------------------------------------
import collections
import threading
_STARTED = time.time()
_JOBS = collections.deque(maxlen=50)
DETAIL_JOBS = 10                                         # jobs that keep their SRT and translation detail in memory
_job_local = threading.local()
_job_counter = [0]


# Profiles change module level settings, so jobs running at the same time must agree on the profile. Jobs of the
# same profile (or none) run in parallel; a job needing another profile waits until the running ones are done.
_gate = threading.Condition()
_gate_state = {'sig': None, 'active': 0}


def _profile_context(video_file: str | None, task: str, language: str | None) -> dict:
    lang = (language or '').lower()
    return {'path': video_file or '', 'source_language': lang, 'target_language': 'en' if task == 'translate' else lang}


def _enter_job(ctx: dict, overrides: dict | None = None, tag: str | None = None) -> str | None:
    prof = _CFG.match_profile(ctx) if _CFG is not None else None
    sig = tag or (prof.get('name') if prof else None)
    with _gate:
        while _gate_state['active'] and _gate_state['sig'] != sig:
            _gate.wait(timeout=5)
        _gate_state['sig'] = sig
        _gate_state['active'] += 1
        _apply_config(context=ctx, overrides=overrides)    # picks up config changes and the profile of this job
    return sig


def _exit_job() -> None:
    with _gate:
        _gate_state['active'] -= 1
        if not _gate_state['active']:
            _apply_config()                    # back to the base settings, and any save made during the job, while idle
        _gate.notify_all()


# Resource guard: hold a job until the hour is allowed and (for translations) the host has enough free RAM.
def _in_window(spec: str, now: 'datetime.datetime | None' = None) -> bool:
    """'HH:MM-HH:MM' in local time, wrapping past midnight. Empty or unparseable = always allowed."""
    m = re.fullmatch(r'\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*', spec or '')
    if not m:
        return True
    a, b = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
    if a >= 24 * 60 or b >= 24 * 60 or a == b:
        return True
    import datetime
    now = now or datetime.datetime.now()
    cur = now.hour * 60 + now.minute
    return a <= cur < b if a < b else (cur >= a or cur < b)


_host_ram_cache = {'t': 0.0, 'data': None}


def _host_ram() -> dict | None:
    """{'free_gb', 'total_gb'} from the configured host RAM endpoint (cached 10 s); None when unknown."""
    if not GUARD_HOST_RAM_URL:
        return None
    if time.time() - _host_ram_cache['t'] < 10:
        return _host_ram_cache['data']
    data = None
    try:
        j = httpx.get(GUARD_HOST_RAM_URL, timeout=3).json()
        data = {'free_gb': float(j['mem_available_gb']), 'total_gb': float(j.get('mem_total_gb') or 0)}
    except Exception:                                     # noqa: BLE001
        pass
    _host_ram_cache.update(t=time.time(), data=data)
    return data


def _guard_reason(task: str) -> str | None:
    if not _in_window(GUARD_WINDOW):
        return f"outside the allowed hours ({GUARD_WINDOW})"
    if task == 'translate' and GUARD_MIN_FREE_RAM_GB > 0:
        ram = _host_ram()
        if ram and ram['free_gb'] < GUARD_MIN_FREE_RAM_GB:
            return f"host RAM low ({ram['free_gb']:.1f} GB free, {GUARD_MIN_FREE_RAM_GB:g} GB needed)"
    return None


def _guard_wait(job: dict, task: str, ctx: dict) -> None:
    deadline = time.time() + GUARD_MAX_WAIT_S
    reason = _guard_reason(task)
    if reason is None:
        return
    print(f"Resource guard: holding the job: {reason}")
    try:
        while reason is not None:
            if time.time() > deadline:
                raise RuntimeError(f"gave up waiting for the resource guard: {reason}")
            job.update(status='waiting', waiting_for=reason)
            time.sleep(15)
            _apply_config(context=ctx)                    # settings changed in the UI release (or hold) the job
            reason = _guard_reason(task)
    finally:
        job.update(waiting_for='')
    job.update(status='running', started=time.time())
    print("Resource guard: released.")


def _guard_status() -> dict:
    ram = _host_ram()
    return {'window': GUARD_WINDOW, 'in_window': _in_window(GUARD_WINDOW), 'min_free_ram_gb': GUARD_MIN_FREE_RAM_GB,
            'ram_configured': bool(GUARD_HOST_RAM_URL), 'ram': ram, 'max_llm_models': GUARD_MAX_LLM_MODELS,
            'waiting': sum(1 for j in list(_JOBS) if j.get('status') == 'waiting')}


def _apply_config_when_idle() -> None:
    """Called after a save in the web UI: apply right away unless a job is running (it applies at the next job)."""
    with _gate:
        if not _gate_state['active']:
            _apply_config()


def _note_backend(label: str) -> None:
    job = getattr(_job_local, 'job', None)
    if job is not None and label not in job['backends']:
        job['backends'].append(label)


def _get_job(job_id: int) -> dict | None:
    for j in list(_JOBS):
        if j['id'] == job_id:
            return dict(j)
    return None


def _ui_state() -> dict:
    """Read-only snapshot used by the web UI."""
    return {
        'version': version, 'started': _STARTED, 'cfg': _CFG, 'get_job': _get_job, 'guard': _guard_status,
        'jobs': [{**{k: v for k, v in j.items() if k not in ('srt', 'detail')}, 'has_srt': 'srt' in j, 'has_detail': 'detail' in j}
                 for j in list(_JOBS)],
        'enabled': UI_ENABLED, 'token': UI_TOKEN, 'trusted_networks': list(UI_TRUSTED_NETWORKS),
        'speech_urls': list(WHISPER_ASR_URLS), 'prefer_native': WHISPER_PREFER_NATIVE,
        'llm': {'primary': OPENAI_BASE_URL, 'fallback': OPENAI_BASE_URL_FALLBACK, 'dedicated': TRANSLATE_LLM_BASE_URL,
                'model': TRANSLATE_LLM_MODEL, 'model_fallback': TRANSLATE_LLM_MODEL_FALLBACK,
                'model_dedicated': TRANSLATE_LLM_MODEL_DEDICATED},
        'sonarr_url': SONARR_URL, 'radarr_url': RADARR_URL,
        'flags': {'diarization': WHISPER_DIARIZE, 'bible': TRANSLATE_BIBLE, 'review': TRANSLATE_REVIEW,
                  'keep_honorifics': TRANSLATE_KEEP_HONORIFICS},
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/config")
def config_info():
    """Read-only view of the active configuration (secrets masked) and where each value comes from."""
    _apply_config()
    if _CFG is None:
        return {"enabled": False}
    return {"enabled": True, "dir": _CFG.dir, "errors": _CFG.errors, "effective": _CFG.masked(),
            "sources": {'.'.join(p): _CFG.source('.'.join(p)) for p in _CFG._leaf}}


@app.get("/status")
def status():
    return {"version": f"Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version}"}


@app.get("/busy")
def busy():
    """Counts only (no details): lets a low-priority LLM client (tools/idle_gate.py) stay out of the way of a running job."""
    jobs = list(_JOBS)
    return {"active": sum(1 for j in jobs if j.get('status') == 'running') + (1 if _PLAYGROUND.list() and
            _PLAYGROUND.list()[0]['status'] == 'running' else 0),
            "waiting": sum(1 for j in jobs if j.get('status') == 'waiting')}


@app.post("/detect-language")
def detect_language():
    print(f"Forced detected language to {force_detected_language_to}")
    return {
        "detected_language": f"Forced to {force_detected_language_to} from WhisperBridge",
        "language_code": force_detected_language_to,
    }


@app.post("/asr")
def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
):
    job = None
    entered = False
    try:
        ctx = _profile_context(video_file, task, language)
        profile_name = _enter_job(ctx)        # picks up changes made to /config and the profile for this file
        entered = True
        _job_local.ctx = ctx
        start_time = time.time()
        pcm_bytes  = audio_file.file.read()
        _job_counter[0] += 1
        job = {'id': _job_counter[0], 'started': start_time, 'file': os.path.basename(video_file or ''), 'task': task,
               'language': language, 'pcm_mb': round(len(pcm_bytes) / 1024 / 1024, 1), 'status': 'running',
               'backends': [], 'seconds': None, 'error': '', 'profile': profile_name}
        _job_local.job = job
        _JOBS.appendleft(job)
        for old in list(_JOBS)[DETAIL_JOBS:]:           # keep the heavy parts (SRT, draft/final) of recent jobs only
            old.pop('srt', None)
            old.pop('detail', None)

        if should_skip_file(video_file):
            job.update(status='skipped', seconds=0)
            print(f"Skipping '{video_file}': matches OP/ED theme-song filename pattern, no dialogue to transcribe.")
            return StreamingResponse(
                iter([""]),
                media_type="text/plain",
                headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
            )

        _guard_wait(job, task, ctx)
        start_time = job['started']
        print(f"Got a {task} task from Bazarr ({len(pcm_bytes) / 1024 / 1024:.1f} MB PCM)")

        if os.getenv('DEBUG_DUMP_AUDIO'):
            dump_path = f"/tmp/debug_audio_{int(time.time())}.pcm"
            with open(dump_path, "wb") as f:
                f.write(pcm_bytes)
            print(f"DEBUG_DUMP_AUDIO: wrote raw PCM to {dump_path}")

        srt_content = call_transcription(pcm_bytes, task, language, video_file)

        elapsed = time.time() - start_time
        job.update(status='done', seconds=round(elapsed), srt=srt_content or '')
        mins, secs = divmod(int(elapsed), 60)
        print(
            f"Transcription of '{video_file}' complete in {mins}m {secs}s."
            if video_file else f"Transcription complete in {mins}m {secs}s."
        )

        return StreamingResponse(
            iter([srt_content or ""]),
            media_type="text/plain",
            headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
        )

    except Exception as e:
        if job is not None:
            job.update(status='error', seconds=round(time.time() - job['started']), error=str(e)[:200])
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if entered:
            _exit_job()


# ---------------------------------------------------------------------------
# Playground (web UI): run the same input with different settings and compare the results.
# One run at a time; translation variants take the settings gate, so they never mix with a real job.
# ---------------------------------------------------------------------------
_SRT_TIME = re.compile(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)')
PLAY_MAX_LINES = 300
PLAY_MAX_AUDIO_MB = 60


def _srt_seconds(h, m, s, frac) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + float('0.' + frac)


def _parse_srt(text: str) -> list[dict]:
    segments = []
    for block in re.split(r'\r?\n\s*\r?\n', (text or '').strip()):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = _SRT_TIME.search(ln)
            if m:
                g = m.groups()
                body = ' '.join(x.strip() for x in lines[i + 1:])
                if body:
                    segments.append({'start': _srt_seconds(*g[:4]), 'end': _srt_seconds(*g[4:]), 'text': body})
                break
    return segments


class _Playground:
    def __init__(self):
        self.runs: 'collections.OrderedDict[str, dict]' = collections.OrderedDict()
        self.lock = threading.Lock()

    def list(self) -> list[dict]:
        return [{k: r[k] for k in ('id', 'kind', 'status', 'started', 'label')} for r in reversed(self.runs.values())]

    def get(self, rid: str) -> dict | None:
        return self.runs.get(rid)

    def start(self, kind: str, payload: dict) -> dict:
        with self.lock:
            if any(r['status'] == 'running' for r in self.runs.values()):
                raise RuntimeError('another playground run is still in progress')
            rid = f"{int(time.time())}-{len(self.runs) + 1}"
            run = {'id': rid, 'kind': kind, 'status': 'running', 'started': time.time(), 'label': payload.get('label') or kind,
                   'results': [], 'source': payload.get('source'), 'error': ''}
            self.runs[rid] = run
            while len(self.runs) > 12:
                self.runs.popitem(last=False)
        target = self._translate if kind == 'translate' else self._transcribe
        threading.Thread(target=target, args=(run, payload), name=f"play-{rid}", daemon=True).start()
        return run

    def _translate(self, run: dict, payload: dict) -> None:
        _job_local.job = None
        try:
            base = _parse_srt(payload['srt'])[:PLAY_MAX_LINES]
            lang = payload.get('language') or None
            run['source'] = {'lines': [s['text'] for s in base], 'starts': [round(s['start'], 1) for s in base]}
            for n, var in enumerate(payload['variants']):
                res = {'name': var['name'], 'status': 'running', 'lines': [], 'seconds': None, 'error': ''}
                run['results'].append(res)
                t0 = time.time()
                try:
                    _enter_job({}, overrides={'bridge': var.get('overrides', {})}, tag=f"play:{run['id']}:{n}")
                    try:
                        out = translate_segments_via_llm([dict(s) for s in base], payload.get('video_file') or None, lang)
                    finally:
                        _exit_job()
                    res.update(status='done', lines=[s['text'] for s in out])
                except Exception as exc:                       # noqa: BLE001
                    traceback.print_exc()
                    res.update(status='error', error=str(exc)[:300])
                res['seconds'] = round(time.time() - t0)
            run['status'] = 'done'
        except Exception as exc:                               # noqa: BLE001
            run.update(status='error', error=str(exc)[:300])

    def _transcribe(self, run: dict, payload: dict) -> None:
        _job_local.job = None
        try:
            natives = [u for u, _lat in _reachable_natives()]
            if not natives:
                raise RuntimeError('no native speech service (npusubs) is reachable')
            url, audio = natives[0], payload['audio']
            run['source'] = {'service': url, 'audio_mb': round(len(audio) / 1048576, 1)}
            for var in payload['variants']:
                res = {'name': var['name'], 'status': 'running', 'srt': '', 'seconds': None, 'error': ''}
                run['results'].append(res)
                t0 = time.time()
                try:
                    params = {'output': 'srt', 'task': 'transcribe', 'word_timestamps': 'true', **var.get('params', {})}
                    if payload.get('language'):
                        params['language'] = payload['language']
                    r = httpx.post(f"{url}/asr", params=params, files={'audio_file': ('clip', io.BytesIO(audio))},
                                   timeout=WHISPER_ASR_TIMEOUT)
                    r.raise_for_status()
                    res.update(status='done', srt=r.text)
                except Exception as exc:                       # noqa: BLE001
                    res.update(status='error', error=str(exc)[:300])
                res['seconds'] = round(time.time() - t0)
            run['status'] = 'done'
        except Exception as exc:                               # noqa: BLE001
            run.update(status='error', error=str(exc)[:300])


_PLAYGROUND = _Playground()


# Web UI (served from ./ui next to this file). Disable with ui.enabled=false in bridge.json.
try:
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui'))
    import bridge_ui
    bridge_ui.install(app, _ui_state, on_change={'bridge': _apply_config_when_idle, 'profiles': _apply_config_when_idle},
                      playground=_PLAYGROUND, play_limits={'lines': PLAY_MAX_LINES, 'audio_mb': PLAY_MAX_AUDIO_MB})
except Exception as _exc:
    print(f"web UI not available: {_exc}")


if __name__ == "__main__":
    model_info = (
        f"model: {whisper_model} | translate model: {whisper_translate_model}"
        if whisper_translate_model != whisper_model
        else f"model: {whisper_model}"
    )
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} | "
        f"{model_info} | "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps | "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB | "
        f"max line: {MAX_LINE_LENGTH} chars | "
        f"gap split: {GAP_SPLIT_SECS}s"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
