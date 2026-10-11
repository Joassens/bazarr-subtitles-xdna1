"""HTTP API: whisper-asr-webservice compatible (Bazarr) plus OpenAI-compatible transcriptions."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__, audio as audio_mod, config, filters, subtitles
from .diarize import Diarizer, assign_speakers, run_on_regions
from .diarize_local import DiarizerRouter, LocalDiarizer
from .engine import Engine

log = logging.getLogger("npusubs")
app = FastAPI(title="npusubs", version=__version__)
engine = Engine()
diarizer = DiarizerRouter(Diarizer(), LocalDiarizer())
_state = {"loading": True, "error": None}


def _diarize_default() -> bool:
    return bool(config.get("diarization.default"))


def _boot() -> None:
    try:
        engine.load()
        engine.warmup()
        _state["loading"] = False
    except Exception as exc:                       # noqa: BLE001
        log.exception("startup failed")
        _state["error"] = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_boot, name="boot", daemon=True).start()


def _need_ready() -> None:
    if _state["error"]:
        raise HTTPException(503, f"engine failed to start: {_state['error']}")
    if _state["loading"]:
        raise HTTPException(503, "model still loading")


def _lang(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().lower()
    return code if code not in ("auto", "none", "") else None


def _run(raw: bytes, *, raw_pcm: bool, language, task, prompt, vad, diarize, min_spk, max_spk, filter_: bool = True,
         beam_size: Optional[int] = None, diarize_engine: Optional[str] = None):
    try:
        wav = audio_mod.decode(raw, raw_pcm=raw_pcm)
    except audio_mod.AudioError as exc:
        raise HTTPException(400, str(exc))
    if len(wav) < 1600:
        raise HTTPException(400, "audio shorter than 0.1 s")
    # Diarization runs on the CPU. By default it runs after the transcription (lower peak memory: beam search
    # plus pyannote at the same time can exhaust a small container). NPUSUBS_DIARIZE_PARALLEL=1 overlaps them.
    box: dict = {}
    do_diarize = bool(diarize)
    chosen = diarizer.pick(diarize_engine)
    if do_diarize and not chosen.available:
        log.warning("diarize requested but the %s engine is unavailable (%s); continuing without speakers",
                    diarizer.engine_name(diarize_engine), chosen.status)
        do_diarize = False

    def _diarize() -> None:
        try:
            t0 = time.time()
            # speech AND music (songs, openings, endings): lyrics are subtitled too, so they get speakers
            regions = engine.vad.regions(wav) if engine.use_vad else []
            box["turns"] = run_on_regions(chosen, wav, regions, min_spk, max_spk)
            log.info("diarization: %d turns in %.1fs", len(box["turns"]), time.time() - t0)
        except Exception:                                            # noqa: BLE001
            log.exception("diarization failed; continuing without speakers")

    worker = None
    if do_diarize and config.get("diarization.parallel"):
        worker = threading.Thread(target=_diarize, name="diarize", daemon=True)
        worker.start()
    try:
        res = engine.transcribe(wav, language=_lang(language), task=task, initial_prompt=prompt or None, vad=vad,
                                beam_size=beam_size)
    except TimeoutError as exc:
        raise HTTPException(503, str(exc))
    if filter_:
        res["segments"] = filters.apply_all(res["segments"])
        res["text"] = " ".join(s["text"] for s in res["segments"])
    if worker is not None:
        worker.join()
    elif do_diarize:
        _diarize()
    if box.get("turns"):
        assign_speakers(res["segments"], box["turns"])
    return res


def _context(video_file: Optional[str], language: Optional[str]) -> dict:
    """What a profile can match on: the media path and the spoken language."""
    return {"path": video_file or "", "source_language": _lang(language) or ""}


def _render(res: dict, output: str, ctx: Optional[dict] = None):
    if output == "json":
        return JSONResponse({"text": res["text"], "language": res["language"], "segments": res["segments"],
                             "profile": res.get("profile")})
    if output == "txt":
        return PlainTextResponse(subtitles.to_txt(res["segments"]))
    if output == "tsv":
        return PlainTextResponse(subtitles.to_tsv(res["segments"]))
    cues = subtitles.build_cues(res["segments"], subtitles.Options.from_config(ctx))
    if output == "vtt":
        return PlainTextResponse(subtitles.to_vtt(cues))
    return PlainTextResponse(subtitles.to_srt(cues))


# ------------------------------------------------------------------ service info
@app.get("/")
def root():
    return {"service": "npusubs", "version": __version__, "model": engine.model_name,
            "status": "ok" if not _state["loading"] and not _state["error"] else ("error" if _state["error"] else "loading")}


@app.get("/health")
def health():
    st = engine.stats
    cur = engine.current                      # read once: it can be cleared by the worker at any moment
    return {"status": "error" if _state["error"] else ("loading" if _state["loading"] else "ok"),
            "error": _state["error"], "model": engine.model_name, "version": __version__,
            "vad": engine.use_vad, "diarization_available": diarizer.available, "diarization": diarizer.status,
            "requests": st["requests"], "audio_seconds": round(st["audio_s"], 1), "wall_seconds": round(st["wall_s"], 1),
            "uptime_s": round(time.time() - st["started"], 1),
            "npu_encoder_s": round(engine.encoder.busy_s, 1) if engine.encoder else None,
            "current": ({**cur, "running_s": round(time.time() - cur["started"])} if cur else None),
            "beam": config.get("beam"),
            "config": {"dir": config.store.dir, "file": os.path.exists(os.path.join(config.store.dir, "npusubs.json")),
                       "errors": config.store.errors, "beam": config.get("beam"), "beam_source": config.store.source("beam")}}


# ------------------------------------------------- whisper-asr-webservice API
@app.post("/asr")
def asr(audio_file: UploadFile = File(...),
        encode: bool = Query(True, description="false = raw 16 kHz mono s16le PCM"),
        task: str = Query("transcribe"), language: Optional[str] = Query(None),
        initial_prompt: Optional[str] = Query(None), vad_filter: Optional[bool] = Query(None),
        word_timestamps: bool = Query(True), output: str = Query("txt"),
        diarize: Optional[bool] = Query(None), min_speakers: Optional[int] = Query(None),
        max_speakers: Optional[int] = Query(None), video_file: Optional[str] = Query(None),
        beam_size: Optional[int] = Query(None, description="1 = greedy (fast); default NPUSUBS_BEAM"),
        diarization_engine: Optional[str] = Query(None, description="auto | pyannote | embedding (token-free); default from the config")):
    _need_ready()
    if diarization_engine and diarization_engine not in DiarizerRouter.ENGINES:
        raise HTTPException(400, f"diarization_engine must be one of {', '.join(DiarizerRouter.ENGINES)}")
    # A profile (config/profiles/*.json matching the media path and language) changes these defaults per request;
    # explicit query parameters from the caller still win.
    ctx = _context(video_file, language)
    prof = config.store.match_profile(ctx)
    if prof:
        log.info("profile '%s' applies to %s", prof.get("name"), video_file)
    if beam_size is None and prof:
        beam_size = int(config.get("beam", context=ctx))
    if vad_filter is None and prof:
        vad_filter = bool(config.get("vad.enabled", context=ctx))
    if diarize is None:
        diarize = bool(config.get("diarization.default", context=ctx))
    res = _run(audio_file.file.read(), raw_pcm=not encode, language=language, task=task, prompt=initial_prompt,
               vad=vad_filter, diarize=diarize, min_spk=min_speakers, max_spk=max_speakers, beam_size=beam_size,
               diarize_engine=diarization_engine)
    res["profile"] = prof.get("name") if prof else None
    return _render(res, output.lower(), ctx)


@app.post("/detect-language")
def detect_language(audio_file: UploadFile = File(...), encode: bool = Query(True)):
    _need_ready()
    try:
        wav = audio_mod.decode(audio_file.file.read(), raw_pcm=not encode)
    except audio_mod.AudioError as exc:
        raise HTTPException(400, str(exc))
    return engine.detect_language(wav)


# --------------------------------------------------- OpenAI-compatible API
@app.post("/v1/audio/transcriptions")
def openai_transcriptions(file: UploadFile = File(...), model: str = Form("whisper-1"),
                          language: Optional[str] = Form(None), prompt: Optional[str] = Form(None),
                          response_format: str = Form("json"), temperature: float = Form(0.0),
                          diarize: Optional[bool] = Form(None)):
    _need_ready()
    res = _run(file.file.read(), raw_pcm=False, language=language, task="transcribe", prompt=prompt, vad=None,
               diarize=_diarize_default() if diarize is None else diarize, min_spk=None, max_spk=None)
    if response_format == "verbose_json":
        words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                 for s in res["segments"] for w in s["words"]]
        return {"task": "transcribe", "language": res["language"], "duration": res["duration"], "text": res["text"],
                "segments": [{k: s[k] for k in ("id", "start", "end", "text") if k in s} | {"words": s["words"]} |
                             ({"speaker": s["speaker"]} if "speaker" in s else {}) for s in res["segments"]],
                "words": words}
    if response_format in ("srt", "vtt", "text"):
        return _render(res, "txt" if response_format == "text" else response_format)
    return {"text": res["text"]}
