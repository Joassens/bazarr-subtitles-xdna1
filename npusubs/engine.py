"""Whisper on the XDNA1 NPU: encoder on the NPU, everything else on the CPU.

The trick: openai-whisper accepts any module as `model.encoder`. We swap in one that runs
the encoder on the NPU (using the open drakosha/whisper-xdna stack: XRT + MLIR-AIE),
which keeps every openai-whisper feature working: word timestamps (DTW over the
decoder's cross-attention), language detection, temperature fallback, prompts.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import torch

from . import config
from .vad import TimeMap, Vad

log = logging.getLogger("npusubs")
SR = 16000
CTX = 1500            # encoder context for a full 30 s window (what DTW alignment needs)


def _import_xdna():
    root = os.environ.get("XDNA_ROOT", "/opt/npu-whisper")
    if not os.path.isdir(os.path.join(root, "src")):
        raise RuntimeError(f"whisper-xdna checkout not found at XDNA_ROOT={root}")
    os.chdir(root)                              # rawxrt resolves its overlay paths from here
    sys.path.insert(0, os.path.join(root, "src"))
    import npu_whisper_encoder as E             # noqa: WPS433
    import rawxrt                               # noqa: WPS433
    return E, rawxrt


class NpuAudioEncoder(torch.nn.Module):
    """Drop-in for whisper.model.AudioEncoder: mel [B, n_mels, 3000] -> features [B, 1500, d]."""

    def __init__(self, E, W, rt, n_layer: int, n_head: int, ctx: int = CTX):
        super().__init__()
        self.E, self.W, self.rt, self.n_layer, self.n_head, self.ctx = E, W, rt, n_layer, n_head, ctx
        self._memo: dict[bytes, torch.Tensor] = {}   # whisper encodes the same window twice (decode + DTW)
        self.calls = self.hits = 0
        self.busy_s = 0.0

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        outs = []
        for b in range(mel.shape[0]):
            m = np.ascontiguousarray(mel[b].detach().cpu().numpy().astype(np.float32)[:, : 2 * self.ctx])
            key = hashlib.blake2b(m.tobytes(), digest_size=16).digest()
            self.calls += 1
            if key in self._memo:
                self.hits += 1
                outs.append(self._memo[key])
                continue
            t0 = time.perf_counter()
            feats = self.E.encoder(self.W, m, self.rt.matmul, self.rt.matmul, prof={}, npu_softmax=False,
                                   n_layer=self.n_layer, n_head=self.n_head, attn_fn=self.rt.attn_chain,
                                   mlp_fn=self.rt.mlp, fold_bias=True)
            self.busy_s += time.perf_counter() - t0
            t = torch.from_numpy(np.ascontiguousarray(feats)).float()
            if len(self._memo) >= 4:
                self._memo.pop(next(iter(self._memo)))
            self._memo[key] = t
            outs.append(t)
        return torch.stack(outs)


def _clean_segments(result: dict, tmap: TimeMap, regions: list[tuple[float, float]]) -> list[dict]:
    """Normalise openai-whisper output and map timestamps back to the original timeline."""
    out = []
    for i, seg in enumerate(result.get("segments", [])):
        words = []
        for w in seg.get("words", []) or []:
            s, e = tmap.to_orig(float(w["start"])), tmap.to_orig(float(w["end"]), end=True)
            words.append({"word": w["word"], "start": round(s, 3), "end": round(max(e, s), 3),
                          "score": round(float(w.get("probability", 0.0)), 3)})
        text = seg["text"].strip()
        if not text:
            continue
        start = words[0]["start"] if words else round(tmap.to_orig(float(seg["start"])), 3)
        end = words[-1]["end"] if words else round(tmap.to_orig(float(seg["end"]), end=True), 3)
        out.append({"id": len(out), "start": start, "end": end, "text": text, "words": words,
                    "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 3),
                    "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 3)})
    return out


class Engine:
    def __init__(self, model_name: str | None = None, threads: int | None = None, use_vad: bool | None = None):
        # model and threads are read once (restart to change); the rest below is live configuration.
        self.model_name = model_name or config.get("model")
        self.threads = threads or config.get("threads")
        self._use_vad_override = use_vad
        self.vad = Vad()
        self.model = None
        self.encoder: NpuAudioEncoder | None = None
        self._lock = threading.Semaphore(1)
        self.current: dict | None = None          # the job running on the NPU right now (for dashboards)
        self.stats = {"requests": 0, "audio_s": 0.0, "wall_s": 0.0, "started": time.time(), "ready": False}

    # live settings (re-read on every use, so a changed config file applies to the next request)
    @property
    def use_vad(self) -> bool:
        return self._use_vad_override if self._use_vad_override is not None else bool(config.get("vad.enabled"))

    @property
    def queue_wait(self) -> float:
        return float(config.get("queue_wait_s"))

    @property
    def beam_size(self) -> int:
        """Accuracy over speed by default: beam search with 5 candidates (1 = greedy, fastest)."""
        return int(config.get("beam"))

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import whisper
        E, rawxrt = _import_xdna()
        torch.set_num_threads(self.threads)
        torch.set_grad_enabled(False)
        t0 = time.perf_counter()
        model = whisper.load_model(self.model_name).eval()
        W = {"w." + k: v.numpy().astype(np.float32) for k, v in model.encoder.state_dict().items()}
        n_layer = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("w.blocks."))
        d_model = W["w.ln_post.weight"].shape[0]
        n_head = d_model // 64
        for k in [k for k in list(W) if k.endswith(".weight") and W[k].ndim == 2 and not k.endswith("_ln.weight")]:
            W[k[: -len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
        extra = E.prepare_mlp_chunks(W, n_layer, d_model, 0)
        rt = rawxrt.RawRT()
        rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)] + extra)
        E.gelu = rt.gelu
        self.encoder = NpuAudioEncoder(E, W, rt, n_layer, n_head)
        model.encoder = self.encoder                      # <- the whole trick
        self.model = model
        log.info("model %s loaded, encoder on NPU (%d layers, d=%d) in %.1fs", self.model_name, n_layer, d_model,
                 time.perf_counter() - t0)

    def warmup(self) -> None:
        """Compile/attach the NPU overlays (minutes on the very first start, cached afterwards)."""
        t0 = time.perf_counter()
        n_mels = self.model.dims.n_mels
        self.model.encoder(torch.zeros(1, n_mels, 2 * CTX))
        self.model.transcribe(np.zeros(SR * 2, dtype=np.float32), fp16=False, language="en")
        self.stats["ready"] = True
        log.info("warmup done in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------ helpers
    def _acquire(self):
        if not self._lock.acquire(timeout=self.queue_wait):
            raise TimeoutError("NPU busy: queue wait exceeded")

    # -------------------------------------------------------------- language
    def detect_language(self, audio: np.ndarray) -> dict[str, Any]:
        import whisper
        from whisper.audio import log_mel_spectrogram, pad_or_trim
        regions = self.vad.speech_regions(audio) if self.use_vad else []
        start = int((regions[0][0] if regions else 0.0) * SR)
        chunk = audio[start: start + 30 * SR]
        self._acquire()
        try:
            mel = log_mel_spectrogram(pad_or_trim(chunk), self.model.dims.n_mels)
            _, probs = self.model.detect_language(mel)
        finally:
            self._lock.release()
        code = max(probs, key=probs.get)
        return {"language_code": code, "detected_language": whisper.tokenizer.LANGUAGES.get(code, code),
                "confidence": round(float(probs[code]), 4)}

    # ------------------------------------------------------------ transcribe
    def transcribe(self, audio: np.ndarray, *, language: str | None = None, task: str = "transcribe",
                   initial_prompt: str | None = None, vad: bool | None = None, beam_size: int | None = None,
                   temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> dict[str, Any]:
        t0 = time.perf_counter()
        beam = self.beam_size if beam_size is None else beam_size
        duration = len(audio) / SR
        use_vad = self.use_vad if vad is None else vad
        regions: list[tuple[float, float]] = []
        tmap = TimeMap([])
        work = audio
        if use_vad:
            work, tmap, regions = self.vad.condense(audio)
        self._acquire()
        self.current = {"audio_s": round(duration), "started": time.time(), "beam": beam, "vad": bool(use_vad)}
        try:
            result = self.model.transcribe(
                work, language=language or None, task=task, initial_prompt=initial_prompt, fp16=False,
                word_timestamps=True, condition_on_previous_text=False, temperature=temperature,
                compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6,
                hallucination_silence_threshold=2.0, verbose=None,
                **({"beam_size": beam, "best_of": beam} if beam and beam > 1 else {}))
        finally:
            self.current = None
            self._lock.release()
        segments = _clean_segments(result, tmap, regions)
        wall = time.perf_counter() - t0
        self.stats["requests"] += 1
        self.stats["audio_s"] += duration
        self.stats["wall_s"] += wall
        log.info("transcribed %.1fs audio in %.1fs (x%.1f realtime), %d segments, vad=%s",
                 duration, wall, duration / max(wall, 1e-6), len(segments), bool(regions) and tmap.pieces != [])
        return {"language": result.get("language", language), "duration": round(duration, 3),
                "text": " ".join(s["text"] for s in segments), "segments": segments,
                "speech_regions": [[round(s, 2), round(e, 2)] for s, e in regions]}
