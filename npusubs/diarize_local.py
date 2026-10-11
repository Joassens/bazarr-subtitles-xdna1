"""Token-free speaker diarization: speaker embeddings of short windows, clustered.

No Hugging Face token, no gated terms and no pyannote.audio needed. It uses the WeSpeaker ResNet34 speaker embedding
(the same network pyannote uses) in the TorchScript form published as `spurtcarl/finesub-speaker-embedding`
(CC BY 4.0: credit WeSpeaker, Wang et al., ICASSP 2023, and pyannote). The files are downloaded once from the Hugging Face
hub *without* a token, pinned to one revision and checked against a SHA-256 before they are loaded.

How it works: the (speech-only) audio is cut into overlapping windows (1.5 s, every 0.5 s), every window gets a
256-dimensional embedding, the embeddings are reduced with the LDA transform from pyannote's community-1 pipeline and
clustered hierarchically (average linkage, cosine distance). Tiny clusters are merged into the nearest big one and the
labels are smoothed over time. It is much cheaper than pyannote on the CPU (about 15x faster in a test) but coarser: it
has no overlapped-speech detection and cannot split a back-and-forth faster than about a window.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time

import numpy as np

log = logging.getLogger("npusubs")
SR = 16000

REPO = "spurtcarl/finesub-speaker-embedding"
REVISION = "92e0ffb2cf50cdedd7f9b5e3849eeca4c0c54598"
FILES = {                                           # file -> SHA-256 (a changed file is refused, never loaded)
    "wespeaker-cpu.ts": "0a3e9baaa186ffffb6b5a7e5d4899549f3beafb1e87724638b449eda623e86da",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smooth_labels(labels: np.ndarray, size: int = 5) -> np.ndarray:
    """Majority vote over a sliding window (removes one-window flickers)."""
    out = labels.copy()
    half = size // 2
    for i in range(len(labels)):
        out[i] = np.bincount(labels[max(0, i - half): i + half + 1]).argmax()
    return out


def merge_small_clusters(emb: np.ndarray, labels: np.ndarray, min_windows: int) -> np.ndarray:
    """Clusters with fewer than `min_windows` windows are merged into the nearest bigger cluster (by centroid)."""
    labels = labels.copy()
    ids, counts = np.unique(labels, return_counts=True)
    big = [i for i, c in zip(ids, counts) if c >= min_windows]
    if not big or len(big) == len(ids):
        return labels
    cent = {i: _l2(emb[labels == i].mean(axis=0)) for i in ids}
    for i in ids:
        if i not in big:
            nearest = max(big, key=lambda b: float(cent[i] @ cent[b]))
            labels[labels == i] = nearest
    return labels


def cluster_embeddings(emb: np.ndarray, threshold: float, min_speakers: int | None, max_speakers: int | None,
                       min_windows: int) -> np.ndarray:
    """Cluster L2-normalised embeddings; returns one label (0..k-1) per row."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    z = linkage(emb, method="average", metric="cosine")
    labels = fcluster(z, threshold, criterion="distance") - 1
    labels = merge_small_clusters(emb, labels, min_windows)
    k = len(np.unique(labels))
    lo, hi = (min_speakers or 1), (max_speakers or len(emb))
    if k < lo or k > hi:                               # the caller knows better than the threshold
        labels = fcluster(z, max(lo, min(k, hi)), criterion="maxclust") - 1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def labels_to_turns(labels: np.ndarray, starts: np.ndarray, window: float, hop: float, duration: float
                    ) -> list[tuple[float, float, str]]:
    """Each window owns the stretch around its centre; the first and the last own the audio's ends."""
    turns: list[list] = []
    for i, (s, lab) in enumerate(zip(starts, labels)):
        a = 0.0 if i == 0 else s + window / 2 - hop / 2
        b = duration if i == len(starts) - 1 else s + window / 2 + hop / 2
        if turns and turns[-1][2] == lab:
            turns[-1][1] = b
        else:
            turns.append([a, b, int(lab)])
    return [(float(a), float(min(b, duration)), f"SPEAKER_{lab:02d}") for a, b, lab in turns if b > a]


class LocalDiarizer:
    """Same interface as `Diarizer` (available, status, run)."""

    def __init__(self):
        self._net = None
        self._xv = None
        self._lock = threading.Lock()
        self._fail_until = 0.0
        self._fail_reason = ""

    @staticmethod
    def _deps_ok() -> bool:
        try:
            import scipy  # noqa: F401
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            return True
        except Exception:                               # noqa: BLE001
            return False

    @property
    def available(self) -> bool:
        return self._deps_ok() and time.monotonic() >= self._fail_until

    @property
    def status(self) -> str:
        if not self._deps_ok():
            return "needs torch, torchaudio and scipy"
        if time.monotonic() < self._fail_until:
            return f"model unavailable: {self._fail_reason}"
        return "ready" if self._net is not None else "ready (the model is downloaded on first use)"

    def _fetch(self) -> dict[str, str]:
        from huggingface_hub import hf_hub_download
        paths = {}
        for name, digest in FILES.items():
            p = hf_hub_download(REPO, name, revision=REVISION, token=False)     # no token: the repository is public
            if _sha256(p) != digest:
                raise RuntimeError(f"{name} does not match the pinned checksum; refusing to load it")
            paths[name] = p
        return paths

    def _load(self) -> None:
        if self._net is not None:
            return
        import torch
        try:
            paths = self._fetch()
            net = torch.jit.load(paths["wespeaker-cpu.ts"], map_location="cpu").eval()
            xv = np.load(paths["plda/xvec_transform.npz"])
            self._xv = {k: xv[k] for k in xv.files}
            self._net = net
        except Exception as exc:                        # noqa: BLE001
            self._fail_until = time.monotonic() + 600.0
            self._fail_reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            log.warning("local diarization model unavailable (%s); retrying in 10 min", self._fail_reason)
            raise

    def _lda(self, e: np.ndarray) -> np.ndarray:
        xv = self._xv
        z = (np.sqrt(256) * _l2(e - xv["mean1"])) @ xv["lda"] - xv["mean2"]
        return _l2(np.sqrt(128) * _l2(z))

    def _embed(self, audio: np.ndarray, window: float, hop: float) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torchaudio.compliance.kaldi as kaldi
        n, h = int(window * SR), int(hop * SR)
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        starts = list(range(0, len(audio) - n + 1, h))
        embs = []
        for i in range(0, len(starts), 32):
            feats = []
            for s in starts[i:i + 32]:
                w = torch.from_numpy(np.ascontiguousarray(audio[s:s + n])).unsqueeze(0) * 32768
                f = kaldi.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=SR,
                                window_type="hamming", use_energy=False)
                feats.append(f - f.mean(0, keepdim=True))
            frames = min(f.shape[0] for f in feats)
            x = torch.stack([f[:frames] for f in feats])
            with torch.no_grad():
                embs.append(self._net(x, torch.ones(x.shape[0], frames)).numpy())
        return np.concatenate(embs), np.array(starts) / SR

    def run(self, audio: np.ndarray, min_speakers: int | None = None, max_speakers: int | None = None
            ) -> list[tuple[float, float, str]]:
        from . import config
        window = float(config.get("diarization.embedding.window_s", 1.5))
        hop = float(config.get("diarization.embedding.hop_s", 0.5))
        with self._lock:
            self._load()
            emb, starts = self._embed(audio, window, hop)
        emb = self._lda(emb)
        min_windows = max(1, int(round(float(config.get("diarization.embedding.min_speaker_s", 3.0)) / hop)))
        labels = cluster_embeddings(emb, float(config.get("diarization.embedding.threshold", 0.8)), min_speakers, max_speakers,
                                    min_windows)
        labels = smooth_labels(labels)
        return labels_to_turns(labels, starts, window, hop, len(audio) / SR)


class DiarizerRouter:
    """Picks the engine: `pyannote` (best, needs a Hugging Face token), `embedding` (token-free) or `auto`
    (pyannote when it is usable, otherwise embedding). The engine can be chosen per request."""

    ENGINES = ("auto", "pyannote", "embedding")

    def __init__(self, pyannote, local: LocalDiarizer):
        self.pyannote, self.local = pyannote, local

    def pick(self, engine: str | None = None):
        from . import config
        engine = engine or config.get("diarization.engine", "auto")
        if engine == "pyannote":
            return self.pyannote
        if engine == "embedding":
            return self.local
        return self.pyannote if self.pyannote.available else self.local

    def engine_name(self, engine: str | None = None) -> str:
        return "pyannote" if self.pick(engine) is self.pyannote else "embedding"

    @property
    def available(self) -> bool:
        return self.pick().available

    @property
    def status(self) -> str:
        d = self.pick()
        return f"{d.status} [{self.engine_name()}]"
