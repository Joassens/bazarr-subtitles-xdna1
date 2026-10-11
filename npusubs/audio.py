"""Audio decoding helpers: anything ffmpeg understands, or raw 16 kHz mono PCM."""
from __future__ import annotations

import subprocess

import numpy as np

SR = 16000


class AudioError(ValueError):
    pass


def decode(data: bytes, raw_pcm: bool = False, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Return mono float32 audio at 16 kHz.

    raw_pcm=True means `data` is signed 16-bit little-endian mono PCM at 16 kHz,
    which is what Bazarr sends when it passes encode=false.
    """
    if not data:
        raise AudioError("empty audio")
    if raw_pcm:
        if len(data) % 2:
            data = data[:-1]
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        input=data, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("ffmpeg could not decode the upload: " + proc.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(proc.stdout, dtype="<f4").copy()
