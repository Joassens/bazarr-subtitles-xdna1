"""npusubs: subtitle-grade speech-to-text on the AMD XDNA1 NPU.

The Whisper encoder runs on the NPU (via drakosha/whisper-xdna, MIT); everything
else (decoding, word timestamps, VAD, speaker diarization, subtitle formatting)
runs on the CPU. Speaks the whisper-asr-webservice API that Bazarr's "Whisper"
provider uses, plus an OpenAI-compatible /v1/audio/transcriptions endpoint.
"""
__version__ = "0.1.0"
