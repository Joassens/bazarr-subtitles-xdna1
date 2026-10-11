# Changelog

## 0.1.0 (first release)

### Web UI and configuration (bridge)
- **Shared configuration**: settings live in JSON files with schemas (`config.example/`), hot reloaded, invalid files
  keep the last good configuration, secrets in `secrets.env`, environment variables still override. Used by npusubs
  and the bridge.
- **Web UI** at `/ui`: dashboard, backends, configuration editor with validation and conflict detection, history with
  compare and restore, profiles applied per job, A/B playground (translation and transcription), request detail
  (source, draft, reviewed text, reasons, SRT download), resource guard (allowed hours, host RAM).
- **Profiles**: `config/profiles/*.json` match on path and languages; the bridge and npusubs apply them per request.
- `tools/hostmem_agent.py`: read-only host RAM endpoint for the resource guard.
- `tools/idle_gate.py`: low-priority lane to an LLM server for background clients (for example Lingarr); the bridge
  reports `GET /busy` (job counts only) so the gate stays closed while a subtitle job is running.
- CI runs the unit tests and checks that the generated schemas are current.

### npusubs
- Token-free speaker diarization (`diarization.engine`: auto, pyannote, embedding), also selectable per request and in the bridge.
- Whisper large-v3-turbo with the encoder on the AMD XDNA 1 NPU, `whisper-asr-webservice` and OpenAI-compatible APIs.
- Silero VAD with time mapping, music/lyrics regions, hallucination filters, subtitle formatting (CJK aware).
- Optional speaker diarization (pyannote community-1), beam search by default (accuracy over speed).

### Bridge
- Modified copy of McCloudS/bazarr-openai-whisperbridge: several speech backends with failover, long timeouts,
  dedicated translation LLM, translation bible and review pass, speaker-aware cue splitting.
