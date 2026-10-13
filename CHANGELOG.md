# Changelog

## Unreleased

### gpuwatch: Priorities tab
- `gpuwatch --arbiter-url`: passes the gpuarbiter's page and API through under `/arbiter/`; the page shows it as a **Priorities** tab. The arbiter can then
  listen on localhost only.

### gpuarbiter (new tool)
- `tools/gpuarbiter.py` + `gpuarbiter.html`: pauses a Tdarr node while a higher-priority class (for example a media server) uses the same GPU
  engine, or when host RAM is low, and resumes it after a quiet period. Per-engine priorities, dry-run mode, a log of every decision, an editable
  page, respects manual pauses/resumes. See [docs/gpuarbiter.md](docs/gpuarbiter.md).

### gpuwatch (new tool)
- `tools/gpuwatch.py` + `gpuwatch.html`: read-only service that shows which container/process uses the AMD GPU (per-client
  fdinfo counters), with a live page, a history in SQLite (load per class, sessions, overlaps, unexplained GPU load), JSON
  and Prometheus metrics. See [docs/gpuwatch.md](docs/gpuwatch.md).

### Idle gate looks at the real GPU load
- `tools/hostmem_agent.py` also reports the amdgpu busy percentage (`gpu_busy_pct`, `gpu_busy_max_10s`, `gpu_busy_avg_10s`).
- `tools/idle_gate.py --gpu-url`: the low-priority lane stays closed only while the GPU is really busy (over `--gpu-busy-pct`,
  default 15 %), not merely because a subtitle job runs somewhere. The gate's own GPU load is ignored. `--busy-url` is optional now.

### Packaging and documentation
- **One stack**: `compose.yml` now starts the bridge and npusubs together (`docker compose up -d`); the bridge has its own
  `bridge/Dockerfile`. `docker compose up -d bridge` runs the bridge alone (no NPU).
- `config.example/bridge.json` points the bridge at the `npusubs` service by default.
- README: how the pieces fit (diagram and table), what you need for which use, a "No NPU?" path, Bazarr points at the bridge.
- `docs/huggingface.md`: when a Hugging Face account is needed (only the pyannote engine), how to accept the licence and make a token.

## 0.1.0 (first release)

### Web UI and configuration (bridge)
- **Shared configuration**: settings live in JSON files with schemas (`config.example/`), hot reloaded, invalid files
  keep the last good configuration, secrets in `secrets.env`, environment variables still override. Used by npusubs
  and the bridge.
- **Web UI** at `/ui` (user name and password chosen on first use; optional API token for scripts): dashboard, backends, configuration editor with validation and conflict detection, history with
  compare and restore, profiles applied per job, A/B playground (translation and transcription), request detail
  (source, draft, reviewed text, reasons, SRT download), resource guard (allowed hours, host RAM).
- **Profiles**: `config/profiles/*.json` match on path and languages; the bridge and npusubs apply them per request.
- `tools/hostmem_agent.py`: read-only host RAM endpoint for the resource guard.
- `tools/idle_gate.py`: low-priority lane to an LLM server for background clients (for example Lingarr); the bridge
  reports `GET /busy` (job counts only) so the gate stays closed while a subtitle job is running.
- CI runs the unit tests and checks that the generated schemas are current.

### npusubs
- Live progress: `/health` shows what Whisper has transcribed so far (percent, latest lines, phase), and the bridge dashboard shows it as a progress bar with the latest lines.
- One request at a time from start to finish (a first-come-first-served queue, `max_concurrent_jobs`), so a diarization never overlaps the next transcription.
- Token-free speaker diarization (`diarization.engine`: auto, pyannote, embedding), also selectable per request and in the bridge.
- Whisper large-v3-turbo with the encoder on the AMD XDNA 1 NPU, `whisper-asr-webservice` and OpenAI-compatible APIs.
- Silero VAD with time mapping, music/lyrics regions, hallucination filters, subtitle formatting (CJK aware).
- Optional speaker diarization (pyannote community-1), beam search by default (accuracy over speed).

### Bridge
- Modified copy of McCloudS/bazarr-openai-whisperbridge: several speech backends with failover, long timeouts,
  dedicated translation LLM, translation bible and review pass, speaker-aware cue splitting.
