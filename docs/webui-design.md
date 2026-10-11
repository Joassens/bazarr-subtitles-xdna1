# Web UI and shared configuration: design and status

Status: **M1 to M5 are done** (M4: profiles applied per job and edited in the UI, plus the A/B playground. M5: job detail
with draft against reviewed text and SRT download, and the resource guard with allowed hours and a host RAM check).
Not done from M5: the Lingarr refiner integration (the refiner is a separate script on another host, disabled until a
better reviewer model is chosen). M1: shared config library `subcfg`, schemas, npusubs and the bridge read
`config/*.json` (environment variables still override). M2: dashboard, backends and configuration view. M3: editing with
schema validation, hot reload, conflict detection, history and rollback, all served by the bridge at `/ui` on its
existing lightweight web server (FastAPI/uvicorn), not as a separate container; the config folder is mounted read-write
into the bridge and writing needs `ui.token`. Everything from M6 on (packaging, publishing) is still open.

## Goal

Tune the whole subtitle chain (speech service, bridge, review/refiner, Lingarr prompt) from one web page,
without editing environment variables or restarting containers. Settings live as **plain JSON files in one
config folder** (optionally a git repo). Every tool reads that folder, validates it and reloads it by itself.

```
 Browser ──> webui (FastAPI + static pages) ──writes──> /config  (JSON, prompts, history)
                         │                                  ▲ read-only mounts, hot reload
                         ├── GET /health of every service    │
                         └── Playground runs through the    npusubs   bridge   refiner   lingarr-watcher
                             normal API of the tools
```

The web UI is its own small container. Tools never depend on it: if it is down they keep running on the
last good config.

## 1. The config folder

```
config/
  settings.json            # shared: named backends, endpoints, resource guard
  npusubs.json             # speech service: beam, VAD, music, diarization, line rules
  bridge.json              # backend order, timeouts, translation + accuracy passes, filters
  refiner.json             # post-pass over Lingarr output: target language, roots, review on/off
  profiles/
    anime-ja.json          # overrides, chosen per series/path/language pair
    live-action-en.json
  prompts/
    translate.system.md    # editable prompt templates with {placeholders}
    review.system.md
    bible.system.md
    lingarr.system.md      # what the Lingarr watcher pushes into Lingarr
  history/                 # one snapshot per save (or: the folder is a git repo and every save is a commit)
  secrets.env              # chmod 600, never shown or returned by the UI; referenced as ${NAME}
```

Rules:

* **Schema first.** Every file has `schema_version` and a JSON Schema. The UI builds its forms from the
  schemas, and tools reject invalid files and keep the last good one (the UI shows the error).
* **Precedence**: built-in defaults < `config/*.json` < environment variables < matching profile.
  Environment variables stay supported, so today's compose files keep working while we migrate.
* **Secrets stay out of JSON.** API keys and the HF token live in `secrets.env` (or the environment) and are
  referenced as `"hf_token": "${HF_TOKEN}"`. The UI has write-only fields for them.
* **Atomic writes** (temp file + rename), so a tool never reads a half file.
* **Hot reload**: each tool checks the folder every few seconds (mtime) or on `POST /reload`. Settings that
  need a restart (for example the model, thread count) are marked in the schema and shown as "applies after restart".
* **History and rollback**: every save is a snapshot (or a git commit with the UI user as author). The UI can
  diff and roll back. Because it is just a folder, you can also keep it in a private git repo yourself.

### Example

```json
// config/bridge.json
{
  "schema_version": 1,
  "backends": {
    "speech": [
      { "name": "gpu-box",  "url": "http://<gpu-host>:9090", "kind": "whisper-asr" },
      { "name": "npu",        "url": "http://<npu-host>:9091",  "kind": "whisper-asr" }
    ],
    "prefer_first_reachable": true,
    "retry_offline_after_s": 60,
    "timeout_s": 28800
  },
  "translation": {
    "llm": { "name": "qwen30b", "url": "http://<llm-host>:13305/api/v1", "model": "Qwen3-30B-A3B-Instruct-2507-GGUF" },
    "bible": true,
    "review": { "enabled": true, "accept_reasons": ["mistranslation","pronoun-gender","name","grammar","misheard","literal-idiom"] },
    "keep_honorifics": true,
    "temperature": 0.2,
    "batch_size": 5,
    "context_lines": 6
  },
  "diarization": { "enabled": true, "speaker_style": "dash" },
  "filters": { "repetition_loops": true, "recurring_lines": true, "skip_filename_pattern": "(?i)\\b(NC)?(OP|ED)\\d{0,2}\\b" }
}
```

### Profiles

A profile is a partial config plus a match rule. First match wins.

```json
// config/profiles/anime-ja.json
{
  "match": { "path_glob": "*/Anime/*", "source_language": "ja" },
  "overrides": {
    "npusubs": { "beam": 5, "music": true },
    "bridge":  { "translation": { "keep_honorifics": true } }
  }
}
```

The bridge already knows the video path and language of each request, so it can resolve the profile per job.

## 2. Pages

1. **Dashboard**: services up/down with latency (npusubs, bridge, LLM, Lingarr), NPU busy/idle, queue length,
   host memory, last jobs. A warning bar for the *resource guard* (below).
2. **Backends**: ordered list (drag to reorder) of speech and LLM backends with health, a test button, and
   "disable for now".
3. **Transcription**: beam, VAD thresholds, music handling, diarization and speaker style, line length, gap split.
   A live preview on a test clip.
4. **Translation**: LLM per language pair, bible/review toggles, honorifics, temperature, and the **prompt editor**
   (templates with placeholders and a "dry run on 20 lines").
5. **Profiles**: rule list with "which profile applies to this file?" tester.
6. **Playground**: upload a clip or paste an SRT, run it with config A and config B, see the results side by
   side (like the model comparison table in this repo's history), keep or discard.
7. **Jobs**: requests Bazarr sent (file, duration, backend used, wall time, number of review edits) with the draft vs
   reviewed text and the SRT for download.
8. **History**: config snapshots, diff, rollback, export/import (zip).

## 3. Resource guard (lesson learned)

An iGPU takes its model memory from the host's RAM. Loading a second large model once made the host's OOM killer
terminate an unrelated virtual machine. The UI and the tools therefore get a small guard in `settings.json`:

```json
"resource_guard": { "min_free_host_ram_gb": 12, "max_concurrent_llm_models": 1, "heavy_jobs_window": "01:00-07:00" }
```

* Before activating a model or backend the UI checks free host RAM (a small read-only agent or a read-only Proxmox
  API token) and refuses with an explanation when it would not fit.
* Optional time window for heavy jobs (beam search plus diarization of a whole season).

## 4. Security

* Bound to the LAN address only. Optional shared token / basic auth in front (`WEBUI_TOKEN`).
* The UI never returns secrets; saved values are masked. CSRF protection on writes. All writes are validated
  against the schema server-side.
* The config folder is mounted read-only into the tools, read-write only into the UI container.

## 5. Implementation sketch

* `subcfg`: a tiny Python package (loader, defaults, JSON Schema validation, hot reload, profile resolution).
  npusubs, the bridge and the refiner use it; env vars keep working as overrides.
* `webui`: FastAPI plus a no-build front end (plain JS with small components), forms generated from the schemas,
  one container, one port.
* Non-Python consumers (shell, the Lingarr prompt watcher) read `GET /config/effective?path=...` or the files.

## 6. Plan and status

| Step | Content | Status |
|---|---|---|
| M1 | `subcfg` + schemas; npusubs and bridge read `config/*.json` (env still overrides) | done |
| M2 | read-only UI: dashboard, backends, effective-config view | done |
| M3 | editing with validation, hot reload, history and rollback | done |
| M4 | profiles (applied per job) and the A/B playground | done |
| M5 | request detail (draft vs reviewed, SRT), resource guard with allowed hours and host RAM | done; refiner integration not done |
| M6 | docs, CI, packaging | done |

## 7. Decisions on the open questions

1. **Where the UI runs:** inside the bridge, on its existing FastAPI/uvicorn port, with no build step and no extra
   container.
2. **Config folder:** a plain folder with a snapshot per save (`history/`); you can put it in your own git repo.
3. **Lingarr prompt and Bazarr provider settings:** not managed. The Lingarr refiner (a separate script) stays outside
   until a better reviewer model is chosen.
4. **UI language:** English only.
5. **Auth:** a shared token (`ui.token`) in a cookie or `X-Token` header, writes need it plus a CSRF header. No login
   page; keep the bridge on a trusted LAN.
