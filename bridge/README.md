# Bridge: Bazarr, speech service and translation

An optional bridge between **Bazarr's Whisper provider** and one or more speech services (npusubs,
whisper-asr-webservice, any OpenAI-compatible endpoint). It does what Bazarr cannot:

* **Several backends with failover.** `WHISPER_ASR_URL` is a comma separated list in order of preference
  (for example a fast GPU box first, the NPU second). The first one that answers is used; one that is offline is
  skipped for a minute so it does not slow every request. An OpenAI-compatible endpoint can serve as last resort.
* **Translation with an LLM**, for example Japanese audio to English or Dutch subtitles: the speech service
  transcribes in the source language, the bridge translates with context.
* **Accuracy passes** (on by default): a scene "bible" (characters, gender, names, recurring terms) from the
  whole script, then a conservative review that only accepts real corrections (mistranslation, pronoun/gender,
  name, grammar, misheard, literal idiom). Style preferences are ignored on purpose.
* **Speaker aware subtitles**: with diarization, a subtitle is split when the speaker changes.
* Filters for repetition loops, phantom first cues and recurring hallucinated lines; OP/ED filename skip.

It is a modified copy of [McCloudS/bazarr-openai-whisperbridge](https://github.com/McCloudS/bazarr-openai-whisperbridge)
(Apache-2.0). `bridge.py` replaces the file inside the upstream image, see [docker-compose.example.yml](docker-compose.example.yml)
and [NOTICE-CHANGES.md](NOTICE-CHANGES.md).

## Web UI

The bridge serves a small web UI at `/ui` on its existing port, with no build step and no extra dependencies.
Mount the `ui/` folder next to `bridge.py` (`./ui:/app/ui:ro`, see the compose example).

* **Dashboard**: bridge version and uptime, config health, live status of the speech services (for npusubs also the
  model, beam, diarization and the job running on the NPU right now), the translation LLMs (which one is used, what
  Lemonade has loaded), Sonarr/Radarr, and the last 50 requests (file, task, backends used, status, time).
* **Backends**: the speech services in order of preference and the LLM endpoints, with latency and which one is in use.
* **Configuration**: every setting of `bridge`, `npusubs` and `settings` with description, current value, and where it
  comes from (default, file, or an environment variable that overrides the file). Edit toggles, numbers, choices, text
  and lists (one per line), then **Save**: the change is validated against the schema (errors show next to the field),
  written atomically and picked up by the running service within seconds, no restart. Settings marked *restart needed*
  apply after the next restart; settings overridden by an environment variable are shown read-only. "Reset to default"
  removes the key from the file. Secrets are never shown or edited here (they live in `secrets.env`).
* **History**: every save keeps the previous file in `config/history/<component>/`. Compare a snapshot with the current
  file and restore it (the current file is snapshotted first, so a restore can be undone).

* **Request detail**: click a request on the dashboard to open it: for a translation the source line, the **draft**, the **final** text after the review pass
  and the reason for each correction, plus the translation bible; for any request the finished SRT with a download
  button. The 10 most recent requests keep this detail in memory (until the bridge restarts).
* **Resource guard** (dashboard card, settings under `guard.*`): an iGPU takes its model memory from the host's RAM, so a
  big model can starve the host (in one setup it got a virtual machine killed). The guard can hold jobs until it is safe:
  *allowed hours* (`guard.heavy_window`, for example `01:00-07:00`; jobs outside it wait), and *minimum free host RAM*
  for translation jobs (`guard.min_free_host_ram_gb`, read from `guard.host_ram_url`, an endpoint on the machine that
  owns the memory: run [../tools/hostmem_agent.py](../tools/hostmem_agent.py) there). Waiting jobs show their reason
  on the dashboard and are released within seconds of a settings change; after `guard.max_wait_s` they fail instead of
  waiting forever. The card also warns when more LLMs are loaded than `guard.max_llm_models`. It guards what comes
  through the bridge; programs that call the LLM directly (Lingarr, Home Assistant) are not held back.
* **Low-priority lane for other LLM clients** ([../tools/idle_gate.py](../tools/idle_gate.py)): a background client such
  as Lingarr should not compete with Home Assistant or a subtitle job for the GPU. Point it at the gate instead of the
  LLM server and its generation requests are held until nobody else has used the LLM for a while (default 10 minutes:
  another client's request, a busy model, or a running bridge job, which the bridge reports on `GET /busy`, counts
  only). The client sees a slow answer, not an error, as long as it tolerates a long wait (Lingarr does). Everyone else
  keeps using the LLM server directly and never waits. `GET /gate/status` on the gate shows what it sees. Run it as a
  service (see the comment in the file) and set the client's endpoint to the gate's address.
* **Profiles**: settings that apply only to some files (a path pattern and/or a language). Create, edit and delete them
  with a form that offers every setting, and check which profile applies to a path. See
  [../config.example](../config.example/README.md) for what a profile can change. The Dashboard shows the profile each
  request used.
* **Playground**: run one input with up to four sets of settings and compare the results side by side.
  *Translation*: paste an SRT and give each variant the bridge settings that differ (for example the draft without
  bible and review against your current settings); lines that differ from the first variant are highlighted.
  *Transcription*: upload a short clip and compare beam, VAD and speaker labelling on the first reachable speech
  service. One run at a time (shown with its elapsed time); runs stay in memory until the bridge restarts. A translation
  run waits for real jobs and holds them back while it runs, because it changes the bridge's settings.

**New here? See [../docs/web-ui-getting-started.md](../docs/web-ui-getting-started.md): one command creates the token.**

Editing is only enabled when a token is set **and** the config folder is mounted read-write (`../config:/config`).
Set `ui.token` (in `config/secrets.env` as `WEBUI_TOKEN`, referenced from `bridge.json` as `"${WEBUI_TOKEN}"`; restart the
bridge after changing it): open `/ui?token=...` once to log in (a cookie is set), or send an `X-Token` header to the
JSON endpoints. Writes additionally need an `X-Requested-With: ui` header (CSRF guard) and are refused when the file
changed since it was loaded (HTTP 409). Without a token the UI is read-only and open to everyone who can reach the
bridge, so keep it on a trusted LAN. `ui.enabled=false` turns it off.

JSON API: `GET /ui/api/status`, `GET /ui/api/config?component=`, `POST /ui/api/config/<component>`
(`{"changes": {"beam": 4}, "remove": ["music.enabled"], "etag": "..."}`), `GET /ui/api/history/<component>[/<snapshot>]`,
`POST /ui/api/history/<component>/<snapshot>/restore`, `GET /ui/api/profiles`, `GET /ui/api/profiles/match?path=`,
`POST /ui/api/profiles/<name>` and `/delete`, `GET /ui/api/jobs/<id>[/srt]`, `GET/POST /ui/api/playground[/translate|/transcribe|/<run>]`.

## Configuration

Settings live in `config/bridge.json` (see [../config.example](../config.example/README.md)): defaults, validation and
descriptions come from `config/schemas/bridge.schema.json`, changes are picked up between requests without a restart,
and an invalid file keeps the last good configuration. `GET /config` shows the active values (secrets masked) and
where each one comes from. API keys belong in `config/secrets.env` and are referenced as `"${NAME}"`.

The environment variables below keep working and **override** the file when set (useful for existing deployments);
use `python subcfg/subcfg.py seed bridge --dir config` to turn your current environment into `bridge.json`.

## Settings (environment variables, optional)

| Variable | Meaning |
|---|---|
| `WHISPER_ASR_URL` | comma separated whisper-asr-webservice compatible services, first reachable wins |
| `WHISPER_PREFER_NATIVE` | `true` = use them whenever reachable, not only when faster than the fallback |
| `WHISPER_ASR_TIMEOUT` | seconds to wait for a transcription (a film on an NPU takes hours; default 600 is too short) |
| `WHISPER_DIARIZE` | ask the service for speaker labels (default `true`) |
| `OPENAI_BASE_URL`, `OPENAI_BASE_URL_FALLBACK` | OpenAI-compatible LLM (and whisper fallback) endpoints |
| `TRANSLATE_LLM_MODEL`, `TRANSLATE_LLM_MODEL_FALLBACK` | model names on those endpoints |
| `TRANSLATE_LLM_BASE_URL`, `TRANSLATE_LLM_MODEL_DEDICATED` | optional dedicated translation LLM that wins when reachable |
| `TRANSLATE_BIBLE`, `TRANSLATE_REVIEW` | accuracy passes (default `true`) |
| `TRANSLATE_KEEP_HONORIFICS` | keep -san/-kun/-chan in Japanese translations (default `true`) |
| `TRANSLATE_TEMPERATURE`, `TRANSLATE_BATCH_SIZE`, `TRANSLATE_CONTEXT_LINES` | translation tuning |
| `TRANSLATE_DEBUG` | log the bible and every review edit |
| `SONARR_URL`/`SONARR_API_KEY`, `RADARR_URL`/`RADARR_API_KEY` | optional: show/movie context for prompts |

Memory note for self-hosters: an iGPU shares the host's RAM. Check the host's free memory before loading a second large
model next to an existing one; an out-of-memory kill takes down whatever is biggest, which may be an unrelated VM.
