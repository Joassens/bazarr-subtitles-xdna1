# gpuarbiter: lower-priority GPU work steps aside

`tools/gpuarbiter.py` pauses a background consumer (today: a **Tdarr** node) while something more important uses the same part of the GPU, or
while the host is short of RAM, and resumes it by itself afterwards. It builds on [gpuwatch](gpuwatch.md) (who uses the GPU) and the
[host agent](../tools/hostmem_agent.py) (free RAM).

## Why per engine

The GPU is not one resource. A media server and a transcoder both use the **video engine** (encode/decode), so they really compete. An LLM mostly
uses the **graphics/compute** engines and barely touches the video engine. gpuarbiter therefore only lets a consumer yield to classes that share
an engine with it:

```
Plex, Jellyfin   priority 1   media
LLM, Whisper     priority 2   gfx, compute          (no shared engine with Tdarr: Tdarr does not yield to them)
Tdarr            priority 4   media     controller: tdarr
```

A class with a `controller` yields to every class with a **lower priority number** that is busy (`active_pct`, default 3 %) on a shared engine.

## What it does and does not do

* Pausing a Tdarr node stops **new** jobs from starting; a running job finishes (small jobs take seconds).
* It resumes only what **it** paused. A node somebody paused by hand is left alone. If somebody resumes it by hand while a rule holds it, that is
  respected for `manual_override_s`.
* `hold_s` (default 10 s) ignores blips; `release_s` (default 180 s) keeps it paused until the other work has been quiet that long.
* **RAM guard**: pauses below `pause_below_gb` free RAM, resumes above `resume_above_gb` (hysteresis, resumes at once).
* No view of the GPU (gpuwatch down) means no change: no data is not the same as quiet.
* `dry_run: true` only writes what it *would* do to the log.
* Every decision (pause, resume, would pause, error, a config change) is stored in SQLite and shown on the page.

## Run it

```bash
mkdir -p /opt/gpuarbiter && cp tools/gpuarbiter.py tools/gpuarbiter.html /opt/gpuarbiter/
python3 /opt/gpuarbiter/gpuarbiter.py --config /etc/gpuarbiter/config.json --bind <host-ip> --port 9620 --db /var/lib/gpuarbiter/events.db
```

A systemd unit is in the comment at the bottom of the script. Open `http://<host-ip>:9620/`: what is busy, the state of each consumer, the priority
table and thresholds (editable, saved with a copy of the previous version in `history/`), and the log of decisions. A missing config file means the
defaults, which start in **dry-run mode** with example classes. Class names must match what gpuwatch shows.

No authentication: bind it to a trusted network. Changing the configuration needs the `X-Requested-With: arbiter` header (the page sends it), which
keeps a stray cross-site request from changing it.

## Config

```json
{
  "dry_run": false,
  "sources": {"gpuwatch": "http://<host>:9610/api/now", "hostmem": "http://<host>:9600/"},
  "controllers": {"tdarr": {"type": "tdarr", "url": "http://<tdarr-host>:8265"}},
  "classes": [
    {"name": "Plex", "priority": 1, "engines": ["media"]},
    {"name": "Tdarr", "priority": 4, "engines": ["media"], "controller": "tdarr"}
  ],
  "active_pct": 3.0, "hold_s": 10, "release_s": 180, "manual_override_s": 600,
  "ram_guard": {"enabled": true, "pause_below_gb": 3.0, "resume_above_gb": 5.0}
}
```

The Lingarr/LLM side is handled separately by the [idle gate](../tools/idle_gate.py).
