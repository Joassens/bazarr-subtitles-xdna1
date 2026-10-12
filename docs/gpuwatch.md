# gpuwatch: who uses the GPU, and when

`tools/gpuwatch.py` is a small **read-only** service for a host whose AMD GPU is shared by several containers (a media server,
a transcoder, an LLM, speech models...). It answers "who is using the GPU right now" and keeps a history, so you can see
who competes with whom before deciding who should give way. It never pauses or changes anything.

## How it works

The amdgpu driver exposes per-client counters in `/proc/<pid>/fdinfo/<fd>` (the same source `amdgpu_top` shows): how many
nanoseconds a client kept each engine busy (graphics, compute, video decode/encode, DMA) and how much VRAM/GTT it holds.
gpuwatch samples them every 2 s, turns the differences into percentages, and groups processes into **classes** by LXC
container and process name (`Plex`, `Tdarr`, `LLM (llama-server)`...). It needs root: reading other containers' `/proc` entries.

It records, in SQLite (30 days by default):

* **load per class** every 5 s (the busiest of graphics, compute and video) plus the whole GPU's busy percentage,
* **sessions**: a class started / stopped using the GPU, with duration, average and peak,
* **overlaps**: two or more classes busy at the same time, which ones, for how long,
* the GPU load that **no known class explains** (a consumer you have not named yet).

## Run it

```bash
mkdir -p /opt/gpuwatch && cp tools/gpuwatch.py tools/gpuwatch.html /opt/gpuwatch/
python3 /opt/gpuwatch/gpuwatch.py --bind <host-ip> --port 9610 --db /var/lib/gpuwatch/gpuwatch.db
```

A systemd unit is in the comment at the bottom of `gpuwatch.py`. Open `http://<host-ip>:9610/`: live view, history chart,
a table of who used the GPU, which pairs overlapped, and the event log. No authentication: bind it to a LAN address.

## Name your classes

The built-in rules fit one particular host (container ids 114, 117, 104). Give your own with `--rules rules.json`:

```json
[
  {"class": "Plex",   "container": "114", "comm": "^Plex"},
  {"class": "Tdarr",  "container": "117", "comm": ".*"},
  {"class": "LLM",    "container": "104", "comm": "llama"}
]
```

`container` is the LXC id (`"host"` for processes outside any container, `"*"` for any), `comm` and `cmd` are regular
expressions on the process name and command line; the first matching rule wins. Anything unmatched shows up as
`<name> (CT<id>)`, so new consumers are visible without configuration.

## Endpoints

| | |
|---|---|
| `/` | the page |
| `/api/now` | the latest sample, per class and process |
| `/api/history?minutes=60` | load per class over time (about 120 points) |
| `/api/stats?hours=24` | active time, average/peak, overlaps between classes, unexplained busy time |
| `/api/events?limit=100` | sessions and overlaps, newest first |
| `/metrics` | Prometheus text format (`gpuwatch_class_gfx_percent{class="..."}` and friends) |

Prometheus scrape config: `- job_name: gpuwatch` / `static_configs: [{targets: ["<host-ip>:9610"]}]`.

## Reading the numbers

* A class's load is a percentage of the sampling interval, so a video encoder at 100 % in `media` is not the same
  thing as the graphics engine being full: that is why graphics, compute and video are shown separately.
* Memory is what the class holds, busy or not: an LLM that sits loaded and idle shows 0 % load and many GB.
* The GPU's own busy percentage (the line in the chart) counts all engines of the chip; if it is high while no class
  is active, "no known class explains it" tells you how long that happened.
