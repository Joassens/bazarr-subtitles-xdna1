#!/usr/bin/env python3
"""gpuwatch: who uses the GPU, and when. Read-only, with a history.

Run it as root on the host that owns an AMD GPU shared by several containers. Every `--interval` seconds it reads the amdgpu
per-client counters the kernel exposes in /proc/<pid>/fdinfo (the same source amdgpu_top uses), works out how busy each process
kept the graphics/compute engines, the video engines (encode/decode) and the DMA engine, and groups the processes into
*classes* (for example "Plex", "Tdarr", "LLM") by container and process name. It then

  * serves a live page and a history (`/`), JSON (`/api/*`) and Prometheus metrics (`/metrics`),
  * records everything in a small SQLite database: load per class every 5 s, *sessions* (when a class starts and stops using
    the GPU), *overlaps* (when two classes were busy at the same time) and GPU load nobody could be blamed for. That history
    is the evidence for deciding who should give way to whom.

It only reads: it never pauses or changes anything itself. (With `--arbiter-url` it also passes a few fixed paths through to a gpuarbiter next to
it, so the page can show that as a "Priorities" tab; the arbiter does the controlling.) No authentication, so bind it to a LAN address. Python standard library
only. Classes come from built-in rules for a typical Proxmox media host; override them with `--rules rules.json`:

    [{"class": "Plex", "container": "114", "comm": "^Plex"}, {"class": "LLM", "container": "104", "comm": "llama"}]

`container` is the LXC id (or "host", or "*"), `comm` and `cmd` are regular expressions on the process name and command line.
The first matching rule wins; a process no rule matches becomes "<name> (CT<id>)".

    python3 gpuwatch.py --bind <host-ip> --port 9610 --db /var/lib/gpuwatch/gpuwatch.db
"""
import argparse
import collections
import glob
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("gpuwatch")

DEFAULT_RULES = [
    {"class": "Plex", "container": "114", "comm": "^Plex"},
    {"class": "Jellyfin", "container": "114", "comm": "^(ffmpeg|jellyfin)"},
    {"class": "Tdarr", "container": "117", "comm": ".*"},
    {"class": "LLM (llama-server)", "container": "104", "comm": "llama"},
    {"class": "Whisper (Lemonade)", "container": "104", "comm": "whisper"},
    {"class": "amdgpu_top", "container": "*", "comm": "^amdgpu_top$"},
]
UNITS = {"": 1, "B": 1, "KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}
GROUPS = ("gfx", "compute", "media", "dma")


# ---------------------------------------------------------------------------------------------------- reading the kernel

def engine_group(name: str) -> str:
    n = name.lower()
    if n.startswith("gfx"):
        return "gfx"
    if n.startswith("compute"):
        return "compute"
    if n.startswith("dma"):
        return "dma"
    if any(k in n for k in ("dec", "enc", "jpeg", "vcn", "vpe")):
        return "media"
    return "other"


def parse_fdinfo(text: str):
    """One DRM client of the amdgpu driver, or None when this fd is something else."""
    d = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        d[k.strip()] = v.strip()
    if "amdgpu" not in d.get("drm-driver", ""):
        return None
    engines, mem = {}, {}
    for k, v in d.items():
        if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity"):
            m = re.match(r"(\d+)", v)
            engines[k[len("drm-engine-"):]] = int(m.group(1)) if m else 0
        elif k.startswith("drm-memory-"):
            m = re.match(r"(\d+)\s*(\w*)", v)
            if m:
                mem[k[len("drm-memory-"):]] = int(m.group(1)) * UNITS.get(m.group(2), 1)
    return {"client": d.get("drm-client-id", ""), "name": d.get("drm-client-name", ""), "engines": engines, "mem": mem}


def compile_rules(rules):
    out = []
    for r in rules:
        out.append({"class": r["class"], "container": str(r.get("container", "*")),
                    "comm": re.compile(r.get("comm", "")), "cmd": re.compile(r.get("cmd", ""))})
    return out


class GpuLoad:
    """The amdgpu busy percentage of the whole GPU (the number amdgpu_top shows at the top)."""

    def __init__(self, paths=None):
        self.paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent") if paths is None else paths

    def read(self):
        best = None
        for p in self.paths:
            try:
                with open(p) as f:
                    v = int(f.read().strip())
            except (OSError, ValueError):
                continue
            best = v if best is None else max(best, v)
        return best


class Collector:
    """Finds the processes that hold an amdgpu render node and turns the engine counters into percentages."""

    def __init__(self, rules, proc="/proc", rescan_s=10.0, gpu_paths=None, clock=time.time):
        self.proc, self.rescan_s, self.clock = proc, rescan_s, clock
        self.rules = compile_rules(rules)
        self.gpu = GpuLoad(gpu_paths)
        self.fds = {}                                    # pid -> fd numbers that point at /dev/dri/*
        self.meta = {}                                   # pid -> comm, container, cmd
        self.prev = {}                                   # (pid, client) -> (time, {group: ns})
        self._last_scan = 0.0

    # -- overridable for tests
    def _link(self, path: str) -> str:
        try:
            return os.readlink(path)
        except OSError:
            return ""

    def _read(self, path: str) -> str:
        try:
            with open(path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _scan(self):
        found = {}
        for name in os.listdir(self.proc):
            if not name.isdigit():
                continue
            fddir = f"{self.proc}/{name}/fd"
            try:
                entries = os.listdir(fddir)
            except OSError:
                continue
            for fd in entries:
                if self._link(f"{fddir}/{fd}").startswith("/dev/dri/"):
                    found.setdefault(int(name), []).append(fd)
        self.fds = found
        self.meta = {p: m for p, m in self.meta.items() if p in found}

    def _meta(self, pid: int) -> dict:
        m = self.meta.get(pid)
        if m is None:
            cg = self._read(f"{self.proc}/{pid}/cgroup").splitlines()
            lxc = re.search(r"/lxc/(\d+)/", cg[0]) if cg else None
            m = self.meta[pid] = {
                "pid": pid, "comm": self._read(f"{self.proc}/{pid}/comm").strip(),
                "container": lxc.group(1) if lxc else "host",
                "cmd": self._read(f"{self.proc}/{pid}/cmdline").replace("\0", " ").strip()[:160]}
        return m

    def classify(self, m: dict) -> str:
        for r in self.rules:
            if r["container"] in ("*", m["container"]) and r["comm"].search(m["comm"]) and r["cmd"].search(m["cmd"]):
                return r["class"]
        return f"{m['comm'] or 'pid ' + str(m['pid'])} ({'host' if m['container'] == 'host' else 'CT' + m['container']})"

    def _clients(self, pid: int):
        seen, out = set(), []
        for fd in self.fds.get(pid, []):
            c = parse_fdinfo(self._read(f"{self.proc}/{pid}/fdinfo/{fd}"))
            if c and c["client"] not in seen:
                seen.add(c["client"])
                out.append(c)
        return out

    def sample(self) -> dict:
        now = self.clock()
        if now - self._last_scan >= self.rescan_s:
            self._scan()
            self._last_scan = now
        classes, cur = {}, {}
        for pid in list(self.fds):
            m = self._meta(pid)
            cls = self.classify(m)
            for c in self._clients(pid):
                groups = collections.Counter()
                for k, ns in c["engines"].items():
                    groups[engine_group(k)] += ns
                key = (pid, c["client"])
                cur[key] = (now, groups)
                e = classes.setdefault(cls, {"gfx": 0.0, "compute": 0.0, "media": 0.0, "dma": 0.0,
                                             "vram_mb": 0.0, "gtt_mb": 0.0, "procs": {}})
                prev = self.prev.get(key)
                if prev and now > prev[0]:
                    span = (now - prev[0]) * 1e9
                    for g in GROUPS:
                        e[g] += max(0, groups.get(g, 0) - prev[1].get(g, 0)) / span * 100
                e["vram_mb"] += c["mem"].get("vram", 0) / (1 << 20)
                e["gtt_mb"] += c["mem"].get("gtt", 0) / (1 << 20)
                e["procs"][pid] = {"pid": pid, "comm": m["comm"], "container": m["container"], "cmd": m["cmd"]}
        self.prev = cur
        for e in classes.values():
            for g in GROUPS:
                e[g] = round(min(100.0, e[g]), 1)
            e["vram_mb"], e["gtt_mb"] = round(e["vram_mb"], 1), round(e["gtt_mb"], 1)
            e["procs"] = list(e["procs"].values())
        return {"ts": now, "gpu_busy": self.gpu.read(), "classes": classes}


def load_of(e: dict) -> float:
    """One number for 'how hard this class works the GPU right now'."""
    return max(e.get("gfx", 0), e.get("compute", 0), e.get("media", 0))


# ------------------------------------------------------------------------------------------------------ the history

class Recorder:
    """SQLite history: load per class every `flush_s`, sessions and overlaps as events."""

    def __init__(self, path, active_pct=3.0, flush_s=5.0, quiet_s=15.0, idle_row_s=60.0, retention_days=30):
        self.active_pct, self.flush_s, self.quiet_s, self.idle_row_s = active_pct, flush_s, quiet_s, idle_row_s
        self.retention_s = retention_days * 86400
        self.lock = threading.Lock()
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS samples(ts INTEGER, dt REAL, class TEXT, gfx REAL, compute REAL, media REAL, dma REAL,
                                               vram_mb REAL, gtt_mb REAL, active INTEGER);
            CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
            CREATE INDEX IF NOT EXISTS samples_class_ts ON samples(class, ts);
            CREATE TABLE IF NOT EXISTS events(ts REAL, kind TEXT, class TEXT, detail TEXT);
            CREATE INDEX IF NOT EXISTS events_ts ON events(ts);""")
        self._acc, self._win_start, self._last_t = {}, None, None
        self._idle_row = {}                              # class -> when its last idle row was written
        self.sessions = {}                               # class -> {start, last, peak, sum, n}
        self.overlap = None                              # {start, classes, gpu_max}

    # -- events
    def event(self, ts, kind, cls, **detail):
        log.info("%s %s %s", kind, cls or "", json.dumps(detail) if detail else "")
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (ts, kind, cls, json.dumps(detail)))
            self.db.commit()

    def add(self, s: dict):
        now = s["ts"]
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if self._win_start is None:
            self._win_start = now
        rows = dict(s["classes"])
        rows["__gpu__"] = {"gfx": float(s["gpu_busy"] or 0), "compute": 0.0, "media": 0.0, "dma": 0.0, "vram_mb": 0.0, "gtt_mb": 0.0}
        for cls, e in rows.items():
            a = self._acc.setdefault(cls, {"t": 0.0, "gfx": 0.0, "compute": 0.0, "media": 0.0, "dma": 0.0, "vram_mb": 0.0, "gtt_mb": 0.0})
            a["t"] += dt
            for g in GROUPS:
                a[g] += e[g] * dt
            a["vram_mb"], a["gtt_mb"] = max(a["vram_mb"], e["vram_mb"]), max(a["gtt_mb"], e["gtt_mb"])
        self._track(s)
        if now - self._win_start >= self.flush_s:
            self._flush(now)

    def _flush(self, now):
        span = max(now - self._win_start, 0.001)
        with self.lock:
            for cls, a in self._acc.items():
                avg = {g: a[g] / span for g in GROUPS}              # time a class was absent counts as no load
                active = int(cls != "__gpu__" and max(avg["gfx"], avg["compute"], avg["media"]) >= self.active_pct)
                if cls == "__gpu__" or active or now - self._idle_row.get(cls, 0) >= self.idle_row_s:
                    self._idle_row[cls] = now
                    self.db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)",
                                    (int(now), span, cls, *(round(avg[g], 1) for g in GROUPS),
                                     round(a["vram_mb"], 1), round(a["gtt_mb"], 1), active))
            self.db.execute("DELETE FROM samples WHERE ts < ?", (int(now - self.retention_s),))
            self.db.execute("DELETE FROM events WHERE ts < ?", (now - self.retention_s,))
            self.db.commit()
        self._acc, self._win_start = {}, now

    def _track(self, s):
        now = s["ts"]
        for cls, e in s["classes"].items():
            ld = load_of(e)
            if ld >= self.active_pct:
                ss = self.sessions.get(cls)
                cmds = [p.get("cmd", "")[:160] for p in (e.get("procs") or []) if p.get("cmd")]
                if ss is None:
                    ss = self.sessions[cls] = {"start": now, "last": now, "peak": 0.0, "sum": 0.0, "n": 0, "cmds": []}
                    self.event(now, "start", cls, load=ld, cmd=cmds[0] if cmds else "")      # what was it doing? (playback or background work)
                for c in cmds:
                    if c not in ss["cmds"] and len(ss["cmds"]) < 3:
                        ss["cmds"].append(c)
                ss["last"] = now
                ss["peak"] = max(ss["peak"], ld)
                ss["sum"] += ld
                ss["n"] += 1
        for cls in [c for c, ss in self.sessions.items() if now - ss["last"] >= self.quiet_s]:
            ss = self.sessions.pop(cls)
            self.event(now, "stop", cls, seconds=round(ss["last"] - ss["start"]), peak=round(ss["peak"], 1),
                       avg=round(ss["sum"] / max(ss["n"], 1), 1), cmds=ss["cmds"])
        running = set(self.sessions)
        gpu = s["gpu_busy"] or 0
        if len(running) >= 2:
            if self.overlap is None:
                self.overlap = {"start": now, "classes": set(running), "gpu_max": gpu}
                self.event(now, "overlap_start", "", classes=sorted(running), gpu_busy=gpu)
            self.overlap["classes"] |= running
            self.overlap["gpu_max"] = max(self.overlap["gpu_max"], gpu)
        elif self.overlap is not None:
            o, self.overlap = self.overlap, None
            self.event(now, "overlap_end", "", classes=sorted(o["classes"]), seconds=round(now - o["start"]), gpu_busy_max=o["gpu_max"])

    # -- queries
    def _q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def history(self, minutes: int, now=None):
        now = time.time() if now is None else now
        minutes = max(1, min(int(minutes), 60 * 24 * 30))
        bucket = max(5, int(minutes * 60 / 120) // 5 * 5)
        t0 = int((now - minutes * 60) // bucket * bucket)
        n = int((now - t0) // bucket) + 1
        series = {}
        for cls, b, load in self._q(
                "SELECT class, CAST(ts/? AS INTEGER)*?, SUM(MAX(gfx,compute,media)*dt)/? FROM samples WHERE ts>=? GROUP BY 1,2",
                (bucket, bucket, bucket, t0)):
            series.setdefault(cls, [0.0] * n)
            i = int((b - t0) // bucket)
            if 0 <= i < n:
                series[cls][i] = round(min(100.0, load), 1)
        gpu = series.pop("__gpu__", [0.0] * n)
        return {"t0": t0, "bucket_s": bucket, "gpu": gpu, "classes": series}

    def events(self, limit=100):
        return [{"ts": ts, "kind": k, "class": c, "detail": json.loads(d or "{}")}
                for ts, k, c, d in self._q("SELECT ts, kind, class, detail FROM events ORDER BY ts DESC LIMIT ?", (int(limit),))]

    def stats(self, hours: int, now=None):
        now = time.time() if now is None else now
        since = int(now - hours * 3600)
        classes = [{"class": c, "active_s": round(a or 0), "avg_load": round(avg or 0, 1), "peak_load": round(pk or 0, 1),
                    "vram_mb": round(vr or 0), "gtt_mb": round(gt or 0)}
                   for c, a, avg, pk, vr, gt in self._q(
            """SELECT class, SUM(CASE WHEN active THEN dt END), AVG(CASE WHEN active THEN MAX(gfx,compute,media) END),
                      MAX(MAX(gfx,compute,media)), MAX(vram_mb), MAX(gtt_mb)
               FROM samples WHERE ts>=? AND class!='__gpu__' GROUP BY class ORDER BY 2 DESC""", (since,))]
        pairs = [{"a": a, "b": b, "seconds": round(s or 0)} for a, b, s in self._q(
            """SELECT x.class, y.class, SUM(x.dt) FROM samples x JOIN samples y ON x.ts=y.ts AND x.class<y.class
               WHERE x.ts>=? AND x.active AND y.active AND x.class!='__gpu__' AND y.class!='__gpu__'
               GROUP BY 1,2 ORDER BY 3 DESC""", (since,))]
        unexplained = self._q(
            """SELECT SUM(g.dt) FROM samples g WHERE g.ts>=? AND g.class='__gpu__' AND g.gfx>=15 AND NOT EXISTS
               (SELECT 1 FROM samples c WHERE c.ts=g.ts AND c.class!='__gpu__' AND c.active)""", (since,))[0][0] or 0
        busy = self._q("SELECT SUM(dt), SUM(gfx*dt) FROM samples WHERE ts>=? AND class='__gpu__'", (since,))[0]
        return {"hours": hours, "classes": classes, "overlaps": pairs, "unattributed_busy_s": round(unexplained),
                "observed_s": round(busy[0] or 0), "gpu_avg_busy": round((busy[1] or 0) / busy[0], 1) if busy[0] else 0}


# --------------------------------------------------------------------------------------------------------- the server

class State:
    def __init__(self, collector, recorder, interval):
        self.collector, self.recorder, self.interval = collector, recorder, interval
        self.latest = {"ts": 0, "gpu_busy": None, "classes": {}}
        self.started = time.time()

    def loop(self):
        while True:
            try:
                s = self.collector.sample()
                self.latest = s
                self.recorder.add(s)
            except Exception:                            # noqa: BLE001
                log.exception("sample failed")
            time.sleep(self.interval)


def metrics(st: State) -> str:
    """Prometheus text format of the latest sample (gfx/compute/media/dma are percent of the sampling interval)."""
    s, out = st.latest, []
    lab = lambda cls: cls.replace("\\", "\\\\").replace('"', '\\"')                   # noqa: E731
    out.append("# TYPE gpuwatch_gpu_busy_percent gauge")
    out.append(f"gpuwatch_gpu_busy_percent {s['gpu_busy'] if s['gpu_busy'] is not None else 'NaN'}")
    for name in GROUPS:
        out.append(f"# TYPE gpuwatch_class_{name}_percent gauge")
        for cls, e in s["classes"].items():
            out.append(f'gpuwatch_class_{name}_percent{{class="{lab(cls)}"}} {e[name]}')
    for key in ("vram", "gtt"):
        out.append(f"# TYPE gpuwatch_class_{key}_bytes gauge")
        for cls, e in s["classes"].items():
            out.append(f'gpuwatch_class_{key}_bytes{{class="{lab(cls)}"}} {int(e[key + "_mb"] * (1 << 20))}')
    return "\n".join(out) + "\n"


ARBITER_PATHS = {("GET", "/arbiter/"): "/", ("GET", "/arbiter/api/state"): "/api/state", ("GET", "/arbiter/api/events"): "/api/events",
                 ("GET", "/arbiter/api/config"): "/api/config", ("POST", "/arbiter/api/config"): "/api/config"}


def make_handler(st: State, page: bytes, arbiter_url: str = ""):
    """`arbiter_url` (optional): a gpuarbiter running next to this service. Its page and API are passed through under /arbiter/ so the page
    can show it as a tab; only those few fixed paths are forwarded, to that one address."""
    arbiter_url = arbiter_url.rstrip("/")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _proxy(self, method, path):
            body = None
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 262144:
                    return self._send({"error": "request too large"}, code=413)
                body = self.rfile.read(n)                  # consume the body before answering, whatever the answer is
            upstream = ARBITER_PATHS.get((method, path))
            if not arbiter_url or upstream is None:
                return self._send({"error": "not found"}, code=404)
            req = urllib.request.Request(arbiter_url + upstream, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json"),
                                                  "X-Requested-With": self.headers.get("X-Requested-With", "")})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    return self._send(r.read(), r.headers.get("Content-Type", "application/json"), r.status)
            except urllib.error.HTTPError as exc:
                return self._send(exc.read(), exc.headers.get("Content-Type", "application/json"), exc.code)
            except Exception:                            # noqa: BLE001
                return self._send({"error": "the arbiter does not answer"}, code=502)

        def do_POST(self):                               # noqa: N802
            self._proxy("POST", urllib.parse.urlsplit(self.path).path)

        def _send(self, body, ctype="application/json", code=200):
            raw = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):                                # noqa: N802
            u = urllib.parse.urlsplit(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            num = lambda k, d: int(q[k]) if q.get(k, "").isdigit() else d          # noqa: E731
            if u.path == "/arbiter":
                self.send_response(302)
                self.send_header("Location", "/arbiter/")
                self.end_headers()
                return
            if u.path.startswith("/arbiter/"):
                return self._proxy("GET", u.path)
            if u.path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if u.path == "/api/now":
                return self._send({**st.latest, "uptime_s": round(time.time() - st.started)})
            if u.path == "/api/history":
                return self._send(st.recorder.history(num("minutes", 60)))
            if u.path == "/api/events":
                return self._send(st.recorder.events(num("limit", 100)))
            if u.path == "/api/stats":
                return self._send(st.recorder.stats(num("hours", 24)))
            if u.path == "/metrics":
                return self._send(metrics(st), "text/plain; version=0.0.4")
            self._send({"error": "not found"}, code=404)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9610)
    ap.add_argument("--db", default="/var/lib/gpuwatch/gpuwatch.db")
    ap.add_argument("--rules", default="", help="JSON file with the class rules (replaces the built-in ones)")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--active-pct", type=float, default=3.0, help="load from which a class counts as using the GPU")
    ap.add_argument("--quiet-s", type=float, default=15.0, help="seconds of quiet that end a session")
    ap.add_argument("--retention-days", type=int, default=30)
    ap.add_argument("--arbiter-url", default="", help="a gpuarbiter next to this service (e.g. http://127.0.0.1:9620): shown as the Priorities tab")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rules = json.load(open(a.rules)) if a.rules else DEFAULT_RULES
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpuwatch.html"), "rb").read()
    st = State(Collector(rules), Recorder(a.db, active_pct=a.active_pct, quiet_s=a.quiet_s, retention_days=a.retention_days), a.interval)
    threading.Thread(target=st.loop, daemon=True).start()
    log.info("gpuwatch on %s:%s, database %s%s", a.bind, a.port, a.db, f", arbiter {a.arbiter_url}" if a.arbiter_url else "")
    ThreadingHTTPServer((a.bind, a.port), make_handler(st, page, a.arbiter_url)).serve_forever()


if __name__ == "__main__":
    main()

# /etc/systemd/system/gpuwatch.service   (root: it has to read /proc/<pid>/fdinfo of other containers' processes)
# [Unit]
# Description=gpuwatch: who uses the GPU (read-only)
# After=network-online.target
# [Service]
# ExecStart=/usr/bin/python3 /opt/gpuwatch/gpuwatch.py --bind <host-ip> --port 9610
# StateDirectory=gpuwatch
# ProtectSystem=strict
# ProtectHome=yes
# PrivateTmp=yes
# NoNewPrivileges=yes
# Restart=on-failure
# [Install]
# WantedBy=multi-user.target
