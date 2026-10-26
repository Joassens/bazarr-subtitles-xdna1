#!/usr/bin/env python3
"""gpuarbiter: lower-priority GPU work steps aside for higher-priority work.

It watches who uses the GPU (tools/gpuwatch.py) and how much RAM is free (tools/hostmem_agent.py), and PAUSES a background consumer that can be
paused (today: a Tdarr node) while a class with a higher priority uses the same GPU engine, or while the host is short of RAM. It resumes it by
itself after a quiet period. Only the node it paused itself is resumed; a node somebody paused by hand is left alone, and a manual resume is
respected for a while. A running job is never killed: pausing stops *new* jobs from starting.

Priorities live in a JSON config (edit it on the page). A class that has a `controller` yields to every class with a LOWER priority number that
shares at least one engine with it (engines: gfx, compute, media, dma) and is busy on that engine. Everything it decides is logged to SQLite and
shown on the page. `dry_run: true` only logs what it would do. No authentication: bind it to a LAN address. Python standard library only.

    python3 gpuarbiter.py --config /etc/gpuarbiter/config.json --bind <ip> --port 9620 --db /var/lib/gpuarbiter/events.db
"""
import argparse
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("gpuarbiter")
ENGINES = ("gfx", "compute", "media", "dma")

DEFAULT_CONFIG = {
    "dry_run": True,
    "poll_s": 5,
    "sources": {"gpuwatch": "http://127.0.0.1:9610/api/now", "hostmem": "http://127.0.0.1:9600/"},
    "controllers": {"tdarr": {"type": "tdarr", "url": "http://127.0.0.1:8265"}},
    "classes": [
        {"name": "Plex", "priority": 1, "engines": ["media"]},
        {"name": "Jellyfin", "priority": 1, "engines": ["media"]},
        {"name": "LLM (llama-server)", "priority": 2, "engines": ["gfx", "compute"]},
        {"name": "Whisper (Lemonade)", "priority": 2, "engines": ["compute"]},
        {"name": "Tdarr", "priority": 4, "engines": ["media"], "controller": "tdarr"},
    ],
    "active_pct": 3.0,          # load (percent of the sampling interval) from which a class counts as busy on an engine
    "hold_s": 10,               # a reason must last this long before the consumer is paused (ignores blips)
    "release_s": 180,           # ... and must be gone this long before it is resumed
    "manual_override_s": 600,   # after somebody resumes the consumer by hand, do not pause it again for this long
    "ram_guard": {"enabled": True, "pause_below_gb": 3.0, "resume_above_gb": 5.0},
}


def validate(cfg: dict) -> list:
    """Problems with a config, as readable strings (empty list = fine)."""
    errs = []
    if not isinstance(cfg.get("classes"), list) or not cfg["classes"]:
        return ["classes: a non-empty list is required"]
    names = set()
    for i, c in enumerate(cfg["classes"]):
        where = f"classes[{i}]"
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            errs.append(f"{where}: name is required")
            continue
        if c["name"] in names:
            errs.append(f"{where}: duplicate class {c['name']!r}")
        names.add(c["name"])
        if not isinstance(c.get("priority"), (int, float)) or isinstance(c.get("priority"), bool):
            errs.append(f"{where} ({c['name']}): priority must be a number (1 = most important)")
        eng = c.get("engines", [])
        if not isinstance(eng, list) or any(e not in ENGINES for e in eng):
            errs.append(f"{where} ({c['name']}): engines must be a list of {', '.join(ENGINES)}")
        if c.get("controller") and c["controller"] not in cfg.get("controllers", {}):
            errs.append(f"{where} ({c['name']}): unknown controller {c['controller']!r}")
    for k, lo in (("poll_s", 1), ("hold_s", 0), ("release_s", 0), ("manual_override_s", 0), ("active_pct", 0)):
        if not isinstance(cfg.get(k), (int, float)) or isinstance(cfg.get(k), bool) or cfg[k] < lo:
            errs.append(f"{k}: a number of at least {lo} is required")
    rg = cfg.get("ram_guard") or {}
    if rg.get("enabled") and not (isinstance(rg.get("pause_below_gb"), (int, float)) and isinstance(rg.get("resume_above_gb"), (int, float))
                                  and rg["resume_above_gb"] > rg["pause_below_gb"]):
        errs.append("ram_guard: resume_above_gb must be a number above pause_below_gb")
    return errs


def merged(user: dict) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def fetch_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


class TdarrControl:
    """Pause/resume the nodes of a Tdarr server (pausing only stops new jobs from starting). `nodes` (names or ids) limits it to the nodes that share the
    GPU engine with the other consumers; without it every node is controlled."""

    def __init__(self, url, nodes=None, passive_nodes=None):
        self.url = url.rstrip("/") + "/api/v2"
        self.nodes = [str(n) for n in nodes] if nodes else None
        self.passive = [str(n) for n in passive_nodes] if passive_nodes else []

    def _selected(self):
        nodes = self._call("get-nodes") or {}
        if self.nodes is None:
            return nodes
        return {nid: n for nid, n in nodes.items() if nid in self.nodes or n.get("nodeName") in self.nodes}

    def _passive_nodes(self):
        nodes = self._call("get-nodes") or {}
        return {nid: n for nid, n in nodes.items() if nid in self.passive or n.get("nodeName") in self.passive}

    def _call(self, path, payload=None):
        req = urllib.request.Request(f"{self.url}/{path}", data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode()
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    def paused(self):
        """True when all nodes are paused, False when at least one runs."""
        nodes = self._selected()
        return all(bool(n.get("nodePaused")) for n in nodes.values()) if nodes else False

    def set_paused(self, value: bool):
        for nid in self._selected():
            self._call("update-node", {"data": {"nodeID": nid, "nodeUpdates": {"nodePaused": bool(value)}}})
        # passive (failover) nodes do the opposite: they work while the active nodes are held back, and wait while the active nodes work
        for nid in self._passive_nodes():
            self._call("update-node", {"data": {"nodeID": nid, "nodeUpdates": {"nodePaused": not bool(value)}}})


class Events:
    def __init__(self, path, retention_days=30):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.retention_s = retention_days * 86400
        self.db.executescript("CREATE TABLE IF NOT EXISTS events(ts REAL, kind TEXT, consumer TEXT, detail TEXT);"
                              "CREATE INDEX IF NOT EXISTS events_ts ON events(ts);")

    def add(self, ts, kind, consumer, detail):
        log.info("%s %s: %s", kind, consumer, detail)
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (ts, kind, consumer, detail))
            self.db.execute("DELETE FROM events WHERE ts < ?", (ts - self.retention_s,))
            self.db.commit()

    def recent(self, limit=100):
        with self.lock:
            rows = self.db.execute("SELECT ts, kind, consumer, detail FROM events ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"ts": t, "kind": k, "consumer": c, "detail": d} for t, k, c, d in rows]


class Arbiter:
    def __init__(self, cfg, events, controllers=None, clock=time.time, fetch=fetch_json, state_path=None):
        self.cfg, self.events, self.clock, self.fetch = merged(cfg), events, clock, fetch
        self.controllers = controllers if controllers is not None else {}
        self.reasons = {}            # consumer -> {reason: {"since": t, "last": t}}
        self.active = {}             # consumer -> {reason: text} currently holding it paused
        self.we_paused = {}          # consumer -> bool
        self.override_until = {}     # consumer -> time
        self.last = {"ts": 0, "classes": {}, "ram_gb": None, "errors": []}
        self.last_paused = {}        # consumer -> last seen state of the controller
        self.note_at = {}            # consumer -> when the "paused by somebody else" note was last written
        self.lock = threading.Lock()
        self.state_path = state_path  # remembers `we_paused` over restarts and reboots
        self._saved = None
        self._load_state()

    def _load_state(self):
        if not self.state_path:
            return
        try:
            with open(self.state_path) as f:
                self.we_paused = {k: bool(v) for k, v in json.load(f).get("we_paused", {}).items()}
            self._saved = dict(self.we_paused)
            log.info("restored state: we_paused=%s", self.we_paused)
        except (OSError, ValueError):
            pass

    def _save_state(self):
        if not self.state_path or self.we_paused == self._saved:
            return
        try:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"we_paused": self.we_paused}, f)
            os.replace(tmp, self.state_path)
            self._saved = dict(self.we_paused)
        except OSError as exc:
            log.warning("cannot save state: %s", exc)

    def set_config(self, cfg):
        with self.lock:
            self.cfg = merged(cfg)

    # -- what is firing right now
    def _firing(self, consumer, classes, ram):
        c = self.cfg
        by_name = {x["name"]: x for x in c["classes"]}
        me = by_name.get(consumer)
        out = {}
        if not me:
            return out
        for other in c["classes"]:
            if other["name"] == consumer or other["priority"] >= me["priority"]:
                continue
            shared = [e for e in other.get("engines", []) if e in me.get("engines", [])]
            cur = classes.get(other["name"])
            if cur and any(cur.get(e, 0) >= c["active_pct"] for e in shared):
                busy = max(cur.get(e, 0) for e in shared)
                out[f"yield:{other['name']}"] = f"{other['name']} (priority {other['priority']}) is busy on {'/'.join(shared)} ({busy}%)"
        rg = c.get("ram_guard") or {}
        if rg.get("enabled") and ram is not None:
            was = "ram" in self.active.get(consumer, {})
            limit = rg["resume_above_gb"] if was else rg["pause_below_gb"]
            if ram < limit:
                out["ram"] = f"only {ram:.1f} GB RAM free on the host (guard: pause below {rg['pause_below_gb']} GB, resume above {rg['resume_above_gb']} GB)"
        return out

    def step(self):
        with self.lock:
            now = self.clock()
            c = self.cfg
            errors = []
            classes, ram, gpu_ok = {}, None, True
            try:
                data = self.fetch(c["sources"]["gpuwatch"])
                classes = {n: {e: v.get(e, 0) for e in ENGINES} for n, v in data.get("classes", {}).items()}
            except Exception as exc:                        # noqa: BLE001
                gpu_ok = False
                errors.append(f"gpuwatch: {exc}")
            try:
                ram = float(self.fetch(c["sources"]["hostmem"])["mem_available_gb"])
            except Exception as exc:                        # noqa: BLE001
                errors.append(f"hostmem: {exc}")
            self.last = {"ts": now, "classes": classes, "ram_gb": ram, "errors": errors}
            if not gpu_ok:
                return                                       # no view of the GPU: change nothing rather than guess
            for consumer in [x["name"] for x in c["classes"] if x.get("controller")]:
                self._consumer(consumer, classes, ram)
            self._save_state()

    def _consumer(self, consumer, classes, ram):
        c, now = self.cfg, self.clock()
        ctl_name = next(x["controller"] for x in c["classes"] if x["name"] == consumer)
        ctl = self.controllers.get(ctl_name)
        firing = self._firing(consumer, classes, ram)
        mem = self.reasons.setdefault(consumer, {})          # reason -> {since: start of this episode, last: last time it fired, text}
        prev = self.active.get(consumer, {})                 # reasons that held the consumer paused at the previous step
        for key, text in firing.items():
            r = mem.get(key)
            if r is None or now - r["last"] > c["poll_s"] * 3:    # a new episode (a gap in between)
                r = mem[key] = {"since": now, "last": now, "text": text}
            r["last"], r["text"] = now, text
        holding = {}
        for key, r in list(mem.items()):
            if key in firing:
                if now - r["since"] >= c["hold_s"] or key in prev:       # lasted long enough, or already holding
                    holding[key] = r["text"]
            elif key != "ram" and key in prev and now - r["last"] < c["release_s"]:
                holding[key] = r["text"] + " (ended; waiting for it to stay quiet)"
            elif key == "ram" or now - r["last"] >= c["release_s"]:
                del mem[key]                                             # RAM has its own hysteresis; the others are quiet long enough
        want_pause = bool(holding)
        self.active[consumer] = holding
        # what the controller says now
        try:
            paused = ctl.paused() if ctl else None
        except Exception as exc:                             # noqa: BLE001
            self.events.add(now, "error", consumer, f"cannot read the state of {ctl_name}: {exc}")
            return
        before = self.last_paused.get(consumer)
        self.last_paused[consumer] = paused
        if self.we_paused.get(consumer) and paused is False and before is True:
            self.we_paused[consumer] = False
            self.override_until[consumer] = now + c["manual_override_s"]
            self.events.add(now, "external_resume", consumer, "somebody resumed it by hand: leaving it running for a while")
        dry = c.get("dry_run", True)
        try:
            if want_pause and paused is False and now >= self.override_until.get(consumer, 0):
                why = "; ".join(holding.values())
                if dry:
                    if not self.we_paused.get(consumer):
                        self.we_paused[consumer] = True      # pretend, so the matching resume is also only logged once
                        self.events.add(now, "would_pause", consumer, why)
                else:
                    ctl.set_paused(True)
                    self.we_paused[consumer] = True
                    self.last_paused[consumer] = True
                    self.events.add(now, "pause", consumer, why)
            elif not want_pause and self.we_paused.get(consumer):
                if dry:
                    self.events.add(now, "would_resume", consumer, "nothing holds it back any more")
                else:
                    ctl.set_paused(False)
                    self.last_paused[consumer] = False
                    self.events.add(now, "resume", consumer, "nothing holds it back any more")
                self.we_paused[consumer] = False
            elif want_pause and paused is True and not self.we_paused.get(consumer) and now - self.note_at.get(consumer, 0) > 3600:
                self.note_at[consumer] = now
                self.events.add(now, "note", consumer, "already paused by somebody else: leaving it alone")
        except Exception as exc:                             # noqa: BLE001
            self.events.add(now, "error", consumer, f"could not change {ctl_name}: {exc}")

    def state(self):
        with self.lock:
            return {"dry_run": self.cfg.get("dry_run", True), "now": self.clock(), "last_poll": self.last["ts"], "errors": self.last["errors"],
                    "ram_gb": self.last["ram_gb"], "busy_classes": {n: v for n, v in self.last["classes"].items() if max(v.values() or [0]) >= self.cfg["active_pct"]},
                    "consumers": {k: {"paused": self.last_paused.get(k), "paused_by_arbiter": bool(self.we_paused.get(k)),
                                      "holding": self.active.get(k, {}), "manual_override_until": self.override_until.get(k, 0)}
                                  for k in [x["name"] for x in self.cfg["classes"] if x.get("controller")]}}


class ConfigFile:
    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save(self, cfg):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if os.path.exists(self.path):                        # keep the previous version
            hist = os.path.join(os.path.dirname(os.path.abspath(self.path)), "history")
            os.makedirs(hist, exist_ok=True)
            with open(self.path) as f, open(os.path.join(hist, time.strftime("config-%Y%m%d-%H%M%S.json")), "w") as g:
                g.write(f.read())
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, self.path)


def make_handler(arb: Arbiter, cfgfile: ConfigFile, page: bytes):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            raw = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):                                    # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(arb.state())
            if path == "/api/events":
                return self._send(arb.events.recent(100))
            if path == "/api/config":
                return self._send(arb.cfg)
            self._send({"error": "not found"}, code=404)

        def do_POST(self):                                   # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if 0 < n <= 262144 else b""      # always consume the body before answering
            if self.path.split("?")[0] != "/api/config":
                return self._send({"error": "not found"}, code=404)
            if self.headers.get("X-Requested-With") != "arbiter":
                return self._send({"error": "missing X-Requested-With header"}, code=403)
            try:
                new = json.loads(raw.decode())
            except ValueError:
                return self._send({"error": "not valid JSON"}, code=400)
            errs = validate(merged(new))
            if errs:
                return self._send({"error": "invalid config", "problems": errs}, code=422)
            cfgfile.save(new)
            arb.set_config(new)
            arb.events.add(arb.clock(), "config", "-", "configuration saved from the page" + (" (dry run is ON)" if merged(new).get("dry_run") else ""))
            self._send({"ok": True})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="/etc/gpuarbiter/config.json")
    ap.add_argument("--db", default="/var/lib/gpuarbiter/events.db")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9620)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfgfile = ConfigFile(a.config)
    cfg = merged(cfgfile.load())
    problems = validate(cfg)
    if problems:
        raise SystemExit("config problems: " + "; ".join(problems))
    controllers = {k: TdarrControl(v["url"], v.get("nodes"), v.get("passive_nodes")) for k, v in cfg["controllers"].items() if v.get("type") == "tdarr"}
    arb = Arbiter(cfg, Events(a.db), controllers, state_path=os.path.join(os.path.dirname(os.path.abspath(a.db)), "state.json"))
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpuarbiter.html"), "rb").read()

    def loop():
        while True:
            try:
                arb.step()
            except Exception:                                # noqa: BLE001
                log.exception("step failed")
            time.sleep(arb.cfg["poll_s"])

    threading.Thread(target=loop, daemon=True).start()
    log.info("gpuarbiter on %s:%s (dry_run=%s)", a.bind, a.port, cfg.get("dry_run"))
    ThreadingHTTPServer((a.bind, a.port), make_handler(arb, cfgfile, page)).serve_forever()


if __name__ == "__main__":
    main()

# /etc/systemd/system/gpuarbiter.service
# [Unit]
# Description=gpuarbiter: lower-priority GPU work steps aside
# After=network-online.target
# [Service]
# ExecStart=/usr/bin/python3 /opt/gpuarbiter/gpuarbiter.py --config /etc/gpuarbiter/config.json --bind <host-ip> --port 9620
# StateDirectory=gpuarbiter
# ConfigurationDirectory=gpuarbiter
# ProtectSystem=strict
# ReadWritePaths=/etc/gpuarbiter
# ProtectHome=yes
# NoNewPrivileges=yes
# Restart=on-failure
# [Install]
# WantedBy=multi-user.target
