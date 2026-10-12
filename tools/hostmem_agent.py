#!/usr/bin/env python3
"""Tiny read-only endpoint that reports the free RAM and the GPU load of the machine it runs on.

Run it on the host whose memory the models use (an iGPU takes its model memory from the host's RAM, so the
container that runs the LLM cannot see the real number, and cannot see the GPU's load either):

    python3 hostmem_agent.py --bind <host-ip> --port 9600

    curl http://<host-ip>:9600/
    {"mem_total_gb": 85.7, "mem_available_gb": 61.2, "swap_used_gb": 0.0,
     "gpu_busy_pct": 0, "gpu_busy_max_10s": 3, "gpu_busy_avg_10s": 0.6}

Set `guard.host_ram_url` in bridge.json to that URL for the resource guard, and give the same URL to the idle gate
(`--gpu-url`) so it only lets a low-priority client run while the GPU is really idle. The GPU numbers are the amdgpu
busy percentage (`/sys/class/drm/card*/device/gpu_busy_percent`, the counter amdgpu_top shows), sampled once a second; they
are left out on a machine without it. It reads /proc/meminfo and that file and nothing else, takes no input and has no
authentication, so bind it to a LAN address only. Python standard library only. A systemd unit is in the comment at the bottom.
"""
import argparse
import collections
import glob
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

WINDOW_S = 10


class GpuSampler:
    """Samples the amdgpu busy percentage once a second and keeps the last WINDOW_S seconds."""

    def __init__(self, paths=None, interval=1.0):
        self.paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent") if paths is None else paths
        self.interval = interval
        self.samples = collections.deque()                 # (time, percent)
        self.lock = threading.Lock()

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

    def sample(self, now=None):
        v = self.read()
        if v is None:
            return
        now = time.time() if now is None else now
        with self.lock:
            self.samples.append((now, v))
            while self.samples and self.samples[0][0] < now - WINDOW_S:
                self.samples.popleft()

    def snapshot(self, now=None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            vals = [v for t, v in self.samples if t >= now - WINDOW_S]
        if not vals:
            return {}
        return {"gpu_busy_pct": vals[-1], "gpu_busy_max_10s": max(vals), "gpu_busy_avg_10s": round(sum(vals) / len(vals), 1)}

    def start(self):
        def loop():
            while True:
                self.sample()
                time.sleep(self.interval)
        threading.Thread(target=loop, daemon=True).start()


GPU = GpuSampler([])                                       # replaced in main(); no paths = no GPU numbers


def meminfo() -> dict:
    kb = {}
    with open("/proc/meminfo") as f:
        for line in f:
            name, _, rest = line.partition(":")
            kb[name] = int(rest.split()[0])
    gb = lambda v: round(v / 1048576, 2)              # noqa: E731
    return {"mem_total_gb": gb(kb["MemTotal"]), "mem_available_gb": gb(kb["MemAvailable"]),
            "swap_used_gb": gb(kb.get("SwapTotal", 0) - kb.get("SwapFree", 0))}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):                                  # noqa: N802
        body = json.dumps({**meminfo(), **GPU.snapshot()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):                      # quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9600)
    args = ap.parse_args()
    GPU = GpuSampler()
    GPU.start()
    HTTPServer((args.bind, args.port), Handler).serve_forever()

# /etc/systemd/system/hostmem-agent.service
# [Unit]
# Description=Read-only host RAM endpoint
# After=network-online.target
# [Service]
# ExecStart=/usr/bin/python3 /opt/hostmem-agent/hostmem_agent.py --bind <host-ip> --port 9600
# DynamicUser=yes
# ProtectSystem=strict
# ProtectHome=yes
# Restart=on-failure
# [Install]
# WantedBy=multi-user.target
