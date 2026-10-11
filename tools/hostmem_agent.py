#!/usr/bin/env python3
"""Tiny read-only endpoint that reports the free RAM of the machine it runs on (for the bridge's resource guard).

Run it on the host whose memory the models use (an iGPU takes its model memory from the host's RAM, so the
container that runs the LLM cannot see the real number):

    python3 hostmem_agent.py --bind <host-ip> --port 9600

    curl http://<host-ip>:9600/
    {"mem_total_gb": 85.7, "mem_available_gb": 61.2, "swap_used_gb": 0.0}

Then set `guard.host_ram_url` in bridge.json to that URL. It reads /proc/meminfo and nothing else, takes no input and
has no authentication, so bind it to a LAN address only. Python standard library only. A systemd unit is in the
comment at the bottom.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


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
        body = json.dumps(meminfo()).encode()
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
