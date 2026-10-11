#!/usr/bin/env python3
"""Idle gate: a low-priority lane to an OpenAI-compatible LLM server (tested with Lemonade).

Point a background client (for example Lingarr) at this gate instead of the LLM server. Its generation requests
(`POST .../chat/completions`, `.../completions`) are held until nobody else has used the LLM for `--idle-s` seconds,
then forwarded. Everything else (model lists, health) passes straight through. Other clients keep talking to the LLM
server directly and never wait for the gate.

"Nobody else" is judged from three signals, polled every `--poll-s`:
  * the server's request counter went up by more than the gate itself forwarded (someone else sent a request),
  * a model reports `is_busy` / `is_streaming` while the gate has nothing in flight,
  * (optional) `--busy-url` answers {"active": n} with n > 0, for example the bridge's `/busy`: a subtitle job that is
    still transcribing will need the LLM soon, so the lane stays closed.

A request that waits longer than `--hold-max-s` gets `503` with `Retry-After`; the client retries later.
`GET /gate/status` shows what the gate sees. Python standard library only; no authentication: bind it to a LAN address.

    python3 idle_gate.py --listen <ip>:13306 --upstream http://127.0.0.1:13305 --busy-url http://127.0.0.1:9500/busy
"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATED = ("/chat/completions", "/completions")


class State:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.inflight = 0
        self.waiting = 0
        self.forwarded = 0                       # requests the gate sent upstream
        self.completed = 0                       # ... of which are finished (the server counts a request when it finishes)
        self._suspect = None                     # since when the server's counter is ahead of what the gate explains
        self.last_other = 0.0                    # last time someone else was seen using the LLM
        self.reason = "starting"
        self._base = None                        # request counter at the start, minus what the gate forwarded

    def get_json(self, url, timeout=3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:                        # noqa: BLE001
            return None

    def poll(self):
        a, now = self.a, time.time()
        stats = self.get_json(a.upstream + "/api/v1/stats")
        health = self.get_json(a.upstream + "/api/v1/health")
        with self.lock:
            own, inflight = self.completed, self.inflight
        if stats and "request_count_total" in stats:
            total = stats["request_count_total"]
            if self._base is None:
                self._base = total - own
            expected = self._base + own                      # what the counter reads if only the gate sent requests
            if total > expected:
                # Maybe just the instant between the server counting a request and the gate noting it finished: only
                # a difference that is still there at the next poll is another client.
                if self._suspect is None:
                    self._suspect = now
                elif now - self._suspect >= self.a.poll_s * 0.9:
                    self.last_other, self.reason = now, "another client sent a request"
                    self._base, self._suspect = total - own, None
            else:
                self._suspect = None
                if total < expected and not inflight:        # an own request that was never counted (error): resync
                    self._base = total - own
        if health and not inflight:
            if any(m.get("is_busy") or m.get("is_streaming") for m in health.get("all_models_loaded", [])):
                self.last_other, self.reason = now, "the LLM is busy"
        if a.busy_url:
            b = self.get_json(a.busy_url)
            if b and (b.get("active") or b.get("waiting")):
                self.last_other, self.reason = now, "a subtitle job is running"
        if stats is None and health is None:
            self.last_other, self.reason = now, "the LLM server does not answer"

    def idle_for(self):
        return time.time() - self.last_other

    def open(self):
        return self.idle_for() >= self.a.idle_s


def make_handler(st: State):
    a = st.a
    up = urllib.parse.urlsplit(a.upstream)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _forward(self, body=None):
            gated = self.command == "POST" and self.path.split("?")[0].endswith(GATED)
            if gated:
                deadline = time.time() + a.hold_max_s
                with st.lock:
                    st.waiting += 1
                try:
                    while not st.open():
                        if time.time() > deadline:
                            return self._send(503, {"error": {"message": f"LLM busy with other work ({st.reason}); retry later"}},
                                              extra={"Retry-After": "60"})
                        time.sleep(1)
                finally:
                    with st.lock:
                        st.waiting -= 1
                with st.lock:
                    st.inflight += 1
                    st.forwarded += 1
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=a.timeout_s)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "content-length")}
                if body is not None:
                    headers["Content-Length"] = str(len(body))
                conn.request(self.command, self.path, body=body, headers=headers)
                r = conn.getresponse()
                self.send_response(r.status)
                chunked = r.getheader("Transfer-Encoding", "").lower() == "chunked"
                for k, v in r.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(k, v)
                if chunked or r.getheader("Content-Length") is None:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = r.read(8192) if not chunked else r.read1(8192)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    data = r.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                conn.close()
            except Exception as exc:             # noqa: BLE001
                try:
                    self._send(502, {"error": {"message": f"upstream error: {exc}"}})
                except Exception:                # noqa: BLE001
                    pass
            finally:
                if gated:
                    with st.lock:
                        st.inflight -= 1
                        st.completed += 1

        def do_GET(self):                        # noqa: N802
            if self.path.split("?")[0] == "/gate/status":
                with st.lock:
                    return self._send(200, {"open": st.open(), "idle_for_s": round(st.idle_for()), "idle_needed_s": a.idle_s,
                                            "last_reason": st.reason, "waiting": st.waiting, "in_flight": st.inflight,
                                            "forwarded": st.forwarded})
            self._forward()

        def do_POST(self):                       # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            self._forward(self.rfile.read(n) if n else None)

        do_PUT = do_DELETE = do_POST             # noqa: N815

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", default="127.0.0.1:13306", help="ip:port")
    ap.add_argument("--upstream", default="http://127.0.0.1:13305")
    ap.add_argument("--idle-s", type=float, default=600, help="seconds of quiet before the lane opens")
    ap.add_argument("--hold-max-s", type=float, default=85, help="longest a request waits before it gets a 503")
    ap.add_argument("--poll-s", type=float, default=2)
    ap.add_argument("--timeout-s", type=float, default=900, help="upstream timeout of a forwarded request")
    ap.add_argument("--busy-url", default="", help="optional JSON endpoint answering {\"active\": n}")
    a = ap.parse_args()
    st = State(a)
    st.last_other = time.time()                  # start closed: wait one idle period before the first request

    def loop():
        while True:
            try:
                st.poll()
            except Exception:                    # noqa: BLE001
                pass
            time.sleep(a.poll_s)

    threading.Thread(target=loop, daemon=True).start()
    host, _, port = a.listen.rpartition(":")
    ThreadingHTTPServer((host or "127.0.0.1", int(port)), make_handler(st)).serve_forever()


if __name__ == "__main__":
    main()

# /etc/systemd/system/idle-gate.service
# [Unit]
# Description=Idle gate: low-priority lane to the LLM server
# After=network-online.target
# [Service]
# ExecStart=/usr/bin/python3 /opt/idle-gate/idle_gate.py --listen <lan-ip>:13306 --upstream http://127.0.0.1:13305 --idle-s 600 --hold-max-s 3300
# DynamicUser=yes
# ProtectSystem=strict
# ProtectHome=yes
# Restart=on-failure
# [Install]
# WantedBy=multi-user.target
