"""The idle gate holds a background client's requests while someone else uses the LLM (fake upstream, no LLM needed)."""
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import idle_gate  # noqa: E402


class Upstream:
    """Counts chat requests like Lemonade's /api/v1/stats does."""
    def __init__(self, delay=0.0):
        self.count = 0
        self.delay = delay
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):                              # noqa: N802
                if self.path.endswith("/stats"):
                    self._json({"request_count_total": outer.count})
                elif self.path.endswith("/health"):
                    self._json({"all_models_loaded": []})
                else:
                    self._json({"data": []})

            def do_POST(self):                             # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                time.sleep(outer.delay)
                outer.count += 1                           # counted when it finishes, like the real server
                self._json({"choices": [{"message": {"content": "OK"}}]})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


class GpuAgent:
    """Stands in for tools/hostmem_agent.py: reports whatever `busy` is set to."""
    def __init__(self):
        self.busy = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):                              # noqa: N802
                raw = json.dumps({"gpu_busy_pct": outer.busy, "gpu_busy_max_10s": outer.busy}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def start_gate(upstream_port, idle_s, hold_s, gpu_url="", gpu_grace_s=0.5):
    class A:
        pass
    a = A()
    a.upstream, a.idle_s, a.hold_max_s, a.poll_s, a.timeout_s, a.busy_url = f"http://127.0.0.1:{upstream_port}", idle_s, hold_s, 0.2, 30, ""
    a.gpu_url, a.gpu_busy_pct, a.gpu_grace_s = gpu_url, 15, gpu_grace_s
    st = idle_gate.State(a)
    st.last_other = time.time() - 1000                    # start open
    srv = ThreadingHTTPServer(("127.0.0.1", 0), idle_gate.make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    def loop():
        while True:
            st.poll()
            time.sleep(a.poll_s)
    threading.Thread(target=loop, daemon=True).start()
    return st, srv.server_address[1]


def post(port, path="/api/v1/chat/completions"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b'{"model":"m"}', headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


class GateTest(unittest.TestCase):
    def test_own_traffic_keeps_the_lane_open_and_other_traffic_closes_it(self):
        up = Upstream()
        st, port = start_gate(up.port, idle_s=2.0, hold_s=0.3)
        time.sleep(0.5)
        for _ in range(3):
            self.assertEqual(post(port), 200)
            time.sleep(0.5)
        self.assertEqual(st.forwarded, 3)
        self.assertTrue(st.open(), "the gate's own requests must not count as someone else")
        post(up.port)                                      # another client talks to the LLM directly
        time.sleep(0.8)
        self.assertFalse(st.open())
        self.assertEqual(post(port), 503)                  # held for hold_max_s, then told to retry
        time.sleep(2.0)                                    # the quiet period passes
        self.assertEqual(post(port), 200)

    def test_other_client_is_noticed_while_the_gate_is_busy_itself(self):
        up = Upstream(delay=0.4)
        st, port = start_gate(up.port, idle_s=5.0, hold_s=0.3)
        stop = threading.Event()

        def background():                                  # a client that always has a request in flight
            while not stop.is_set():
                post(port)
        t = threading.Thread(target=background, daemon=True)
        t.start()
        time.sleep(2.5)
        self.assertTrue(st.open(), "continuous own traffic alone must not close the lane")
        post(up.port)                                      # someone else sends one request
        time.sleep(1.5)
        stop.set()
        t.join(timeout=5)
        self.assertFalse(st.open(), "another client's request must close the lane even while the gate is busy")

    def test_a_busy_gpu_closes_the_lane_and_an_idle_one_opens_it(self):
        up, gpu = Upstream(), GpuAgent()
        st, port = start_gate(up.port, idle_s=1.0, hold_s=0.3, gpu_url=gpu.url)
        time.sleep(0.6)
        self.assertEqual(post(port), 200)                  # GPU idle: lane open
        gpu.busy = 80                                      # something else (not through the gate) loads the GPU
        time.sleep(1.5)                                    # past the gate's own grace period
        self.assertFalse(st.open())
        self.assertIn("GPU is busy", st.reason)
        self.assertEqual(post(port), 503)
        gpu.busy = 0
        time.sleep(2.0)                                    # quiet for idle_s
        self.assertEqual(post(port), 200)

    def test_the_gates_own_gpu_load_does_not_close_the_lane(self):
        up, gpu = Upstream(delay=0.3), GpuAgent()
        st, port = start_gate(up.port, idle_s=1.0, hold_s=0.3, gpu_url=gpu.url, gpu_grace_s=1.5)
        time.sleep(0.6)
        gpu.busy = 90                                      # the load comes from the gate's own request
        for _ in range(4):
            self.assertEqual(post(port), 200)
            time.sleep(0.2)
        gpu.busy = 0
        time.sleep(1.0)
        self.assertTrue(st.open(), "GPU load right after our own request is ours, not someone else's")

    def test_an_unreachable_gpu_agent_does_not_block(self):
        up = Upstream()
        st, port = start_gate(up.port, idle_s=1.0, hold_s=0.3, gpu_url="http://127.0.0.1:9/")
        time.sleep(0.6)
        self.assertEqual(post(port), 200)

    def test_other_endpoints_pass_straight_through(self):
        up = Upstream()
        st, port = start_gate(up.port, idle_s=60, hold_s=1.0)
        st.last_other = time.time()                        # closed
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/models", timeout=10) as r:
            self.assertEqual(json.loads(r.read()), {"data": []})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/gate/status", timeout=10) as r:
            self.assertFalse(json.loads(r.read())["open"])


if __name__ == "__main__":
    unittest.main()
