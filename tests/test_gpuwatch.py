"""gpuwatch: per-client GPU counters become per-class load, sessions, overlaps and a history (fake /proc, no GPU needed)."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import gpuwatch  # noqa: E402


def fdinfo(client, gfx=0, dec=0, enc=0, vram_kib=0, driver="amdgpu"):
    return (f"pos:\t0\nflags:\t02100002\ndrm-driver:\t{driver}\ndrm-pdev:\t0000:c5:00.0\ndrm-client-id:\t{client}\n"
            f"drm-memory-vram:\t{vram_kib} KiB\ndrm-memory-gtt: \t2048 KiB\ndrm-memory-cpu: \t0 KiB\n"
            f"drm-engine-gfx:\t{gfx} ns\ndrm-engine-compute:\t0 ns\ndrm-engine-dec:\t{dec} ns\ndrm-engine-enc:\t{enc} ns\n"
            f"drm-engine-capacity-gfx:\t1\n")


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def make_proc(root, pid, comm, cgroup, fds):
    d = os.path.join(root, str(pid))
    put(f"{d}/comm", comm + "\n")
    put(f"{d}/cgroup", cgroup + "\n")
    put(f"{d}/cmdline", comm.replace(" ", "\0") + "\0")
    for fd, text in fds.items():
        put(f"{d}/fd/{fd}", "")
        put(f"{d}/fdinfo/{fd}", text)


class FakeCollector(gpuwatch.Collector):
    def _link(self, path):                                 # fd entries are plain files in the fake /proc
        return "/dev/dri/renderD128" if os.path.isfile(path) else ""


class ParseTest(unittest.TestCase):
    def test_amdgpu_client(self):
        c = gpuwatch.parse_fdinfo(fdinfo(7, gfx=500, dec=20, vram_kib=4096))
        self.assertEqual(c["client"], "7")
        self.assertEqual(c["engines"], {"gfx": 500, "compute": 0, "dec": 20, "enc": 0})   # "capacity" is not an engine time
        self.assertEqual(c["mem"]["vram"], 4096 * 1024)
        self.assertEqual(c["mem"]["gtt"], 2048 * 1024)

    def test_other_fds_are_ignored(self):
        self.assertIsNone(gpuwatch.parse_fdinfo(fdinfo(1, driver="i915")))
        self.assertIsNone(gpuwatch.parse_fdinfo("pos:\t0\nflags:\t0\n"))

    def test_engine_groups(self):
        g = gpuwatch.engine_group
        self.assertEqual([g(x) for x in ("gfx", "compute", "dma", "dec", "enc", "vcn_unified", "jpeg", "weird")],
                         ["gfx", "compute", "dma", "media", "media", "media", "media", "other"])


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.t = [1000.0]
        make_proc(self.root, 11, "llama-server", "0::/lxc/104/ns/system.slice/lemond.service",
                  {"5": fdinfo(1, gfx=0, vram_kib=1 << 20), "6": fdinfo(1, gfx=0, vram_kib=1 << 20)})   # same client on two fds
        make_proc(self.root, 12, "ffmpeg", "0::/lxc/117/ns/system.slice/docker-0123456789ab.scope", {"4": fdinfo(2, dec=0, enc=0)})
        make_proc(self.root, 13, "Plex Transcoder", "0::/lxc/114/ns/system.slice/plexmediaserver.service", {"4": fdinfo(3)})
        make_proc(self.root, 14, "sleep", "0::/user.slice", {})                                   # no GPU
        make_proc(self.root, 15, "somethingelse", "0::/lxc/120/ns/x", {"4": fdinfo(4)})
        self.c = FakeCollector(gpuwatch.DEFAULT_RULES, proc=self.root, clock=lambda: self.t[0], gpu_paths=[])

    def test_classes_and_load(self):
        first = self.c.sample()
        self.assertEqual(sorted(first["classes"]), ["LLM (llama-server)", "Plex", "Tdarr", "somethingelse (CT120)"])
        self.assertEqual(first["classes"]["Tdarr"]["gfx"], 0.0)                                   # no previous reading yet
        self.t[0] = 1002.0
        for fd in ("5", "6"):                                                                     # both fds of one client report the same counters
            put(f"{self.root}/11/fdinfo/{fd}", fdinfo(1, gfx=1_000_000_000, vram_kib=1 << 20))    # 1 s of 2 s on the gfx engine
        put(f"{self.root}/12/fdinfo/4", fdinfo(2, dec=500_000_000, enc=500_000_000))              # media: 1 s in total
        s = self.c.sample()
        llm = s["classes"]["LLM (llama-server)"]
        self.assertEqual(llm["gfx"], 50.0)
        self.assertEqual(llm["vram_mb"], 1024.0, "one client on two fds is counted once")
        self.assertEqual([p["pid"] for p in llm["procs"]], [11])
        self.assertEqual(s["classes"]["Tdarr"]["media"], 50.0)
        self.assertEqual(s["classes"]["Plex"]["media"], 0.0)

    def test_a_process_that_leaves_disappears(self):
        self.c.sample()
        for name in ("fd", "fdinfo"):
            for f in os.listdir(f"{self.root}/13/{name}"):
                os.remove(f"{self.root}/13/{name}/{f}")
        self.t[0] = 1020.0                                                                          # past the rescan interval
        self.assertNotIn("Plex", self.c.sample()["classes"])


def sample(ts, gpu=0, **classes):
    return {"ts": ts, "gpu_busy": gpu, "classes": {
        c: {"gfx": v, "compute": 0.0, "media": 0.0, "dma": 0.0, "vram_mb": 100.0, "gtt_mb": 10.0, "procs": []} for c, v in classes.items()}}


class RecorderTest(unittest.TestCase):
    def run_scenario(self):
        r = gpuwatch.Recorder(":memory:")
        for t in range(1000, 1061):
            a = 50.0 if t < 1030 else None
            b = 40.0 if 1010 <= t < 1020 else None
            cl = {k: v for k, v in (("A", a), ("B", b)) if v is not None}
            r.add(sample(float(t), gpu=60 if cl else 0, **cl))
        return r

    def test_sessions_and_overlap_events(self):
        r = self.run_scenario()
        ev = list(reversed(r.events(50)))
        self.assertEqual([(e["kind"], e["class"]) for e in ev],
                         [("start", "A"), ("start", "B"), ("overlap_start", ""), ("stop", "B"), ("overlap_end", ""), ("stop", "A")])
        stop_b = ev[3]["detail"]
        self.assertEqual((stop_b["peak"], stop_b["avg"]), (40.0, 40.0))
        self.assertEqual(ev[2]["detail"]["classes"], ["A", "B"])

    def test_history_stats_and_pairs(self):
        r = self.run_scenario()
        h = r.history(minutes=1, now=1060.0)
        self.assertNotIn("__gpu__", h["classes"])
        self.assertEqual(len(h["gpu"]), int((1060 - h["t0"]) // h["bucket_s"]) + 1)
        self.assertGreaterEqual(max(h["classes"]["A"]), 40.0)
        st = r.stats(hours=1, now=1060.0)
        by = {c["class"]: c for c in st["classes"]}
        self.assertGreater(by["A"]["active_s"], by["B"]["active_s"])
        self.assertEqual(by["A"]["vram_mb"], 100)
        pair = st["overlaps"][0]
        self.assertEqual((pair["a"], pair["b"]), ("A", "B"))
        self.assertTrue(5 <= pair["seconds"] <= 15, pair)

    def test_sessions_remember_what_the_class_was_running(self):
        r = gpuwatch.Recorder(":memory:")
        for t in range(3000, 3030):
            s = sample(float(t), gpu=40, Jellyfin=40.0)
            s["classes"]["Jellyfin"]["procs"] = [{"pid": 1, "comm": "ffmpeg", "container": "114", "cmd": "ffmpeg -skip_frame nokey -i /media/x.mkv -vf tile"}]
            r.add(s)
        for t in range(3030, 3060):
            r.add(sample(float(t), gpu=0))
        ev = {e["kind"]: e["detail"] for e in r.events(10) if e["class"] == "Jellyfin"}
        self.assertIn("-skip_frame nokey", ev["start"]["cmd"])
        self.assertEqual(len(ev["stop"]["cmds"]), 1)

    def test_busy_gpu_nobody_is_blamed_for(self):
        r = gpuwatch.Recorder(":memory:")
        for t in range(2000, 2041):
            r.add(sample(float(t), gpu=80))                                                         # GPU busy, no class active
        self.assertGreaterEqual(r.stats(1, now=2040.0)["unattributed_busy_s"], 25)

    def test_old_rows_are_pruned(self):
        r = gpuwatch.Recorder(":memory:", retention_days=1)
        for t in range(1000, 1011):
            r.add(sample(float(t), gpu=10))
        r.add(sample(1000.0 + 3 * 86400, gpu=10))
        r.add(sample(1000.0 + 3 * 86400 + 6, gpu=10))
        self.assertEqual(r._q("SELECT COUNT(*) FROM samples WHERE ts < ?", (1000 + 86400,))[0][0], 0)


class ServerTest(unittest.TestCase):
    def test_endpoints(self):
        rec = gpuwatch.Recorder(":memory:")
        st = gpuwatch.State(None, rec, 2)
        st.latest = sample(1000.0, gpu=42, **{'Weird "name"': 12.5})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), gpuwatch.make_handler(st, b"<html>page</html>"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        get = lambda p: urllib.request.urlopen(base + p, timeout=10).read().decode()           # noqa: E731
        self.assertIn("page", get("/"))
        self.assertEqual(json.loads(get("/api/now"))["gpu_busy"], 42)
        self.assertEqual(json.loads(get("/api/history?minutes=5"))["classes"], {})
        self.assertEqual(json.loads(get("/api/events")), [])
        self.assertIn("overlaps", json.loads(get("/api/stats?hours=1")))
        m = get("/metrics")
        self.assertIn("gpuwatch_gpu_busy_percent 42", m)
        self.assertIn('gpuwatch_class_gfx_percent{class="Weird \\"name\\""} 12.5', m)
        srv.shutdown()
        srv.server_close()


class ArbiterTabTest(unittest.TestCase):
    """gpuwatch passes a few fixed paths through to a gpuarbiter so its page can show it as a tab."""

    def serve(self, handler):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def setUp(self):
        seen = self.seen = []
        from http.server import BaseHTTPRequestHandler

        class Fake(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):                              # noqa: N802
                seen.append(("GET", self.path))
                if self.path == "/":
                    return self._reply(200, b"<html>arbiter page</html>", "text/html")
                self._reply(200, json.dumps({"path": self.path}).encode())

            def do_POST(self):                             # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                seen.append(("POST", self.path, self.headers.get("X-Requested-With"), body))
                self._reply(422 if b"bad" in body else 200, json.dumps({"ok": b"bad" not in body}).encode())

        self.upstream = self.serve(Fake)
        st = gpuwatch.State(None, gpuwatch.Recorder(":memory:"), 2)
        self.base = self.serve(gpuwatch.make_handler(st, b"<html>gpuwatch</html>", self.upstream))
        self.plain = self.serve(gpuwatch.make_handler(st, b"<html>gpuwatch</html>"))

    def get(self, base, path):
        try:
            with urllib.request.urlopen(base + path, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_the_arbiter_page_and_api_are_passed_through(self):
        self.assertEqual(self.get(self.base, "/arbiter/"), (200, b"<html>arbiter page</html>"))
        code, body = self.get(self.base, "/arbiter/api/state")
        self.assertEqual((code, json.loads(body)), (200, {"path": "/api/state"}))
        self.assertIn(b"gpuwatch", self.get(self.base, "/")[1], "the normal page is untouched")

    def test_saving_the_config_is_forwarded_with_its_header_and_the_status_comes_back(self):
        def post(body):
            req = urllib.request.Request(self.base + "/arbiter/api/config", data=body, method="POST", headers={"X-Requested-With": "arbiter", "Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req, timeout=10).status
            except urllib.error.HTTPError as exc:
                return exc.code
        self.assertEqual(post(b'{"hold_s": 5}'), 200)
        self.assertEqual(post(b'{"bad": 1}'), 422)
        self.assertEqual(self.seen[0][:3], ("POST", "/api/config", "arbiter"))

    def test_only_the_fixed_paths_are_forwarded(self):
        self.assertEqual(self.get(self.base, "/arbiter/api/secret")[0], 404)
        self.assertEqual(self.get(self.base, "/arbiter/../../etc/passwd")[0], 404)
        self.assertEqual(self.get(self.plain, "/arbiter/api/state")[0], 404, "without --arbiter-url nothing is forwarded")
        self.assertEqual([s for s in self.seen if s[1] not in ("/", "/api/state", "/api/config")], [])

    def test_a_missing_arbiter_gives_502_not_a_crash(self):
        st = gpuwatch.State(None, gpuwatch.Recorder(":memory:"), 2)
        dead = self.serve(gpuwatch.make_handler(st, b"x", "http://127.0.0.1:9"))
        self.assertEqual(self.get(dead, "/arbiter/api/state")[0], 502)

    def test_the_bare_path_redirects(self):
        import http.client
        host, port = self.base.rsplit(":", 1)
        conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=10)
        conn.request("GET", "/arbiter")
        r = conn.getresponse()
        self.assertEqual((r.status, r.getheader("Location")), (302, "/arbiter/"))


if __name__ == "__main__":
    unittest.main()
