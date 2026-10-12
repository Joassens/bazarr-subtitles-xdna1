"""gpuarbiter: pauses a background consumer for higher-priority GPU work or low RAM, resumes by itself, and respects manual actions (fake everything)."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import gpuarbiter  # noqa: E402


class FakeController:
    def __init__(self):
        self.p = False
        self.calls = []

    def paused(self):
        return self.p

    def set_paused(self, v):
        self.calls.append(v)
        self.p = bool(v)


class Rig:
    """A fake clock, fake gpuwatch/hostmem data and a fake Tdarr."""
    def __init__(self, **cfg):
        self.t = 1000.0
        self.classes = {}
        self.ram = 20.0
        self.gpuwatch_ok = True
        self.ctl = FakeController()
        base = {"dry_run": False, "poll_s": 5, "hold_s": 10, "release_s": 60, "manual_override_s": 300}
        base.update(cfg)
        self.arb = gpuarbiter.Arbiter(base, gpuarbiter.Events(":memory:"), {"tdarr": self.ctl}, clock=lambda: self.t, fetch=self._fetch)

    def _fetch(self, url):
        if "9610" in url or "gpuwatch" in url:
            if not self.gpuwatch_ok:
                raise OSError("gpuwatch is down")
            return {"classes": self.classes}
        return {"mem_available_gb": self.ram}

    def run(self, seconds, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        for _ in range(int(seconds // 5)):
            self.arb.step()
            self.t += 5

    def kinds(self):
        return [e["kind"] for e in reversed(self.arb.events.recent(50))]


def busy(media=0, gfx=0, compute=0):
    return {"gfx": gfx, "compute": compute, "media": media, "dma": 0}


class ArbiterTest(unittest.TestCase):
    def test_a_media_server_makes_tdarr_wait_after_the_hold_time_and_resume_after_quiet(self):
        r = Rig()
        r.run(10, classes={"Plex": busy(media=40)})
        self.assertFalse(r.ctl.p, "a short blip must not pause anything")
        r.run(30)
        self.assertTrue(r.ctl.p)
        self.assertEqual(r.ctl.calls, [True])
        r.run(40, classes={})                              # Plex is gone, but not long enough
        self.assertTrue(r.ctl.p)
        r.run(60)
        self.assertFalse(r.ctl.p)
        self.assertEqual(r.kinds(), ["pause", "resume"])

    def test_blips_shorter_than_the_hold_time_are_ignored(self):
        r = Rig()
        for _ in range(6):
            r.run(5, classes={"Plex": busy(media=50)})
            r.run(15, classes={})
        self.assertEqual(r.ctl.calls, [])

    def test_the_llm_does_not_stop_tdarr_because_they_share_no_engine(self):
        r = Rig()
        r.run(60, classes={"LLM (llama-server)": busy(gfx=90, compute=80)})
        self.assertEqual(r.ctl.calls, [])

    def test_a_lower_priority_class_never_stops_a_higher_one(self):
        r = Rig()
        r.run(60, classes={"Tdarr": busy(media=95)})
        self.assertEqual(r.ctl.calls, [])

    def test_ram_guard_has_hysteresis_and_resumes_at_once(self):
        r = Rig()
        r.run(30, ram=2.0)
        self.assertTrue(r.ctl.p)
        r.run(30, ram=4.0)                                 # between the limits: stays paused
        self.assertTrue(r.ctl.p)
        r.run(10, ram=6.0)                                 # above resume_above_gb: no extra waiting
        self.assertFalse(r.ctl.p)

    def test_somebody_elses_pause_is_left_alone(self):
        r = Rig()
        r.ctl.p = True                                     # paused by hand
        r.run(30, classes={"Plex": busy(media=50)})
        r.run(120, classes={})
        self.assertTrue(r.ctl.p, "we did not pause it, so we must not resume it")
        self.assertEqual(r.ctl.calls, [])

    def test_a_manual_resume_is_respected_for_a_while(self):
        r = Rig()
        r.run(40, classes={"Plex": busy(media=50)})
        self.assertTrue(r.ctl.p)
        r.ctl.p = False                                    # the user resumes it by hand
        r.run(30)
        self.assertFalse(r.ctl.p)
        self.assertIn("external_resume", r.kinds())
        self.assertEqual(r.ctl.calls, [True], "must not pause again during the override period")
        r.run(400)                                         # override over, Plex still busy
        self.assertTrue(r.ctl.p)

    def test_dry_run_only_writes_to_the_log(self):
        r = Rig(dry_run=True)
        r.run(40, classes={"Plex": busy(media=50)})
        r.run(120, classes={})
        self.assertEqual(r.ctl.calls, [])
        self.assertEqual(r.kinds(), ["would_pause", "would_resume"])

    def test_without_a_view_of_the_gpu_nothing_changes(self):
        r = Rig()
        r.run(40, classes={"Plex": busy(media=50)})
        self.assertTrue(r.ctl.p)
        r.run(300, gpuwatch_ok=False)
        self.assertTrue(r.ctl.p, "no data is not the same as quiet")
        self.assertTrue(r.arb.state()["errors"])

    def test_priorities_come_from_the_config(self):
        r = Rig(classes=[{"name": "A", "priority": 1, "engines": ["media"]}, {"name": "B", "priority": 2, "engines": ["media"], "controller": "tdarr"}])
        r.run(40, classes={"A": busy(media=50)})
        self.assertTrue(r.ctl.p)


class ConfigTest(unittest.TestCase):
    def test_validation(self):
        good = gpuarbiter.merged({})
        self.assertEqual(gpuarbiter.validate(good), [])
        bad = gpuarbiter.merged({"classes": [{"name": "X", "priority": "high", "engines": ["video"], "controller": "nope"}], "hold_s": -1})
        problems = " ".join(gpuarbiter.validate(bad))
        for word in ("priority must be a number", "engines must be", "unknown controller", "hold_s"):
            self.assertIn(word, problems)
        self.assertTrue(gpuarbiter.validate(gpuarbiter.merged({"ram_guard": {"enabled": True, "pause_below_gb": 5, "resume_above_gb": 3}})))

    def test_http_api_saves_a_valid_config_and_rejects_the_rest(self):
        d = tempfile.mkdtemp()
        cfgfile = gpuarbiter.ConfigFile(os.path.join(d, "config.json"))
        r = Rig()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), gpuarbiter.make_handler(r.arb, cfgfile, b"<html>page</html>"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def post(body, header=True):
            req = urllib.request.Request(base + "/api/config", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **({"X-Requested-With": "arbiter"} if header else {})})
            try:
                return urllib.request.urlopen(req, timeout=10).status
            except urllib.error.HTTPError as exc:
                return exc.code

        self.assertEqual(post({"dry_run": True}, header=False), 403)
        self.assertEqual(post({"classes": []}), 422)
        self.assertEqual(post({"dry_run": True, "hold_s": 30}), 200)
        self.assertEqual(json.load(open(cfgfile.path))["hold_s"], 30)
        self.assertEqual(r.arb.cfg["hold_s"], 30)
        self.assertEqual(post({"dry_run": False, "hold_s": 20}), 200)
        self.assertTrue(os.listdir(os.path.join(d, "history")), "the previous version is kept")
        self.assertEqual(json.load(urllib.request.urlopen(base + "/api/state", timeout=10))["dry_run"], False)
        self.assertIn(b"page", urllib.request.urlopen(base + "/", timeout=10).read())
        self.assertEqual(json.load(urllib.request.urlopen(base + "/api/events", timeout=10))[0]["kind"], "config")
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    unittest.main()
