"""API tests for the bridge web UI (skipped when fastapi/httpx are not installed)."""
import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "subcfg"))
sys.path.insert(0, os.path.join(ROOT, "bridge", "ui"))
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import bridge_ui
    HAVE = True
except Exception:                                           # noqa: BLE001
    HAVE = False
import subcfg  # noqa: E402

SCHEMA = {"type": "object", "properties": {
    "beam": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10, "x-restart": False},
    "vad": {"type": "object", "properties": {"enabled": {"type": "boolean", "default": True}}},
    "urls": {"type": "array", "items": {"type": "string"}, "default": []},
    "key": {"type": "string", "default": "", "x-secret": True},
}}
H = {"X-Requested-With": "ui"}


@unittest.skipUnless(HAVE, "fastapi not installed")
class UiTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "schemas"))
        with open(os.path.join(self.d, "schemas", "comp.schema.json"), "w") as f:
            json.dump(SCHEMA, f)
        self.store = subcfg.ConfigStore("comp", self.d, env={}, min_interval=0)
        self.applied = []
        self.state = {"enabled": True, "token": "tok", "cfg": self.store}
        app = FastAPI()
        bridge_ui.install(app, lambda: self.state, on_change={"comp": lambda: self.applied.append(1), "profiles": lambda: self.applied.append(1)})
        self.c = TestClient(app)
        self.c.headers.update({"X-Token": "tok"})

    def test_auth_required(self):
        self.assertEqual(TestClient(self.c.app).get("/ui/api/config?component=comp").status_code, 401)
        self.assertEqual(self.c.get("/ui/api/config?component=comp").status_code, 200)

    def test_stale_cookie_does_not_beat_a_valid_query_token(self):
        c = TestClient(self.c.app, cookies={"subtitle_ui": "old-token"})
        self.assertEqual(c.get("/ui/api/config?component=comp&token=tok").status_code, 200)
        self.assertEqual(c.get("/ui/api/config?component=comp&token=wrong").status_code, 401)

    def test_write_needs_csrf_header_and_token(self):
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3}})
        self.assertEqual(r.status_code, 403)
        self.state["token"] = ""
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3}}, headers=H)
        self.assertEqual(r.status_code, 403)

    def test_save_validate_conflict_history_restore(self):
        j = self.c.get("/ui/api/config?component=comp").json()
        self.assertTrue(j["can_write"])
        etag = j["etag"]
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3, "vad.enabled": False, "urls": ["a"]}, "etag": etag}, headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.store.get("beam"), 3)
        self.assertFalse(self.store.get("vad.enabled"))
        self.assertEqual(self.applied, [1])
        # stale etag -> 409
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 4}, "etag": etag}, headers=H)
        self.assertEqual(r.status_code, 409)
        # invalid value -> 422, nothing written
        et = self.c.get("/ui/api/config?component=comp").json()["etag"]
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 99}, "etag": et}, headers=H)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.store.get("beam"), 3)
        # unknown and secret keys are refused
        r = self.c.post("/ui/api/config/comp", json={"changes": {"nope": 1, "key": "x"}}, headers=H)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(len(r.json()["errors"]), 2)
        # remove -> back to default, empty section pruned
        r = self.c.post("/ui/api/config/comp", json={"remove": ["vad.enabled"]}, headers=H)
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("vad", self.store.raw_file())
        self.assertTrue(self.store.get("vad.enabled"))
        # history + restore
        time.sleep(0.01)
        items = self.c.get("/ui/api/history/comp").json()["items"]
        self.assertEqual(len(items), 1)                      # the first save had no earlier file to snapshot
        oldest = items[-1]["name"]
        snap = self.c.get(f"/ui/api/history/comp/{oldest}").json()
        self.assertIn("diff", snap)
        self.assertEqual(self.c.get("/ui/api/history/comp/nope.json").status_code, 404)
        r = self.c.post(f"/ui/api/history/comp/{oldest}/restore", headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.c.post("/ui/api/history/comp/nope.json/restore", headers=H).status_code, 422)

    def test_profiles_crud_validation_and_match(self):
        body = {"description": "anime", "match": {"path_glob": "*/Anime/*", "source_language": "ja"},
                "overrides": {"comp": {"beam": 8, "vad.enabled": False}}}
        self.assertEqual(self.c.post("/ui/api/profiles/20-anime", json=body).status_code, 403)       # no CSRF header
        r = self.c.post("/ui/api/profiles/20-anime", json=body, headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.applied, [1])
        listed = self.c.get("/ui/api/profiles").json()
        self.assertEqual([p["file"] for p in listed["profiles"]], ["20-anime"])
        self.assertEqual(listed["profiles"][0]["overrides"]["comp"], {"beam": 8, "vad.enabled": False})
        self.assertIn("comp", listed["settings"])
        self.assertFalse(any(s["key"] == "key" for s in listed["settings"]["comp"]))                  # secrets are not offered
        # the store applies it
        ctx = {"path": "/m/Anime/x.mkv", "source_language": "ja"}
        self.assertEqual(self.store.get("beam", context=ctx), 8)
        self.assertEqual(self.store.get("beam"), 5)
        m = self.c.get("/ui/api/profiles/match", params={"path": "/m/Anime/x.mkv", "source_language": "ja"}).json()
        self.assertEqual(m["match"]["file"], "20-anime")
        self.assertIsNone(self.c.get("/ui/api/profiles/match", params={"path": "/m/Series/x.mkv"}).json()["match"])
        # validation
        for bad in ({"overrides": {"comp": {"beam": 99}}, "match": {}}, {"overrides": {"comp": {"nope": 1}}, "match": {}},
                    {"overrides": {"comp": {"key": "x"}}, "match": {}}, {"overrides": {"zzz": {"a": 1}}, "match": {}},
                    {"overrides": {}, "match": {"weird": "x"}}):
            self.assertEqual(self.c.post("/ui/api/profiles/bad", json=bad, headers=H).status_code, 422, bad)
        self.assertEqual(self.c.post("/ui/api/profiles/..%2Fx", json=body, headers=H).status_code, 404)
        self.assertEqual(self.c.post("/ui/api/profiles/.hidden", json=body, headers=H).status_code, 422)
        # delete keeps a snapshot
        self.assertEqual(self.c.post("/ui/api/profiles/20-anime/delete", headers=H).status_code, 200)
        self.assertEqual(self.c.get("/ui/api/profiles").json()["profiles"], [])
        self.assertTrue(os.listdir(os.path.join(self.d, "history", "profiles", "20-anime")))
        self.assertEqual(self.c.post("/ui/api/profiles/20-anime/delete", headers=H).status_code, 404)

    def test_playground_validation_and_start(self):
        started = []

        class Fake:
            def list(self): return []
            def get(self, rid): return {"id": rid, "status": "done", "results": []}
            def start(self, kind, payload):
                started.append((kind, payload)); return {"id": "r1"}
        st = {"enabled": True, "token": "tok", "cfg": self.store}
        app = FastAPI()
        bridge_ui.install(app, lambda: st, playground=Fake())
        c = TestClient(app)
        c.headers.update({"X-Token": "tok"})
        srt = "1\n00:00:01,000 --> 00:00:02,000\nhello\n"
        ok = {"srt": srt, "language": "ja", "variants": [{"name": "A", "overrides": {"comp.beam": 3}}, {"name": "B", "overrides": {}}]}
        # the playground's own overrides use the bridge component; here the test component is 'comp', so use a flat key of 'bridge'
        r = c.post("/ui/api/playground/translate", json=ok, headers=H)
        self.assertEqual(r.status_code, 422)                                    # 'bridge' has no schema in this test
        self.assertEqual(c.post("/ui/api/playground/translate", json={**ok, "variants": []}, headers=H).status_code, 422)
        self.assertEqual(c.post("/ui/api/playground/translate", json={**ok, "srt": " "}, headers=H).status_code, 422)
        self.assertEqual(c.post("/ui/api/playground/translate", json=ok).status_code, 403)
        meta = {"variants": [{"name": "greedy", "params": {"beam_size": 1}}, {"name": "beam", "params": {"beam_size": 5, "diarize": True}}]}
        import json as _json
        r = c.post("/ui/api/playground/transcribe", data={"meta": _json.dumps(meta)}, files={"file": ("a.wav", b"RIFFxxxx")}, headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(started[-1][1]["variants"][1]["params"], {"beam_size": 5, "diarize": "true"})
        bad = {"variants": [{"name": "x", "params": {"beam_size": 99}}]}
        r = c.post("/ui/api/playground/transcribe", data={"meta": _json.dumps(bad)}, files={"file": ("a.wav", b"x")}, headers=H)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(c.get("/ui/api/playground/r1").json()["status"], "done")

    def test_job_detail_and_srt_download(self):
        jobs = {7: {"id": 7, "file": "Show S01E01.mkv", "task": "translate", "status": "done", "seconds": 12,
                    "srt": "1\n00:00:01,000 --> 00:00:02,000\nHi\n", "detail": {"source": ["x"], "draft": ["a"], "final": ["b"], "why": ["name"], "bible": ""}},
                8: {"id": 8, "file": "old.mkv", "task": "transcribe", "status": "done", "seconds": 1}}
        self.state["get_job"] = lambda i: dict(jobs[i]) if i in jobs else None
        j = self.c.get("/ui/api/jobs/7").json()
        self.assertEqual(j["detail"]["final"], ["b"])
        self.assertNotIn("srt", j["job"])
        r = self.c.get("/ui/api/jobs/7/srt")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertIn('filename="Show_S01E01.srt"', r.headers["content-disposition"])
        self.assertIn("Hi", r.text)
        self.assertIn("10 most recent", self.c.get("/ui/api/jobs/8").json()["note"])      # details already dropped
        self.assertEqual(self.c.get("/ui/api/jobs/8/srt").status_code, 404)
        self.assertEqual(self.c.get("/ui/api/jobs/99").status_code, 404)
        self.assertEqual(TestClient(self.c.app).get("/ui/api/jobs/7").status_code, 401)

    def test_write_hint_explains_why_editing_is_off(self):
        self.assertIsNone(self.c.get("/ui/api/config?component=comp").json()["write_hint"])
        self.state["token"] = ""
        hint = self.c.get("/ui/api/config?component=comp").json()["write_hint"]
        self.assertIn("init-token", hint)
        self.state["token"] = "tok"
        self.state["cfg"] = None
        self.assertEqual(self.c.get("/ui/api/profiles").json(), {"enabled": False})

    def test_unknown_component_and_path_tricks(self):
        self.assertEqual(self.c.get("/ui/api/config?component=nope").status_code, 404)
        self.assertEqual(self.c.get("/ui/api/config?component=../x").status_code, 404)

    def test_secret_values_never_returned(self):
        with open(os.path.join(self.d, "comp.json"), "w") as f:
            json.dump({"key": "plaintext-secret"}, f)
        self.store.refresh(force=True)
        text = self.c.get("/ui/api/config?component=comp").text
        self.assertNotIn("plaintext-secret", text)


if __name__ == "__main__":
    unittest.main()
