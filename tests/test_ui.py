"""API tests for the bridge web UI (skipped when fastapi/httpx are not installed)."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

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

    def test_write_needs_csrf_header_and_token(self):
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3}})
        self.assertEqual(r.status_code, 403)
        self.state["token"] = ""                                   # the API token is switched off: the header no longer counts
        r = self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3}}, headers=H)
        self.assertEqual(r.status_code, 401)

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
        with mock.patch("bridge_ui.os.access", return_value=False):
            j = self.c.get("/ui/api/config?component=comp").json()
            self.assertFalse(j["can_write"])
            self.assertIn("read-only", j["write_hint"])
            self.assertEqual(self.c.post("/ui/api/config/comp", json={"changes": {"beam": 3}}, headers=H).status_code, 403)
        self.state["cfg"] = None
        self.assertEqual(self.c.get("/ui/api/profiles").status_code, 503)

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


def _auth_app(directory, token=""):
    store = subcfg.ConfigStore("comp", directory, env={}, min_interval=0)
    state = {"enabled": True, "token": token, "cfg": store}
    app = FastAPI()
    bridge_ui.install(app, lambda: state)
    return app, state


@unittest.skipUnless(HAVE, "fastapi not installed")
class AuthTest(unittest.TestCase):
    PW = "correct horse battery"

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "schemas"))
        with open(os.path.join(self.d, "schemas", "comp.schema.json"), "w") as f:
            json.dump(SCHEMA, f)
        bridge_ui._throttle = bridge_ui.ui_auth.Throttle()
        bridge_ui._auth_stores.clear()
        self.app, self.state = _auth_app(self.d)
        self.c = TestClient(self.app, client=("192.168.1.20", 50000), follow_redirects=False)

    def setup_ok(self, client=None):
        return (client or self.c).post("/ui/api/setup", json={"username": "admin", "password": self.PW, "confirm": self.PW}, headers=H)

    def test_first_visit_goes_to_setup_and_nothing_else_is_reachable(self):
        r = self.c.get("/ui")
        self.assertEqual((r.status_code, r.headers["location"]), (303, "/ui/setup"))
        self.assertEqual(self.c.get("/ui/login").headers["location"], "/ui/setup")
        r = self.c.get("/ui/api/status")
        self.assertEqual((r.status_code, r.json()["login"]), (401, "/ui/setup"))
        page = self.c.get("/ui/setup")
        self.assertEqual(page.status_code, 200)
        self.assertIn('data-mode="setup"', page.text)

    def test_setup_is_refused_from_the_internet_and_without_the_csrf_header(self):
        out = TestClient(self.app, client=("8.8.8.8", 4000), follow_redirects=False)
        self.assertEqual(out.get("/ui/setup").status_code, 403)
        self.assertEqual(self.setup_ok(out).status_code, 403)
        r = self.c.post("/ui/api/setup", json={"username": "admin", "password": self.PW, "confirm": self.PW})
        self.assertEqual(r.status_code, 403)

    def test_a_lan_on_a_public_looking_range_can_be_trusted_explicitly(self):
        lan = TestClient(self.app, client=("172.57.56.240", 4000), follow_redirects=False)
        refused = lan.get("/ui/setup")
        self.assertEqual(refused.status_code, 403)
        self.assertIn("trusted_networks", refused.text)                    # it says how to fix it
        self.state["trusted_networks"] = ["172.57.56.0/24"]
        self.assertEqual(lan.get("/ui/setup").status_code, 200)
        self.assertEqual(self.setup_ok(lan).status_code, 200)

    def test_setup_enforces_a_decent_password(self):
        for pw, confirm in (("short", "short"), ("password", "password"), ("admin", "admin"), ("aaaaaaaaaa", "aaaaaaaaaa"),
                            (self.PW, "something else entirely")):
            r = self.c.post("/ui/api/setup", json={"username": "admin", "password": pw, "confirm": confirm}, headers=H)
            self.assertEqual(r.status_code, 422, pw)
        r = self.c.post("/ui/api/setup", json={"username": "bad name!", "password": self.PW, "confirm": self.PW}, headers=H)
        self.assertEqual(r.status_code, 422)
        self.assertFalse(os.path.exists(os.path.join(self.d, "auth.json")))

    def test_setup_signs_you_in_and_only_works_once(self):
        r = self.setup_ok()
        self.assertEqual(r.status_code, 200, r.text)
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        self.assertEqual(self.c.get("/ui").status_code, 200)
        self.assertEqual(self.c.get("/ui/api/me").json()["user"], "admin")
        self.assertEqual(self.setup_ok().status_code, 409)
        stored = open(os.path.join(self.d, "auth.json")).read()
        self.assertNotIn(self.PW, stored)                                  # only a salted hash is stored
        self.assertIn("scrypt$", stored)
        if os.name == "posix":
            self.assertEqual(os.stat(os.path.join(self.d, "auth.json")).st_mode & 0o777, 0o600)

    def test_login_logout_and_wrong_password(self):
        self.setup_ok()
        other = TestClient(self.app, client=("192.168.1.30", 1), follow_redirects=False)
        self.assertEqual(other.get("/ui").headers["location"], "/ui/login")
        bad = other.post("/ui/api/login", json={"username": "admin", "password": "nope-nope"}, headers=H)
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(other.post("/ui/api/login", json={"username": "admin", "password": self.PW}).status_code, 403)   # no CSRF header
        ok = other.post("/ui/api/login", json={"username": "admin", "password": self.PW}, headers=H)
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(other.get("/ui").status_code, 200)
        self.assertEqual(other.post("/ui/api/logout", headers=H).status_code, 200)
        self.assertEqual(other.get("/ui/api/status").status_code, 401)

    def test_login_is_slowed_down_after_repeated_failures(self):
        self.setup_ok()
        other = TestClient(self.app, client=("192.168.1.31", 1), follow_redirects=False)
        codes = [other.post("/ui/api/login", json={"username": "admin", "password": f"wrong-{i}"}, headers=H).status_code for i in range(7)]
        self.assertEqual(codes[:5], [401] * 5)
        self.assertEqual(codes[5:], [429, 429])
        self.assertEqual(other.post("/ui/api/login", json={"username": "admin", "password": self.PW}, headers=H).status_code, 429)

    def test_a_forged_or_expired_session_is_not_accepted(self):
        self.setup_ok()
        auth = bridge_ui._auth_store(self.state)
        good = auth.new_session()
        other = TestClient(self.app, client=("192.168.1.32", 1), follow_redirects=False)
        other.cookies.set(bridge_ui.COOKIE, good[:-1] + ("0" if good[-1] != "0" else "1"))
        self.assertEqual(other.get("/ui/api/status").status_code, 401)
        with mock.patch("ui_auth.time.time", return_value=time.time() + 15 * 86400):
            self.assertIsNone(auth.session_user(good))

    def test_change_password_signs_the_other_sessions_out(self):
        self.setup_ok()
        old_cookie = self.c.cookies.get(bridge_ui.COOKIE)
        r = self.c.post("/ui/api/password", json={"current": "wrong", "new": "another long phrase", "confirm": "another long phrase"}, headers=H)
        self.assertEqual(r.status_code, 422)
        r = self.c.post("/ui/api/password", json={"current": self.PW, "new": "short", "confirm": "short"}, headers=H)
        self.assertEqual(r.status_code, 422)
        r = self.c.post("/ui/api/password", json={"current": self.PW, "new": "another long phrase", "confirm": "another long phrase"}, headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.c.get("/ui/api/me").status_code, 200)                       # this browser got a fresh session
        stale = TestClient(self.app, client=("192.168.1.33", 1), follow_redirects=False)
        stale.cookies.set(bridge_ui.COOKIE, old_cookie)
        self.assertEqual(stale.get("/ui/api/me").status_code, 401)
        login = stale.post("/ui/api/login", json={"username": "admin", "password": "another long phrase"}, headers=H)
        self.assertEqual(login.status_code, 200)

    def test_api_token_header_still_works_for_scripts_but_cannot_change_the_password(self):
        app, state = _auth_app(self.d, token="scripttoken")
        c = TestClient(app, client=("192.168.1.40", 1), follow_redirects=False)
        self.assertEqual(c.get("/ui/api/config?component=comp").status_code, 401)
        c.headers.update({"X-Token": "scripttoken"})
        self.assertEqual(c.get("/ui/api/config?component=comp").status_code, 200)
        self.assertEqual(c.post("/ui/api/config/comp", json={"changes": {"beam": 4}}, headers=H).status_code, 200)
        self.assertEqual(c.post("/ui/api/password", json={"current": "x", "new": "y", "confirm": "y"}, headers=H).status_code, 403)
        self.assertEqual(c.get("/ui/api/me").json()["via"], "token")

    def test_readonly_folder_cannot_store_a_password(self):
        with mock.patch("ui_auth.os.access", return_value=False):
            page = self.c.get("/ui/setup")
            self.assertIn('data-mode="readonly"', page.text)
            self.assertEqual(self.setup_ok().status_code, 403)

    def test_without_a_config_folder_the_ui_says_so(self):
        self.state["cfg"] = None
        self.assertEqual(self.c.get("/ui").status_code, 503)


class UiAuthUnitTest(unittest.TestCase):
    def test_hash_is_salted_and_verifies(self):
        import ui_auth
        a, b = ui_auth.hash_password("one two three four"), ui_auth.hash_password("one two three four")
        self.assertNotEqual(a, b)
        self.assertTrue(ui_auth.check_hash("one two three four", a))
        self.assertFalse(ui_auth.check_hash("one two three five", a))
        self.assertFalse(ui_auth.check_hash("x", "garbage"))

    def test_local_addresses(self):
        import ui_auth
        for ok in ("192.168.1.5", "10.0.0.2", "172.16.4.4", "127.0.0.1", "::1", "::ffff:192.168.1.5", "fe80::1", "100.64.3.2"):
            self.assertTrue(ui_auth.is_local_address(ok), ok)
        for bad in ("8.8.8.8", "1.1.1.1", "2001:4860:4860::8888", "", None, "testclient", "172.57.56.240"):
            self.assertFalse(ui_auth.is_local_address(bad), bad)

    def test_trusted_networks_cover_a_lan_with_a_non_standard_range(self):
        import ui_auth
        self.assertTrue(ui_auth.is_local_address("172.57.56.240", ["172.57.56.0/24"]))
        self.assertFalse(ui_auth.is_local_address("172.57.57.1", ["172.57.56.0/24"]))
        self.assertTrue(ui_auth.is_local_address("172.57.56.240", ["nonsense", " 172.57.0.0/16 "]))      # bad entries are ignored

    def test_password_policy(self):
        import ui_auth
        self.assertEqual(ui_auth.validate_password("a long enough phrase", "admin"), [])
        for bad in ("short", "password123", "ADMINISTRATOR", "aaaaaaaaaaaa", "x" * 300):
            self.assertTrue(ui_auth.validate_password(bad, "admin"), bad)
        self.assertTrue(ui_auth.validate_password("Sandra-01", "sandra-01"))      # not the user name
