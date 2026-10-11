"""Unit tests for the shared configuration library.  python -m unittest discover -s tests -v"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "subcfg"))
import subcfg  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "beam": {"type": "integer", "default": 5, "x-env": "T_BEAM", "minimum": 1, "maximum": 10},
        "vad": {"type": "object", "properties": {
            "enabled": {"type": "boolean", "default": True, "x-env": "T_VAD"},
            "threshold": {"type": "number", "default": 0.35},
        }},
        "urls": {"type": "array", "items": {"type": "string"}, "default": [], "x-env": "T_URLS"},
        "style": {"type": "string", "default": "dash", "enum": ["none", "dash", "label"]},
        "api_key": {"type": "string", "default": "", "x-env": "T_KEY", "x-secret": True},
        "model": {"type": "string", "default": "m", "x-restart": True},
    },
}


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f) if isinstance(data, (dict, list)) else f.write(data)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def store(self, env=None, **kw):
        return subcfg.ConfigStore("comp", self.d, schema=SCHEMA, env=env or {}, min_interval=0, **kw)

    def test_defaults_without_files(self):
        s = self.store()
        self.assertEqual(s.get("beam"), 5)
        self.assertEqual(s.get("vad.threshold"), 0.35)
        self.assertEqual(s.source("beam"), "default")

    def test_file_overrides_default_env_overrides_file(self):
        write(os.path.join(self.d, "comp.json"), {"beam": 3, "vad": {"enabled": False}})
        s = self.store()
        self.assertEqual(s.get("beam"), 3)
        self.assertFalse(s.get("vad.enabled"))
        self.assertEqual(s.source("beam"), "file")
        s2 = self.store(env={"T_BEAM": "1", "T_VAD": "true", "T_URLS": "a, b"})
        self.assertEqual(s2.get("beam"), 1)
        self.assertTrue(s2.get("vad.enabled"))
        self.assertEqual(s2.get("urls"), ["a", "b"])
        self.assertEqual(s2.source("beam"), "env")

    def test_hot_reload(self):
        p = os.path.join(self.d, "comp.json")
        write(p, {"beam": 2})
        s = self.store()
        self.assertEqual(s.get("beam"), 2)
        time.sleep(0.01)
        write(p, {"beam": 7})
        os.utime(p, None)
        self.assertTrue(s.refresh())
        self.assertEqual(s.get("beam"), 7)
        self.assertFalse(s.refresh())

    def test_bad_file_keeps_last_good(self):
        p = os.path.join(self.d, "comp.json")
        write(p, {"beam": 4})
        s = self.store()
        write(p, "{ not json")
        s.refresh(force=True)
        self.assertEqual(s.get("beam"), 4)
        self.assertTrue(s.errors)
        write(p, {"beam": 99})                     # violates maximum
        s.refresh(force=True)
        self.assertEqual(s.get("beam"), 4)
        self.assertTrue(any("maximum" in e for e in s.errors))

    def test_unknown_setting_is_a_warning_not_a_failure(self):
        write(os.path.join(self.d, "comp.json"), {"beam": 2, "typo": 1})
        s = self.store()
        self.assertEqual(s.get("beam"), 2)
        self.assertTrue(any("unknown setting" in e for e in s.errors))

    def test_secrets_are_interpolated_and_masked(self):
        write(os.path.join(self.d, "comp.json"), {"api_key": "${MY_KEY}"})
        write(os.path.join(self.d, "secrets.env"), "# comment\nMY_KEY=abc123\n")
        s = self.store()
        self.assertEqual(s.get("api_key"), "abc123")
        self.assertEqual(s.masked()["api_key"], "********")
        self.assertEqual(s.raw_file()["api_key"], "${MY_KEY}")

    def test_profile_overrides_per_context(self):
        write(os.path.join(self.d, "comp.json"), {"beam": 5})
        write(os.path.join(self.d, "profiles", "anime.json"),
              {"name": "anime", "match": {"path_glob": "*/Anime/*", "source_language": "ja"}, "overrides": {"comp": {"beam": 8}}})
        s = self.store()
        self.assertEqual(s.get("beam", context={"path": "/m/Anime/x.mkv", "source_language": "ja"}), 8)
        self.assertEqual(s.get("beam", context={"path": "/m/Series/x.mkv", "source_language": "ja"}), 5)
        self.assertEqual(s.get("beam"), 5)

    def test_save_validates_snapshots_and_is_atomic(self):
        s = self.store()
        self.assertTrue(s.save({"beam": 0}))                    # invalid -> errors, nothing written
        self.assertFalse(os.path.exists(os.path.join(self.d, "comp.json")))
        self.assertEqual(s.save({"beam": 3}), [])
        self.assertEqual(s.get("beam"), 3)
        self.assertEqual(s.save({"beam": 4}, author="tester"), [])
        snaps = os.listdir(os.path.join(self.d, "history", "comp"))
        self.assertEqual(len(snaps), 1)
        self.assertTrue(snaps[0].endswith("-tester.json"))
        self.assertEqual(json.load(open(os.path.join(self.d, "comp.json")))["beam"], 4)
        self.assertFalse([n for n in os.listdir(self.d) if n.endswith(".tmp")])

    def test_seed_from_env_skips_defaults_and_secrets(self):
        s = self.store(env={"T_BEAM": "3", "T_VAD": "true", "T_KEY": "supersecret"})
        seed = s.seed_from_env()
        self.assertEqual(seed["beam"], 3)
        self.assertNotIn("vad", seed)                            # equals the default
        self.assertEqual(seed["api_key"], "${T_KEY}")

    def test_number_is_always_float(self):
        schema = {"type": "object", "properties": {"x": {"type": "number", "default": 3}}}
        s = subcfg.ConfigStore("n", self.d, schema=schema, env={}, min_interval=0)
        self.assertIsInstance(s.get("x"), float)
        write(os.path.join(self.d, "n.json"), {"x": 4})
        s.refresh(force=True)
        self.assertEqual(s.get("x"), 4.0)
        self.assertIsInstance(s.get("x"), float)

    def test_history_etag_generation_and_restore(self):
        s = self.store()
        g0 = s.generation
        self.assertEqual(s.etag(), "none")
        self.assertEqual(s.save({"beam": 3}, author="alice"), [])
        e1 = s.etag()
        self.assertGreater(s.generation, g0)
        self.assertEqual(s.save({"beam": 4}, author="bob smith!"), [])
        self.assertNotEqual(s.etag(), e1)
        self.assertEqual(s.save({"beam": 5}, author="bob smith!"), [])          # same second: both snapshots kept
        hist = s.history()
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(h["author"].startswith("bobsmith") for h in hist))
        self.assertEqual(s.read_snapshot("../comp.json"), None)                  # no path tricks
        oldest = hist[-1]["name"]
        self.assertEqual(s.read_snapshot(oldest)["beam"], 3)
        self.assertEqual(s.restore(oldest, author="alice"), [])
        self.assertEqual(s.get("beam"), 3)
        self.assertEqual(len(s.history()), 3)                                   # the replaced file was snapshotted
        self.assertTrue(s.restore("missing.json"))

    def test_describe_lists_every_setting_with_source(self):
        write(os.path.join(self.d, "comp.json"), {"beam": 3, "api_key": "${K}"})
        write(os.path.join(self.d, "secrets.env"), "K=hidden\n")
        s = self.store(env={"T_VAD": "false"})
        rows = {r["key"]: r for r in s.describe()}
        self.assertEqual(rows["beam"]["value"], 3)
        self.assertEqual(rows["beam"]["source"], "file")
        self.assertEqual(rows["vad.enabled"]["source"], "env")
        self.assertEqual(rows["vad.threshold"]["source"], "default")
        self.assertEqual(rows["vad.threshold"]["section"], "vad")
        self.assertEqual(rows["api_key"]["value"], "********")
        self.assertTrue(rows["api_key"]["secret"])
        self.assertTrue(rows["model"]["restart"])

    def test_init_token_is_idempotent_and_keeps_other_settings(self):
        write(os.path.join(self.d, "bridge.json"), {"schema_version": 1, "ui": {"enabled": True}, "llm": {"model": "m"}})
        write(os.path.join(self.d, "secrets.env"), "HF_TOKEN=abc\n")
        token, changed = subcfg.init_token(self.d)
        self.assertEqual(len(token), 48)
        self.assertEqual(len(changed), 2)
        env = open(os.path.join(self.d, "secrets.env")).read()
        self.assertIn("HF_TOKEN=abc", env)
        self.assertIn(f"WEBUI_TOKEN={token}", env)
        cfg = json.load(open(os.path.join(self.d, "bridge.json")))
        self.assertEqual(cfg["ui"], {"enabled": True, "token": "${WEBUI_TOKEN}"})
        self.assertEqual(cfg["llm"], {"model": "m"})
        again, changed = subcfg.init_token(self.d)
        self.assertEqual((again, changed), (token, []))
        new, changed = subcfg.init_token(self.d, rotate=True)
        self.assertNotEqual(new, token)
        self.assertEqual(open(os.path.join(self.d, "secrets.env")).read().count("WEBUI_TOKEN="), 1)

    def test_reset_password_removes_only_the_password_file(self):
        write(os.path.join(self.d, "auth.json"), {"username": "admin"})
        write(os.path.join(self.d, "bridge.json"), {"schema_version": 1})
        self.assertTrue(subcfg.reset_password(self.d))
        self.assertFalse(os.path.exists(os.path.join(self.d, "auth.json")))
        self.assertTrue(os.path.exists(os.path.join(self.d, "bridge.json")))
        self.assertFalse(subcfg.reset_password(self.d))

    def test_init_token_in_an_empty_folder(self):
        d = os.path.join(self.d, "fresh")
        token, changed = subcfg.init_token(d)
        self.assertTrue(os.path.exists(os.path.join(d, "bridge.json")))
        self.assertEqual(json.load(open(os.path.join(d, "bridge.json")))["ui"]["token"], "${WEBUI_TOKEN}")

    def test_restart_flag(self):
        s = self.store()
        self.assertTrue(s.needs_restart("model"))
        self.assertFalse(s.needs_restart("beam"))

    def test_schema_loaded_from_config_dir(self):
        write(os.path.join(self.d, "schemas", "other.schema.json"), {"type": "object", "properties": {"x": {"type": "integer", "default": 9}}})
        s = subcfg.ConfigStore("other", self.d, env={})
        self.assertEqual(s.get("x"), 9)


class ValidateTest(unittest.TestCase):
    def test_types_enum_ranges(self):
        self.assertEqual(subcfg.validate({"beam": 5}, SCHEMA), [])
        self.assertTrue(subcfg.validate({"beam": "5"}, SCHEMA))
        self.assertTrue(subcfg.validate({"beam": True}, SCHEMA))
        self.assertTrue(subcfg.validate({"style": "wild"}, SCHEMA))
        self.assertTrue(subcfg.validate({"urls": [1]}, SCHEMA))
        self.assertTrue(subcfg.validate({"vad": {"enabled": "yes"}}, SCHEMA))

    def test_pattern(self):
        schema = {"type": "object", "properties": {"w": {"type": "string", "pattern": r"^$|^\d{1,2}:\d{2}-\d{1,2}:\d{2}$"}}}
        for ok in ("", "01:00-07:00", "22:30-6:00"):
            self.assertEqual(subcfg.validate({"w": ok}, schema), [], ok)
        for bad in ("night", "1-7", "01:00"):
            self.assertTrue(subcfg.validate({"w": bad}, schema), bad)


if __name__ == "__main__":
    unittest.main()
