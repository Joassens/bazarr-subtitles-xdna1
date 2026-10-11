"""subcfg: one shared, hot-reloading JSON configuration for the subtitle tools (standard library only).

Layout of a config directory (everything optional):

    <dir>/settings.json              shared settings
    <dir>/<component>.json           one file per tool (npusubs.json, bridge.json, ...)
    <dir>/schemas/<component>.schema.json   JSON Schema (subset) with "default", "x-env", "x-secret", "x-restart"
    <dir>/profiles/*.json            {"name", "match": {...}, "overrides": {"<component>": {...}}}
    <dir>/history/<component>/       snapshots of replaced files
    <dir>/secrets.env                NAME=value lines; referenced from JSON as "${NAME}"

Precedence (lowest to highest): schema defaults < <component>.json < environment variable < matching profile.
A bad file never takes a tool down: the last good configuration stays active and `errors` explains why.
"""
from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
import tempfile
import threading
import time
from typing import Any

SECRET_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")


# --------------------------------------------------------------------------- schema helpers
def _is_leaf(schema: dict) -> bool:
    return schema.get("type") != "object"


def walk(schema: dict, path: tuple = ()):
    """Yield (path, leaf_schema) for every non-object property."""
    for key, sub in (schema.get("properties") or {}).items():
        if _is_leaf(sub):
            yield path + (key,), sub
        else:
            yield from walk(sub, path + (key,))


def _set(d: dict, path: tuple, value: Any) -> None:
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def _get(d: dict, path: tuple, default: Any = None) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def defaults_from_schema(schema: dict) -> dict:
    out: dict = {}
    for path, leaf in walk(schema):
        if "default" in leaf:
            _set(out, path, copy.deepcopy(leaf["default"]))
    return out


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def coerce_env(value: str, leaf: dict) -> Any:
    t = leaf.get("type")
    v = value.strip()
    if t == "boolean":
        return v.lower() not in ("0", "false", "no", "off", "")
    if t == "integer":
        return int(float(v))
    if t == "number":
        return float(v)
    if t == "array":
        return [p.strip() for p in v.split(",") if p.strip()]
    return value


def validate(data: Any, schema: dict, path: str = "") -> list[str]:
    """Small JSON Schema subset: type, enum, pattern, minimum, maximum, properties, items. Unknown keys are reported."""
    errs: list[str] = []
    t = schema.get("type")
    here = path or "(root)"
    if t == "object":
        if not isinstance(data, dict):
            return [f"{here}: expected object"]
        props = schema.get("properties") or {}
        for k, v in data.items():
            if k in ("schema_version", "$schema"):
                continue
            if k not in props:
                errs.append(f"{path + '.' if path else ''}{k}: unknown setting")
            else:
                errs += validate(v, props[k], f"{path + '.' if path else ''}{k}")
        return errs
    types = {"boolean": bool, "integer": int, "number": (int, float), "string": str, "array": list}
    if t in types:
        ok = isinstance(data, types[t]) and not (t in ("integer", "number") and isinstance(data, bool))
        if t == "integer" and isinstance(data, float):
            ok = data.is_integer()
        if not ok:
            return [f"{here}: expected {t}, got {type(data).__name__}"]
    if "enum" in schema and data not in schema["enum"]:
        errs.append(f"{here}: must be one of {schema['enum']}")
    if isinstance(data, str) and "pattern" in schema and not re.search(schema["pattern"], data):
        errs.append(f"{here}: does not match the expected format ({schema.get('x-format', schema['pattern'])})")
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errs.append(f"{here}: below minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errs.append(f"{here}: above maximum {schema['maximum']}")
    if t == "array" and isinstance(data, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(data):
            errs += validate(item, schema["items"], f"{here}[{i}]")
    return errs


# --------------------------------------------------------------------------------- store
class ConfigStore:
    def __init__(self, component: str, directory: str | None = None, schema: dict | str | None = None,
                 schema_dirs: tuple = (), min_interval: float = 2.0, env: dict | None = None):
        self.component = component
        self.dir = directory if directory is not None else os.environ.get("SUBCFG_DIR", "/config")
        self.min_interval = min_interval
        self._env = env if env is not None else os.environ
        self._lock = threading.RLock()
        self.errors: list[str] = []
        self.schema = self._find_schema(schema, schema_dirs)
        self._defaults = defaults_from_schema(self.schema)
        self._env_map = {p: leaf["x-env"] for p, leaf in walk(self.schema) if leaf.get("x-env")}
        self._leaf = {p: leaf for p, leaf in walk(self.schema)}
        self._sig: tuple | None = None
        self.generation = 0               # increases every time the configuration is (re)loaded
        self._last_check = 0.0
        self._file: dict = {}
        self._base: dict = copy.deepcopy(self._defaults)
        self._sources: dict[tuple, str] = {}
        self._secrets: dict[str, str] = {}
        self._profiles: list[dict] = []
        self.refresh(force=True)

    # ---------------------------------------------------------------- loading
    def _find_schema(self, schema, schema_dirs) -> dict:
        if isinstance(schema, dict):
            return schema
        candidates = [schema] if isinstance(schema, str) else []
        candidates += [os.path.join(self.dir, "schemas", f"{self.component}.schema.json")]
        candidates += [os.path.join(d, f"{self.component}.schema.json") for d in schema_dirs]
        for p in candidates:
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                continue
        return {"type": "object", "properties": {}}

    def _path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    def _signature(self) -> tuple:
        sig = []
        for p in (self._path(f"{self.component}.json"), self._path("settings.json"), self._path("secrets.env")):
            try:
                st = os.stat(p)
                sig.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((p, 0, 0))
        pdir = self._path("profiles")
        try:
            for name in sorted(os.listdir(pdir)):
                if name.endswith(".json"):
                    st = os.stat(os.path.join(pdir, name))
                    sig.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
        return tuple(sig)

    def refresh(self, force: bool = False) -> bool:
        """Re-read changed files (cheap: an mtime check at most every min_interval seconds). True if anything changed."""
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_check < self.min_interval:
                return False
            self._last_check = now
            sig = self._signature()
            if not force and sig == self._sig:
                return False
            self._sig = sig
            self._load()
            return True

    def _read_json(self, path: str) -> dict | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self.errors.append(f"{os.path.basename(path)}: {exc}")
            return None

    def _load(self) -> None:
        self.generation += 1
        self.errors = []
        # secrets.env
        secrets: dict[str, str] = {}
        try:
            with open(self._path("secrets.env"), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        secrets[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
        self._secrets = secrets
        # component file (keep the last good one on any problem)
        data = self._read_json(self._path(f"{self.component}.json"))
        if data is not None:
            errs = validate(data, self.schema) if self.schema.get("properties") else []
            hard = [e for e in errs if "unknown setting" not in e]
            self.errors += [f"{self.component}.json: {e}" for e in errs]
            if not hard:
                self._file = {k: v for k, v in data.items() if k not in ("schema_version", "$schema")}
        # profiles
        profiles = []
        pdir = self._path("profiles")
        try:
            for name in sorted(os.listdir(pdir)):
                if name.endswith(".json"):
                    p = self._read_json(os.path.join(pdir, name))
                    if p and isinstance(p.get("overrides"), dict):
                        p.setdefault("name", name[:-5])
                        p["_file"] = name[:-5]
                        profiles.append(p)
        except OSError:
            pass
        self._profiles = profiles
        self._compute()

    def _compute(self) -> None:
        eff = deep_merge(self._defaults, self._file)
        sources: dict[tuple, str] = {}
        for path, _leaf in self._leaf.items():
            sources[path] = "file" if _get(self._file, path, _MISSING) is not _MISSING else "default"
        for path, env_name in self._env_map.items():
            if env_name in self._env and str(self._env[env_name]) != "":
                try:
                    _set(eff, path, coerce_env(str(self._env[env_name]), self._leaf[path]))
                    sources[path] = "env"
                except (ValueError, TypeError) as exc:
                    self.errors.append(f"env {env_name}: {exc}")
        for path, leaf in self._leaf.items():                        # "number" is always a float
            val = _get(eff, path, _MISSING)
            if leaf.get("type") == "number" and isinstance(val, int) and not isinstance(val, bool):
                _set(eff, path, float(val))
        self._base, self._sources = eff, sources

    # ----------------------------------------------------------------- reading
    def _resolve_secrets(self, value: Any) -> Any:
        if isinstance(value, str) and "${" in value:
            return SECRET_RE.sub(lambda m: self._secrets.get(m.group(1), self._env.get(m.group(1), "")), value)
        if isinstance(value, list):
            return [self._resolve_secrets(v) for v in value]
        if isinstance(value, dict):
            return {k: self._resolve_secrets(v) for k, v in value.items()}
        return value

    def match_profile(self, context: dict | None) -> dict | None:
        if not context:
            return None
        for prof in self._profiles:
            m = prof.get("match") or {}
            ok = True
            if "path_glob" in m:
                ok = ok and fnmatch.fnmatch(str(context.get("path", "")).replace("\\", "/"), m["path_glob"])
            for key in ("source_language", "target_language"):
                if key in m:
                    ok = ok and str(context.get(key, "")).lower() == str(m[key]).lower()
            if ok:
                return prof
        return None

    def effective(self, context: dict | None = None) -> dict:
        with self._lock:
            eff = copy.deepcopy(self._base)
            prof = self.match_profile(context)
            if prof:
                eff = deep_merge(eff, (prof.get("overrides") or {}).get(self.component, {}))
            return self._resolve_secrets(eff)

    def get(self, path: str, default: Any = None, context: dict | None = None) -> Any:
        return _get(self.effective(context), tuple(path.split(".")), default)

    def secret(self, name: str, default: str = "") -> str:
        """A secret by name: environment first, then secrets.env in the config directory."""
        with self._lock:
            return self._env.get(name) or self._secrets.get(name) or default

    def source(self, path: str) -> str:
        """'default', 'file' or 'env' for a dotted key (what currently wins)."""
        return self._sources.get(tuple(path.split(".")), "unknown")

    def needs_restart(self, path: str) -> bool:
        return bool(self._leaf.get(tuple(path.split(".")), {}).get("x-restart"))

    def describe(self, context: dict | None = None) -> list[dict]:
        """Every setting with title, description, type, default, current value (secrets masked), source and flags.
        This is what a UI needs to render a settings page."""
        eff = self.masked(context)
        out = []
        for path, leaf in self._leaf.items():
            key = ".".join(path)
            out.append({
                "key": key, "section": path[0] if len(path) > 1 else "general",
                "title": leaf.get("title", key), "description": leaf.get("description", ""), "type": leaf.get("type"),
                "default": None if leaf.get("x-secret") else leaf.get("default"), "value": _get(eff, path),
                "source": self._sources.get(path, "default"), "restart": bool(leaf.get("x-restart")),
                "secret": bool(leaf.get("x-secret")), "env": leaf.get("x-env"), "enum": leaf.get("enum"),
                "min": leaf.get("minimum"), "max": leaf.get("maximum"),
            })
        return out

    def raw_file(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._file)

    def masked(self, context: dict | None = None) -> dict:
        """Effective config with x-secret values hidden (safe to show in a UI)."""
        eff = self.effective(context)
        for path, leaf in self._leaf.items():
            if leaf.get("x-secret") and _get(eff, path):
                _set(eff, path, "********")
        return eff

    # ----------------------------------------------------------------- writing
    def save(self, data: dict, author: str = "ui", keep_history: int = 100) -> list[str]:
        """Validate, snapshot the old file to history/, write atomically. Returns validation errors (nothing written)."""
        errs = validate(data, self.schema) if self.schema.get("properties") else []
        if errs:
            return errs
        os.makedirs(self.dir, exist_ok=True)
        target = self._path(f"{self.component}.json")
        if os.path.exists(target):
            hdir = self._path("history", self.component)
            os.makedirs(hdir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            who = re.sub(r"[^A-Za-z0-9_]", "", author) or "unknown"
            snap = os.path.join(hdir, f"{stamp}-{who}.json")
            n = 1
            while os.path.exists(snap):                       # two saves within one second: keep both
                n += 1
                snap = os.path.join(hdir, f"{stamp}-{who}{n}.json")
            with open(target, "rb") as src, open(snap, "wb") as dst:
                dst.write(src.read())
            snaps = sorted(os.listdir(hdir))
            for old in snaps[:-keep_history]:
                os.remove(os.path.join(hdir, old))
        out = {"schema_version": 1, **{k: v for k, v in data.items() if k != "schema_version"}}
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{self.component}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, target)
        self.refresh(force=True)
        return []

    # ---------------------------------------------------------- history, conflicts, restore
    def etag(self) -> str:
        """Fingerprint of the component file, for detecting edits made by someone else in the meantime."""
        import hashlib
        try:
            with open(self._path(f"{self.component}.json"), "rb") as f:
                return hashlib.sha1(f.read()).hexdigest()[:16]
        except OSError:
            return "none"

    def history(self) -> list[dict]:
        """Snapshots of earlier versions of the file, newest first."""
        hdir = self._path("history", self.component)
        out = []
        try:
            for name in sorted(os.listdir(hdir), reverse=True):
                if not name.endswith(".json"):
                    continue
                stamp, author = name[:15], name[16:-5]            # YYYYmmdd-HHMMSS-<author>.json
                st = os.stat(os.path.join(hdir, name))
                out.append({"name": name, "time": stamp, "author": author, "size": st.st_size})
        except OSError:
            pass
        return out

    def read_snapshot(self, name: str) -> dict | None:
        if "/" in name or "\\" in name or not name.endswith(".json"):
            return None
        path = os.path.join(self._path("history", self.component), name)
        if not os.path.isfile(path):
            return None
        return self._read_json(path)

    def restore(self, name: str, author: str = "ui") -> list[str]:
        """Make an earlier snapshot the current file again (the current one is snapshotted first)."""
        data = self.read_snapshot(name)
        if data is None:
            return [f"snapshot {name} not found or unreadable"]
        return self.save({k: v for k, v in data.items() if k != "schema_version"}, author=author)

    # ------------------------------------------------------------------ profiles
    def profiles(self) -> list[dict]:
        """All profiles in the order they are tried (by file name, first match wins)."""
        with self._lock:
            return copy.deepcopy(self._profiles)

    def _profile_file(self, name: str) -> str | None:
        return self._path("profiles", f"{name}.json") if PROFILE_NAME_RE.match(name or "") else None

    def _snapshot_profile(self, name: str, author: str) -> None:
        src = self._profile_file(name)
        if src and os.path.exists(src):
            hdir = self._path("history", "profiles", name)
            os.makedirs(hdir, exist_ok=True)
            who = re.sub(r"[^A-Za-z0-9_]", "", author) or "unknown"
            stamp, n = time.strftime("%Y%m%d-%H%M%S"), 1
            snap = os.path.join(hdir, f"{stamp}-{who}.json")
            while os.path.exists(snap):
                n += 1
                snap = os.path.join(hdir, f"{stamp}-{who}{n}.json")
            with open(src, "rb") as a, open(snap, "wb") as b:
                b.write(a.read())

    def save_profile(self, name: str, data: dict, author: str = "ui") -> list[str]:
        """Write profiles/<name>.json atomically (the old file is snapshotted). Structure is checked here; the values
        of the overrides are the caller's job (they belong to other components' schemas)."""
        path = self._profile_file(name)
        if not path:
            return [f"invalid profile name '{name}' (letters, digits, _ . - ; max 61 characters)"]
        errs = []
        match, over = data.get("match", {}), data.get("overrides")
        if not isinstance(match, dict) or not set(match) <= {"path_glob", "source_language", "target_language"}:
            errs.append("match: only path_glob, source_language and target_language are allowed")
        elif not all(isinstance(v, str) and v for v in match.values()):
            errs.append("match: values must be non-empty text")
        if not isinstance(over, dict) or not all(isinstance(v, dict) for v in over.values()):
            errs.append("overrides: expected {component: {setting: value}}")
        if errs:
            return errs
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._snapshot_profile(name, author)
        out = {"schema_version": 1, "name": data.get("name") or name, "match": match, "overrides": over}
        if data.get("description"):
            out["description"] = str(data["description"])
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=f".{name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
        self.refresh(force=True)
        return []

    def delete_profile(self, name: str, author: str = "ui") -> list[str]:
        path = self._profile_file(name)
        if not path or not os.path.exists(path):
            return [f"profile '{name}' not found"]
        self._snapshot_profile(name, author)           # kept in history/profiles/<name>/ so it can be put back
        os.remove(path)
        self.refresh(force=True)
        return []

    def seed_from_env(self) -> dict:
        """The JSON you would need to reproduce what the environment variables currently set (differences from defaults)."""
        out: dict = {}
        for path, env_name in self._env_map.items():
            if env_name in self._env and str(self._env[env_name]) != "":
                leaf = self._leaf[path]
                if leaf.get("x-secret"):
                    _set(out, path, "${%s}" % env_name)          # never write secrets into JSON
                    continue
                try:
                    val = coerce_env(str(self._env[env_name]), leaf)
                except (ValueError, TypeError):
                    continue
                if val != _get(self._defaults, path, _MISSING):
                    _set(out, path, val)
        return out


_MISSING = object()


def init_token(directory: str, rotate: bool = False) -> tuple[str, list[str]]:
    """Make the web UI editable: put WEBUI_TOKEN in secrets.env (kept if it already exists, unless `rotate`) and point
    `ui.token` in bridge.json at it. Returns (token, what was changed). Safe to run again."""
    import secrets
    os.makedirs(directory, exist_ok=True)
    changed = []
    env_path = os.path.join(directory, "secrets.env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    token = ""
    for ln in lines:
        if ln.strip().startswith("WEBUI_TOKEN="):
            token = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if rotate or not token:
        token = secrets.token_hex(24)
        lines = [ln for ln in lines if not ln.strip().startswith("WEBUI_TOKEN=")] + [f"WEBUI_TOKEN={token}"]
        with open(env_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        changed.append("secrets.env: WEBUI_TOKEN " + ("replaced" if rotate else "created"))
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    cfg_path = os.path.join(directory, "bridge.json")
    data: dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    if (data.get("ui") or {}).get("token") != "${WEBUI_TOKEN}":
        data.setdefault("schema_version", 1)
        data.setdefault("ui", {})["token"] = "${WEBUI_TOKEN}"
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".bridge.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, cfg_path)
        changed.append('bridge.json: ui.token = "${WEBUI_TOKEN}"')
    return token, changed


def main() -> None:                       # python -m subcfg ...
    import argparse
    ap = argparse.ArgumentParser(prog="subcfg")
    ap.add_argument("command", choices=["check", "seed", "show", "init-token"])
    ap.add_argument("component", nargs="?", help="npusubs, bridge, ... (not needed for init-token)")
    ap.add_argument("--dir", default=os.environ.get("SUBCFG_DIR", "/config"))
    ap.add_argument("--schemas", default="", help="extra schema directory")
    ap.add_argument("--rotate", action="store_true", help="init-token: replace an existing token")
    ap.add_argument("--url", default="http://<bridge-host>:9500", help="init-token: address of the bridge, for the login link")
    a = ap.parse_args()
    if a.command == "init-token":
        token, changed = init_token(a.dir, a.rotate)
        print("\n".join(changed) if changed else "Nothing to change: the token is already set up (use --rotate for a new one).")
        print(f"\nLog in to the web UI with this address (it sets a cookie, so you only need it once per browser):\n"
              f"  {a.url.rstrip('/')}/ui?token={token}\n\n"
              "Keep it private. If the bridge is already running, restart it once so it picks the token up.\n"
              "Editing also needs the config folder mounted read-write into the bridge.")
        return
    if not a.component:
        ap.error("component is required for this command")
    store = ConfigStore(a.component, a.dir, schema_dirs=(a.schemas,) if a.schemas else ())
    if a.command == "seed":
        print(json.dumps({"schema_version": 1, **store.seed_from_env()}, indent=2, ensure_ascii=False))
    elif a.command == "show":
        eff = store.masked()
        for path in sorted(store._leaf):
            key = ".".join(path)
            print(f"{key:40s} {json.dumps(_get(eff, path), ensure_ascii=False):40s} [{store.source(key)}]"
                  + (" (restart)" if store.needs_restart(key) else ""))
    else:
        print("OK" if not store.errors else "\n".join(store.errors))


if __name__ == "__main__":
    main()
