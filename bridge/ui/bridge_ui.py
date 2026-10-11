"""Web UI for the bridge: dashboard, backends, configuration editor with history (M1-M3).

Served by the bridge's own FastAPI app, no build step and no extra dependencies (httpx is already used by the bridge).
Reading is open on the LAN unless `ui.token` is set. Writing is only possible when a token is set, the config folder
is writable, and the request carries the token plus an `X-Requested-With: ui` header (CSRF guard).
Secrets (x-secret settings) are masked and can not be edited here: they belong in secrets.env.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import hmac
import os
import re
import time

import httpx
from fastapi import APIRouter, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

HERE = os.path.dirname(os.path.abspath(__file__))
COOKIE = "subtitle_ui"
_cache: dict = {"t": 0.0, "data": None}
_stores: dict = {}
CACHE_S = 4.0
COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


# ------------------------------------------------------------------------------------------- probes
def _probe(url: str, timeout: float = 3.0) -> dict:
    t0 = time.monotonic()
    try:
        r = httpx.get(url, timeout=timeout)
        data = None
        try:
            data = r.json()
        except Exception:                                   # noqa: BLE001
            pass
        return {"ok": r.status_code < 500, "http": r.status_code, "ms": round((time.monotonic() - t0) * 1000),
                "json": data, "error": None}
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "http": None, "ms": None, "json": None, "error": type(exc).__name__}


def _speech(url: str) -> dict:
    h = _probe(url + "/health")
    info = {"url": url, "kind": "unreachable", "ok": False, "ms": h["ms"], "detail": {}}
    if h["http"] == 200 and isinstance(h["json"], dict) and "status" in h["json"]:        # npusubs
        d = h["json"]
        info.update(kind="npusubs", ok=d.get("status") == "ok", detail={
            "status": d.get("status"), "model": d.get("model"), "version": d.get("version"), "beam": d.get("beam"),
            "vad": d.get("vad"), "diarization": d.get("diarization"), "requests": d.get("requests"),
            "current": d.get("current"), "config_errors": (d.get("config") or {}).get("errors") or []})
        return info
    r = _probe(url + "/")
    if r["ok"]:
        info.update(kind="whisper-asr-webservice", ok=True, ms=r["ms"])
    else:
        info["error"] = r["error"] or f"HTTP {r['http']}"
    return info


def _llm(role: str, base: str | None, model: str | None) -> dict:
    if not base:
        return {"role": role, "configured": False}
    base = base.rstrip("/")
    m = _probe(base + "/models")
    out = {"role": role, "configured": True, "url": base, "model": model, "ok": m["http"] == 200, "ms": m["ms"],
           "error": None if m["http"] == 200 else (m["error"] or f"HTTP {m['http']}"), "available": [], "loaded": []}
    if isinstance(m["json"], dict) and isinstance(m["json"].get("data"), list):
        out["available"] = [str(x.get("id")) for x in m["json"]["data"] if isinstance(x, dict)][:12]
    if out["ok"] and base.endswith("/api/v1"):                                            # Lemonade: loaded models
        h = _probe(base + "/health")
        if isinstance(h["json"], dict):
            models = [x for x in h["json"].get("all_models_loaded", []) if isinstance(x, dict)]
            out["loaded"] = [str(x.get("model_name")) for x in models]
            out["loaded_llm"] = [str(x.get("model_name")) for x in models if x.get("type", "llm") == "llm"]
    return out


# ------------------------------------------------------------------------------------- config helpers
def _components(cfg) -> list[str]:
    names = {os.path.basename(p)[: -len(".schema.json")] for p in glob.glob(os.path.join(cfg.dir, "schemas", "*.schema.json"))}
    return sorted(names)


def _get_store(st: dict, component: str):
    cfg = st["cfg"]
    if cfg is None or not COMPONENT_RE.match(component) or component not in _components(cfg):
        return None
    if component == cfg.component:
        return cfg
    if component not in _stores:
        import subcfg
        _stores[component] = subcfg.ConfigStore(component, cfg.dir)
    return _stores[component]


def _can_write(st: dict) -> bool:
    cfg = st["cfg"]
    return bool(st["token"]) and cfg is not None and os.access(cfg.dir, os.W_OK)


def _write_hint(st: dict) -> str | None:
    """Why editing is off (None when it is on). Shown in the UI so a newcomer knows what to do."""
    cfg = st["cfg"]
    if cfg is None:
        return "The configuration layer is off: there is no config folder to edit."
    if not st["token"]:
        return ("No access token is set. Create one with  python subcfg/subcfg.py init-token --dir <your config folder>"
                "  (or see docs/web-ui-getting-started.md), then restart the bridge and open the login link it prints.")
    if not os.access(cfg.dir, os.W_OK):
        return "The config folder is mounted read-only. Mount it read-write into the bridge (remove :ro) and restart it."
    return None


def _set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _del_path(d: dict, dotted: str) -> None:
    keys = dotted.split(".")
    chain = [d]
    for k in keys[:-1]:
        if not isinstance(chain[-1].get(k), dict):
            return
        chain.append(chain[-1][k])
    chain[-1].pop(keys[-1], None)
    for i in range(len(keys) - 1, 0, -1):                     # prune sections that became empty
        if not chain[i]:
            chain[i - 1].pop(keys[i - 1], None)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k == "schema_version":
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _diff(old: dict, new: dict) -> list[dict]:
    a, b = _flatten(old), _flatten(new)
    return [{"key": k, "old": a.get(k, None), "new": b.get(k, None), "old_set": k in a, "new_set": k in b}
            for k in sorted(set(a) | set(b)) if a.get(k, object()) != b.get(k, object())]


# -------------------------------------------------------------------------------------------- status
def _build_status(st: dict) -> dict:
    cfg = st["cfg"]
    llm_cfg = st["llm"]
    jobs = []
    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        f_speech = [pool.submit(_speech, u) for u in st["speech_urls"]]
        f_llm = [pool.submit(_llm, "dedicated", llm_cfg["dedicated"], llm_cfg["model_dedicated"]),
                 pool.submit(_llm, "primary", llm_cfg["primary"], llm_cfg["model"]),
                 pool.submit(_llm, "fallback", llm_cfg["fallback"], llm_cfg["model_fallback"])]
        f_arr = [pool.submit(lambda n, u: {"name": n, "url": u, **{k: v for k, v in _probe(u + "/ping").items() if k in ("ok", "ms")}}, n, u)
                 for n, u in (("Sonarr", st["sonarr_url"]), ("Radarr", st["radarr_url"])) if u]
        speech = [f.result() for f in f_speech]
        llm = [f.result() for f in f_llm]
        arr = [f.result() for f in f_arr]
    speech_use = next((i for i, s in enumerate(speech) if s["ok"]), None)
    by_role = {x["role"]: x for x in llm}
    pick = None
    if by_role["dedicated"].get("ok"):
        pick = "dedicated"
    else:
        p, f = by_role["primary"], by_role["fallback"]
        if f.get("ok") and (not p.get("ok") or (f["ms"] or 1e9) < (p["ms"] or 1e9)):
            pick = "fallback"
        elif p.get("configured"):
            pick = "primary"
    now = time.time()
    for j in st["jobs"]:
        jobs.append({**j, "age_s": round(now - j["started"]), "running_s": round(now - j["started"]) if j["status"] == "running" else j["seconds"]})
    files = []
    if cfg is not None:
        for name in ("bridge.json", "npusubs.json", "settings.json", "secrets.env"):
            files.append({"name": name, "present": os.path.exists(os.path.join(cfg.dir, name))})
    guard = None
    if callable(st.get("guard")):
        guard = st["guard"]()
        guard["llm_loaded"] = sorted({m for x in llm for m in (x.get("loaded_llm") or [])})
    return {
        "time": now, "guard": guard,
        "bridge": {"version": st["version"], "uptime_s": round(now - st["started"]), "flags": st["flags"],
                   "prefer_native": st["prefer_native"],
                   "config": None if cfg is None else {"dir": cfg.dir, "errors": cfg.errors, "files": files}},
        "speech": speech, "speech_in_use": speech_use, "llm": llm, "llm_in_use": pick, "arr": arr, "jobs": jobs[:30],
    }


# ---------------------------------------------------------------------------------------------- routes
def _validate_overrides(st: dict, flat: dict) -> tuple[dict, list[str]]:
    """{component: {dotted.key: value}} -> ({component: nested}, errors). Unknown and secret settings are refused."""
    import subcfg
    nested_all, errs = {}, []
    if not isinstance(flat, dict) or not all(isinstance(v, dict) for v in flat.values()):
        return {}, ["overrides: expected {component: {setting: value}}"]
    for comp, items in flat.items():
        store = _get_store(st, comp)
        if store is None:
            errs.append(f"{comp}: unknown component")
            continue
        nested = {}
        for key, value in items.items():
            leaf = store._leaf.get(tuple(str(key).split(".")))
            if leaf is None:
                errs.append(f"{comp}.{key}: unknown setting")
            elif leaf.get("x-secret"):
                errs.append(f"{comp}.{key}: secrets can not be overridden")
            else:
                _set_path(nested, key, value)
        errs += [f"{comp}.{e}" for e in subcfg.validate(nested, store.schema)]
        if nested:
            nested_all[comp] = nested
    return nested_all, errs


def install(app, get_state, on_change: dict | None = None, playground=None, play_limits: dict | None = None) -> None:
    router = APIRouter()
    on_change = on_change or {}
    play_limits = play_limits or {"lines": 300, "audio_mb": 60}

    def check(request: Request, write: bool = False, needs_folder: bool = True):
        st = get_state()
        if not st["enabled"]:
            return JSONResponse({"error": "web UI disabled (ui.enabled=false)"}, status_code=404)
        token = st["token"]
        if token:
            # any of them may be the right one: a stale cookie must not beat a fresh ?token=
            given = (request.query_params.get("token"), request.headers.get("x-token"), request.cookies.get(COOKIE))
            if not any(g and hmac.compare_digest(g.encode(), token.encode()) for g in given):
                return JSONResponse({"error": "token required: open /ui?token=..."}, status_code=401)
        if write:
            if not token:
                return JSONResponse({"error": "editing needs ui.token to be set (WEBUI_TOKEN in secrets.env)"}, status_code=403)
            if request.headers.get("x-requested-with") != "ui":
                return JSONResponse({"error": "missing X-Requested-With: ui header"}, status_code=403)
            if needs_folder and not _can_write(st):
                return JSONResponse({"error": "the config folder is read-only: mount it read-write into the bridge"}, status_code=403)
        return None

    @router.get("/ui", response_class=HTMLResponse)
    def ui(request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            resp = HTMLResponse(f.read())
        token = get_state()["token"]
        if token and request.query_params.get("token") == token:
            resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=30 * 86400)
        return resp

    @router.get("/ui/api/status")
    def status(request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        now = time.monotonic()
        if _cache["data"] is None or now - _cache["t"] > CACHE_S:
            _cache["data"], _cache["t"] = _build_status(get_state()), now
        return {**_cache["data"], "can_write": _can_write(get_state()), "write_hint": _write_hint(get_state())}

    @router.get("/ui/api/config")
    def config(request: Request, component: str = "bridge"):
        denied = check(request)
        if denied is not None:
            return denied
        st = get_state()
        if st["cfg"] is None:
            return {"enabled": False}
        store = _get_store(st, component)
        if store is None:
            return JSONResponse({"error": f"unknown component '{component}'"}, status_code=404)
        store.refresh()
        secret = {".".join(p) for p, leaf in store._leaf.items() if leaf.get("x-secret")}
        in_file = {k: v for k, v in _flatten(store.raw_file()).items() if k not in secret}
        return {"enabled": True, "component": component, "components": _components(st["cfg"]), "dir": store.dir,
                "errors": store.errors, "etag": store.etag(), "can_write": _can_write(st), "write_hint": _write_hint(st),
                "rows": store.describe(),
                "file": in_file}

    @router.post("/ui/api/config/{component}")
    def save(component: str, request: Request, body: dict = Body(...)):
        denied = check(request, write=True)
        if denied is not None:
            return denied
        st = get_state()
        store = _get_store(st, component)
        if store is None:
            return JSONResponse({"error": f"unknown component '{component}'"}, status_code=404)
        store.refresh()
        if body.get("etag") not in (None, store.etag()):
            return JSONResponse({"error": "The file changed since you loaded it. Reload and try again."}, status_code=409)
        changes, remove = body.get("changes") or {}, body.get("remove") or []
        if not isinstance(changes, dict) or not isinstance(remove, list):
            return JSONResponse({"error": "changes must be an object and remove a list"}, status_code=400)
        errs = []
        for key in list(changes) + list(remove):
            leaf = store._leaf.get(tuple(str(key).split(".")))
            if leaf is None:
                errs.append(f"{key}: unknown setting")
            elif leaf.get("x-secret"):
                errs.append(f"{key}: secrets are set in secrets.env, not here")
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        raw = store.raw_file()
        for key, value in changes.items():
            _set_path(raw, key, value)
        for key in remove:
            _del_path(raw, key)
        try:
            errs = store.save(raw, author="ui")
        except OSError as exc:
            return JSONResponse({"error": f"could not write the config file: {exc}"}, status_code=500)
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        hook = on_change.get(component)
        if hook:
            hook()
        restart = [k for k in list(changes) + list(remove) if store.needs_restart(k)]
        return {"ok": True, "etag": store.etag(), "restart": restart, "errors": store.errors}

    @router.get("/ui/api/history/{component}")
    def history(component: str, request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        store = _get_store(get_state(), component)
        if store is None:
            return JSONResponse({"error": f"unknown component '{component}'"}, status_code=404)
        return {"component": component, "items": store.history()}

    @router.get("/ui/api/history/{component}/{name}")
    def snapshot(component: str, name: str, request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        store = _get_store(get_state(), component)
        snap = store.read_snapshot(name) if store else None
        if snap is None:
            return JSONResponse({"error": "snapshot not found"}, status_code=404)
        return {"name": name, "snapshot": snap, "diff": _diff(snap, store.raw_file())}

    @router.post("/ui/api/history/{component}/{name}/restore")
    def restore(component: str, name: str, request: Request):
        denied = check(request, write=True)
        if denied is not None:
            return denied
        store = _get_store(get_state(), component)
        if store is None:
            return JSONResponse({"error": f"unknown component '{component}'"}, status_code=404)
        try:
            errs = store.restore(name, author="ui-restore")
        except OSError as exc:
            return JSONResponse({"error": f"could not write the config file: {exc}"}, status_code=500)
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        hook = on_change.get(component)
        if hook:
            hook()
        return {"ok": True, "etag": store.etag()}

    # ----------------------------------------------------------------------------------------- jobs
    @router.get("/ui/api/jobs/{job_id}")
    def job_detail(job_id: int, request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        job = get_state().get("get_job", lambda _i: None)(job_id)
        if job is None:
            return JSONResponse({"error": "job not found (the bridge keeps the last 50 requests)"}, status_code=404)
        srt = job.pop("srt", None)
        detail = job.pop("detail", None)
        return {"job": job, "detail": detail, "srt": srt,
                "note": None if (srt is not None or detail is not None) else "Details are kept for the 10 most recent jobs only."}

    @router.get("/ui/api/jobs/{job_id}/srt")
    def job_srt(job_id: int, request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        job = get_state().get("get_job", lambda _i: None)(job_id)
        if job is None or job.get("srt") is None:
            return JSONResponse({"error": "no subtitle kept for this job"}, status_code=404)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(job.get("file") or f"job{job_id}")[0]) or f"job{job_id}"
        return PlainTextResponse(job["srt"], headers={"Content-Disposition": f'attachment; filename="{name}.srt"'})

    # ------------------------------------------------------------------------------------ profiles
    def _profile_view(p: dict) -> dict:
        return {"file": p.get("_file"), "name": p.get("name"), "description": p.get("description", ""),
                "match": p.get("match") or {},
                "overrides": {c: _flatten(v) for c, v in (p.get("overrides") or {}).items() if isinstance(v, dict)}}

    @router.get("/ui/api/profiles")
    def profiles(request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        st = get_state()
        if st["cfg"] is None:
            return {"enabled": False}
        st["cfg"].refresh()
        settings = {}
        for comp in _components(st["cfg"]):
            store = _get_store(st, comp)
            settings[comp] = [{k: r[k] for k in ("key", "section", "title", "description", "type", "default", "enum", "min", "max")}
                              for r in store.describe() if not r["secret"]]
        return {"enabled": True, "can_write": _can_write(st), "write_hint": _write_hint(st), "dir": st["cfg"].dir,
                "profiles": [_profile_view(p) for p in st["cfg"].profiles()], "settings": settings}

    @router.get("/ui/api/profiles/match")
    def profile_match(request: Request, path: str = "", source_language: str = "", target_language: str = ""):
        denied = check(request)
        if denied is not None:
            return denied
        cfg = get_state()["cfg"]
        if cfg is None:
            return {"match": None}
        cfg.refresh()
        ctx = {"path": path, "source_language": source_language, "target_language": target_language}
        prof = cfg.match_profile(ctx)
        return {"match": _profile_view(prof) if prof else None, "context": ctx}

    @router.post("/ui/api/profiles/{name}")
    def profile_save(name: str, request: Request, body: dict = Body(...)):
        denied = check(request, write=True)
        if denied is not None:
            return denied
        st = get_state()
        nested, errs = _validate_overrides(st, body.get("overrides") or {})
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        try:
            errs = st["cfg"].save_profile(name, {"name": name, "description": body.get("description", ""),
                                                 "match": body.get("match") or {}, "overrides": nested}, author="ui")
        except OSError as exc:
            return JSONResponse({"error": f"could not write the profile: {exc}"}, status_code=500)
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        for hook in (on_change.get("profiles"),):
            if hook:
                hook()
        return {"ok": True}

    @router.post("/ui/api/profiles/{name}/delete")
    def profile_delete(name: str, request: Request):
        denied = check(request, write=True)
        if denied is not None:
            return denied
        errs = get_state()["cfg"].delete_profile(name, author="ui")
        if errs:
            return JSONResponse({"errors": errs}, status_code=404)
        if on_change.get("profiles"):
            on_change["profiles"]()
        return {"ok": True}

    # ---------------------------------------------------------------------------------- playground
    @router.get("/ui/api/playground")
    def play_list(request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        st = get_state()
        return {"available": playground is not None, "can_run": bool(st["token"]) and playground is not None,
                "write_hint": _write_hint(st),
                "limits": play_limits, "runs": playground.list() if playground else []}

    @router.get("/ui/api/playground/{rid}")
    def play_get(rid: str, request: Request):
        denied = check(request)
        if denied is not None:
            return denied
        run = playground.get(rid) if playground else None
        if run is None:
            return JSONResponse({"error": "run not found (runs are kept in memory until the bridge restarts)"}, status_code=404)
        return run

    def _variants(raw, kind: str, st: dict):
        if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
            return None, ["give 1 to 4 variants"]
        out, errs = [], []
        for i, v in enumerate(raw):
            name = str(v.get("name") or f"Variant {i + 1}")[:40] if isinstance(v, dict) else None
            if name is None:
                errs.append(f"variant {i + 1}: expected an object")
                continue
            if kind == "translate":
                nested, e = _validate_overrides(st, {"bridge": v.get("overrides") or {}})
                errs += [f"{name}: {x}" for x in e]
                out.append({"name": name, "overrides": nested.get("bridge", {})})
            else:
                p, params = v.get("params") or {}, {}
                if p.get("beam_size") not in (None, ""):
                    if not isinstance(p["beam_size"], int) or not 1 <= p["beam_size"] <= 10:
                        errs.append(f"{name}: beam_size must be 1-10")
                    else:
                        params["beam_size"] = p["beam_size"]
                for flag in ("vad_filter", "diarize"):
                    if p.get(flag) not in (None, ""):
                        params[flag] = "true" if p[flag] in (True, "true") else "false"
                out.append({"name": name, "params": params})
        return out, errs

    def _start(kind: str, payload: dict):
        try:
            return playground.start(kind, payload)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    @router.post("/ui/api/playground/translate")
    def play_translate(request: Request, body: dict = Body(...)):
        denied = check(request, write=True, needs_folder=False)
        if denied is not None:
            return denied
        if playground is None:
            return JSONResponse({"error": "playground not available"}, status_code=404)
        variants, errs = _variants(body.get("variants"), "translate", get_state())
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        srt = str(body.get("srt") or "")
        if len(srt) > 400_000 or not srt.strip():
            return JSONResponse({"error": "paste an SRT (at most 400 kB)"}, status_code=422)
        run = _start("translate", {"label": str(body.get("label") or "Translation")[:60], "srt": srt, "variants": variants,
                                   "language": str(body.get("language") or "")[:8], "video_file": str(body.get("video_file") or "")[:500]})
        return run if isinstance(run, JSONResponse) else {"id": run["id"]}

    @router.post("/ui/api/playground/transcribe")
    async def play_transcribe(request: Request):
        denied = check(request, write=True, needs_folder=False)
        if denied is not None:
            return denied
        if playground is None:
            return JSONResponse({"error": "playground not available"}, status_code=404)
        import json as _json
        form = await request.form()
        upload = form.get("file")
        try:
            meta = _json.loads(form.get("meta") or "{}")
        except ValueError:
            return JSONResponse({"error": "meta must be JSON"}, status_code=400)
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "attach an audio or video clip as 'file'"}, status_code=422)
        audio = await upload.read()
        if len(audio) > play_limits["audio_mb"] * 1048576:
            return JSONResponse({"error": f"clip larger than {play_limits['audio_mb']} MB"}, status_code=413)
        variants, errs = _variants(meta.get("variants"), "transcribe", get_state())
        if errs:
            return JSONResponse({"errors": errs}, status_code=422)
        run = _start("transcribe", {"label": str(meta.get("label") or upload.filename or "Transcription")[:60], "audio": audio,
                                    "variants": variants, "language": str(meta.get("language") or "")[:8]})
        return run if isinstance(run, JSONResponse) else {"id": run["id"]}

    app.include_router(router)
