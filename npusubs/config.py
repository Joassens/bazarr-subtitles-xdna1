"""npusubs settings: defaults and validation come from the schema, values from /config/npusubs.json and the
environment (see subcfg). Everything except the model and thread count is re-read on the fly."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "subcfg"), "/opt/npusubs"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

import subcfg  # noqa: E402

SCHEMA_DIRS = tuple(d for d in (
    os.environ.get("SUBCFG_SCHEMA_DIR", ""),
    "/opt/npusubs/schemas",
    os.path.join(_HERE, "..", "config.example", "schemas"),
) if d)

store = subcfg.ConfigStore("npusubs", schema_dirs=SCHEMA_DIRS)


def get(path: str, default=None, context: dict | None = None):
    """Current value of a dotted setting (cheap; changed files are picked up within a few seconds)."""
    store.refresh()
    return store.get(path, default, context)


def secret(name: str) -> str:
    store.refresh()
    return store.secret(name)
