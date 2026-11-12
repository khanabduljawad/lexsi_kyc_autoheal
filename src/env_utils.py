"""Lightweight, zero-dependency .env loader for local AutoHeal-KYC CLI entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(env_path: Path | None = None, *, override: bool = False) -> Path | None:
    """Load key=value pairs from project .env into os.environ without printing secrets.

    Explicitly exported environment variables win by default (``override=False``).
    """
    target = Path(env_path) if env_path is not None else (PROJECT_ROOT / ".env")
    if not target.is_file():
        return None

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if override or key not in os.environ or not os.environ[key]:
            os.environ[key] = val

    return target
