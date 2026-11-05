"""Result saving with provenance."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "wildfire", "scripts"], cwd=ROOT)
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def results_dir() -> Path:
    """Committed full results live in results/; `make reproduce` sets
    WILDFIRE_OUT to a separate directory so it never touches them."""
    d = Path(os.environ.get("WILDFIRE_OUT", ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(name: str, payload: dict, seed, config: dict) -> Path:
    out = {
        "name": name,
        "seed": seed,
        "config": config,
        "git_hash": git_hash(),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": payload,
    }
    p = results_dir() / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True))
    return p


def load_result(name: str, base: Path | None = None) -> dict:
    p = (base or results_dir()) / f"{name}.json"
    return json.loads(p.read_text())


def smoke() -> bool:
    return os.environ.get("WILDFIRE_SMOKE", "0") == "1"
