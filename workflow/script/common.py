"""Common helpers for the lightweight paper agent.

Provides:
    * Path constants for the ``workflow/`` workspace.
  * ``.env`` loading (config.env) into ``os.environ``.
  * Slug / citekey helpers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# workflow/script/common.py  ->  workflow/
RELATED_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = RELATED_DIR / "script"
PAPERS_DIR = RELATED_DIR / "papers"
DOC_DIR = RELATED_DIR / "doc"
GRAPH_DIR = RELATED_DIR / "graph"
INDEX_PATH = RELATED_DIR / "index.json"

# Relevance levels (see workflow/readme.md).
LEVELS = {
    1: "weak — borrowed in intro/method",
    2: "related — compared in Related Work",
    3: "high — baseline; comparison experiments / shared eval",
    4: "near-duplicate — same task or method (publication risk)",
}


def load_env(config_path: Path | None = None) -> dict[str, str]:
    """Load ``config.env`` (or ``.env``) into ``os.environ`` and return the keys read.

    Lines are ``KEY=VALUE``; ``#`` comments and blank lines are ignored.
    Existing environment variables are *not* overwritten.
    """
    loaded: dict[str, str] = {}
    candidates = [config_path] if config_path else [SCRIPT_DIR / "config.env", SCRIPT_DIR / ".env"]
    for path in candidates:
        if not path or not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            loaded[key] = value
            os.environ.setdefault(key, value)
        break
    return loaded


def slugify(text: str) -> str:
    """Turn arbitrary text into a filesystem/identifier-safe slug."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def ensure_dirs() -> None:
    for d in (PAPERS_DIR, DOC_DIR, GRAPH_DIR):
        d.mkdir(parents=True, exist_ok=True)
