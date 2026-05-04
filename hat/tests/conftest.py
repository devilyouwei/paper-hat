from __future__ import annotations

import pytest

from hat.api import deps
from hat.config.settings import get_settings


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect every persistent path to a per-test tmp dir.

    Prevents tests from accumulating chat sessions, neocortex training rows,
    or HF cache state in the repo's ``runs/`` and ``model/`` trees.
    """
    s = get_settings()
    monkeypatch.setattr(s, "raw_root", tmp_path / "raw")
    monkeypatch.setattr(s, "raw_log_path", tmp_path / "raw_log.jsonl")
    monkeypatch.setattr(s, "neocortex_path", tmp_path / "neocortex" / "train.jsonl")
    monkeypatch.setattr(
        s, "neocortex_traces_path", tmp_path / "neocortex" / "traces.jsonl"
    )
    # Clear any singletons populated by prior tests.
    deps.get_session_store.cache_clear()
    deps.get_raw_log.cache_clear()
    deps.get_loop.cache_clear()
    deps._initial_cortex.cache_clear()
    yield
    deps.get_session_store.cache_clear()
    deps.get_raw_log.cache_clear()
    deps.get_loop.cache_clear()
