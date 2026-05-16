"""Per-session raw chat history.

A *session* is a topic-scoped conversation, in the spirit of ChatGPT-style
chat tools. Sessions live under ``settings.raw_root/sessions/<id>.jsonl``
(one :class:`Interaction` per line) plus a single ``index.json`` that holds
session metadata (id, title, timestamps, message count).

This module is the **only** writer of raw chat data. ADR-002 still applies:
no training pipeline reads from here. The Hippocampus Agent / wake step is
the only path that promotes an interaction into the curated Neocortex.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel, Field

from ...core.schemas import Interaction


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    # short prefixed id, easier to spot in URLs and the filesystem.
    return f"s-{uuid4().hex[:12]}"


class Session(BaseModel):
    """Chat-session metadata. Stored in ``index.json`` as one entry per session."""

    id: str = Field(default_factory=_new_id)
    title: str = "New chat"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    message_count: int = 0
    # Optional model info captured on the first turn for UI display.
    model: str | None = None


class SessionStoreError(RuntimeError):
    pass


class JsonlSessionStore:
    """Filesystem-backed session store.

    Layout::

        <root>/
        ├── index.json            # list[Session]
        └── sessions/
            └── <session_id>.jsonl  # one Interaction per line

    All public methods take a lock so concurrent FastAPI workers don't corrupt
    the index. Writes are atomic via ``os.replace``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.index_path = self.root / "index.json"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ----- index --------------------------------------------------------

    def _load_index(self) -> list[Session]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return [Session.model_validate(row) for row in data]

    def _save_index(self, sessions: list[Session]) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        payload = [s.model_dump(mode="json") for s in sessions]
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.index_path)

    def _session_path(self, session_id: str) -> Path:
        # guard against path traversal
        if "/" in session_id or ".." in session_id:
            raise SessionStoreError(f"invalid session id {session_id!r}")
        return self.sessions_dir / f"{session_id}.jsonl"

    # ----- public API ---------------------------------------------------

    def list(self) -> list[Session]:
        with self._lock:
            sessions = self._load_index()
        # newest first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def latest(self) -> Session | None:
        sessions = self.list()
        return sessions[0] if sessions else None

    def get(self, session_id: str) -> Session:
        with self._lock:
            for s in self._load_index():
                if s.id == session_id:
                    return s
        raise SessionStoreError(f"unknown session {session_id!r}")

    def create(self, title: str | None = None, model: str | None = None) -> Session:
        s = Session(title=title or "New chat", model=model)
        with self._lock:
            sessions = self._load_index()
            sessions.append(s)
            self._save_index(sessions)
        # touch the messages file so the directory listing is consistent.
        self._session_path(s.id).touch(exist_ok=True)
        return s

    def rename(self, session_id: str, title: str) -> Session:
        title = title.strip() or "New chat"
        with self._lock:
            sessions = self._load_index()
            for i, s in enumerate(sessions):
                if s.id == session_id:
                    s.title = title
                    s.updated_at = _now()
                    sessions[i] = s
                    self._save_index(sessions)
                    return s
        raise SessionStoreError(f"unknown session {session_id!r}")

    def delete(self, session_id: str) -> bool:
        with self._lock:
            sessions = self._load_index()
            kept = [s for s in sessions if s.id != session_id]
            if len(kept) == len(sessions):
                return False
            self._save_index(kept)
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        return True

    def append(self, session_id: str, interaction: Interaction) -> Session:
        """Append an interaction line and bump the session's metadata."""
        path = self._session_path(session_id)
        line = interaction.model_dump_json() + "\n"
        with self._lock:
            sessions = self._load_index()
            target_idx: int | None = None
            for i, s in enumerate(sessions):
                if s.id == session_id:
                    target_idx = i
                    break
            if target_idx is None:
                raise SessionStoreError(f"unknown session {session_id!r}")
            target = sessions[target_idx]
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
            target.message_count += 1
            target.updated_at = _now()
            sessions[target_idx] = target
            self._save_index(sessions)
            return target

    def messages(self, session_id: str) -> list[Interaction]:
        path = self._session_path(session_id)
        if not path.exists():
            # ensure the session exists in the index
            self.get(session_id)
            return []
        out: list[Interaction] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Interaction.model_validate_json(line))
        return out

    # ----- compat with the legacy RawInteractionLog API ---------------

    def append_anonymous(self, interaction: Interaction) -> Session:
        """Append to a synthetic 'default' session — kept for callers that
        don't track session ids yet (CLI smoke tests, /chat, etc.)."""
        sid = "default"
        try:
            self.get(sid)
        except SessionStoreError:
            with self._lock:
                sessions = self._load_index()
                # insert with a stable id so subsequent calls reuse it.
                default = Session(id=sid, title="Default")
                sessions.append(default)
                self._save_index(sessions)
            self._session_path(sid).touch(exist_ok=True)
        return self.append(sid, interaction)


__all__ = [
    "Session",
    "SessionStoreError",
    "JsonlSessionStore",
]
