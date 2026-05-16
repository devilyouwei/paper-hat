"""Schemas for the curated-memory (Neocortex) management API.

Each entry is the on-disk SFT row plus a derived ``query`` / ``response``
projection so the UI can render and edit it without re-parsing the messages
list."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _user(messages: list[dict[str, Any]]) -> str:
    return next((m.get("content", "") for m in messages if m.get("role") == "user"), "")


def _assistant(messages: list[dict[str, Any]]) -> str:
    return next(
        (m.get("content", "") for m in messages if m.get("role") == "assistant"), ""
    )


class NeocortexEntry(BaseModel):
    """A single curated memory row, in a UI-friendly shape."""

    trace_id: str
    interaction_id: str | None = None
    session_id: str | None = None
    interaction_ids: list[str] = Field(default_factory=list)
    query: str
    response: str
    score: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "NeocortexEntry":
        msgs = row.get("messages") or []
        return cls(
            trace_id=row.get("trace_id") or "",
            interaction_id=row.get("interaction_id"),
            session_id=row.get("session_id"),
            interaction_ids=list(row.get("interaction_ids") or []),
            query=_user(msgs),
            response=_assistant(msgs),
            score=float(row.get("score") or 0.0),
            signals=row.get("signals") or {},
            metadata=row.get("metadata") or {},
        )


class NeocortexList(BaseModel):
    object: str = "list"
    data: list[NeocortexEntry]


class NeocortexEntryUpdate(BaseModel):
    """Patch payload. Any field omitted leaves the existing value untouched.

    Score is intentionally NOT editable: it is derived from the cortex's
    uncertainty at write time and rewriting it would falsify the training
    signal. Operators who disagree with a trace should edit or delete it.
    """

    query: str | None = None
    response: str | None = None
