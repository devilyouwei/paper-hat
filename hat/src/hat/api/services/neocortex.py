"""Curated-memory (Neocortex) management service.

Writes still go exclusively through the wake/sleep loop's ``WriteDecision``
contract (ADR-002); these helpers only edit or remove entries that were
already accepted, which is a manual-curation surface for the operator.
"""

from __future__ import annotations

from ...memory.curated.jsonl_store import JsonlNeocortex
from .container import get_loop


class NeocortexUnsupportedError(Exception):
    """Active neocortex backend does not support manual curation."""


class TraceNotFoundError(Exception):
    pass


def get_store() -> JsonlNeocortex:
    store = get_loop().neocortex
    if not isinstance(store, JsonlNeocortex):
        raise NeocortexUnsupportedError(
            "active neocortex store does not support manual curation"
        )
    return store


def _row_tag(row: dict) -> str | None:
    extras = ((row.get("metadata") or {}).get("extras")) or {}
    tag = extras.get("embed_model")
    return tag if isinstance(tag, str) else None


def list_entries(
    session_id: str | None = None, embed_model: str | None = None
) -> list[dict]:
    store = get_store()
    rows = (
        store.entries_by_session(session_id) if session_id else store.entries()
    )
    if embed_model is not None:
        rows = [r for r in rows if _row_tag(r) == embed_model]
    return rows


def get_entry(trace_id: str) -> dict:
    row = get_store().get_entry(trace_id)
    if row is None:
        raise TraceNotFoundError(trace_id)
    return row


def update_entry(
    trace_id: str, *, query: str | None, response: str | None
) -> dict:
    store = get_store()
    row = store.update(trace_id, query=query, response=response)
    if row is None:
        raise TraceNotFoundError(trace_id)
    # If the canonical query changed, the vector index must be re-embedded
    # for this trace; otherwise dedup keeps matching paraphrases of the
    # *old* query.
    if query is not None:
        loop = get_loop()
        if loop.deduper is not None:
            try:
                vec = loop.deduper.embedder.embed([query])[0]
                loop.deduper.index.update(trace_id, vec)
            except Exception:  # noqa: BLE001
                pass
    return row


def delete_entry(trace_id: str) -> None:
    store = get_store()
    if not store.delete(trace_id):
        raise TraceNotFoundError(trace_id)
    loop = get_loop()
    if loop.deduper is not None:
        try:
            loop.deduper.index.remove(trace_id)
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "NeocortexUnsupportedError",
    "TraceNotFoundError",
    "get_store",
    "list_entries",
    "get_entry",
    "update_entry",
    "delete_entry",
]
