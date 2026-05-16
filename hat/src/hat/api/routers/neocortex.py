"""Curated-memory (Neocortex) management API.

Endpoints (mounted under ``/api/neocortex``):

* ``GET    /``               — list every curated entry
* ``GET    /{trace_id}``     — fetch a single entry
* ``PATCH  /{trace_id}``     — edit query / response / score
* ``DELETE /{trace_id}``     — drop the entry from both the SFT file and sidecar

Writes still go exclusively through the wake/sleep loop's ``WriteDecision``
contract (ADR-002); these endpoints only edit or remove entries that were
already accepted, which is a manual-curation surface for the operator.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...memory.curated.jsonl_store import JsonlNeocortex
from ..deps import get_loop
from ..schemas.neocortex import (
    NeocortexEntry,
    NeocortexEntryUpdate,
    NeocortexList,
)

router = APIRouter()


def get_neocortex() -> JsonlNeocortex:
    store = get_loop().neocortex
    if not isinstance(store, JsonlNeocortex):
        raise HTTPException(
            501, "active neocortex store does not support manual curation"
        )
    return store


@router.get("", response_model=NeocortexList)
def list_entries(
    session_id: str | None = Query(default=None),
    store: JsonlNeocortex = Depends(get_neocortex),
) -> NeocortexList:
    rows = (
        store.entries_by_session(session_id) if session_id else store.entries()
    )
    return NeocortexList(data=[NeocortexEntry.from_row(r) for r in rows])


@router.get("/{trace_id}", response_model=NeocortexEntry)
def get_entry(
    trace_id: str, store: JsonlNeocortex = Depends(get_neocortex)
) -> NeocortexEntry:
    row = store.get_entry(trace_id)
    if row is None:
        raise HTTPException(404, f"unknown trace {trace_id!r}")
    return NeocortexEntry.from_row(row)


@router.patch("/{trace_id}", response_model=NeocortexEntry)
def patch_entry(
    trace_id: str,
    req: NeocortexEntryUpdate,
    store: JsonlNeocortex = Depends(get_neocortex),
) -> NeocortexEntry:
    row = store.update(
        trace_id, query=req.query, response=req.response
    )
    if row is None:
        raise HTTPException(404, f"unknown trace {trace_id!r}")
    return NeocortexEntry.from_row(row)


@router.delete("/{trace_id}")
def delete_entry(
    trace_id: str, store: JsonlNeocortex = Depends(get_neocortex)
) -> dict:
    if not store.delete(trace_id):
        raise HTTPException(404, f"unknown trace {trace_id!r}")
    return {"deleted": trace_id}
