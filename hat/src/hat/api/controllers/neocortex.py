"""Curated-memory (Neocortex) endpoints (mounted at ``/api/neocortex``).

* ``GET    /``               — list every curated entry
* ``GET    /{trace_id}``     — fetch a single entry
* ``PATCH  /{trace_id}``     — edit query / response / score
* ``DELETE /{trace_id}``     — drop the entry from both the SFT file and sidecar
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..schemas.neocortex import (
    NeocortexEntry,
    NeocortexEntryUpdate,
    NeocortexList,
)
from ..services import neocortex as svc

router = APIRouter()


@router.get("", response_model=NeocortexList)
def list_entries(
    session_id: str | None = Query(default=None),
    embed_model: str | None = Query(
        default=None,
        description=(
            "Filter rows by the embedder that wrote them "
            "(``metadata.extras.embed_model`` == this value)."
        ),
    ),
) -> NeocortexList:
    try:
        rows = svc.list_entries(session_id=session_id, embed_model=embed_model)
    except svc.NeocortexUnsupportedError as e:
        raise HTTPException(501, str(e)) from e
    return NeocortexList(data=[NeocortexEntry.from_row(r) for r in rows])


@router.get("/{trace_id}", response_model=NeocortexEntry)
def get_entry(trace_id: str) -> NeocortexEntry:
    try:
        row = svc.get_entry(trace_id)
    except svc.NeocortexUnsupportedError as e:
        raise HTTPException(501, str(e)) from e
    except svc.TraceNotFoundError as e:
        raise HTTPException(404, f"unknown trace {str(e)!r}") from e
    return NeocortexEntry.from_row(row)


@router.patch("/{trace_id}", response_model=NeocortexEntry)
def patch_entry(trace_id: str, req: NeocortexEntryUpdate) -> NeocortexEntry:
    try:
        row = svc.update_entry(
            trace_id, query=req.query, response=req.response
        )
    except svc.NeocortexUnsupportedError as e:
        raise HTTPException(501, str(e)) from e
    except svc.TraceNotFoundError as e:
        raise HTTPException(404, f"unknown trace {str(e)!r}") from e
    return NeocortexEntry.from_row(row)


@router.delete("/{trace_id}")
def delete_entry(trace_id: str) -> dict:
    try:
        svc.delete_entry(trace_id)
    except svc.NeocortexUnsupportedError as e:
        raise HTTPException(501, str(e)) from e
    except svc.TraceNotFoundError as e:
        raise HTTPException(404, f"unknown trace {str(e)!r}") from e
    return {"deleted": trace_id}
