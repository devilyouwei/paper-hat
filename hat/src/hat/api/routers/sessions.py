"""Chat-session management.

Endpoints:

* ``GET    /api/sessions``               — list all sessions (newest first)
* ``POST   /api/sessions``               — create a new (empty) session
* ``GET    /api/sessions/{id}``          — session metadata + full message log
* ``PATCH  /api/sessions/{id}``          — rename a session
* ``DELETE /api/sessions/{id}``          — delete a session and its log
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...memory.raw.sessions import SessionStoreError
from ..deps import get_session_store
from ..schemas.sessions import (
    Session,
    SessionCreateRequest,
    SessionList,
    SessionMessages,
    SessionRenameRequest,
)

router = APIRouter()


@router.get("", response_model=SessionList)
def list_sessions(store=Depends(get_session_store)) -> SessionList:
    return SessionList(data=store.list())


@router.post("", response_model=Session)
def create_session(
    req: SessionCreateRequest | None = None,
    store=Depends(get_session_store),
) -> Session:
    req = req or SessionCreateRequest()
    return store.create(title=req.title, model=req.model)


@router.get("/{session_id}", response_model=SessionMessages)
def get_session(session_id: str, store=Depends(get_session_store)) -> SessionMessages:
    try:
        session = store.get(session_id)
    except SessionStoreError as e:
        raise HTTPException(404, str(e)) from e
    return SessionMessages(session=session, messages=store.messages(session_id))


@router.patch("/{session_id}", response_model=Session)
def rename_session(
    session_id: str,
    req: SessionRenameRequest,
    store=Depends(get_session_store),
) -> Session:
    try:
        return store.rename(session_id, req.title)
    except SessionStoreError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/{session_id}")
def delete_session(session_id: str, store=Depends(get_session_store)) -> dict:
    ok = store.delete(session_id)
    if not ok:
        raise HTTPException(404, f"unknown session {session_id!r}")
    return {"deleted": session_id}
