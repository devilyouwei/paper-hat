"""Schemas for the chat-session REST surface (``/api/sessions``)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ...core.schemas import Interaction
from ...memory.raw.sessions import Session


class SessionList(BaseModel):
    object: str = "list"
    data: list[Session]


class SessionMessages(BaseModel):
    session: Session
    messages: list[Interaction]


class SessionCreateRequest(BaseModel):
    title: str | None = None
    model: str | None = None


class SessionRenameRequest(BaseModel):
    title: str


__all__ = [
    "Session",
    "SessionList",
    "SessionMessages",
    "SessionCreateRequest",
    "SessionRenameRequest",
    # re-exported for convenience
    "Interaction",
    "datetime",
]
