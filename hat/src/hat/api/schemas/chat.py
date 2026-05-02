from __future__ import annotations

from pydantic import BaseModel

from ...core.schemas import ScoreSignals


class ChatRequest(BaseModel):
    query: str
    context: str | None = None
    feedback: float | None = None
    user_correction: str | None = None


class ChatResponse(BaseModel):
    response: str
    consolidated: bool
    trace_id: str | None = None
    signals: ScoreSignals | None = None
