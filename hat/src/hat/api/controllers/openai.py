"""OpenAI-compatible chat-completions endpoint.

Mounted at ``/v1`` so any standard OpenAI client (``OpenAI(base_url=…)``)
can talk to HAT directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...config.settings import get_settings
from ..schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
)
from hat.core.runtime.container import get_loop, get_raw_log, get_session_store
from ..services.openai import OpenAIChatService

router = APIRouter()


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    s = get_settings()
    return ModelList(data=[ModelCard(id=s.ui_model)])


@router.post("/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    loop=Depends(get_loop),
    log=Depends(get_raw_log),
    sessions=Depends(get_session_store),
):
    service = OpenAIChatService(loop=loop, raw_log=log, sessions=sessions)
    try:
        if req.stream:
            return StreamingResponse(
                service.handle_stream(req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return service.handle(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


__all__ = ["router", "ChatCompletionResponse"]
