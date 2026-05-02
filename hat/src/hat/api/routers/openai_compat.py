"""OpenAI-compatible chat-completions endpoint.

Mounted at ``/v1`` so any standard OpenAI client (``OpenAI(base_url=…)``) can
talk to HAT directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...config.settings import get_settings
from ..controllers.openai_compat import OpenAIChatController
from ..deps import get_loop, get_raw_log
from ..schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
)

router = APIRouter()


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    s = get_settings()
    return ModelList(data=[ModelCard(id=s.ui_model)])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    req: ChatCompletionRequest,
    loop=Depends(get_loop),
    log=Depends(get_raw_log),
) -> ChatCompletionResponse:
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="streaming not implemented yet; set stream=false",
        )
    try:
        return OpenAIChatController(loop=loop, raw_log=log).handle(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
