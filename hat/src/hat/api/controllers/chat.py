"""POST /chat — single-turn legacy endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat import ChatService
from ..services.container import get_loop, get_raw_log

router = APIRouter()


@router.post("", response_model=ChatResponse)
def post_chat(
    req: ChatRequest,
    loop=Depends(get_loop),
    log=Depends(get_raw_log),
) -> ChatResponse:
    return ChatService(loop=loop, raw_log=log).handle(req)
