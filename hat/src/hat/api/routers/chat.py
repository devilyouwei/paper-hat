from __future__ import annotations

from fastapi import APIRouter, Depends

from ..controllers.chat import ChatController
from ..deps import get_loop, get_raw_log
from ..schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("", response_model=ChatResponse)
def post_chat(
    req: ChatRequest,
    loop=Depends(get_loop),
    log=Depends(get_raw_log),
) -> ChatResponse:
    return ChatController(loop=loop, raw_log=log).handle(req)
