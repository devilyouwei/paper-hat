"""OpenAI Chat-Completions compatible request/response DTOs.

We support the subset needed for the standard OpenAI Python client to talk to
HAT. Streaming is intentionally out of scope for the first cut.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    user: str | None = None  # optional client-supplied user id
    # HAT extension: forwarded to the tokenizer's chat template (e.g. Qwen3.5
    # uses ``enable_thinking`` to toggle the <think> phase).
    chat_template_kwargs: dict | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    # HAT-specific extras (clients can ignore these)
    hat_consolidated: bool = False
    hat_trace_id: str | None = None


# ---- Streaming chunks (chat.completion.chunk) -----------------------------


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta = Field(default_factory=ChatCompletionDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    # HAT extras attached to the final chunk
    hat_consolidated: bool | None = None
    hat_trace_id: str | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "hat"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
