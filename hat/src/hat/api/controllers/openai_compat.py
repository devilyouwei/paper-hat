"""Controller for OpenAI-compatible chat completions.

Strategy:

* The conversation prefix (everything except the last user message) is fed to
  the Cortex as a flat ``context`` string. If the Cortex has a native
  :meth:`chat` method (e.g. :class:`HFCortex`), the full message list is
  forwarded so the model's chat template handles role formatting natively.
* The last user message is then run through the wake step so the Hippocampus
  Agent can score and (selectively) consolidate the turn.
* A single response is returned in OpenAI ``chat.completion`` shape, or a
  generator of ``chat.completion.chunk`` SSE payloads when streaming.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from ...core.loop import WakeSleepLoop
from ...core.schemas import Interaction
from ...memory.raw.log import RawInteractionLog
from ..schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


def _split_messages(messages: list[ChatMessage]) -> tuple[list[ChatMessage], ChatMessage]:
    if not messages:
        raise ValueError("messages must be non-empty")
    last = messages[-1]
    if last.role != "user":
        raise ValueError("the final message must have role='user'")
    return messages[:-1], last


def _flatten_history(history: list[ChatMessage]) -> str | None:
    if not history:
        return None
    parts = [f"{m.role}: {m.content}" for m in history]
    return "\n".join(parts)


def _build_gen_kwargs(req: ChatCompletionRequest) -> dict:
    gen_kwargs: dict = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_kwargs["max_tokens"] = req.max_tokens
    if req.chat_template_kwargs:
        gen_kwargs["chat_template_kwargs"] = dict(req.chat_template_kwargs)
    return gen_kwargs


@dataclass
class OpenAIChatController:
    loop: WakeSleepLoop
    raw_log: RawInteractionLog

    def handle(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        history, last = _split_messages(req.messages)
        gen_kwargs = _build_gen_kwargs(req)

        cortex = self.loop.cortex
        response_text: str
        if hasattr(cortex, "chat") and callable(cortex.chat):
            response_text = cortex.chat(
                [m.model_dump() for m in req.messages], **gen_kwargs
            )
            interaction = Interaction(
                context=_flatten_history(history),
                query=last.content,
                response=response_text,
            )
            trace = self.loop.wake_step(interaction)
        else:
            interaction = Interaction(
                context=_flatten_history(history),
                query=last.content,
            )
            trace = self.loop.wake_step(interaction)
            response_text = interaction.response or ""

        self.raw_log.append(interaction)

        return ChatCompletionResponse(
            model=getattr(cortex, "name", req.model),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            hat_consolidated=trace is not None,
            hat_trace_id=trace.id if trace else None,
        )

    # ------------------------------------------------------------------ stream

    def handle_stream(self, req: ChatCompletionRequest) -> Iterator[str]:
        """Yield SSE-formatted ``data: {...}\\n\\n`` strings (plus terminator).

        Falls back to a single-chunk emit if the active Cortex does not
        implement ``stream_chat``.
        """
        history, last = _split_messages(req.messages)
        gen_kwargs = _build_gen_kwargs(req)

        cortex = self.loop.cortex
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = getattr(cortex, "name", req.model)

        def chunk(delta: ChatCompletionDelta, finish: str | None = None,
                  *, hat_consolidated: bool | None = None,
                  hat_trace_id: str | None = None) -> str:
            payload = ChatCompletionChunk(
                id=chat_id,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0, delta=delta, finish_reason=finish
                    )
                ],
                hat_consolidated=hat_consolidated,
                hat_trace_id=hat_trace_id,
            )
            return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

        # Opening role chunk (mirrors OpenAI behaviour).
        yield chunk(ChatCompletionDelta(role="assistant"))

        msgs = [m.model_dump() for m in req.messages]
        collected: list[str] = []

        if hasattr(cortex, "stream_chat") and callable(cortex.stream_chat):
            try:
                for piece in cortex.stream_chat(msgs, **gen_kwargs):
                    if not piece:
                        continue
                    collected.append(piece)
                    yield chunk(ChatCompletionDelta(content=piece))
            except Exception as e:  # pragma: no cover - runtime safety net
                yield chunk(
                    ChatCompletionDelta(content=f"\n[stream error] {type(e).__name__}: {e}")
                )
        else:
            # No streaming support — fall back to a single blocking chat() call.
            text = (
                cortex.chat(msgs, **gen_kwargs)
                if hasattr(cortex, "chat") and callable(cortex.chat)
                else cortex.generate(last.content, context=_flatten_history(history))
            )
            collected.append(text)
            yield chunk(ChatCompletionDelta(content=text))

        full = "".join(collected)
        interaction = Interaction(
            context=_flatten_history(history),
            query=last.content,
            response=full,
        )
        trace = self.loop.wake_step(interaction)
        self.raw_log.append(interaction)

        # Closing chunk + DONE marker.
        yield chunk(
            ChatCompletionDelta(),
            finish="stop",
            hat_consolidated=trace is not None,
            hat_trace_id=trace.id if trace else None,
        )
        yield "data: [DONE]\n\n"
