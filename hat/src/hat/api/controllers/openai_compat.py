"""Controller for OpenAI-compatible chat completions.

Strategy:

* The conversation prefix (everything except the last user message) is fed to
  the Cortex as a flat ``context`` string. If the Cortex has a native
  :meth:`chat` method (e.g. :class:`HFCortex`), the full message list is
  forwarded so the model's chat template handles role formatting natively.
* The last user message is then run through the wake step so the Hippocampus
  Agent can score and (selectively) consolidate the turn.
* A single response is returned in OpenAI ``chat.completion`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.loop import WakeSleepLoop
from ...core.schemas import Interaction
from ...memory.raw.log import RawInteractionLog
from ..schemas.openai import (
    ChatCompletionChoice,
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


@dataclass
class OpenAIChatController:
    loop: WakeSleepLoop
    raw_log: RawInteractionLog

    def handle(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        history, last = _split_messages(req.messages)

        # Per-request generation overrides — only forwarded if the client
        # actually set them; otherwise the LM uses its built-in defaults.
        gen_kwargs: dict = {}
        if req.temperature is not None:
            gen_kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            gen_kwargs["max_tokens"] = req.max_tokens

        # If the Cortex supports native chat-template generation, use it for
        # higher fidelity multi-turn behavior; otherwise fall back to flattened
        # context.
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
