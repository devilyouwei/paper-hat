"""Cortex implementation backed by an HF chat-tuned model."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import Interaction
from .base import Cortex


class HFCortex(Cortex):
    """Wraps an :class:`HFLanguageModel` and exposes the Cortex API.

    Adds a :meth:`chat` method so the OpenAI-compatible router can pass full
    multi-turn message lists into the model's chat template, while the
    wake/sleep loop still calls :meth:`generate` for single-shot use.
    """

    def __init__(self, lm, name: str | None = None) -> None:
        self.lm = lm
        self.name = name or getattr(lm, "name", "hf-cortex")

    def generate(self, query: str, *, context: str | None = None, **kwargs) -> str:
        return self.lm.generate(query, context=context, **kwargs)

    def chat(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        return self.lm.chat(messages, **kwargs)

    def stream_chat(self, messages: Sequence[dict[str, str]], **kwargs):
        yield from self.lm.stream_chat(messages, **kwargs)

    def uncertainty(self, interaction: Interaction) -> float:
        # TODO: predictive entropy via token_logprobs once implemented.
        return 0.5
