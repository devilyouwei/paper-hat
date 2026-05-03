"""Cortex implementation backed by an MLX (Apple Silicon) model."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import Interaction
from .base import Cortex


class MLXCortex(Cortex):
    """Wraps an :class:`MLXLanguageModel`. Same shape as :class:`HFCortex`."""

    def __init__(self, lm, name: str | None = None) -> None:
        self.lm = lm
        self.name = name or getattr(lm, "name", "mlx-cortex")

    def generate(self, query: str, *, context: str | None = None, **kwargs) -> str:
        return self.lm.generate(query, context=context, **kwargs)

    def chat(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        return self.lm.chat(messages, **kwargs)

    def uncertainty(self, interaction: Interaction) -> float:
        # TODO: predictive entropy from mlx-lm `generate_step` logprobs.
        return 0.5
