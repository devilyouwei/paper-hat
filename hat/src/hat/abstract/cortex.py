"""Cortex interfaces (paper §3.3).

The Cortex is the online interaction model. Concrete implementations
wrap a :class:`LanguageModel` backend (HuggingFace, MLX, ...). The
:class:`Cortex` ABC defines the canonical surface that the wake step
calls; the :class:`LanguageModel` Protocol describes the structural
seam used by ``LogprobUncertainty`` and other consumers that talk
directly to the LM rather than through the Cortex façade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .schemas import Interaction


class Cortex(ABC):
    """Online interaction model (paper §3.3).

    Responsibilities:

    1. Generate a response to a user query (single-shot :meth:`generate`,
       or multi-turn :meth:`chat` / :meth:`stream_chat` for chat-tuned
       models that own their own template).
    2. Report an uncertainty estimate for that interaction (predictive
       entropy, self-consistency variance, ...) used by the
       Hippocampus' write policy.
    """

    name: str = "cortex"

    @abstractmethod
    def generate(self, query: str, *, context: str | None = None) -> str: ...

    @abstractmethod
    def chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> str: ...

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> Iterator[str]: ...

    @abstractmethod
    def uncertainty(self, interaction: Interaction) -> float: ...


@runtime_checkable
class LanguageModel(Protocol):
    """Structural interface for an underlying LM backend.

    Implementations live next to their Cortex adapter (e.g.
    ``hat.core.cortex.hf.HFLanguageModel``,
    ``hat.core.cortex.mlx.MLXLanguageModel``).
    """

    name: str

    def generate(
        self, prompt: str, *, context: str | None = None, **kwargs: Any
    ) -> str: ...
    def token_logprobs(self, prompt: str, response: str) -> list[float]: ...


__all__ = ["Cortex", "LanguageModel"]
