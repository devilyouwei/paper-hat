from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

from ..schemas import Interaction


class Cortex(ABC):
    """Online interaction model (paper §3.3).

    Concrete implementations wrap a ``LanguageModel`` backend (HF, MLX, …).
    The Cortex is responsible for:

    1. Generating a response to a user query (single-shot :meth:`generate`,
       or multi-turn :meth:`chat` / :meth:`stream_chat` for chat-tuned
       models that own their own template).
    2. Reporting an uncertainty estimate for that interaction (predictive
       entropy, self-consistency variance, …) used by the Hippocampus'
       write policy.
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
