from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Interaction


class Cortex(ABC):
    """Online interaction model (paper §3.3).

    Concrete implementations wrap a ``LanguageModel`` backend (HF, vLLM, Ollama).
    The Cortex is responsible for two things:

    1. Generating a response to a user query.
    2. Reporting an uncertainty estimate for that interaction (predictive entropy,
       self-consistency variance, …) used by the Hippocampus' write policy.
    """

    name: str = "cortex"

    @abstractmethod
    def generate(self, query: str, *, context: str | None = None) -> str: ...

    @abstractmethod
    def uncertainty(self, interaction: Interaction) -> float: ...
