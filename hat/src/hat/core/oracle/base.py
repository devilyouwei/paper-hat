from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Interaction


class Oracle(ABC):
    """External teacher consulted on demand (paper §3.5).

    Concrete implementations: ``OpenAICompatibleOracle`` (``hat.models.backends``)
    over GPT-4o, vLLM-served Llama-70B, or Ollama. The ``WakeSleepLoop`` only
    queries the oracle when ``U(x) > τ_O`` or ``F(x) > 0`` — paper Eq.
    ``oracle_query``.
    """

    name: str = "oracle"

    @abstractmethod
    def consult(self, interaction: Interaction) -> str: ...


class NoopOracle(Oracle):
    """Returns the user correction or the cortex response. For tests."""

    name = "noop-oracle"

    def consult(self, interaction: Interaction) -> str:
        return interaction.user_correction or interaction.response or ""
