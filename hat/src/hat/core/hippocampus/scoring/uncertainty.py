"""Uncertainty estimators.

The default :class:`ConstantUncertainty` is a placeholder so the loop runs
without a real LM; :class:`LogprobUncertainty` is the production path used
when the active Cortex exposes per-token log-probabilities.

Math (paper §3.4.2, ``uncertainty`` equation):

    U(x) = 1 - exp( mean_t log p(y_t | x, y_<t) )

i.e. one minus the geometric-mean per-token probability of the response. A
confident greedy decode → ``mean log p`` near 0 → ``U`` near 0; a wandering
sample with low per-token confidence → ``U`` near 1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ...schemas import Interaction


class UncertaintyEstimator(ABC):
    """Returns ``U(x) ∈ [0, 1]`` for an interaction.

    Real implementations: predictive entropy (paper Eq. ``uncertainty``),
    token-level confidence, self-consistency variance, sampled disagreement.
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> float: ...


class ConstantUncertainty(UncertaintyEstimator):
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(self, interaction: Interaction) -> float:
        return self.value


class LogprobUncertainty(UncertaintyEstimator):
    """Uncertainty derived from per-token logprobs of the actual response.

    The cortex must expose an underlying language model with a
    ``chat_logprobs(messages, response) -> list[float]`` (preferred) or
    ``token_logprobs(prompt, response) -> list[float]`` method. If neither
    is available — or the call returns an empty list — we fall back to
    ``fallback`` so the loop never crashes on a backend that doesn't yet
    implement the API (e.g. mlx, ollama).
    """

    def __init__(self, cortex, *, fallback: float = 0.5) -> None:
        self.cortex = cortex
        self.fallback = fallback

    def _logprobs(self, interaction: Interaction) -> list[float]:
        # Reach for the underlying backend. ``HFCortex`` stores it as ``.lm``;
        # other adapters may expose the same attribute or be the LM directly.
        lm = getattr(self.cortex, "lm", self.cortex)
        if interaction.response is None:
            return []
        messages: list[dict[str, str]] = []
        if interaction.context:
            messages.append({"role": "system", "content": interaction.context})
        messages.append({"role": "user", "content": interaction.query})
        try:
            if hasattr(lm, "chat_logprobs") and callable(lm.chat_logprobs):
                return list(lm.chat_logprobs(messages, interaction.response))
            if hasattr(lm, "token_logprobs") and callable(lm.token_logprobs):
                # No chat template available — concatenate query/response.
                prompt = interaction.query
                if interaction.context:
                    prompt = f"{interaction.context}\n\n{prompt}"
                return list(lm.token_logprobs(prompt, interaction.response))
        except Exception:
            return []
        return []

    def __call__(self, interaction: Interaction) -> float:
        lps = self._logprobs(interaction)
        if not lps:
            return self.fallback
        mean_lp = sum(lps) / len(lps)
        # 1 - geometric mean probability, clipped for numerical safety.
        return max(0.0, min(1.0, 1.0 - math.exp(mean_lp)))
