"""Feedback extractors.

The default :class:`BinaryFeedback` is a deterministic short-circuit: if the
caller supplied an explicit correction or a numeric ``feedback`` field, the
trace is scored without consulting the model. Otherwise (and for richer
implicit-signal scoring), :class:`LLMFeedbackJudge` asks the Cortex itself
to grade the strength of supervision present in the turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import Interaction
from .llm_judge import call_judge, load_prompt, parse_score, render


class FeedbackExtractor(ABC):
    """Maps explicit / implicit user supervision to a scalar."""

    @abstractmethod
    def __call__(self, interaction: Interaction) -> float: ...


class BinaryFeedback(FeedbackExtractor):
    """``1.0`` if a correction or explicit feedback exists, else ``0.0``.

    A correction always wins over a numeric ``feedback`` field so verified
    errors (paper §3.4.2 "Feedback") are guaranteed to be consolidated.
    """

    def __call__(self, interaction: Interaction) -> float:
        if interaction.user_correction:
            return 1.0
        if interaction.feedback is not None:
            return float(interaction.feedback)
        return 0.0


class LLMFeedbackJudge(FeedbackExtractor):
    """Ask the Cortex to grade how informative the user's reaction is.

    Strong explicit signals (correction string, numeric feedback) short-circuit
    to deterministic values so we don't waste a forward pass on the obvious
    case. Otherwise we fall through to a prompted call.
    """

    def __init__(self, cortex, *, fallback: float = 0.0, max_tokens: int = 32) -> None:
        self.cortex = cortex
        self.fallback = fallback
        self.max_tokens = max_tokens
        self._template = load_prompt("feedback")

    def __call__(self, interaction: Interaction) -> float:
        # Deterministic short-circuit for unambiguous cases.
        if interaction.user_correction:
            return 1.0
        if interaction.feedback is not None:
            return max(0.0, min(1.0, float(interaction.feedback)))

        marker = "## Input"
        if marker in self._template:
            system, body = self._template.split(marker, 1)
            user = marker + body
        else:
            system, user = self._template, ""
        rendered = render(
            user,
            query=interaction.query or "",
            response=interaction.response or "",
            correction="",
            feedback="",
        )
        raw = call_judge(
            self.cortex, system=system.strip(), user=rendered.strip(),
            max_tokens=self.max_tokens,
        )
        return parse_score(raw, fallback=self.fallback)
