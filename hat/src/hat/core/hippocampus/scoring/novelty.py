"""Novelty estimators.

Novelty is "is this new **to the model**" — *not* "is this new content in the
world". We deliberately do not encode the trace and compare against a vector
store, because that measures dataset overlap, not model knowledge. Instead
the model itself is asked, under a strict prompt that ignores truthfulness,
legality, and safety, whether the content was already in its repertoire.

The default :class:`AlwaysNovel` is a stand-in for tests / noop runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import MemoryTrace
from .llm_judge import call_judge, load_prompt, parse_score, render


class NoveltyEstimator(ABC):
    """``N(m) ∈ [0, 1]`` — see paper §3.4.2 'Novelty'.

    The default :class:`AlwaysNovel` returns ``1.0`` so every trace passes
    during dev. The production path is :class:`LLMNoveltyJudge`, which asks
    the active Cortex itself how unfamiliar the content is.
    """

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> float: ...


class AlwaysNovel(NoveltyEstimator):
    """Returns ``1.0`` so every trace passes the novelty channel during dev."""

    def __call__(self, trace: MemoryTrace) -> float:
        return 1.0


class LLMNoveltyJudge(NoveltyEstimator):
    """Self-novelty: ask the Cortex whether the **user input** is new to it.

    Important: novelty is scored against information that arrived from the
    user (``query`` plus any ``user_correction``/reference they supplied).
    The model's own response is *not* used as input to the judge — letting
    the model evaluate its own output would be a circular reference and
    would systematically depress novelty scores. To preserve that
    invariant we recover the originating ``Interaction`` fields via
    :class:`MemoryTrace.metadata.extras` when the abstractor passes them
    through; otherwise we fall back to ``trace.query`` alone.

    Loads :file:`hippocampus/prompts/novelty.md`, substitutes ``{query}`` and
    ``{correction}``, and calls ``cortex.chat`` with low max_tokens and
    zero temperature. The response is parsed permissively (see
    ``parse_score``). Any failure falls back to ``fallback`` so the wake
    step never raises.
    """

    def __init__(self, cortex, *, fallback: float = 0.5, max_tokens: int = 32) -> None:
        self.cortex = cortex
        self.fallback = fallback
        self.max_tokens = max_tokens
        self._template = load_prompt("novelty")

    def __call__(self, trace: MemoryTrace) -> float:
        # Split into a system prompt (the rules + contract) and a user prompt
        # (the input fields). The .md file already segregates them via the
        # "## Input" header, so we slice on that.
        marker = "## Input"
        if marker in self._template:
            system, body = self._template.split(marker, 1)
            user = marker + body
        else:
            system, user = self._template, ""
        # Recover user-side fields if the abstractor stashed them in extras;
        # otherwise the trace's ``query`` is still safe (it always mirrors
        # the user's input). We deliberately do **not** read
        # ``cortex_response`` / ``target_response`` here.
        extras = trace.metadata.extras or {}
        correction = extras.get("user_correction") or ""
        rendered = render(
            user,
            query=trace.query or "",
            correction=correction,
        )
        raw = call_judge(
            self.cortex, system=system.strip(), user=rendered.strip(),
            max_tokens=self.max_tokens,
        )
        return parse_score(raw, fallback=self.fallback)
