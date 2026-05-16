"""Novelty estimators.

Novelty is "is this new **to the model**" — *not* "is this new content in the
world". We deliberately do not encode the trace and compare against a vector
store, because that measures dataset overlap, not model knowledge. Instead
the model itself is asked, under a strict prompt that ignores truthfulness,
legality, and safety, whether the content was already in its repertoire.

The default :class:`AlwaysNovel` is a stand-in for tests / noop runs.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# --- Brain-inspired multi-layer novelty scaffolding -----------------------
#
# The literature-friendly decomposition used in this project is:
#
#   novelty = w_i * intrinsic + w_s * state + w_m * memory (+ optional llm)
#
# where:
# - intrinsic: model prediction mismatch / uncertainty signal
# - state: distance to the recent hidden-state distribution
# - memory: distance to consolidated long-term traces
#
# This block intentionally ships as structure-first scaffolding so we can
# review the API and integration points before committing to a concrete math
# backend (PCA, kNN, density, etc.). Existing ``LLMNoveltyJudge`` behavior
# remains unchanged.


@dataclass
class NoveltyBreakdown:
    """Per-layer novelty components in ``[0, 1]``.

    ``combined`` is the weighted blend used by the write policy.
    ``llm`` is optional and can be used as a semantic referee when the cheap
    channels disagree (kept optional to control latency/cost).
    """

    intrinsic: float
    state: float
    memory: float
    llm: float | None
    combined: float


class StateNoveltyChannel(ABC):
    """Novelty relative to recent cortex activation patterns.

    Implementations are expected to consume/maintain query embeddings or
    hidden states and return an outlier score in ``[0, 1]``.
    """

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> float: ...


class MemoryNoveltyChannel(ABC):
    """Novelty relative to consolidated long-term traces.

    Implementations typically compare the current query embedding to stored
    neocortex embeddings (kNN / density / reconstruction error).
    """

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> float: ...


class ConstantStateNovelty(StateNoveltyChannel):
    """Deterministic placeholder for ablations/tests."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(self, trace: MemoryTrace) -> float:
        return self.value


class ConstantMemoryNovelty(MemoryNoveltyChannel):
    """Deterministic placeholder for ablations/tests."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(self, trace: MemoryTrace) -> float:
        return self.value


class CompositeNoveltyJudge(NoveltyEstimator):
    """Three-layer novelty combiner (intrinsic/state/memory) with optional
    LLM semantic referee.

    Design goals:
    - Preserve ``NoveltyEstimator`` compatibility (``__call__(trace)``).
    - Allow the wake loop to pass intrinsic novelty explicitly via
      :meth:`score_with_intrinsic` to avoid double counting.
    - Store per-layer diagnostics under
      ``trace.metadata.extras['novelty_breakdown']`` for UI/debugging.
    """

    def __init__(
        self,
        *,
        state: StateNoveltyChannel,
        memory: MemoryNoveltyChannel,
        llm: NoveltyEstimator | None = None,
        w_intrinsic: float = 0.34,
        w_state: float = 0.33,
        w_memory: float = 0.33,
        w_llm: float = 0.0,
        intrinsic_fallback: float = 0.5,
    ) -> None:
        self.state = state
        self.memory = memory
        self.llm = llm
        self.w_intrinsic = w_intrinsic
        self.w_state = w_state
        self.w_memory = w_memory
        self.w_llm = w_llm
        self.intrinsic_fallback = intrinsic_fallback

    @staticmethod
    def _clip(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _combine(
        self,
        *,
        intrinsic: float,
        state: float,
        memory: float,
        llm: float | None,
    ) -> float:
        total = self.w_intrinsic + self.w_state + self.w_memory
        score = (
            self.w_intrinsic * intrinsic
            + self.w_state * state
            + self.w_memory * memory
        )
        if llm is not None and self.w_llm > 0.0:
            score += self.w_llm * llm
            total += self.w_llm
        if total <= 0:
            return self.intrinsic_fallback
        return self._clip(score / total)

    def score_with_intrinsic(
        self,
        trace: MemoryTrace,
        *,
        intrinsic: float,
    ) -> NoveltyBreakdown:
        i = self._clip(intrinsic)
        s = self._clip(self.state(trace))
        m = self._clip(self.memory(trace))
        l = self._clip(self.llm(trace)) if self.llm is not None else None
        c = self._combine(intrinsic=i, state=s, memory=m, llm=l)
        trace.metadata.extras["novelty_breakdown"] = {
            "intrinsic": i,
            "state": s,
            "memory": m,
            "llm": l,
            "combined": c,
        }
        return NoveltyBreakdown(intrinsic=i, state=s, memory=m, llm=l, combined=c)

    def __call__(self, trace: MemoryTrace) -> float:
        # Compatibility path: if the wake loop does not pass intrinsic yet,
        # keep running with a neutral fallback so integration can proceed in
        # stages. The preferred path is ``score_with_intrinsic``.
        return self.score_with_intrinsic(
            trace,
            intrinsic=self.intrinsic_fallback,
        ).combined


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
