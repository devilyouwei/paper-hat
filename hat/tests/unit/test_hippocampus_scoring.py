"""Unit tests for the hippocampus scoring/abstraction upgrades.

Covers:
- ``parse_score`` for the various number formats a chatty model might emit.
- ``LLMNoveltyJudge`` / ``LLMFeedbackJudge`` end-to-end against a stub cortex,
  including the deterministic short-circuits in ``LLMFeedbackJudge``.
- ``LLMAbstractor`` with valid/invalid JSON output.
- ``LogprobUncertainty`` against a stub LM and its fallback path.
"""

from __future__ import annotations

import math

from hat.core.hippocampus import LLMAbstractor
from hat.core.hippocampus.scoring import (
    LLMFeedbackJudge,
    LLMNoveltyJudge,
    LogprobUncertainty,
)
from hat.core.hippocampus.scoring.llm_judge import parse_score
from hat.core.schemas import Interaction, MemoryTrace


# ---------- parse_score ---------------------------------------------------


def test_parse_score_decimal_in_range():
    assert parse_score("0.7") == 0.7
    assert parse_score("the score is 0.42 because…") == 0.42


def test_parse_score_strips_think_blocks():
    assert parse_score("<think>blah blah 0.99</think>0.3") == 0.3


def test_parse_score_percentage():
    assert math.isclose(parse_score("70%"), 0.7)


def test_parse_score_fraction():
    assert math.isclose(parse_score("7/10"), 0.7)


def test_parse_score_likert_zero_to_ten():
    assert math.isclose(parse_score("score: 8 (out of 10)"), 0.8)


def test_parse_score_fallback_on_garbage():
    assert parse_score("no number here", fallback=0.42) == 0.42
    assert parse_score("", fallback=0.1) == 0.1


# ---------- judges --------------------------------------------------------


class _FixedCortex:
    """Minimal cortex stub that always returns the same string."""

    name = "stub"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, **_kwargs) -> str:
        self.calls.append(list(messages))
        return self._reply


def _trace(query: str = "q", response: str = "r") -> MemoryTrace:
    return MemoryTrace(interaction_id="i", query=query, cortex_response=response)


def test_novelty_judge_parses_score():
    cortex = _FixedCortex("0.83")
    score = LLMNoveltyJudge(cortex)(_trace())
    assert math.isclose(score, 0.83)
    # Two messages: a system prompt with the rules, a user prompt with input.
    assert len(cortex.calls[0]) == 2
    assert cortex.calls[0][0]["role"] == "system"


def test_novelty_judge_does_not_leak_model_response():
    """Regression: novelty must score user input only, never the model's
    own response. The judge prompt must not contain the response text."""
    secret = "MODEL_RESPONSE_LEAK_CANARY"
    cortex = _FixedCortex("0.5")
    LLMNoveltyJudge(cortex)(_trace(query="hello", response=secret))
    flat = "\n".join(m["content"] for m in cortex.calls[0])
    assert secret not in flat


def test_novelty_judge_reads_user_correction_from_extras():
    """When the abstractor stashes ``user_correction`` in metadata.extras,
    the novelty judge must surface it as a user-side input field."""
    from hat.core.schemas import TraceMetadata

    cortex = _FixedCortex("0.7")
    trace = MemoryTrace(
        interaction_id="i",
        query="who won the 2025 final?",
        cortex_response="model guess",
        metadata=TraceMetadata(extras={"user_correction": "Argentina won 2-1"}),
    )
    LLMNoveltyJudge(cortex)(trace)
    flat = "\n".join(m["content"] for m in cortex.calls[0])
    assert "Argentina won 2-1" in flat
    assert "model guess" not in flat


def test_novelty_judge_fallback_on_unparseable():
    cortex = _FixedCortex("I refuse to score.")
    assert LLMNoveltyJudge(cortex, fallback=0.25)(_trace()) == 0.25


def test_feedback_judge_short_circuits_on_correction():
    # The correction path must not invoke the model at all.
    cortex = _FixedCortex("0.0")
    inter = Interaction(query="q", response="r", user_correction="actually it's 42")
    assert LLMFeedbackJudge(cortex)(inter) == 1.0
    assert cortex.calls == []


def test_feedback_judge_short_circuits_on_explicit_score():
    cortex = _FixedCortex("0.0")
    inter = Interaction(query="q", response="r", feedback=0.6)
    assert math.isclose(LLMFeedbackJudge(cortex)(inter), 0.6)
    assert cortex.calls == []


def test_feedback_judge_falls_through_to_llm_when_no_signal():
    cortex = _FixedCortex("0.4")
    inter = Interaction(query="q", response="r")
    assert math.isclose(LLMFeedbackJudge(cortex)(inter), 0.4)
    assert len(cortex.calls) == 1


# ---------- abstractor ----------------------------------------------------


def test_llm_abstractor_parses_json():
    payload = (
        '{"summary": "user asked about X",'
        ' "target": "the answer is X",'
        ' "rationale": "novel topic"}'
    )
    cortex = _FixedCortex(payload)
    inter = Interaction(query="what is X?", response="X is foo")
    trace = LLMAbstractor(cortex)(inter)
    assert trace.target_response == "the answer is X"
    assert trace.rationale == "novel topic"
    assert trace.query == "what is X?"


def test_llm_abstractor_falls_back_on_invalid_json():
    cortex = _FixedCortex("not json at all")
    inter = Interaction(query="q", response="r", user_correction="c")
    trace = LLMAbstractor(cortex)(inter)
    # Identity fallback uses the correction as the target response.
    assert trace.target_response == "c"


# ---------- logprob uncertainty ------------------------------------------


class _StubLM:
    def __init__(self, logps: list[float]) -> None:
        self._logps = logps

    def chat_logprobs(self, messages, response):  # noqa: ARG002
        return list(self._logps)


class _LMCortex:
    """Cortex with a real ``.lm`` attribute (mirrors HFCortex shape)."""

    name = "lm-cortex"

    def __init__(self, lm) -> None:
        self.lm = lm


def test_logprob_uncertainty_high_confidence():
    # mean log p ≈ -0.05  ⇒  U ≈ 1 - e^{-0.05} ≈ 0.0488
    lm = _StubLM([-0.05, -0.05, -0.05])
    cortex = _LMCortex(lm)
    u = LogprobUncertainty(cortex)(Interaction(query="q", response="r"))
    assert 0.0 < u < 0.1


def test_logprob_uncertainty_low_confidence():
    # mean log p = -3.0  ⇒  U = 1 - e^{-3} ≈ 0.95
    lm = _StubLM([-3.0, -3.0])
    cortex = _LMCortex(lm)
    u = LogprobUncertainty(cortex)(Interaction(query="q", response="r"))
    assert u > 0.9


def test_logprob_uncertainty_falls_back_when_unsupported():
    class _NoLogprobs:
        pass

    cortex = _LMCortex(_NoLogprobs())
    u = LogprobUncertainty(cortex, fallback=0.42)(
        Interaction(query="q", response="r")
    )
    assert u == 0.42
