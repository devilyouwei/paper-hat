"""Unit tests for oracle integration: cost guard, wake-step trigger, and the
hard-feedback bypass in :class:`LinearWritePolicy`.

We don't hit the network — the oracle is replaced with an in-memory stub
that records every call. The :class:`CostGuard` is exercised against time
directly (no real waits) by patching ``time.sleep``.
"""

from __future__ import annotations

import time

import pytest

from hat.core.cortex.noop import NoopCortex
from hat.core.hippocampus import (
    IdentityAbstractor,
    LinearWritePolicy,
    SupervisedReplayBuilder,
)
from hat.core.hippocampus.scoring import (
    AlwaysNovel,
    BinaryFeedback,
    ConstantUncertainty,
)
from hat.core.loop import WakeSleepLoop
from hat.core.neocortex.store import InMemoryNeocortex
from hat.core.oracle import CostGuard, NoopOracle, OracleQuotaExceeded
from hat.core.oracle.base import Oracle
from hat.core.schemas import Interaction, MemoryTrace, ScoreSignals
from hat.core.sws.trainer import DryRunTrainer


# ---------- LinearWritePolicy hard-feedback bypass ----------------------


def test_feedback_one_forces_accept_below_threshold():
    """A verified user correction (F=1) must enter the dataset even if
    U and N are 0 — the user is the teacher of last resort."""
    pol = LinearWritePolicy(0.4, 0.4, 0.2, threshold=0.5)
    trace = MemoryTrace(interaction_id="i", query="q")
    decision = pol.decide(
        trace, ScoreSignals(uncertainty=0.0, feedback=1.0, novelty=0.0)
    )
    # Soft score is 0.4·0 + 0.4·1 + 0.2·0 = 0.4 < 0.5, but forced=True.
    assert decision.score == pytest.approx(0.4)
    assert decision.accepted is True


def test_partial_feedback_does_not_bypass():
    """Implicit / partial feedback (F<1) goes through the soft policy."""
    pol = LinearWritePolicy(0.4, 0.4, 0.2, threshold=0.5)
    trace = MemoryTrace(interaction_id="i", query="q")
    decision = pol.decide(
        trace, ScoreSignals(uncertainty=0.0, feedback=0.6, novelty=0.0)
    )
    assert decision.accepted is False


# ---------- CostGuard ---------------------------------------------------


def test_cost_guard_daily_budget(tmp_path):
    g = CostGuard(rps=0, daily_calls=2, audit_path=tmp_path / "audit.jsonl")
    g.acquire()
    g.acquire()
    with pytest.raises(OracleQuotaExceeded):
        g.acquire()
    # Audit log records the two successful calls.
    rows = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(rows) == 2


def test_cost_guard_rps_throttle(monkeypatch):
    sleeps: list[float] = []

    def _record(d):
        sleeps.append(d)

    monkeypatch.setattr(time, "sleep", _record)
    g = CostGuard(rps=2.0, daily_calls=0)  # 0 = no daily cap
    g.acquire()
    g.acquire()  # Second call within 0.5s window → must request a sleep.
    assert sleeps and sleeps[0] > 0


def test_cost_guard_disabled_means_pass_through(tmp_path):
    g = CostGuard(rps=0, daily_calls=0, audit_path=None)
    for _ in range(50):
        g.acquire()  # No raise, no sleep.


# ---------- wake_step trigger logic -------------------------------------


class _RecordingOracle(Oracle):
    name = "recording-oracle"

    def __init__(self, reply: str = "the truth") -> None:
        self.reply = reply
        self.consulted: list[Interaction] = []

    def consult(self, interaction):
        self.consulted.append(interaction)
        return self.reply


def _loop(oracle: Oracle | None, *, u: float = 0.5, threshold: float = 0.3) -> WakeSleepLoop:
    return WakeSleepLoop(
        cortex=NoopCortex(),
        abstractor=IdentityAbstractor(),
        uncertainty=ConstantUncertainty(u),
        feedback=BinaryFeedback(),
        novelty=AlwaysNovel(),
        write_policy=LinearWritePolicy(0.4, 0.4, 0.2, threshold=0.0),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=InMemoryNeocortex(),
        trainer=DryRunTrainer(),
        oracle=oracle,
        oracle_threshold=threshold,
    )


def test_oracle_triggers_when_uncertainty_high_and_no_correction():
    oracle = _RecordingOracle("better answer")
    loop = _loop(oracle, u=0.9, threshold=0.5)
    inter = Interaction(query="q", response="r")
    trace = loop.wake_step(inter)
    assert trace is not None
    assert oracle.consulted, "oracle must be consulted when U > threshold"
    assert inter.user_correction == "better answer"
    # Trace metadata records the augmentation.
    assert trace.metadata.extras.get("oracle") is True
    assert "oracle" in trace.metadata.source


def test_oracle_skipped_when_user_already_corrected():
    """If the user already supplied a correction, the oracle must not be
    consulted — the human supervisor outranks the external teacher."""
    oracle = _RecordingOracle("better answer")
    loop = _loop(oracle, u=0.9, threshold=0.5)
    inter = Interaction(query="q", response="r", user_correction="user said so")
    trace = loop.wake_step(inter)
    assert oracle.consulted == []
    assert inter.user_correction == "user said so"
    assert trace.metadata.extras.get("oracle") is not True


def test_oracle_skipped_when_uncertainty_below_threshold():
    oracle = _RecordingOracle()
    loop = _loop(oracle, u=0.2, threshold=0.5)
    loop.wake_step(Interaction(query="q", response="r"))
    assert oracle.consulted == []


def test_oracle_empty_reply_does_not_inject_correction():
    """A network/quota failure surfaces as ``""``; we must not poison the
    interaction with an empty correction string."""
    class _Empty(Oracle):
        name = "empty"

        def consult(self, _interaction):
            return ""

    loop = _loop(_Empty(), u=0.9, threshold=0.5)
    inter = Interaction(query="q", response="r")
    trace = loop.wake_step(inter)
    assert inter.user_correction is None
    assert trace.metadata.extras.get("oracle") is not True


def test_noop_oracle_is_importable_and_usable():
    """Smoke test that the default :class:`NoopOracle` still works for
    tests / dev environments without API keys."""
    oracle = NoopOracle()
    inter = Interaction(query="q", response="r", user_correction="c")
    assert oracle.consult(inter) == "c"
