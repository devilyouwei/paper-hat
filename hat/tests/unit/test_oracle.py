"""Unit tests for oracle integration: cost guard and wake-step trigger.

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
    SupervisedReplayBuilder,
    UncertaintyGatePolicy,
)
from hat.core.hippocampus.scoring import ConstantUncertainty
from hat.core.loop import WakeSleepLoop
from hat.core.neocortex.store import InMemoryNeocortex
from hat.core.oracle import CostGuard, NoopOracle, OracleQuotaExceeded
from hat.core.oracle.base import Oracle
from hat.core.schemas import Interaction
from hat.core.sws.trainer import DryRunTrainer


# ---------- CostGuard ---------------------------------------------------


def test_cost_guard_daily_budget(tmp_path):
    g = CostGuard(rps=0, daily_calls=2, audit_path=tmp_path / "audit.jsonl")
    g.acquire()
    g.acquire()
    with pytest.raises(OracleQuotaExceeded):
        g.acquire()
    rows = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(rows) == 2


def test_cost_guard_rps_throttle(monkeypatch):
    sleeps: list[float] = []

    def _record(d):
        sleeps.append(d)

    monkeypatch.setattr(time, "sleep", _record)
    g = CostGuard(rps=2.0, daily_calls=0)
    g.acquire()
    g.acquire()
    assert sleeps and sleeps[0] > 0


def test_cost_guard_disabled_means_pass_through(tmp_path):
    g = CostGuard(rps=0, daily_calls=0, audit_path=None)
    for _ in range(50):
        g.acquire()


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
        write_policy=UncertaintyGatePolicy(threshold=0.0),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=InMemoryNeocortex(),
        trainer=DryRunTrainer(),
        oracle=oracle,
        oracle_threshold=threshold,
    )


def test_oracle_triggers_when_uncertainty_high():
    oracle = _RecordingOracle("better answer")
    loop = _loop(oracle, u=0.9, threshold=0.5)
    inter = Interaction(query="q", response="r")
    trace = loop.wake_step(inter)
    assert trace is not None
    assert oracle.consulted, "oracle must be consulted when U > threshold"
    # Oracle output overrides the response that gets persisted.
    assert inter.response == "better answer"
    assert trace.metadata.extras.get("oracle") is True
    assert "oracle" in trace.metadata.source


def test_oracle_skipped_when_uncertainty_below_threshold():
    oracle = _RecordingOracle()
    loop = _loop(oracle, u=0.2, threshold=0.5)
    loop.wake_step(Interaction(query="q", response="r"))
    assert oracle.consulted == []


def test_oracle_empty_reply_does_not_overwrite_response():
    """A network/quota failure surfaces as ``""``; we must not poison the
    interaction with an empty response."""
    class _Empty(Oracle):
        name = "empty"

        def consult(self, _interaction):
            return ""

    loop = _loop(_Empty(), u=0.9, threshold=0.5)
    inter = Interaction(query="q", response="r")
    trace = loop.wake_step(inter)
    assert inter.response == "r"
    assert trace.metadata.extras.get("oracle") is not True


def test_noop_oracle_returns_response():
    oracle = NoopOracle()
    inter = Interaction(query="q", response="r")
    assert oracle.consult(inter) == "r"
