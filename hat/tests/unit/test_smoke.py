from __future__ import annotations

import pytest

from hat.core.cortex.noop import NoopCortex
from hat.core.hippocampus import (
    IdentityAbstractor,
    SupervisedReplayBuilder,
    UncertaintyGatePolicy,
)
from hat.core.hippocampus.scoring import ConstantUncertainty
from hat.core.loop import WakeSleepLoop
from hat.core.neocortex.store import (
    InMemoryNeocortex,
    NeocortexWriteError,
)
from hat.abstract.schemas import (
    Interaction,
    MemoryTrace,
    ScoreSignals,
    WriteDecision,
)
from hat.core.sws.trainer import DryRunTrainer


def test_imports() -> None:
    import hat.abstract  # noqa: F401
    import hat.core  # noqa: F401
    import hat.core.loop  # noqa: F401


def test_neocortex_rejects_unauthorized_write() -> None:
    store = InMemoryNeocortex()
    trace = MemoryTrace(interaction_id="i1", query="q")

    bad_id = WriteDecision(
        trace_id="other",
        score=1.0,
        threshold=0.0,
        signals=ScoreSignals(),
        accepted=True,
    )
    with pytest.raises(NeocortexWriteError):
        store.write(trace, bad_id)

    rejected = WriteDecision(
        trace_id=trace.id,
        score=0.0,
        threshold=0.5,
        signals=ScoreSignals(),
        accepted=False,
    )
    with pytest.raises(NeocortexWriteError):
        store.write(trace, rejected)


def test_uncertainty_gate_policy() -> None:
    pol = UncertaintyGatePolicy(threshold=0.3)
    trace = MemoryTrace(interaction_id="i", query="q")
    assert pol.score(trace, ScoreSignals(uncertainty=0.8)) == pytest.approx(0.8)
    assert pol.decide(trace, ScoreSignals(uncertainty=0.5)).accepted is True
    assert pol.decide(trace, ScoreSignals(uncertainty=0.1)).accepted is False


def _build_loop(*, uncertainty: float = 1.0) -> WakeSleepLoop:
    return WakeSleepLoop(
        cortex=NoopCortex(),
        abstractor=IdentityAbstractor(),
        uncertainty=ConstantUncertainty(uncertainty),
        write_policy=UncertaintyGatePolicy(threshold=0.3),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=InMemoryNeocortex(),
        trainer=DryRunTrainer(),
    )


def test_wake_sleep_smoke() -> None:
    loop = _build_loop()
    traces = loop.wake_step(Interaction(query="hello"))
    assert traces  # non-empty list
    assert len(loop.neocortex) == 1

    stats = loop.sleep_step(cycle=1, k=8)
    assert stats.cycle == 1
    assert stats.n_replayed >= 1


def test_wake_step_skips_below_threshold() -> None:
    loop = _build_loop(uncertainty=0.05)
    traces = loop.wake_step(Interaction(query="hello"))
    assert traces == []
    assert len(loop.neocortex) == 0


def test_chat_controller_appends_raw_log(tmp_path) -> None:
    from hat.api.services.chat import ChatService
    from hat.api.schemas.chat import ChatRequest
    from hat.core.sessions.raw_log import JsonlRawLog

    log = JsonlRawLog(tmp_path / "raw.jsonl")
    ctrl = ChatService(loop=_build_loop(), raw_log=log)
    res = ctrl.handle(ChatRequest(query="hi"))
    assert res.response.startswith("[noop]")
    assert sum(1 for _ in log) == 1
