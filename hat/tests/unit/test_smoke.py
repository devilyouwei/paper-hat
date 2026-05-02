from __future__ import annotations

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
from hat.core.neocortex.store import (
    InMemoryNeocortex,
    NeocortexWriteError,
)
from hat.core.schemas import (
    Interaction,
    MemoryTrace,
    ScoreSignals,
    WriteDecision,
)
from hat.core.sws.trainer import DryRunTrainer


def test_imports() -> None:
    import hat.core  # noqa: F401
    import hat.core.loop  # noqa: F401
    import hat.core.protocols  # noqa: F401


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


def test_linear_write_policy_score() -> None:
    pol = LinearWritePolicy(0.4, 0.4, 0.2, threshold=0.3)
    trace = MemoryTrace(interaction_id="i", query="q")
    s = pol.score(trace, ScoreSignals(uncertainty=1.0, feedback=1.0, novelty=1.0))
    assert abs(s - 1.0) < 1e-9
    assert pol.decide(trace, ScoreSignals(uncertainty=0.0, feedback=0.0, novelty=0.0)).accepted is False


def _build_loop() -> WakeSleepLoop:
    return WakeSleepLoop(
        cortex=NoopCortex(),
        abstractor=IdentityAbstractor(),
        uncertainty=ConstantUncertainty(1.0),
        feedback=BinaryFeedback(),
        novelty=AlwaysNovel(),
        write_policy=LinearWritePolicy(0.4, 0.4, 0.2, threshold=0.3),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=InMemoryNeocortex(),
        trainer=DryRunTrainer(),
    )


def test_wake_sleep_smoke() -> None:
    loop = _build_loop()
    trace = loop.wake_step(Interaction(query="hello"))
    assert trace is not None
    assert len(loop.neocortex) == 1

    stats = loop.sleep_step(cycle=1, k=8)
    assert stats.cycle == 1
    assert stats.n_replayed >= 1


def test_chat_controller_appends_raw_log(tmp_path) -> None:
    from hat.api.controllers.chat import ChatController
    from hat.api.schemas.chat import ChatRequest
    from hat.memory.raw.log import JsonlRawLog

    log = JsonlRawLog(tmp_path / "raw.jsonl")
    ctrl = ChatController(loop=_build_loop(), raw_log=log)
    res = ctrl.handle(ChatRequest(query="hi"))
    assert res.response.startswith("[noop]")
    assert sum(1 for _ in log) == 1
