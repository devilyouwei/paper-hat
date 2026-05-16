"""Tests for the session-aware (two-step) wake step.

We stub the abstractor instead of invoking a real LLM. The stub records the
``prior_traces`` it received and either returns a brand-new trace (CREATE) or
a trace tagged with ``metadata.extras['revise_of'] = <id>`` (REVISE) so we
can verify the loop's downstream branching, the JSONL store's ``revise``
behaviour, and the SSE event_sink contract end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from hat.core.cortex.noop import NoopCortex
from hat.core.hippocampus import (
    Abstractor,
    SupervisedReplayBuilder,
    UncertaintyGatePolicy,
)
from hat.core.hippocampus.scoring import ConstantUncertainty
from hat.core.loop import WakeSleepLoop
from hat.core.schemas import (
    Interaction,
    MemoryTrace,
    TraceMetadata,
)
from hat.core.sws.trainer import DryRunTrainer
from hat.memory.curated.jsonl_store import JsonlNeocortex


class _ScriptedAbstractor(Abstractor):
    """Deterministic abstractor used to script CREATE / REVISE behaviour."""

    def __init__(self) -> None:
        self.mode: str = "create"          # 'create', 'revise', or 'drop'
        self.revise_of: str | None = None
        self.last_target: str = "answer"
        self.last_rationale: str = "because"
        self.last_query: str | None = None  # override the query on the trace
        self.seen_prior: list[list[dict]] = []

    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace | None:
        self.seen_prior.append(list(prior_traces or []))
        if self.mode == "drop":
            return None
        extras: dict = {}
        if self.mode == "revise" and self.revise_of:
            extras["revise_of"] = self.revise_of
        return MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=self.last_query if self.last_query is not None else interaction.query,
            cortex_response=interaction.response,
            target_response=self.last_target,
            rationale=self.last_rationale,
            metadata=TraceMetadata(source=interaction.source, extras=extras),
        )


def _build_loop(tmp_path: Path, abstractor: Abstractor) -> tuple[WakeSleepLoop, JsonlNeocortex]:
    store = JsonlNeocortex(
        tmp_path / "train.jsonl", traces_path=tmp_path / "traces.jsonl"
    )
    loop = WakeSleepLoop(
        cortex=NoopCortex(),
        abstractor=abstractor,
        uncertainty=ConstantUncertainty(1.0),
        write_policy=UncertaintyGatePolicy(threshold=0.3),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=store,
        trainer=DryRunTrainer(),
    )
    return loop, store


def test_wake_step_create_path_writes_with_session_id(tmp_path: Path) -> None:
    abs_ = _ScriptedAbstractor()
    abs_.last_target = "Paris"
    loop, store = _build_loop(tmp_path, abs_)

    events: list[dict] = []
    interaction = Interaction(
        query="capital of France?", session_id="s-aaa",
    )
    trace = loop.wake_step(interaction, event_sink=lambda s, p: events.append({"stage": s, **p}))

    assert trace is not None
    rows = store.entries()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s-aaa"
    assert rows[0]["interaction_ids"] == [interaction.id]

    stages = [e["stage"] for e in events]
    assert stages == ["uncertainty", "abstracting", "routed", "scored", "created"]
    routed = next(e for e in events if e["stage"] == "routed")
    assert routed["decision"] == "CREATE"


def test_wake_step_revise_path_overwrites_in_place(tmp_path: Path) -> None:
    abs_ = _ScriptedAbstractor()
    loop, store = _build_loop(tmp_path, abs_)

    # First turn: CREATE.
    abs_.mode = "create"
    abs_.last_target = "Paris"
    abs_.last_rationale = "capital of France"
    first = Interaction(query="capital of France?", session_id="s-bbb")
    first_trace = loop.wake_step(first)
    assert first_trace is not None
    original_trace_id = first_trace.id

    # Second turn: REVISE, fed with prior_traces. The router also rewrites the
    # query so the saved Q/A pair stays coherent with the corrected target.
    abs_.mode = "revise"
    abs_.revise_of = original_trace_id
    abs_.last_target = "Paris (capital, ~2.1M people)"
    abs_.last_rationale = "user added population detail"
    abs_.last_query = "What is the capital of France and its population?"
    prior_traces = list(store)  # walks all traces; default __iter__ on JsonlNeocortex
    second = Interaction(
        query="Actually, please include the population.",
        session_id="s-bbb",
    )
    events: list[dict] = []
    second_trace = loop.wake_step(
        second,
        prior_traces=prior_traces,
        event_sink=lambda s, p: events.append({"stage": s, **p}),
    )

    assert second_trace is not None
    assert second_trace.id == original_trace_id  # trace_id preserved

    # The JSONL store should still hold a single row, now updated.
    rows = store.entries()
    assert len(rows) == 1
    row = rows[0]
    assert row["trace_id"] == original_trace_id
    # interaction_ids accumulates both turns.
    assert first.id in row["interaction_ids"]
    assert second.id in row["interaction_ids"]
    # target_response was overwritten via assistant message in messages list.
    assistant = next(m for m in row["messages"] if m["role"] == "assistant")
    assert assistant["content"] == "Paris (capital, ~2.1M people)"
    # The user-side message (= query) is rewritten too so the Q/A pair stays
    # coherent for SFT replay.
    user_msg = next(m for m in row["messages"] if m["role"] == "user")
    assert user_msg["content"] == "What is the capital of France and its population?"
    # History entry captures the old version of BOTH query and target.
    history = row.get("metadata", {}).get("extras", {}).get("history") or []
    assert len(history) == 1
    assert history[0]["target_response"] == "Paris"
    assert history[0]["query"] == "capital of France?"

    stages = [e["stage"] for e in events]
    assert "revised" in stages
    routed = next(e for e in events if e["stage"] == "routed")
    assert routed["decision"] == "REVISE"
    assert routed["trace_id"] == original_trace_id

    # And the abstractor saw the prior trace on the second call.
    assert len(abs_.seen_prior) == 2
    assert abs_.seen_prior[0] == []  # first turn had no prior context
    assert any(
        t.get("trace_id") == original_trace_id for t in abs_.seen_prior[1]
    )


def test_entries_by_session_filters_correctly(tmp_path: Path) -> None:
    abs_ = _ScriptedAbstractor()
    loop, store = _build_loop(tmp_path, abs_)

    loop.wake_step(Interaction(query="q1", session_id="s-1"))
    loop.wake_step(Interaction(query="q2", session_id="s-2"))
    loop.wake_step(Interaction(query="q3", session_id="s-1"))

    s1 = store.entries_by_session("s-1")
    s2 = store.entries_by_session("s-2")
    none = store.entries_by_session("s-missing")
    assert len(s1) == 2
    assert len(s2) == 1
    assert none == []
    assert all(r["session_id"] == "s-1" for r in s1)


def test_wake_step_drop_path_writes_nothing(tmp_path: Path) -> None:
    """When the router judges the turn neither novel nor user-supervised,
    the abstractor returns None and the loop must drop the turn entirely
    (no row written, no `created`/`revised` event, but a `dropped` event)."""
    abs_ = _ScriptedAbstractor()
    abs_.mode = "drop"
    loop, store = _build_loop(tmp_path, abs_)

    events: list[dict] = []
    interaction = Interaction(query="Hi! How are you?", session_id="s-drop")
    result = loop.wake_step(
        interaction, event_sink=lambda s, p: events.append({"stage": s, **p})
    )

    assert result is None
    assert store.entries() == []
    stages = [e["stage"] for e in events]
    assert "dropped" in stages
    assert "created" not in stages
    assert "revised" not in stages
    assert "routed" not in stages  # routed is only emitted when trace survives
    dropped = next(e for e in events if e["stage"] == "dropped")
    assert dropped["interaction_id"] == interaction.id
    assert dropped["session_id"] == "s-drop"
