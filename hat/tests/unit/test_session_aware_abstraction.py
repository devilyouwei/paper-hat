"""Tests for the new wake step (post-routing-redesign).

Routing CREATE-vs-REVISE is no longer the abstractor's job; that lives
in :class:`hat.core.hippocampus.dedup.EmbeddingDeduper`. The abstractor
returns a (possibly multi-element) list of :class:`MemoryTrace` and the
loop walks each one through dedup → write_policy → write/revise.

We stub both the abstractor and the deduper so the tests stay
deterministic without needing a real LLM or real embedder.
"""

from __future__ import annotations

from pathlib import Path

from hat.core.cortex.noop import NoopCortex
from hat.core.hippocampus import (
    Abstractor,
    SupervisedReplayBuilder,
    UncertaintyGatePolicy,
)
from hat.core.hippocampus.dedup import DedupResult
from hat.core.hippocampus.scoring import ConstantUncertainty
from hat.core.loop import WakeSleepLoop
from hat.abstract.schemas import (
    Interaction,
    MemoryTrace,
    TraceMetadata,
)
from hat.core.sws.trainer import DryRunTrainer
from hat.core.neocortex.jsonl_store import JsonlNeocortex


class _ScriptedAbstractor(Abstractor):
    """Deterministic abstractor returning ``list[MemoryTrace]``."""

    def __init__(self) -> None:
        self.mode: str = "create"          # 'create' or 'drop'
        self.last_target: str = "answer"
        self.last_rationale: str = "because"
        self.last_query: str | None = None  # override the query on the trace
        self.kps: list[tuple[str, str]] | None = None  # multi-KP override

    def __call__(
        self,
        interaction: Interaction,
        *,
        event_sink=None,
    ) -> list[MemoryTrace]:
        if self.mode == "drop":
            return []
        if self.kps:
            traces: list[MemoryTrace] = []
            for q, t in self.kps:
                traces.append(
                    MemoryTrace(
                        interaction_id=interaction.id,
                        session_id=interaction.session_id,
                        interaction_ids=[interaction.id],
                        query=q,
                        cortex_response=interaction.response,
                        target_response=t,
                        rationale=self.last_rationale,
                        metadata=TraceMetadata(source=interaction.source),
                    )
                )
            return traces
        q = (
            self.last_query
            if self.last_query is not None
            else interaction.query
        )
        return [
            MemoryTrace(
                interaction_id=interaction.id,
                session_id=interaction.session_id,
                interaction_ids=[interaction.id],
                query=q,
                cortex_response=interaction.response,
                target_response=self.last_target,
                rationale=self.last_rationale,
                metadata=TraceMetadata(source=interaction.source),
            )
        ]


class _FakeIndex:
    """Minimal duck-type stand-in for NpzVectorIndex used by the loop."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, list]] = []
        self.updated: list[tuple[str, list]] = []

    def append(self, trace_id: str, vec) -> None:
        self.appended.append((trace_id, list(vec)))

    def update(self, trace_id: str, vec) -> bool:
        self.updated.append((trace_id, list(vec)))
        return True


class _ScriptedDeduper:
    """Deduper stub: pop from a queue of ``DedupResult``s per call."""

    def __init__(self, results: list[DedupResult] | None = None) -> None:
        self._results: list[DedupResult] = list(results or [])
        self.threshold = 0.82
        self.index = _FakeIndex()
        self.calls: list[MemoryTrace] = []

    def push(self, result: DedupResult) -> None:
        self._results.append(result)

    def route(self, trace: MemoryTrace) -> DedupResult:
        self.calls.append(trace)
        # Stash a deterministic dummy embedding so the loop has
        # something to hand back to the index on append/update.
        trace.metadata.extras["query_embedding"] = [0.1, 0.2, 0.3]
        if not self._results:
            r = DedupResult("create", None, 0.0)
        else:
            r = self._results.pop(0)
        if r.decision == "revise" and r.matched_trace_id:
            trace.metadata.extras["revise_of"] = r.matched_trace_id
        trace.metadata.extras["route_dedup_sim"] = float(r.similarity)
        return r


def _build_loop(
    tmp_path: Path,
    abstractor: Abstractor,
    *,
    deduper: _ScriptedDeduper | None = None,
) -> tuple[WakeSleepLoop, JsonlNeocortex]:
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
        deduper=deduper,
    )
    return loop, store


def test_wake_step_create_path_writes_with_session_id(tmp_path: Path) -> None:
    abs_ = _ScriptedAbstractor()
    abs_.last_target = "Paris"
    dedup = _ScriptedDeduper()  # default = CREATE
    loop, store = _build_loop(tmp_path, abs_, deduper=dedup)

    events: list[dict] = []
    interaction = Interaction(
        query="capital of France?", session_id="s-aaa",
    )
    traces = loop.wake_step(
        interaction, event_sink=lambda s, p: events.append({"stage": s, **p})
    )

    assert len(traces) == 1
    rows = store.entries()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s-aaa"
    assert rows[0]["interaction_ids"] == [interaction.id]

    stages = [e["stage"] for e in events]
    assert stages == [
        "uncertainty",
        "abstracting",
        "extracted",
        "dedup",
        "routed",
        "scored",
        "created",
    ]
    routed = next(e for e in events if e["stage"] == "routed")
    assert routed["decision"] == "CREATE"

    # Vector index received the new entry's embedding.
    assert len(dedup.index.appended) == 1
    assert dedup.index.appended[0][0] == traces[0].id


def test_wake_step_revise_path_overwrites_in_place(tmp_path: Path) -> None:
    abs_ = _ScriptedAbstractor()
    dedup = _ScriptedDeduper()
    loop, store = _build_loop(tmp_path, abs_, deduper=dedup)

    # First turn: CREATE.
    abs_.mode = "create"
    abs_.last_target = "Paris"
    abs_.last_rationale = "capital of France"
    first = Interaction(query="capital of France?", session_id="s-bbb")
    first_traces = loop.wake_step(first)
    assert len(first_traces) == 1
    original_trace_id = first_traces[0].id

    # Second turn: REVISE — deduper says "match the first trace".
    abs_.mode = "create"
    abs_.last_target = "Paris (capital, ~2.1M people)"
    abs_.last_rationale = "user added population detail"
    abs_.last_query = "What is the capital of France and its population?"
    dedup.push(DedupResult("revise", original_trace_id, 0.91))
    second = Interaction(
        query="Actually, please include the population.",
        session_id="s-bbb",
    )
    events: list[dict] = []
    second_traces = loop.wake_step(
        second,
        event_sink=lambda s, p: events.append({"stage": s, **p}),
    )

    assert len(second_traces) == 1
    second_trace = second_traces[0]
    assert second_trace.id == original_trace_id  # trace_id preserved

    # The JSONL store should still hold a single row, now updated.
    rows = store.entries()
    assert len(rows) == 1
    row = rows[0]
    assert row["trace_id"] == original_trace_id
    assert first.id in row["interaction_ids"]
    assert second.id in row["interaction_ids"]
    assistant = next(m for m in row["messages"] if m["role"] == "assistant")
    assert assistant["content"] == "Paris (capital, ~2.1M people)"
    user_msg = next(m for m in row["messages"] if m["role"] == "user")
    assert user_msg["content"] == "What is the capital of France and its population?"
    history = row.get("metadata", {}).get("extras", {}).get("history") or []
    assert len(history) == 1
    assert history[0]["target_response"] == "Paris"
    assert history[0]["query"] == "capital of France?"

    stages = [e["stage"] for e in events]
    assert "revised" in stages
    routed = next(e for e in events if e["stage"] == "routed")
    assert routed["decision"] == "REVISE"
    assert routed["trace_id"] == original_trace_id

    # Vector index updated for the revised entry.
    assert any(tid == original_trace_id for tid, _ in dedup.index.updated)


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
    """When the abstractor returns an empty list (triage drop, or no KPs
    extracted), the loop emits a ``dropped`` event and writes nothing."""
    abs_ = _ScriptedAbstractor()
    abs_.mode = "drop"
    loop, store = _build_loop(tmp_path, abs_)

    events: list[dict] = []
    interaction = Interaction(query="Hi! How are you?", session_id="s-drop")
    traces = loop.wake_step(
        interaction, event_sink=lambda s, p: events.append({"stage": s, **p})
    )

    assert traces == []
    assert store.entries() == []
    stages = [e["stage"] for e in events]
    assert "dropped" in stages
    assert "created" not in stages
    assert "revised" not in stages
    assert "routed" not in stages
    dropped = next(e for e in events if e["stage"] == "dropped")
    assert dropped["interaction_id"] == interaction.id
    assert dropped["session_id"] == "s-drop"


def test_wake_step_extracts_multiple_knowledge_points(tmp_path: Path) -> None:
    """A turn can yield multiple traces — one per KP — each routed
    independently through dedup + write_policy."""
    abs_ = _ScriptedAbstractor()
    abs_.kps = [
        ("我多大了？", "您今年三十岁。"),
        ("我住在哪里？", "您住在北京。"),
        ("我喜欢看什么电影？", "您喜欢看科幻电影。"),
    ]
    dedup = _ScriptedDeduper()
    loop, store = _build_loop(tmp_path, abs_, deduper=dedup)

    events: list[dict] = []
    interaction = Interaction(
        query="我今年三十岁，住在北京，喜欢看科幻电影。", session_id="s-multi",
    )
    traces = loop.wake_step(
        interaction, event_sink=lambda s, p: events.append({"stage": s, **p})
    )

    assert len(traces) == 3
    assert len(store.entries()) == 3
    # One ``created`` event per KP.
    created_events = [e for e in events if e["stage"] == "created"]
    assert len(created_events) == 3
    # ``extracted`` is emitted exactly once and reports the KP count.
    extracted = [e for e in events if e["stage"] == "extracted"]
    assert len(extracted) == 1
    assert extracted[0]["n_kps"] == 3
    # Vector index appended once per CREATE.
    assert len(dedup.index.appended) == 3
