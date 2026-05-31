"""Wake–sleep loop orchestrator (paper §3.8 / Algorithm)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..utils.logging import format_text_block, get_logger, truncate
from .cortex.base import Cortex
from .hippocampus.abstraction import Abstractor
from .hippocampus.dedup import EmbeddingDeduper
from .hippocampus.replay import ReplayBuilder
from .hippocampus.scoring.uncertainty import UncertaintyEstimator
from .hippocampus.selection import WritePolicy
from .neocortex.store import NeocortexStore
from .oracle.base import Oracle
from .schemas import (
    Interaction,
    MemoryTrace,
    ReplayBatch,
    ScoreSignals,
    SWSObjective,
    SWSStats,
)
from .sws.trainer import SWSTrainer

log = get_logger(__name__)


@dataclass
class WakeSleepLoop:
    """Pure-plumbing orchestration of the paper Algorithm.

    Scoring is single-signal: only the cortex's logprob-based uncertainty on
    the original response gates trace creation. The session-aware abstractor
    decides CREATE vs REVISE from the natural multi-turn conversation.
    """

    cortex: Cortex
    abstractor: Abstractor
    uncertainty: UncertaintyEstimator
    write_policy: WritePolicy
    replay_builder: ReplayBuilder
    neocortex: NeocortexStore
    trainer: SWSTrainer
    oracle: Oracle | None = None
    oracle_threshold: float = 0.7
    deduper: EmbeddingDeduper | None = None
    embed_tag: str | None = None
    """Stable identifier of the embedder writing this turn (``"<backend>/<id>"``).
    Stamped on every accepted trace's ``metadata.extras['embed_model']`` so
    memory rows can be filtered by the embedder that wrote them. ``None``
    when no managed embedder is active (in which case dedup is also off
    and rows are not stamped)."""

    def wake_step(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[MemoryTrace] | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> list[MemoryTrace]:
        """Process one interaction; may write zero or more traces.

        Returns the list of accepted traces (each one corresponds to an
        independent knowledge point extracted from the turn). An empty
        list signals "nothing was written" — uncertainty gate failed,
        triage dropped the turn, no knowledge points survived
        extraction, or the write policy rejected every candidate.

        ``prior_traces`` is kept on the public API for backwards
        compatibility (older controllers pass it from
        ``prior_traces_for_session``); it is no longer used to route
        CREATE vs REVISE — that decision is made by the embedding
        deduper against the global vector index.

        ``event_sink`` is a callback invoked at key lifecycle points so
        controllers can forward live progress to the UI.
        """

        def _emit(stage: str, **payload: Any) -> None:
            if event_sink is None:
                return
            try:
                event_sink(stage, dict(payload))
            except Exception:  # noqa: BLE001 - event sinks must not break the loop
                pass

        if interaction.response is None:
            log.info(
                "wake.generate iid={} sid={} query='{}'",
                interaction.id, interaction.session_id,
                truncate(interaction.query or "", limit=160),
            )
            interaction.response = self.cortex.generate(
                interaction.query, context=interaction.context
            )

        log.info(
            "wake.step iid={} sid={} response_chars={}",
            interaction.id, interaction.session_id,
            len(interaction.response or ""),
        )
        log.debug(
            "wake.step.response\n{}",
            format_text_block(interaction.response or "", title="cortex response"),
        )

        u = self.uncertainty(interaction)
        signals = ScoreSignals(uncertainty=u)

        _emit(
            "uncertainty",
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            uncertainty=float(u),
            threshold=float(self.write_policy.threshold),
        )

        # Uncertainty gate: if the cortex was confident enough, the turn isn't
        # worth remembering. Skip abstraction entirely to save model calls.
        if u < self.write_policy.threshold:
            log.info(
                "wake.gate skipped iid={} U={:.4f} < threshold={:.4f}",
                interaction.id, u, self.write_policy.threshold,
            )
            _emit(
                "skipped",
                interaction_id=interaction.id,
                uncertainty=float(u),
                threshold=float(self.write_policy.threshold),
            )
            return []

        log.info(
            "wake.gate pass iid={} U={:.4f} >= threshold={:.4f}",
            interaction.id, u, self.write_policy.threshold,
        )

        # Oracle policy (paper §3.5): consult the external teacher when the
        # cortex was unsure of its own answer. The oracle output overrides
        # the response that gets persisted to the curated trace.
        oracle_used = False
        if self.oracle is not None and u > self.oracle_threshold:
            log.info(
                "wake.oracle.consult iid={} U={:.4f} > oracle_threshold={:.4f} name={}",
                interaction.id, u, self.oracle_threshold,
                getattr(self.oracle, "name", "oracle"),
            )
            correction = self.oracle.consult(interaction)
            if correction:
                log.info(
                    "wake.oracle.applied iid={} correction_chars={}",
                    interaction.id, len(correction),
                )
                interaction.response = correction
                oracle_used = True
            else:
                log.info("wake.oracle.empty iid={} (no correction)", interaction.id)

        _emit(
            "abstracting",
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            prior_trace_ids=[t.id for t in (prior_traces or [])],
        )

        # ---- abstractor: triage + multi-knowledge-point extraction -----
        # Routing (CREATE/REVISE) is no longer the abstractor's job; it
        # only decides keep-vs-drop and emits canonical Q/A pairs.
        try:
            traces = self.abstractor(interaction, event_sink=_emit)
        except TypeError:
            # Backward-compat: third-party abstractors without ``event_sink``.
            try:
                traces = self.abstractor(interaction)
            except TypeError:
                # Very old signature with prior_traces=. Nothing else to do.
                traces = self.abstractor(interaction, prior_traces=None)

        # Tolerate legacy single-trace abstractors during the migration.
        if traces is None:
            traces = []
        elif isinstance(traces, MemoryTrace):
            traces = [traces]

        if not traces:
            log.info(
                "wake.abstractor.dropped iid={} sid={}",
                interaction.id, interaction.session_id,
            )
            _emit(
                "dropped",
                interaction_id=interaction.id,
                session_id=interaction.session_id,
            )
            return []

        _emit(
            "extracted",
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            n_kps=len(traces),
            kps=[
                {
                    "trace_id": t.id,
                    "query": t.query,
                    "target": t.target_response,
                }
                for t in traces
            ],
        )

        written: list[MemoryTrace] = []
        for kp_idx, trace in enumerate(traces):
            trace.metadata.signals = signals
            if self.embed_tag:
                trace.metadata.extras["embed_model"] = self.embed_tag
            if oracle_used:
                trace.metadata.extras["oracle"] = True
                trace.metadata.extras["oracle_name"] = getattr(
                    self.oracle, "name", "oracle"
                )
                if "oracle" not in trace.metadata.source:
                    trace.metadata.source = f"{trace.metadata.source}+oracle"

            # ---- dedup routing ----------------------------------------
            decision_kind = "create"
            matched_trace_id: str | None = None
            similarity = 0.0
            if self.deduper is not None:
                result = self.deduper.route(trace)
                decision_kind = result.decision
                matched_trace_id = result.matched_trace_id
                similarity = result.similarity
                _emit(
                    "dedup",
                    trace_id=trace.id,
                    kp_index=kp_idx,
                    decision=decision_kind,
                    matched_trace_id=matched_trace_id,
                    similarity=float(similarity),
                    threshold=float(self.deduper.threshold),
                )

            _emit(
                "routed",
                decision="REVISE" if decision_kind == "revise" else "CREATE",
                trace_id=matched_trace_id or trace.id,
                kp_index=kp_idx,
                interaction_id=interaction.id,
                similarity=float(similarity),
                reason=trace.metadata.extras.get("extract_rationale"),
            )

            # ---- write policy -----------------------------------------
            decision = self.write_policy.decide(trace, signals)
            log.info(
                "wake.write.decide trace_id={} kp_idx={} score={:.4f} threshold={:.4f} accepted={}",
                trace.id, kp_idx, decision.score, decision.threshold,
                decision.accepted,
            )
            _emit(
                "scored",
                trace_id=trace.id,
                kp_index=kp_idx,
                score=float(decision.score),
                threshold=float(decision.threshold),
                accepted=bool(decision.accepted),
                signals=signals.model_dump(),
            )

            if not decision.accepted:
                _emit(
                    "rejected",
                    trace_id=trace.id,
                    kp_index=kp_idx,
                    score=float(decision.score),
                    threshold=float(decision.threshold),
                )
                continue

            query_vec = trace.metadata.extras.pop("query_embedding", None)

            # ---- REVISE path (matched existing trace) -----------------
            if decision_kind == "revise" and matched_trace_id:
                prior_target = self._fetch_trace_by_id(matched_trace_id)
                history_entry = None
                if prior_target is not None:
                    history_entry = {
                        "interaction_id": prior_target.interaction_id,
                        "query": prior_target.query,
                        "target_response": prior_target.target_response,
                        "rationale": prior_target.rationale,
                        "timestamp": prior_target.metadata.timestamp.isoformat()
                        if prior_target.metadata
                        and prior_target.metadata.timestamp
                        else None,
                    }
                try:
                    self.neocortex.revise(
                        matched_trace_id,
                        query=trace.query,
                        target_response=trace.target_response,
                        rationale=trace.rationale,
                        append_interaction_id=interaction.id,
                        push_history_entry=history_entry,
                    )
                    trace.id = matched_trace_id
                    if prior_target is not None:
                        ids = list(prior_target.interaction_ids or [])
                        if not ids:
                            ids = [prior_target.interaction_id]
                        if interaction.id not in ids:
                            ids.append(interaction.id)
                        trace.interaction_ids = ids
                    if (
                        self.deduper is not None
                        and query_vec is not None
                    ):
                        try:
                            self.deduper.index.update(
                                matched_trace_id, query_vec
                            )
                        except Exception as e:  # noqa: BLE001
                            log.warning(
                                "vector_index.update failed trace_id={}: {}: {}",
                                matched_trace_id, type(e).__name__, e,
                            )
                    log.info(
                        "wake.write.revised trace_id={} iid={} target='{}'",
                        matched_trace_id, interaction.id,
                        truncate(trace.target_response or "", limit=120),
                    )
                    _emit(
                        "revised",
                        trace_id=matched_trace_id,
                        kp_index=kp_idx,
                        interaction_id=interaction.id,
                        target_response=trace.target_response,
                        rationale=trace.rationale,
                    )
                    written.append(trace)
                    continue
                except NotImplementedError:
                    # Backend doesn't support in-place revise; fall through
                    # to CREATE so the data isn't lost.
                    trace.metadata.extras.pop("revise_of", None)

            # ---- CREATE path (default) --------------------------------
            self.neocortex.write(trace, decision)
            if self.deduper is not None and query_vec is not None:
                try:
                    self.deduper.index.append(trace.id, query_vec)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "vector_index.append failed trace_id={}: {}: {}",
                        trace.id, type(e).__name__, e,
                    )
            log.info(
                "wake.write.created trace_id={} iid={} sid={} target='{}'",
                trace.id, interaction.id, trace.session_id,
                truncate(trace.target_response or "", limit=120),
            )
            _emit(
                "created",
                trace_id=trace.id,
                kp_index=kp_idx,
                interaction_id=interaction.id,
                session_id=trace.session_id,
                target_response=trace.target_response,
                rationale=trace.rationale,
            )
            written.append(trace)

        return written

    # ------------------------------------------------------------------
    def _fetch_trace_by_id(self, trace_id: str) -> MemoryTrace | None:
        """Best-effort lookup of an existing trace from the neocortex.

        Used on the REVISE path to preserve the prior target as a
        history entry. Returns ``None`` if the backend doesn't expose a
        single-trace fetch — the revise still proceeds, just without a
        history entry.
        """
        getter = getattr(self.neocortex, "get_entry", None)
        if getter is None:
            return None
        try:
            row = getter(trace_id)
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        try:
            from ..memory.curated.jsonl_store import _sft_to_trace

            trace, _ = _sft_to_trace(row)
            return trace
        except Exception:  # noqa: BLE001
            return None

    def sleep_step(
        self,
        *,
        cycle: int = 0,
        k: int = 32,
        objective: SWSObjective | None = None,
    ) -> SWSStats:
        """Run one SWS cycle: sample replay batch, fit, return stats."""
        objective = objective or SWSObjective()
        traces = list(self.neocortex.sample(k))
        examples = [ex for t in traces for ex in self.replay_builder(t)]
        batch = ReplayBatch(examples=examples, cycle=cycle)
        log.info(
            "sleep.step cycle={} k={} n_traces={} n_examples={} trainer={}",
            cycle, k, len(traces), len(examples), type(self.trainer).__name__,
        )
        stats = self.trainer.fit(batch, objective)
        log.info(
            "sleep.done cycle={} loss_sup={:.4f} duration={:.2f}s",
            cycle, getattr(stats, "loss_sup", 0.0), getattr(stats, "duration_seconds", 0.0),
        )
        return stats
