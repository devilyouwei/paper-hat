"""Wake–sleep loop orchestrator (paper §3.8 / Algorithm)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..utils.logging import format_text_block, get_logger, truncate
from .cortex.base import Cortex
from .hippocampus.abstraction import Abstractor
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
    the original response gates trace creation. Feedback / novelty signals
    were removed in favour of letting the session-aware abstractor decide
    CREATE vs REVISE from the natural multi-turn conversation.
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

    def wake_step(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[MemoryTrace] | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> MemoryTrace | None:
        """Process one interaction; may write a trace into the Neocortex.

        ``prior_traces`` is the (optionally session-scoped) list of traces the
        abstractor may revise instead of creating a new one. ``event_sink`` is
        a callback invoked at key lifecycle points so controllers can forward
        live progress to the UI.
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
            return None

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

        # Build the prompt-friendly view of prior traces for the abstractor.
        # The query is the **canonical** user-side input that the trace
        # will be replayed against; it is the abstractor's main signal on
        # REVISE and must NOT be silently clipped. Keep it whole (queries
        # are short by construction) and only truncate the long target.
        prior_view: list[dict] | None = None
        if prior_traces:
            prior_view = []
            for t in prior_traces:
                q = (t.query or "")[:512]
                tgt = (t.target_response or t.cortex_response or "")[:512]
                prior_view.append(
                    {"trace_id": t.id, "query": q, "target": tgt}
                )

        _emit(
            "abstracting",
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            prior_trace_ids=[t.id for t in (prior_traces or [])],
        )

        try:
            trace = self.abstractor(interaction, prior_traces=prior_view)
        except TypeError:
            # Backward-compat with abstractors that haven't been updated to
            # accept the keyword argument.
            trace = self.abstractor(interaction)

        if trace is None:
            # Abstractor explicitly dropped the turn (router decided neither
            # novel nor user-supervised). Skip writing entirely.
            log.info(
                "wake.abstractor.dropped iid={} sid={}",
                interaction.id, interaction.session_id,
            )
            _emit(
                "dropped",
                interaction_id=interaction.id,
                session_id=interaction.session_id,
            )
            return None

        revise_of = (trace.metadata.extras or {}).get("revise_of")
        prior_by_id = {t.id: t for t in (prior_traces or [])}
        prior_target = prior_by_id.get(revise_of) if revise_of else None

        _emit(
            "routed",
            decision="REVISE" if revise_of else "CREATE",
            trace_id=revise_of or trace.id,
            interaction_id=interaction.id,
            novelty=trace.metadata.extras.get("route_novelty"),
            user_signal=trace.metadata.extras.get("route_user_signal"),
            reason=trace.metadata.extras.get("route_reason"),
        )

        trace.metadata.signals = signals
        if oracle_used:
            trace.metadata.extras["oracle"] = True
            trace.metadata.extras["oracle_name"] = getattr(self.oracle, "name", "oracle")
            if "oracle" not in trace.metadata.source:
                trace.metadata.source = f"{trace.metadata.source}+oracle"

        decision = self.write_policy.decide(trace, signals)
        log.info(
            "wake.write.decide trace_id={} score={:.4f} threshold={:.4f} accepted={} signals={}",
            trace.id, decision.score, decision.threshold,
            decision.accepted, signals.model_dump(),
        )
        _emit(
            "scored",
            trace_id=trace.id,
            score=float(decision.score),
            threshold=float(decision.threshold),
            accepted=bool(decision.accepted),
            signals=signals.model_dump(),
        )

        if not decision.accepted:
            log.info(
                "wake.write.rejected trace_id={} score={:.4f} < threshold={:.4f}",
                trace.id, decision.score, decision.threshold,
            )
            _emit(
                "rejected",
                trace_id=trace.id,
                score=float(decision.score),
                threshold=float(decision.threshold),
            )
            return None

        if revise_of and prior_target is not None:
            history_entry = {
                "interaction_id": prior_target.interaction_id,
                "query": prior_target.query,
                "target_response": prior_target.target_response,
                "rationale": prior_target.rationale,
                "timestamp": prior_target.metadata.timestamp.isoformat()
                if prior_target.metadata and prior_target.metadata.timestamp
                else None,
            }
            try:
                self.neocortex.revise(
                    revise_of,
                    query=trace.query,
                    target_response=trace.target_response,
                    rationale=trace.rationale,
                    append_interaction_id=interaction.id,
                    push_history_entry=history_entry,
                )
                trace.id = revise_of
                ids = list(prior_target.interaction_ids or [])
                if not ids:
                    ids = [prior_target.interaction_id]
                if interaction.id not in ids:
                    ids.append(interaction.id)
                trace.interaction_ids = ids
                log.info(
                    "wake.write.revised trace_id={} iid={} target='{}'",
                    revise_of, interaction.id,
                    truncate(trace.target_response or "", limit=120),
                )
                _emit(
                    "revised",
                    trace_id=revise_of,
                    interaction_id=interaction.id,
                    target_response=trace.target_response,
                    rationale=trace.rationale,
                )
                return trace
            except NotImplementedError:
                trace.metadata.extras.pop("revise_of", None)

        # CREATE path (default).
        self.neocortex.write(trace, decision)
        log.info(
            "wake.write.created trace_id={} iid={} sid={} target='{}'",
            trace.id, interaction.id, trace.session_id,
            truncate(trace.target_response or "", limit=120),
        )
        _emit(
            "created",
            trace_id=trace.id,
            interaction_id=interaction.id,
            session_id=trace.session_id,
            target_response=trace.target_response,
            rationale=trace.rationale,
        )
        return trace

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
