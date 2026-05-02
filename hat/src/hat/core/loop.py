"""Wake–sleep loop orchestrator (paper §3.8 / Algorithm)."""

from __future__ import annotations

from dataclasses import dataclass

from .cortex.base import Cortex
from .hippocampus.abstraction import Abstractor
from .hippocampus.replay import ReplayBuilder
from .hippocampus.scoring.feedback import FeedbackExtractor
from .hippocampus.scoring.novelty import NoveltyEstimator
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


@dataclass
class WakeSleepLoop:
    """Pure-plumbing orchestration of the paper Algorithm.

    All collaborators are injected; the loop only sequences calls and enforces
    the ``raw → curated → replay → params`` flow. Substitute any component to
    run an ablation.
    """

    cortex: Cortex
    abstractor: Abstractor
    uncertainty: UncertaintyEstimator
    feedback: FeedbackExtractor
    novelty: NoveltyEstimator
    write_policy: WritePolicy
    replay_builder: ReplayBuilder
    neocortex: NeocortexStore
    trainer: SWSTrainer
    oracle: Oracle | None = None
    oracle_threshold: float = 0.7

    def wake_step(self, interaction: Interaction) -> MemoryTrace | None:
        """Process one interaction; may write a trace into the Neocortex."""
        if interaction.response is None:
            interaction.response = self.cortex.generate(
                interaction.query, context=interaction.context
            )

        u = self.uncertainty(interaction)
        f = self.feedback(interaction)

        if self.oracle is not None and (u > self.oracle_threshold or f > 0):
            interaction.user_correction = (
                interaction.user_correction or self.oracle.consult(interaction)
            )

        trace = self.abstractor(interaction)
        n = self.novelty(trace)
        signals = ScoreSignals(uncertainty=u, feedback=f, novelty=n)
        trace.metadata.signals = signals

        decision = self.write_policy.decide(trace, signals)
        if decision.accepted:
            self.neocortex.write(trace, decision)
            return trace
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
        return self.trainer.fit(batch, objective)
