"""Embedding-based CREATE-vs-REVISE routing for memory traces.

The abstractor (see :mod:`hat.core.hippocampus.abstraction`) is now a
pure *extractor*: it returns one or more canonical ``(query, target)``
pairs for the current turn but does not decide whether each pair should
overwrite an existing memory or be appended as a new one.

That decision used to live in the abstractor's routing prompt and was
unreliable: small models hallucinated trace_ids, copied meta-correction
utterances into ``query``, or rewrote canonical queries the wrong way.
We move it here, into a deterministic geometric step:

1. Embed the trace's canonical ``query`` with the configured Embedder.
2. Look up the nearest neighbour in the curated memory's vector index.
3. If cosine similarity is above ``threshold``, route to **REVISE** the
   matched trace; otherwise route to **CREATE** a new entry.

Embedding fingerprints are cached on ``trace.metadata.extras`` so the
loop can hand them straight to the index after a successful write,
avoiding a second embed pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...utils.logging import get_logger
from ..protocols import Embedder
from ..schemas import MemoryTrace
from ...memory.curated.vector_index import NpzVectorIndex

log = get_logger(__name__)

Decision = Literal["create", "revise"]


@dataclass(frozen=True)
class DedupResult:
    """Outcome of a single dedup routing call."""

    decision: Decision
    matched_trace_id: str | None
    similarity: float


class EmbeddingDeduper:
    """Geometric router: nearest-neighbour cosine similarity in [0, 1].

    The routed similarity, the matched trace_id (if any), and the raw
    query embedding are stored on ``trace.metadata.extras`` under keys
    ``route_dedup_sim``, ``revise_of`` and ``query_embedding`` so that
    downstream consumers (write policy, jsonl store, vector index)
    don't have to re-embed.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: NpzVectorIndex,
        threshold: float,
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold!r}"
            )
        self.embedder = embedder
        self.index = index
        self.threshold = float(threshold)

    def route(self, trace: MemoryTrace) -> DedupResult:
        query = (trace.query or "").strip()
        if not query:
            log.warning(
                "dedup.empty_query trace_id={} → forcing CREATE", trace.id
            )
            trace.metadata.extras["route_dedup_sim"] = 0.0
            return DedupResult("create", None, 0.0)

        vec = self.embedder.embed([query])[0]
        # Stash as a plain list so it round-trips through pydantic /
        # JSON cleanly. The loop will hand it back to the index on
        # successful CREATE without re-embedding.
        trace.metadata.extras["query_embedding"] = list(vec)

        match = self.index.top1(vec)
        sim = float(match.similarity) if match is not None else 0.0
        trace.metadata.extras["route_dedup_sim"] = sim

        if match is not None and sim >= self.threshold:
            trace.metadata.extras["revise_of"] = match.trace_id
            log.info(
                "dedup.route decision=revise trace_id={} matched={} sim={:.4f} thr={:.2f}",
                trace.id, match.trace_id, sim, self.threshold,
            )
            return DedupResult("revise", match.trace_id, sim)

        log.info(
            "dedup.route decision=create trace_id={} top1_sim={:.4f} thr={:.2f}",
            trace.id, sim, self.threshold,
        )
        return DedupResult("create", None, sim)
