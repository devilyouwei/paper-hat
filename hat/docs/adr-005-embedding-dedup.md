# ADR 005 — Embedding-Based CREATE/REVISE Routing

**Status:** Accepted, 2026-05.

## Context

Until ADR-005 was adopted, the abstractor (paper Eq. *abstraction*) was
responsible for two separate jobs in a single LLM call:

1. **Extraction** — distil the canonical `(query, target)` pair from
   the current turn.
2. **Routing** — decide whether to **CREATE** a new memory entry or
   **REVISE** an existing one (and, if REVISE, return the matched
   `trace_id` and a rewritten canonical query).

In production we observed three failure modes from small (~3B) models:

- **Hallucinated `trace_id`s** — the model produced ids that didn't
  exist in the candidate list, causing silent fall-through to the
  IdentityAbstractor and dataset-poisoning rows like
  `("不对，X 其实是 Z", …)` getting written verbatim.
- **Inconsistent canonical queries** on REVISE — the model copied the
  meta-correction utterance into `query`, breaking the SFT replay
  pair.
- **Routing thrash** — the same fact phrased two slightly different
  ways yielded two CREATEs instead of one CREATE + one REVISE.

The abstractor's *extraction* role is genuinely semantic (free-text → 
canonical Q/A), but the *routing* role is fundamentally **geometric**:
"is the new canonical query close enough to an existing one to count
as a paraphrase?" That comparison does not require an LLM.

## Decision

Split the abstractor into two stages and move routing out of the LLM:

1. **Triage** (LLM) — decide *keep vs drop* for the current turn.
   Returns `{keep: bool, reason: str}`.
2. **Extract** (LLM) — emit a list of canonical knowledge points. A
   single turn may produce multiple `(query, target)` pairs (e.g.
   *"我三十岁，住在北京，喜欢看科幻电影"* → three KPs). Returns
   `{knowledge_points: [{query, target, rationale}, ...]}`. An empty
   list is a valid drop.
3. **Dedup** (deterministic, geometric) — for each extracted KP,
   embed the canonical `query` with a small SentenceTransformer and
   compare against an NPZ-backed vector index of all existing
   memory rows. If cosine similarity ≥ `dedup_threshold` (default
   `0.82`), route to **REVISE** the matched trace; otherwise route
   to **CREATE** a new entry.

The default model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d,
~118 MB, supports zh + en) on `cpu`/`mps`/`cuda` auto-detected.

The vector index is a single NPZ file (`runs/neocortex/embeddings.npz`)
with two arrays: `ids: <U…` and `vecs: float32[N, D]`. Lookup is a
single `vecs @ q` matmul (cosine, since every row is L2-normalised).
This is sized for ≲10⁵ rows; if we ever outgrow that, swap in a real
ANN store behind the same protocol.

## Consequences

### Pros

- Routing is **deterministic and auditable** — `route_dedup_sim` is
  recorded on every trace's `metadata.extras`, and the matched
  `trace_id` (if any) is recorded as `revise_of`.
- The LLM job shrinks: each step has a narrower output schema, so
  small models are more reliable at it.
- Multi-KP turns are now a first-class concept; the wake step writes
  zero or more traces per turn instead of exactly zero or one.
- Adding new triage / extraction prompts no longer risks silently
  changing routing behaviour.

### Cons

- We introduce a new ~120 MB model dependency at import time, even
  for users who only run the noop cortex. Mitigated by lazy loading
  and a graceful fallback to a deterministic hash embedder if the
  SentenceTransformer import fails.
- The vector index is **a sidecar that must stay in sync** with
  `train.jsonl`. The CLI ships `hat reindex-memory` for recovery and
  `JsonlNeocortex.delete` / `update` propagate to the index when an
  embedder is injected.
- Threshold `0.82` is empirical; users who write very short
  paraphrases may need to tune it via `HAT_DEDUP_THRESHOLD`.

## Migration

- `wake_step` now returns `list[MemoryTrace]` instead of
  `MemoryTrace | None`. Existing controllers are updated; the empty
  list signals "nothing written".
- The legacy `prior_traces` argument on `wake_step` is preserved
  for API compatibility but no longer informs routing.
- `Abstractor.__call__` returns `list[MemoryTrace]`. The fallback
  `IdentityAbstractor` wraps its single trace in a 1-element list.
- The old `abstraction_route.md` prompt is deleted.
- The Trace timeline in the UI gains two new lifecycle stages:
  `extracted` (with the KP count) and `dedup` (with the matched
  similarity and threshold).

## Configuration

```python
# hat.config.settings
dedup_enabled: bool = True
dedup_threshold: float = 0.82
embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_device: str = "auto"  # auto | cpu | cuda | mps
embed_index_path: Path = Path("runs/neocortex/embeddings.npz")
```

Override via `HAT_DEDUP_ENABLED`, `HAT_DEDUP_THRESHOLD`, etc.
