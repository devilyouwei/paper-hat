# `memory/` — storage

Strictly two subpackages, with no shared state:

* `raw/` — append-only chat history. Append on every wake step.
* `curated/` — Neocortex. Written **only** through `NeocortexStore.write(trace, decision)`,
  and the store rejects writes without a valid `WriteDecision`. See ADR-002.

The training pipeline must never import from `raw/`.
