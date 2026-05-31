# ADR-006: Embedding-Model Lifecycle

**Status:** Accepted
**Supersedes part of:** ADR-005 (single-embedder assumption)

## Context

ADR-005 introduced an embedding-routed dedup pipeline backed by a single
`Embedder` (`STEmbedder` over a fixed `all-MiniLM-L6-v2` checkpoint) and
a single `runs/neocortex/embeddings.npz` index. As soon as we wanted to
let users **download, swap, and delete** different embedding models —
mirroring the LLM (Cortex) catalog/manager flow — the single-embedder
assumption broke down: vectors from different encoders are not
comparable, so mixing them in one NPZ would produce nonsense cosine
scores.

## Decision

### Backend axis

We treat embedding models as a parallel kind of model with its own
top-level backend names rather than a `kind` sub-axis under the LLM
backends. The supported names are:

| Backend     | Catalog YAML                  | Library       |
|-------------|-------------------------------|---------------|
| `mlx_embed` | `models/catalogs/mlx_embed.yaml` | `mlx-embeddings` |
| `hf_embed`  | `models/catalogs/hf_embed.yaml`  | `sentence-transformers` |

`SUPPORTED_BACKENDS` (LLM) and `SUPPORTED_EMBED_BACKENDS` are kept
disjoint; `ALL_SUPPORTED_BACKENDS` is the union. This keeps the LLM
manager and the embedding manager from ever colliding on a backend
string.

The on-disk layout mirrors LLMs:

    model/<backend>/<model_id>/{config.json, model.safetensors, ...}

### NPZ partitioning

Each `(backend, model_id)` pair gets its own NPZ:

    runs/neocortex/embeddings/<backend>__<id>.npz

resolved via `hat.config.settings.embed_index_path_for(backend, id)`.
There is no fallback NPZ: when no embedder is active, dedup is off
and no vectors are written.

### Source-of-truth tag on memory rows

When an embedder is active, every accepted trace gets

    metadata.extras["embed_model"] = "<backend>/<id>"

stamped on it inside the wake step (`WakeSleepLoop.wake_step`) before
write/revise. This is the single source of truth for "which embedder
wrote this row" — the Memory page filters on it via
`/api/neocortex?embed_model=<tag>`. Rows written without an active
embedder carry no tag and are not surfaced by any filter.

### Manager + REST surface

`EmbeddingManager` (`models/embedding_manager.py`) is a one-to-one
mirror of `ModelManager`: catalog-driven `download` + SSE
`download_streaming`, lazy `load`, `set_active`, `delete` (refuses the
active model), and `unload_all`. The REST surface lives at
`/api/embedding-models` and matches the `/api/models` endpoint shapes
exactly so the UI can reuse its progress/state machinery.

Activation flow:

    POST /api/embedding-models/active {backend, id}
      → EmbeddingManager.set_active(...)
      → deps.swap_active_embedder() rebuilds loop.deduper with the new
        embedder + per-model NpzVectorIndex
      → loop.embed_tag updated so subsequent rows carry the new tag

Deactivation (`DELETE /api/embedding-models/active`) calls
`unload_all()` and drops `loop.deduper` to `None` — dedup is disabled
and subsequent rows are written without an `embed_model` tag.

### Vue UI surfaces

Two switchers, per the user requirement:

1. **Chat header** (`components/chat/ModelBar.vue`) — second `<n-select>`
   bound to the embedding store's `active`. Shows an "Embed: ..." tag.
2. **Memory page** (`views/MemoryView.vue`) — `<n-select>` filter that
   includes "All embedders" and one entry per installed `(backend, id)`
   pair. Selection updates `useMemoryStore().embedFilter` and re-fetches.

A new top-level "Embeddings" tab (`/embedding-models`) hosts the
download/activate/delete UI. It is intentionally a separate page from
`/models` so the LLM and embedder lifecycles stay visually independent.

### Reindex

`hat reindex-memory --backend <b> --id <id>` rebuilds the NPZ for an
arbitrary `(backend, id)` pair from `train.jsonl`. Without flags it
reindexes whichever embedder is currently active; if none is active,
the command exits with an error. There is no auto-reindex on embedder
switch — switching just re-binds the deduper to whatever vectors that
NPZ already contains.

## Consequences

* **Isolation:** rows written by different embedders never collide in a
  single NPZ. Cosine top-1 lookups stay meaningful.
* **No orphaned vectors on switch:** the old NPZ is left intact when a
  user switches embedders; switching back picks up where it left off.
* **HF embed backend ships empty.** The catalog file is wired in but
  `hf_embed.yaml = []` for now. The `HFEmbeddingModel` thin wrapper
  exists so adding entries is a one-line change.
* **Tag granularity = `<backend>/<id>` only.** No revision/hash. If a
  user pulls a newer revision of the same id, vectors mix. We accept
  this trade-off; users who care can manually `reindex-memory`.
* **No legacy / fallback path.** The codebase only supports managed
  embedders. Rows written without an active embedder carry no
  `embed_model` tag and are not surfaced by the Memory page filter.
