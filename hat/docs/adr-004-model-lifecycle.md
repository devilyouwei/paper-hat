# ADR-004 — Model lifecycle: catalog, hot-swap, unload

## Status
Accepted.

## Context
Earlier iterations selected the Cortex's weights via env vars
(`HAT_HF_MODEL_PATH`, …). That worked for a single-model demo but broke
several real workflows:

1. The user wants to compare backends and quantisations interactively from
   the UI, without restarting the server or editing `.env`.
2. On a CUDA box the previous model has to be released before the next is
   loaded — otherwise peak VRAM is `old + new` and large checkpoints OOM.
3. We download multi-GB weights from HuggingFace; duplicating them under
   both `~/.cache/huggingface` *and* the project tree is unacceptable.

## Decision

### Convention-based weights
Models live under `model/<backend>/<id>/`. There is no env var for a
concrete path. Discovery is driven by a YAML **catalog** (one file per
backend in `src/hat/models/catalogs/`); each entry has an `id`, a HuggingFace
`repo_id`, a display name, and an optional size estimate.

### `ModelManager`
A process-wide singleton owns:

* the `(backend, id) → Cortex` cache,
* the active-pointer the loop reads from, and
* `download` / `set_active` / `unload` / `unload_all` lifecycle methods.

`download` calls `huggingface_hub.snapshot_download` with `local_dir=
model/<backend>/<id>` and `cache_dir=model/.hf-cache`, so blobs are reused
across catalog entries that share weights but never duplicated outside the
project tree.

### Evict-before-load on activation
`set_active(backend, id)` first **evicts every other cached Cortex and
releases its allocator memory**, then builds the new one. This guarantees
peak memory is bounded by a single resident model. The loop's `cortex`
attribute is parked on `NoopCortex` during the swap (see
`api.deps.swap_active_cortex`) so a stale strong reference cannot keep the
old weights alive.

### Explicit unload
`DELETE /api/models/active` and the Gradio **Unload** button call
`unload_all`, which flushes the cache and points the loop back at
`NoopCortex`. Operators can free GPU/Metal memory without restarting the
server (useful for long-running interactive sessions and shared dev boxes).

### Allocator hygiene
Dropping Python references is not enough — frameworks cache freed blocks
in their own pools. `_release_cortex` therefore:

1. nulls the `model` / `tokenizer` attributes on the Cortex and its inner
   LM wrapper,
2. runs `gc.collect()`, and
3. calls `torch.cuda.empty_cache()` / `torch.mps.empty_cache()` /
   `mlx.core.clear_cache()` whenever the corresponding framework is
   importable.

## Consequences
* Switching models (and backends) is a `POST /api/models/active` away; the
  UI has buttons but the REST contract is the source of truth.
* Adding a model is a YAML edit, no Python changes.
* `set_active` may briefly run with no active Cortex (during the eviction
  → rebuild window). The loop tolerates this because we explicitly fall
  back to `NoopCortex`.
* The `[mlx]` / `[hf]` extras stay optional; the Manager imports each
  backend lazily so a base install still loads cleanly.
