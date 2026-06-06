# `core/` — paper algorithms & runtime backends

The implementation home for the HAT algorithm. Two kinds of code live here:

1. **Pure-Python algorithm plumbing** — `loop.py` plus the `hippocampus/`,
   `neocortex/`, `oracle/`, and `sws/` packages. These depend only on the
   abstract Protocols / ABCs declared in [`hat/abstract/`](../abstract) and
   carry no FastAPI or web concerns.
2. **Concrete model backends & their lifecycle** — `cortex/`,
   `neocortex/embeddings/`, `lifecycle/`, plus the wiring in `runtime/` and the
   raw-history persistence in `sessions/`. These *do* import heavy deps
   (`mlx-lm`, `transformers`, `sentence-transformers`, `huggingface_hub`) and
   talk to disk / the network — but always lazily, behind an abstract seam.

> **Where do the interfaces live?** All Protocols and ABCs (`Cortex`,
> `LanguageModel`, `Abstractor`, `WritePolicy`, `NeocortexStore`, `Oracle`,
> `SWSTrainer`, `SessionStore`, the `Interaction` / `MemoryTrace` schemas, …)
> are defined under [`hat/abstract/`](../abstract). `core/` only contains
> *implementations* of those seams. The one exception is the `Embedder`
> Protocol, which is co-located with its adapter in
> `neocortex/embeddings/managed.py`.

---

## Directory map

| Path | Paper § | Role |
| --- | --- | --- |
| `loop.py` | §3.8 | `WakeSleepLoop` — wake/sleep orchestrator |
| `cortex/` | §3.3 | Online interaction model backends (`LanguageModel` + `Cortex`) |
| `hippocampus/` | §3.4 | Abstraction, selection (write policy), dedup, replay, scoring |
| `neocortex/` | §3.6 | Long-term curated memory store + vector index + embedders |
| `oracle/` | §3.5 | On-demand external teacher |
| `sws/` | §3.7 | Slow-wave-sleep trainer |
| `lifecycle/` | — | Model catalog, download, load/unload, active-pointer |
| `runtime/` | — | Composition root: wires concrete singletons together |
| `sessions/` | — | Raw chat-history persistence (the *only* writer of raw data) |

`core/__init__.py` intentionally exports just `loop`; everything else is
reached through its subpackage (e.g. `from hat.core.lifecycle import
get_manager`).

---

## `loop.py` — wake/sleep orchestrator (§3.8)

`WakeSleepLoop` is a `@dataclass` of injected collaborators (a `Cortex`, an
`Abstractor`, a `WritePolicy`, an `UncertaintyEstimator`, a `NeocortexStore`,
an optional `Oracle`, an `SWSTrainer`, and an optional `EmbeddingDeduper`). It
owns no construction logic — `runtime/container.py` assembles it.

| Method | Phase | Responsibility |
| --- | --- | --- |
| `wake_step(interaction)` | Wake | Score the cortex's uncertainty, run the abstractor (CREATE vs REVISE), apply the write policy, dedup, and persist accepted traces to the neocortex. |
| `sleep_step(objective)` | Sleep | Build a supervised replay batch from retained traces and hand it to the `SWSTrainer`. |

Single-signal scoring: only the cortex's logprob-based uncertainty gates trace
creation; the session-aware abstractor decides CREATE vs REVISE from the
natural multi-turn conversation.

---

## `cortex/` — online interaction model (§3.3)

The `Cortex` ABC ([`hat/abstract/cortex.py`](../abstract/cortex.py)) defines the
canonical surface the wake step calls:

| Method | Used by |
| --- | --- |
| `generate(query, *, context=None, **kw)` | HAT-native `/chat` |
| `chat(messages, **kw)` | OpenAI-compatible non-streaming path |
| `stream_chat(messages, **kw)` *(generator)* | OpenAI-compatible SSE path |
| `uncertainty(interaction)` | Hippocampus selection gate |

Each model-backed cortex wraps a `LanguageModel` (the structural Protocol in
`hat/abstract/cortex.py`, requiring `name`, `generate`, `token_logprobs`).

| File | Defines | Notes |
| --- | --- | --- |
| `noop.py` | `NoopCortex` | Env-free fallback used when no model is active and by tests; echoes input, `uncertainty()` returns `0.5`. |
| `mlx.py` | `MLXLanguageModel`, `MLXCortex`, `build_mlx_model` | Apple Silicon via `mlx-lm`. |
| `hf.py` | `HFLanguageModel`, `HFCortex`, `build_hf_model` | HuggingFace Transformers (device / dtype / offload / 4-bit). |
| `cloud.py` | `CloudLanguageModel`, `CloudCortex`, `build_cloud_model` | OpenAI-compatible `/v1/chat/completions`. Platform-independent, no local weights. |

`chat` / `stream_chat` pop `enable_thinking` (or a nested
`chat_template_kwargs`) and forward it to the tokenizer template, so think-mode
toggling lives at the backend layer. `CloudCortex.uncertainty()` derives from
API logprobs when available, else falls back to `0.5`. The `lifecycle/`
manager dispatches each backend's `build_*` factory directly from
`ModelManager._build`.

---

## `hippocampus/` — selective consolidation (§3.4)

| File | Defines | Role |
| --- | --- | --- |
| `abstraction.py` | `IdentityAbstractor`, `LLMAbstractor` | Identity copies fields verbatim; the LLM variant runs a two-step judge (triage → extract knowledge points). |
| `selection.py` | `UncertaintyGatePolicy` | Write policy: `score = signals.uncertainty`; a trace is accepted when the score ≥ threshold. |
| `dedup.py` | `EmbeddingDeduper` | Geometric CREATE-vs-REVISE router: embed the query, nearest-neighbour cosine lookup against the active vector index, threshold decision; caches the embedding on the trace. |
| `replay.py` | `SupervisedReplayBuilder` | Converts a retained trace into a `(query, target_response)` training pair. |
| `scoring/uncertainty.py` | `ConstantUncertainty`, `LogprobUncertainty` | `U(x) = 1 − exp(mean_t log p(y_t | x, y_<t))`. |
| `scoring/llm_judge.py` | `load_prompt`, `render`, `call_judge`, score parsing | Shared LLM-as-judge utilities (tolerant of `"0.7"`, `"score: 0.7"`, `"7/10"`, …). |
| `prompts/*.md` | — | Abstraction triage / extraction prompt templates. |

Exports (`__init__.py`) re-surface both the concrete classes above and the
abstract `Abstractor` / `WritePolicy` / `Deduper` / `ReplayBuilder` /
`UncertaintyEstimator` ABCs for convenient typing.

---

## `neocortex/` — long-term curated memory (§3.6)

| File | Defines | Role |
| --- | --- | --- |
| `store.py` | `InMemoryNeocortex` | Reference list-backed store with a score cache. |
| `jsonl_store.py` | `JsonlNeocortex` | JSONL aligned with OpenAI/HF SFT (`messages` + `trace_id` + `score` + metadata); optional sidecar `traces.jsonl`. |
| `vector_index.py` | `NpzVectorIndex` | `(ids, vecs)` table persisted as `.npz`, L2-normalised; `append` / `update` / `lookup` / `load` / `save`. |
| `embeddings/managed.py` | `Embedder` (Protocol), `ManagedEmbedder` | The embedder seam + an adapter that tags vectors with `<backend>/<model_id>`. |
| `embeddings/mlx.py` | `MLXEmbeddingModel`, `build_mlx_embed_model` | `mlx-embeddings` adapter. |
| `embeddings/hf.py` | `HFEmbeddingModel`, `build_hf_embed_model` | `SentenceTransformer` wrapper. |
| `embeddings/cloud.py` | `CloudEmbeddingModel`, `build_cloud_embed_model` | OpenAI-compatible `/v1/embeddings`; L2-normalises client-side. |

**Write-token contract.** Curated memory may only be written through
`NeocortexStore.write(trace, decision)`, and the store rejects anything whose
`decision` is not a `WriteDecision` with `accepted=True` and
`trace_id == trace.id` (raising `NeocortexWriteError`). This enforces the
type-level boundary between raw chat history (`sessions/`) and training data —
see [ADR-002](../../../docs/adr-002-raw-vs-curated.md).

---

## `oracle/` — on-demand teacher (§3.5)

| File | Defines | Role |
| --- | --- | --- |
| `openai_compat.py` | `OpenAICompatibleOracle` | `/v1/chat/completions` client (OpenAI, vLLM, Ollama, Groq, Together, …). Network / HTTP errors degrade to an empty string rather than crashing the wake loop. |
| `noop.py` | `NoopOracle` | Identity oracle — returns the cortex response unchanged. |
| `cost_guard.py` | `CostGuard`, `OracleQuotaExceeded` | Sliding-window RPS limiter + per-UTC-day budget; thread-safe with an optional JSONL audit trail. |
| `prompts.py` | `ORACLE_SYSTEM` | System prompt asking for a clean ground-truth reply (no preamble). |

---

## `sws/` — slow-wave-sleep trainer (§3.7)

| File | Defines | Role |
| --- | --- | --- |
| `trainer.py` | `DryRunTrainer` | No-op `SWSTrainer` used by `hat sleep --dry-run` and tests; `fit()` returns `SWSStats` with the replay count and zero loss. |

---

## `lifecycle/` — model catalog & runtime management

The download / load / unload / activate machinery backing the `/api/models`
and `/api/embedding-models` endpoints. Import-light at the top level (no
torch / mlx); heavy deps are imported lazily inside the build paths.

| File | Defines | Role |
| --- | --- | --- |
| `catalog.py` | `CatalogEntry`, `load_catalog`, `SUPPORTED_BACKENDS`, `SUPPORTED_EMBED_BACKENDS`, `CLOUD_BACKENDS`, `is_cloud_backend` | Catalog schema + backend constants. Defaults ship as package data; override per project at `<HAT_MODEL_ROOT>/<backend>/catalog.yaml`. |
| `base.py` | `BaseModelManager[T]` | Generic install / download / load / activate state machine shared by both managers below: instance cache, active `(backend, id)` pointer, cancellable SSE download, memory-releasing teardown, and the cloud special-casing — all in one place. Subclasses declare the supported backends / weight suffixes / error type and implement only `_build`. |
| `manager.py` | `ModelManager`, `get_manager` | `BaseModelManager[Cortex]`. Builds `mlx` / `hf` / `cloud` cortices. |
| `embedding_manager.py` | `EmbeddingManager`, `get_embedding_manager` | `BaseModelManager[Embedder]`. Builds `mlx_embed` / `hf_embed` / `cloud_embed`. |
| `catalogs/*.yaml` | — | Shipped catalogs: `mlx`, `hf`, `cloud` (LLM) and `mlx_embed`, `hf_embed`, `cloud_embed` (embedding). |

**Cloud backends** (`cloud`, `cloud_embed`) are platform-independent: they call
a remote API, have no local weights, and are therefore treated as always
"installed" — downloads are no-ops and the install gate is skipped. Catalog
entries repurpose `repo_id` as the remote model name and carry a `base_url`
plus `api_key_env` (the env-var *name* holding the key — never the key itself).

---

## `runtime/` — composition root

| File | Role |
| --- | --- |
| `container.py` | Builds and caches the concrete singletons (`get_cortex`, `get_loop`, `get_session_store`, `get_raw_log`) and provides the activation transitions (`swap_active_cortex`, `deactivate_cortex`, `swap_active_embedder`, `deactivate_embedder`). It auto-bootstraps the first installed catalog entry for the configured backend, rebinds the hippocampus / deduper when the active embedder changes, and falls back to `NoopCortex` when nothing is installed. |

This is the only module that knows how to wire abstract seams to concrete
implementations; the FastAPI service and the CLI both consume it.

---

## `sessions/` — raw chat history

The sole writer of *raw* interaction data (kept strictly separate from curated
neocortex memory).

| File | Defines | Role |
| --- | --- | --- |
| `store.py` | `JsonlSessionStore` | Per-session layout: `<root>/index.json` (session list) + `<root>/sessions/<id>.jsonl` (one `Interaction` per line); thread-safe, atomic writes. |
| `raw_log.py` | `JsonlRawLog`, `SessionRawLog` | A single-file JSONL log for tools/tests, and an adapter that makes a `SessionStore` satisfy the `RawInteractionLog` seam. |

---

## Dependency direction

```
hat/abstract/  ← interfaces (Protocols / ABCs / schemas)
      ▲
      │ implemented by
      │
hat/core/      loop.py · hippocampus · neocortex · oracle · sws   (pure)
               cortex · neocortex/embeddings · lifecycle          (heavy deps, lazy)
      ▲
      │ wired by
      │
hat/core/runtime/container.py  →  consumed by  hat/api  and  hat/cli
```

`core/` never imports from `hat/api`. Heavy third-party libraries are imported
inside functions, so importing a subpackage stays cheap until a model is
actually built.
