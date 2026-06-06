# `hat/` — package root

Top-level Python package for the HAT reference implementation.

This README describes the **target architecture** the codebase is
migrating toward. It supersedes the previous layout, in which
abstractions and implementations were sprinkled across `core/`,
`memory/`, and `models/`. The migration plan is summarised at the
bottom of this file.

## Design rules

1. **Abstract vs. concrete are physically separated.** Every ABC and
   `typing.Protocol` lives under `abstract/`. Every concrete class
   lives under `core/` (algorithms, storage, model adapters,
   lifecycle) and imports its base from `abstract/`.
2. **`core/` is the paper library.** Pure algorithm + storage code,
   independent of FastAPI / web concerns. Anything in `core/` can be
   imported by a notebook, a CLI, or another front-end without
   pulling in the HTTP layer.
3. **`core/` exposes a front-end-agnostic composition root.** The
   wiring that turns settings + lifecycle managers + storage into a
   ready-to-use `WakeSleepLoop` lives in `core/runtime/` so both
   `api/` (FastAPI) and `cli.py` (Typer) consume the **same**
   singletons. No business logic in either front-end.
4. **`api/` owns the web surface only.** Controllers translate HTTP
   ↔ `core/runtime` calls. Services in `api/services/` are thin
   request-shaped wrappers; long-lived state (loop, session store,
   active model) is owned by `core/runtime`.
5. **One module per business domain.** Cortex, Hippocampus,
   Neocortex, Oracle, SWS, Sessions, and Model Lifecycle each get a
   single subdirectory under `core/` containing all their concrete
   pieces (algorithm + storage + backend adapters + managers).
6. **No more `memory/` or `models/` siblings.** Their content is
   merged into the matching `core/<domain>/` directory.

## Top-level layout (target)

```
src/hat/
├── __init__.py
├── cli.py
├── README.md                 # this file
├── abstract/                 # all ABCs, Protocols, and shared schemas
├── api/                      # FastAPI surface (controllers, services, schemas)
├── config/                   # pydantic-settings
├── core/                     # paper algorithms + storage + adapters
└── utils/                    # cross-cutting helpers (logging)
```

`memory/`, `models/`, and `core/protocols.py` are **removed** by this
refactor. `ui/` (vanilla web app served by FastAPI) continues to live
beside `api/` since it is a deployment artifact, not Python code; it
stays out of this refactor.

## `abstract/` — interfaces only

All abstractions in one place. No I/O, no heavy deps, no
implementations. Files are grouped by domain so each `core/<domain>/`
folder has a one-to-one base file to inherit from.

| File | Contents | Replaces |
| --- | --- | --- |
| `abstract/schemas.py` | `Interaction`, `MemoryTrace`, `TraceMetadata`, `ScoreSignals`, `WriteDecision`, `ReplayExample`, `ReplayBatch`, `SWSObjective`, `SWSStats`, `Session` | `core/schemas.py`, `Session` model from `memory/raw/sessions.py` |
| `abstract/cortex.py` | `Cortex` ABC, `LanguageModel` Protocol | `core/cortex/base.py`, `LanguageModel` from `core/protocols.py` |
| `abstract/hippocampus.py` | `Abstractor`, `UncertaintyEstimator`, `WritePolicy`, `ReplayBuilder`, `Deduper` ABCs | bases of `core/hippocampus/{abstraction,selection,replay,scoring/uncertainty}.py` plus the Protocols of the same name |
| `abstract/neocortex.py` | `NeocortexStore` ABC (with `WriteDecision` enforcement), `VectorIndex` ABC, `Embedder` Protocol | `core/neocortex/store.py` ABC half, `memory/embeddings.py` Protocol, vector-index interface implied by `memory/curated/vector_index.py` |
| `abstract/oracle.py` | `Oracle` ABC | `core/oracle/base.py` |
| `abstract/sws.py` | `SWSTrainer` ABC | `core/sws/trainer.py` ABC half |
| `abstract/sessions.py` | `SessionStore`, `RawInteractionLog` ABCs | `memory/raw/log.py` ABC, interface implied by `memory/raw/sessions.py` |
| `abstract/__init__.py` | flat re-exports of every public name above | — |

`core/protocols.py` (the Protocol-only file that duplicates several
ABCs) goes away; the canonical interface for each seam is the ABC in
`abstract/`. Where a true `typing.Protocol` is preferred (because the
seam is structural and doesn't need an inheritance relationship — e.g.
`LanguageModel`, `Embedder`), it lives next to its ABC sibling in the
same `abstract/<domain>.py` file.

## `core/` — implementations

Each subdirectory mirrors a paper section and is the **only** home for
that domain's concrete code (algorithm, storage, backend adapters,
prompts).

```
core/
├── __init__.py
├── loop.py                   # WakeSleepLoop (paper §3.8)
├── runtime/                  # composition root + use-cases (front-end agnostic)
├── cortex/                   # paper §3.3 — online interaction model
├── hippocampus/              # paper §3.4 — selective consolidation
├── neocortex/                # paper §3.6 — long-term curated memory
├── oracle/                   # paper §3.5 — on-demand teacher
├── sws/                      # paper §3.7 — slow-wave-sleep trainer
├── sessions/                 # raw chat history (input to wake step)
└── lifecycle/                # model download / catalog / active pointer
```

### `core/runtime/` — composition root and use-cases

This is the **only** place that knows how to wire concrete
implementations together. It replaces today's
`api/services/container.py`. Both FastAPI and the CLI import from
here; neither knows the other exists.

```
core/runtime/
├── __init__.py
├── container.py              # process-wide singletons: get_loop / get_cortex /
│                             # get_session_store / get_raw_log / get_neocortex
├── bootstrap.py              # build_loop(settings), _bootstrap_cortex,
│                             # _make_abstractor / _make_uncertainty / _make_deduper /
│                             # _make_oracle  (was the bottom of api/services/container.py)
├── model_control.py          # swap_active_cortex, deactivate_cortex,
│                             # swap_active_embedder, deactivate_embedder
├── chat.py                   # ChatService: run_turn(session_id, query) -> reply
│                             # (yields events for streaming front-ends)
├── neocortex_admin.py        # list_entries / get_entry / update_entry / delete_entry
│                             # (was api/services/neocortex.py, minus HTTP shapes)
├── sessions.py               # session CRUD use-cases
└── sleep.py                  # run_sleep_cycle(): replay sample → trainer.fit()
```

Conventions for `runtime/`:

- **No FastAPI / no Typer / no urllib** — pure Python.
- Inputs and outputs are pydantic models from `abstract/schemas.py`
  (or plain dicts where the data is already JSON-shaped, e.g. SFT
  rows). Front-ends translate to/from their own schemas.
- Streaming use-cases (chat, downloads) return iterators / generators
  yielding plain dataclasses or dicts. FastAPI wraps them in
  `EventSourceResponse`; the CLI prints them.
- Singletons live in `container.py` behind `@lru_cache` getters, so
  both processes (uvicorn worker, `hat sleep` one-shot) get
  consistent state.

### Two front-ends, one runtime

```
            ┌──────────────────────┐
            │  core/runtime/       │
            │  (singletons +       │
            │   use-cases)         │
            └──┬────────────────┬──┘
               │                │
   imports     │                │     imports
               ▼                ▼
       ┌─────────────┐    ┌─────────────┐
       │   api/      │    │   cli.py    │
       │ FastAPI     │    │ Typer       │
       │ controllers │    │ commands    │
       └─────────────┘    └─────────────┘
```

`api/services/` becomes a **translation layer**: each module wraps a
`core/runtime` use-case to map between Pydantic request schemas /
HTTP errors / SSE framing and the underlying call. `cli.py` uses the
same use-cases directly and prints their output.

### `core/cortex/`

Cortex implementations and the LM backends they wrap. Backend code
moves out of `models/backends/` and joins the cortex it wraps so the
backend + adapter ship as one module per engine.

```
core/cortex/
├── __init__.py               # public re-exports: NoopCortex, HFCortex, MLXCortex
├── noop.py                   # NoopCortex
├── hf.py                     # HFLanguageModel + HFCortex (was models/backends/hf.py + core/cortex/hf_cortex.py)
├── mlx.py                    # MLXLanguageModel + MLXCortex (was models/backends/mlx.py + core/cortex/mlx_cortex.py)
└── cloud.py                  # CloudLanguageModel + CloudCortex (OpenAI-compatible, platform-independent)
```

All three cortex classes inherit from `abstract.cortex.Cortex`. The
LM classes satisfy `abstract.cortex.LanguageModel`.

### `core/hippocampus/`

Unchanged in shape; only its imports change (bases now come from
`abstract/`). All concrete classes here inherit from
`abstract.hippocampus`.

```
core/hippocampus/
├── __init__.py
├── abstraction.py            # IdentityAbstractor, LLMAbstractor
├── selection.py              # UncertaintyGatePolicy
├── replay.py                 # SupervisedReplayBuilder
├── dedup.py                  # EmbeddingDeduper
├── scoring/
│   ├── __init__.py
│   ├── uncertainty.py        # ConstantUncertainty, LogprobUncertainty
│   └── llm_judge.py
└── prompts/                  # markdown prompt templates
```

### `core/neocortex/`

Curated memory + the embedding adapters used by dedup. Absorbs
everything from `memory/curated/` and from the embedding side of
`models/`.

```
core/neocortex/
├── __init__.py               # public re-exports
├── memory_store.py           # InMemoryNeocortex (reference impl)
├── jsonl_store.py            # JsonlNeocortex (was memory/curated/jsonl_store.py)
├── vector_index.py           # NpzVectorIndex (was memory/curated/vector_index.py)
└── embeddings/
    ├── __init__.py           # ManagedEmbedder adapter (was memory/embeddings.py)
    ├── hf.py                 # HF embedding backend (was models/backends/hf_embed.py)
    └── mlx.py                # MLX embedding backend (was models/backends/mlx_embed.py)
```

`InMemoryNeocortex` and `JsonlNeocortex` inherit from
`abstract.neocortex.NeocortexStore`. `NpzVectorIndex` implements
`abstract.neocortex.VectorIndex`. The embedding backends and
`ManagedEmbedder` satisfy `abstract.neocortex.Embedder`.

### `core/oracle/`

Same shape as today, with the `Oracle` ABC moved to `abstract/`.

```
core/oracle/
├── __init__.py
├── noop.py                   # NoopOracle (split out of base.py)
├── openai_compat.py          # OpenAICompatibleOracle
├── cost_guard.py
└── prompts.py
```

### `core/sws/`

Trainer implementations. The empty `models/training/` placeholder is
folded in here for the eventual real (LoRA / EWC) trainer.

```
core/sws/
├── __init__.py
└── trainer.py                # DryRunTrainer (and future PEFT trainers)
```

### `core/sessions/`

Raw chat-history storage. This is the only piece of `memory/raw/`
that stays at the storage layer; it is not curated and not consumed
by training, only by the wake step and the chat UI.

```
core/sessions/
├── __init__.py
├── store.py                  # JsonlSessionStore (was memory/raw/sessions.py)
└── raw_log.py                # JsonlRawLog, SessionRawLog (was memory/raw/log.py)
```

`JsonlSessionStore` implements `abstract.sessions.SessionStore`;
`JsonlRawLog` and `SessionRawLog` implement
`abstract.sessions.RawInteractionLog`. The `Session` pydantic model
moves to `abstract/schemas.py`.

### `core/lifecycle/`

Model download, catalog, and active-pointer logic — the
non-algorithmic infrastructure that produces the `LanguageModel` and
`Embedder` objects the rest of `core/` consumes. Replaces all of
`models/` except backends (which moved to `cortex/` and
`neocortex/embeddings/`).

```
core/lifecycle/
├── __init__.py
├── catalog.py                # CatalogEntry, load_catalog (was models/catalog.py)
├── catalogs/                 # YAML catalogs (was models/catalogs/)
│   ├── hf.yaml
│   ├── hf_embed.yaml
│   ├── mlx.yaml
│   └── mlx_embed.yaml
├── cortex_manager.py         # ModelManager for LLMs (was models/manager.py)
└── embedding_manager.py      # EmbeddingManager (was models/embedding_manager.py)
```

The two managers stay distinct because they own different runtime
caches (a loaded LLM and a loaded embedder are not interchangeable),
but they share `catalog.py` and the YAML directory.

### `core/loop.py`

Unchanged in role: composes a `Cortex`, `Abstractor`,
`UncertaintyEstimator`, `WritePolicy`, `ReplayBuilder`,
`NeocortexStore`, `SWSTrainer`, optional `Oracle`, and optional
`Deduper`. Imports of base classes shift from `core/...` to
`abstract/...`; nothing else changes.

## `api/` — web surface (thin translation layer)

`api/` keeps its current shape (`controllers/`, `services/`,
`schemas/`, `main.py`), but its responsibilities shrink:

- `api/controllers/*.py` — translate HTTP → call into
  `core/runtime` → translate result back to HTTP / SSE.
- `api/services/*.py` — request-shaped wrappers around
  `core/runtime` use-cases (e.g. paginate, build SSE frames). No
  singletons live here any more; they all moved to
  `core/runtime/container.py`.
- `api/schemas/` — Pydantic request/response models for the HTTP
  surface only. Domain types stay in `abstract/schemas.py`.
- `api/main.py` — FastAPI app factory, mounts routers, serves `ui/`.

## `cli.py` — terminal front-end

`cli.py` (Typer) becomes a sibling of `api/`, not a child:

```
hat serve                 → api.main:app via uvicorn
hat chat <session-id>     → core.runtime.chat.run_turn(...)  (interactive REPL)
hat sleep [--dry-run]     → core.runtime.sleep.run_sleep_cycle(...)
hat ingest <path>         → core.runtime.sessions.import_jsonl(...)
hat models list/pull/use  → core.runtime.model_control.* + core.lifecycle.*
hat memory list/show/edit → core.runtime.neocortex_admin.*
hat eval <suite>          → core.runtime.* (future)
```

Every command is a thin wrapper that prints the use-case's return
value or streams its iterator. No command reaches into
`core/<domain>/` directly — the runtime layer is the seam.

## `config/` and `utils/` — unchanged

No structural change. `config/settings.py` keeps owning the
`model_root` / `raw_root` / dedup / oracle settings.

## Inheritance map (one line per implementation)

| Concrete (in `core/`) | Base (in `abstract/`) |
| --- | --- |
| `core.cortex.NoopCortex`, `HFCortex`, `MLXCortex` | `abstract.cortex.Cortex` |
| `core.cortex.hf.HFLanguageModel`, `core.cortex.mlx.MLXLanguageModel` | `abstract.cortex.LanguageModel` (Protocol) |
| `core.hippocampus.abstraction.IdentityAbstractor`, `LLMAbstractor` | `abstract.hippocampus.Abstractor` |
| `core.hippocampus.scoring.uncertainty.ConstantUncertainty`, `LogprobUncertainty` | `abstract.hippocampus.UncertaintyEstimator` |
| `core.hippocampus.selection.UncertaintyGatePolicy` | `abstract.hippocampus.WritePolicy` |
| `core.hippocampus.replay.SupervisedReplayBuilder` | `abstract.hippocampus.ReplayBuilder` |
| `core.hippocampus.dedup.EmbeddingDeduper` | `abstract.hippocampus.Deduper` |
| `core.neocortex.memory_store.InMemoryNeocortex`, `core.neocortex.jsonl_store.JsonlNeocortex` | `abstract.neocortex.NeocortexStore` |
| `core.neocortex.vector_index.NpzVectorIndex` | `abstract.neocortex.VectorIndex` |
| `core.neocortex.embeddings.ManagedEmbedder`, `hf.HFEmbedder`, `mlx.MLXEmbedder` | `abstract.neocortex.Embedder` (Protocol) |
| `core.oracle.NoopOracle`, `OpenAICompatibleOracle` | `abstract.oracle.Oracle` |
| `core.sws.trainer.DryRunTrainer` | `abstract.sws.SWSTrainer` |
| `core.sessions.store.JsonlSessionStore` | `abstract.sessions.SessionStore` |
| `core.sessions.raw_log.JsonlRawLog`, `SessionRawLog` | `abstract.sessions.RawInteractionLog` |

## Migration plan (how we get there)

The refactor is mechanical (move + rename + update imports) and can be
done in a single pass once this design is approved. Suggested order so
each step compiles and the test suite stays green:

1. **Create `abstract/`** with one file per domain (`schemas.py`,
   `cortex.py`, `hippocampus.py`, `neocortex.py`, `oracle.py`,
   `sws.py`, `sessions.py`). Move ABCs and Protocols out of
   `core/...`/`memory/...` and re-export them from their old
   locations for backwards compatibility while migrating.
2. **Move backends into the domain folders.**
   - `models/backends/{hf,mlx}.py` + `core/cortex/{hf,mlx}_cortex.py`
     → `core/cortex/{hf,mlx}.py`.
   - `models/backends/{hf,mlx}_embed.py` + `memory/embeddings.py`
     → `core/neocortex/embeddings/`.
   - `models/registry.py` → `core/cortex/registry.py`.
3. **Move storage layers.**
   - `memory/curated/jsonl_store.py` → `core/neocortex/jsonl_store.py`.
   - `memory/curated/vector_index.py` → `core/neocortex/vector_index.py`.
   - `core/neocortex/store.py` ABC → split: ABC half to
     `abstract/neocortex.py`, `InMemoryNeocortex` to
     `core/neocortex/memory_store.py`.
   - `memory/raw/{sessions,log}.py` → `core/sessions/{store,raw_log}.py`.
4. **Move lifecycle.**
   - `models/{catalog,manager,embedding_manager}.py` and
     `models/catalogs/` → `core/lifecycle/`.
5. **Delete** `core/protocols.py`, `memory/`, `models/training/`,
   `models/`. Update every import site (CLI, API services, tests).
6. **Extract `core/runtime/`** out of `api/services/container.py`:
   - Move `get_loop`, `get_cortex`, `get_session_store`,
     `get_raw_log`, `prior_traces_for_session`, and the
     `_bootstrap_*` / `_make_*` factories to
     `core/runtime/{container,bootstrap}.py`.
   - Move `swap_active_cortex` / `deactivate_cortex` / embedder
     equivalents to `core/runtime/model_control.py`.
   - Lift the meaningful logic of `api/services/{chat,neocortex,
     models,embedding_models,openai}.py` into
     `core/runtime/{chat,neocortex_admin,...}.py`. Leave only HTTP
     framing (SSE, error mapping, pagination) in `api/services/`.
   - Rewrite `cli.py` commands on top of `core/runtime/*`.
7. **Run** `pytest` and the API smoke test; add a CLI smoke test
   (`hat sleep --dry-run`, `hat memory list`) so both front-ends are
   covered. Fix import paths and `__init__.py` re-exports until
   green.

After this completes, `src/hat/` contains exactly:
`abstract/`, `api/`, `config/`, `core/`, `utils/`, plus `__init__.py`
and `cli.py`.
