<p align="center">
  <img src="logo.png" alt="HAT logo" width="320" />
</p>

<h1 align="center">HAT — Hippocampus-Augmented Transformer</h1>

<p align="center">
  Reference implementation of the wake–sleep selective-memory-consolidation system from<br/>
  <em>Learning What to Learn: Hippocampal Memory Consolidation for Continual Model Adaptation</em>.
</p>

<p align="center">
  <a href="docs/"><img alt="docs" src="https://img.shields.io/badge/docs-ADRs-blue"></a>
  <img alt="python" src="https://img.shields.io/badge/python-%E2%89%A53.10-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="status" src="https://img.shields.io/badge/status-skeleton-yellow">
</p>

> **Status:** skeleton. Protocols, ABCs, and a no-op end-to-end path are in
> place; real training and benchmark code land in subsequent iterations.

---

## Features

- **Wake–sleep loop** — online Cortex + offline SWS trainer, separated by a
  curated memory store with a type-enforced write contract.
- **Selective consolidation** — only turns the Cortex is unsure about (low
  log-prob) flow into training data; trivial small-talk is dropped at triage.
- **Local-first** — `mlx` backend for Apple Silicon, `hf` backend for CUDA /
  generic; one-click model download from the web UI, no env-var surgery.
- **OpenAI-compatible API** — point any OpenAI client at `:8000/v1` and every
  call still goes through consolidation.
- **Pure-Python core** — `core/` has no FastAPI / torch / I/O imports; every
  backend is a `typing.Protocol`.

## Quickstart

```bash
uv sync
make serve         # FastAPI + web UI on http://127.0.0.1:8000
make sleep         # run a slow-wave-sleep cycle (dry-run by default)
make test          # run the unit suite
```

The default Cortex is a no-op echo so the loop runs without ML deps. The
web UI is mounted by the same FastAPI process — there is no separate UI
server to launch.

## Local model chat

Pick the backend that matches your machine, install the extra, then
download a model from the **Models** tab in the web UI (or via
`POST /api/models/download`).

### Apple Silicon (M1 / M2 / M3) — recommended on Mac

```bash
uv sync --extra mlx
echo 'HAT_CORTEX_BACKEND=mlx' >> .env
make serve
```

Catalog ships with 8 GB-friendly defaults including
`qwen2.5-0.5b-instruct-4bit` (default, ≈0.3 GB) and `qwen3.5-4b-4bit`
(≈2.3 GB, recommended quality/footprint pick). Weights land under
`model/mlx/<id>/`.

### CUDA / generic (HuggingFace)

```bash
uv sync --extra hf
cp .env.example .env       # then set HAT_CORTEX_BACKEND=hf
make serve
```

Useful env knobs: `HAT_HF_DEVICE` (`auto|cpu|cuda|mps`), `HAT_HF_DTYPE`
(`auto|float16|bfloat16|float32`), `HAT_HF_LOAD_IN_4BIT=true`,
`HAT_HF_OFFLOAD=true` with `HAT_HF_MAX_GPU_GB` / `HAT_HF_MAX_CPU_GB`.
Weights land under `model/hf/<id>/`. See
[`src/hat/config/README.md`](src/hat/config/README.md) for the full list.

### Generation settings

The chat UI exposes per-call temperature, max tokens, and Qwen3
**think-mode** toggles (*Enable thinking*, *Show thinking process*). They
map onto `temperature`, `max_tokens`, and
`extra_body.chat_template_kwargs.enable_thinking` so any OpenAI client can
use them too. Streaming filters `<think>…</think>` client-side when the
toggle is off.

## OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="hat-local")

stream = client.chat.completions.create(
    model="hat-cortex",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for ev in stream:
    if ev.choices and (delta := ev.choices[0].delta.content):
        print(delta, end="", flush=True)
```

Each call still flows through the wake step. The final SSE chunk carries
`hat_consolidated` / `hat_trace_id` / `hat_session_id` extras so clients
can surface that signal. Full endpoint list in
[`src/hat/api/README.md`](src/hat/api/README.md).

## Model management

Weights are managed at runtime via REST (mirrored by the **Models** tab):

| Method & path                              | Purpose                                     |
| ------------------------------------------ | ------------------------------------------- |
| `GET /api/models?backend=mlx\|hf`          | catalog with installed status               |
| `POST /api/models/download`                | `snapshot_download` into `model/<backend>/` |
| `POST /api/models/active`                  | activate (evicts the previous Cortex first) |
| `GET  /api/models/active`                  | currently active `(backend, id)`            |
| `DELETE /api/models/active`                | unload everything, free GPU/Metal memory    |

Catalogs live under [`src/hat/models/catalogs/`](src/hat/models/catalogs/README.md)
(21 MLX entries, 15 HF entries). A project-local
`model/<backend>/catalog.yaml` overrides the bundled one. See
[ADR-004](docs/adr-004-model-lifecycle.md) for the lifecycle contract.

## Architecture

```
                ┌──────────┐  query   ┌──────────────────┐
   user  ──────▶│  Cortex  │─────────▶│  Hippocampus     │
                └──────────┘  resp    │  abs / sel /     │
                                      │  replay          │
                                      └──────┬───────────┘
                                             │ WriteDecision
                                             ▼
                       ┌──────────────┐         ┌──────────────┐
                       │ Raw chat log │         │  Neocortex   │
                       │ (append-only)│         │  (curated)   │
                       └──────────────┘         └──────┬───────┘
                                                       │ replay batch
                                                       ▼
                                              ┌────────────────────┐
                                              │   SWS Trainer      │
                                              │ L_sup+λ_KD+λ_stab  │
                                              └─────────┬──────────┘
                                                        │ θ^{t+1}
                                                        ▼
                                                    (Cortex)
```

### Paper → module map

| Paper section                     | Module                                         |
| --------------------------------- | ---------------------------------------------- |
| §3.3 Cortex                       | `hat.core.cortex`                              |
| §3.4 Hippocampus Agent            | `hat.core.hippocampus`                         |
| §3.4.1 Abstraction                | `hat.core.hippocampus.abstraction`             |
| §3.4.2 Selection                  | `hat.core.hippocampus.selection` + `scoring/`  |
| §3.4.3 Replay construction        | `hat.core.hippocampus.replay`                  |
| §3.5 Oracle (on-demand teacher)   | `hat.core.oracle`                              |
| §3.6 Neocortex (curated memory)   | `hat.core.neocortex` + `hat.memory.curated`    |
| §3.7 SWS trainer                  | `hat.core.sws` + `hat.models.training`         |
| §3.8 Iterative wake–sleep loop    | `hat.core.loop`                                |
| §4 Experiments                    | `hat.data` + `hat.eval` + `experiments/`       |

### Design contracts

1. **Wake/Sleep are deployment-separable.** `api/` + `core/cortex` +
   `core/hippocampus` serve users synchronously. `services/replay_worker` +
   `core/sws` + `models/training` run offline. They communicate **only**
   through `memory/` and `services/job_queue`.
2. **Raw vs curated separation is type-enforced.** `NeocortexStore.write`
   requires an accepted `WriteDecision`; raw log entries cannot accidentally
   become training data ([ADR-002](docs/adr-002-raw-vs-curated.md)).
3. **Backends are protocols.** `LanguageModel`, `SWSTrainer`, `Oracle`,
   `UncertaintyEstimator`, … are `typing.Protocol`s; concrete adapters
   register through `hat.models.registry`
   ([ADR-003](docs/adr-003-backend-protocol.md)).
4. **Controllers are pure Python.** Routers are thin FastAPI handlers;
   controllers depend on protocols. The web UI uses the same REST surface
   any external client would.

## Project layout

```
hat/
├── src/hat/      # Python package — see src/hat/README.md for per-subpackage docs
├── tests/        # pytest suite
├── docs/         # ADRs and architecture notes
├── model/        # downloaded weights (model/hf/<id>/, model/mlx/<id>/)
├── runs/         # runtime artefacts (raw/, neocortex/)
└── pyproject.toml
```

The Python package has per-directory READMEs starting at
[`src/hat/README.md`](src/hat/README.md). Top-level subpackages:

| Module | Role |
| --- | --- |
| [`core/`](src/hat/core/README.md) | Paper algorithms — Cortex, Hippocampus, Neocortex, Oracle, SWS, wake–sleep loop. Pure Python. |
| [`models/`](src/hat/models/README.md) | Lifecycle (registry, catalog, manager) and backends (MLX, HF). |
| [`memory/`](src/hat/memory/README.md) | `raw/` chat history vs `curated/` Neocortex — separation enforced by type. |
| [`api/`](src/hat/api/README.md) | FastAPI: `routers/ → controllers/ → injected protocols`. |
| [`services/`](src/hat/services/README.md) | Long-running orchestration (SWS scheduler, replay worker, job queue). |
| [`ui/`](src/hat/ui/README.md) | Vanilla HTML/CSS/JS web app served by FastAPI at `/`. |
| [`data/`](src/hat/data/README.md), [`eval/`](src/hat/eval/README.md) | Benchmark loaders and metrics (stubs). |
| [`config/`](src/hat/config/README.md), [`utils/`](src/hat/utils/README.md) | Settings and cross-cutting helpers. |

## Development

```bash
make test          # pytest
make lint          # ruff + mypy
make format        # ruff format
```

Tests live in [`tests/`](tests/) and run without any model dependency
against the no-op Cortex.

## Documentation

- ADRs — [`docs/`](docs/README.md)
  - [ADR-001 directory layout](docs/adr-001-directory-layout.md)
  - [ADR-002 raw vs curated memory](docs/adr-002-raw-vs-curated.md)
  - [ADR-003 backend protocol](docs/adr-003-backend-protocol.md)
  - [ADR-004 model lifecycle](docs/adr-004-model-lifecycle.md)
- Per-package READMEs — start at [`src/hat/README.md`](src/hat/README.md).

## Citation

```bibtex
@article{hat2026,
  title  = {Learning What to Learn: Hippocampal Memory Consolidation for
            Continual Model Adaptation},
  year   = {2026},
}
```

## License

MIT — see [`pyproject.toml`](pyproject.toml).
