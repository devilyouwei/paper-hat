# `hat/` — package root

Top-level Python package for the HAT reference implementation. Every
subpackage below corresponds to one architectural layer; see the
per-directory READMEs for details.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exposes `__version__`. No side-effects on import. |
| `cli.py` | `typer`-based command-line entrypoint (`hat serve`, `hat sleep`, `hat train`, `hat eval`, `hat ingest`). Configures logging via `utils.logging.setup`. |

## Subpackages

| Directory | Role |
| --- | --- |
| [`api/`](api/README.md) | FastAPI surface — routers, controllers, schemas, dependency container. Also serves the web UI. |
| [`config/`](config/README.md) | `pydantic-settings` configuration (env / `.env`). |
| [`core/`](core/README.md) | Paper algorithms (Cortex, Hippocampus, Neocortex, Oracle, SWS, wake–sleep loop). Pure Python, no I/O. |
| [`data/`](data/README.md) | Benchmark dataset adapters (stub). |
| [`eval/`](eval/README.md) | Evaluation metrics & harnesses (stub). |
| [`memory/`](memory/README.md) | Storage layer — separate `raw/` and `curated/` stores. |
| [`models/`](models/README.md) | Model lifecycle: registry, catalog, manager, backends, training. |
| [`services/`](services/README.md) | Long-running orchestration (SWS scheduler, replay worker, job queue). |
| [`ui/`](ui/README.md) | Vanilla HTML/CSS/JS web app served by FastAPI. |
| [`utils/`](utils/README.md) | Cross-cutting helpers (logging, …). |

The hard rules between layers (raw vs curated separation, protocol-based
backends, controller purity, etc.) are described in [`../README.md`](../README.md)
and the ADRs under [`../docs/`](../docs/README.md).
