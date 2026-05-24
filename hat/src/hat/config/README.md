# `config/` — runtime configuration

Pydantic-based settings loaded from environment variables (prefix `HAT_`)
and a project-local `.env` file. There is a single `Settings` instance,
cached via `get_settings()`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `Settings` and `get_settings()`. |
| `settings.py` | `Settings(BaseSettings)` definition — storage paths, cortex backend selection, HF / oracle / generation defaults, write/oracle thresholds. |

## Notable keys

* `cortex_backend` — `noop` (default) / `hf` / `mlx`. Concrete model
  paths are **not** in env; the active model is selected at runtime via
  `ModelManager` and the `/api/models` endpoints.
* `raw_root`, `neocortex_path`, `neocortex_traces_path`, `model_root` —
  on-disk locations (see [`../memory/README.md`](../memory/README.md)
  and ADR-002).
* `write_threshold`, `oracle_threshold` — selection gate for trace
  consolidation and oracle consultation (paper §3.4.2 / §3.5).
* `hf_*` — HuggingFace backend hardware preferences (device, dtype,
  offload, 4-bit).
* `oracle_*` — OpenAI-compatible teacher endpoint + cost-guard limits.

Override anything by exporting `HAT_<KEY>=…` or editing `.env`. See
[`.env.example`](../../../.env.example) for a worked example.
