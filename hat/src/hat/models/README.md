# `models/` — backends, catalog, lifecycle

```
models/
├── catalog.py         # YAML loader + CatalogEntry schema
├── catalogs/          # curated YAML catalogs (one per backend)
│   ├── mlx.yaml       # ~18 4-/6-bit Apple-Silicon entries
│   └── hf.yaml        # 6 HF baselines
├── manager.py         # ModelManager: paths · download · load · activate · unload
├── registry.py        # @register("name") for LanguageModel backends
├── backends/          # concrete LanguageModel implementations
│   ├── mlx.py         # mlx-lm; chat() + stream_chat()
│   ├── hf.py          # HF Transformers; chat() + stream_chat() (TextIteratorStreamer)
│   ├── vllm.py        # stub
│   └── ollama.py      # stub
└── training/          # SWS trainer implementations (LoRA / EWC) — stub
```

## Backend extras

| Backend | Extra | Best for |
| --- | --- | --- |
| HuggingFace Transformers | `uv sync --extra hf` | CUDA / generic |
| **MLX** (Apple Metal) | `uv sync --extra mlx` | **Apple Silicon (M1/M2/M3)** |
| vLLM | `uv sync --extra vllm` | server-class GPU (stub) |
| Ollama | `uv sync --extra ollama` | local Ollama daemon (stub) |
| LoRA / EWC trainers | `uv sync --extra train` | SWS fine-tuning (stub) |

Both shipped backends implement a streaming `stream_chat(messages, **kw)`
generator and accept `enable_thinking` (forwarded to the tokenizer's chat
template for Qwen3-style models). HF uses `transformers.TextIteratorStreamer`
on a worker thread; MLX uses `mlx_lm.stream_generate`.

## Catalog

`catalog.py` parses YAML files in `catalogs/` into `CatalogEntry` records
(`id`, `repo_id`, `display`, `size_gb`, …). Entries are surfaced through
`/api/models` so the UI can render the per-backend dropdown with
"installed / not installed" badges. Catalogs are package data — they ship
inside the wheel via `[tool.hatch.build.targets.wheel.force-include]`.

To add a model: append an entry to the matching catalog YAML; no code change
required.

## ModelManager

`ModelManager` (singleton, accessed via `get_manager()`) owns the `(backend,
id) → Cortex` cache and the active-pointer. Key methods:

| Method | Behaviour |
| --- | --- |
| `list_models(backend)` | catalog × installed-on-disk |
| `download(backend, id)` | `huggingface_hub.snapshot_download` → `model/<backend>/<id>/`; HF blob cache pinned to `model/.hf-cache` to avoid duplication under `~/.cache` |
| `set_active(backend, id)` | evict every other cached Cortex **first**, then build the new one — guarantees peak GPU/Metal memory ≈ one model |
| `unload(backend, id)` | drop a single cached Cortex |
| `unload_all()` | drop every cached Cortex; clear the active pointer |

Eviction goes through `_release_cortex`, which nulls the heavy `model` /
`tokenizer` references, runs `gc.collect()`, then calls
`torch.cuda.empty_cache()` / `torch.mps.empty_cache()` /
`mlx.core.clear_cache()` so the framework allocators actually return the
memory. See ADR-004 for the full rationale.
