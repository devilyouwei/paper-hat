# `models/` — backends, catalog, lifecycle

```
models/
├── __init__.py        # re-exports register / create / available from registry
├── registry.py        # @register("name") factory registry for LanguageModel
├── catalog.py         # YAML loader + CatalogEntry (pydantic) schema
├── catalogs/          # curated YAML catalogs shipped as package data
│   ├── mlx.yaml       # 21 Apple-Silicon entries (mostly 4-bit MLX builds)
│   └── hf.yaml        # 15 HuggingFace entries
├── manager.py         # ModelManager: paths · download · load · activate · unload · delete
├── backends/          # concrete LanguageModel implementations
│   ├── mlx.py         # mlx-lm; chat() + stream_chat()
│   └── hf.py          # HF Transformers; chat() + stream_chat() + chat_logprobs()
└── training/          # SWS trainer implementations (LoRA / EWC) — placeholder
```

`SUPPORTED_BACKENDS` (in `catalog.py`) is the source of truth for which
backends can host a model: currently `("mlx", "hf")`. vLLM and Ollama are
*not* shipped — add a backend by registering one with `@register("name")`
and a matching `catalogs/<name>.yaml`.

## Backend extras

| Backend | Extra | Best for |
| --- | --- | --- |
| **MLX** (Apple Metal) | `uv sync --extra mlx` | **Apple Silicon (M1/M2/M3)** |
| HuggingFace Transformers | `uv sync --extra hf` | CUDA / generic |
| LoRA / EWC trainers | `uv sync --extra train` | SWS fine-tuning (placeholder) |

Both shipped backends implement the full `LanguageModel` protocol —
`generate()`, `chat(messages, **kw)`, `stream_chat(messages, **kw)`,
`token_logprobs()` — and accept `enable_thinking` (or a nested
`chat_template_kwargs={...}`), forwarded verbatim to
`tokenizer.apply_chat_template` for Qwen3-style models. HF streaming uses
`transformers.TextIteratorStreamer` on a worker thread; MLX uses
`mlx_lm.stream_generate`.

Two backend details worth knowing:

* **Stop tokens.** Many instruct families (Qwen2.5, Llama-3, ChatML) use a
  turn terminator (`<|im_end|>`, `<|eot_id|>`, `<|end|>`) that is *not* the
  tokenizer's `eos_token`. Both backends collect every plausible terminator
  id once at load time and pass them all to `generate` so chat turns
  actually stop — without this, small / 4-bit models hit `max_new_tokens`
  and loop.
* **HF offload / 4-bit.** `HFLanguageModel` accepts `offload=True` with a
  `max_gpu_gb` / `max_cpu_gb` budget (Accelerate `device_map="auto"` +
  `max_memory`) and `load_in_4bit=True` (bitsandbytes, CUDA only). On a
  plain single-device load it also retries once with offload if it hits a
  CUDA OOM, instead of failing outright.
* **`HFLanguageModel.chat_logprobs(messages, response)`** applies the chat
  template and returns per-token log-probs for `response`, used by the
  uncertainty estimator.

## Catalog

`catalog.py` parses YAML files in `catalogs/` into `CatalogEntry` records
(`id`, `repo_id`, `display`, `size_gb`, `notes`). Entries are surfaced
through `/api/models` so the UI can render the per-backend dropdown with
"installed / not installed" badges. Catalogs are package data — they ship
inside the wheel via `[tool.hatch.build.targets.wheel.force-include]`.

A project-local override at `<HAT_MODEL_ROOT>/<backend>/catalog.yaml`
(same shape as the defaults) wins over the bundled catalog if present, so
private deployments can curate their own list without forking.

To add a model: append an entry to the matching catalog YAML; no code
change required.

## ModelManager

`ModelManager` (singleton, accessed via `get_manager()`) owns the
`(backend, id) → Cortex` cache and the active-pointer. Thread-safe — every
mutating method takes a single `Lock`. Key methods:

| Method | Behaviour |
| --- | --- |
| `list_models(backend)` | catalog × installed-on-disk (a directory is "installed" once it holds at least one `.safetensors` / `.bin` / `.gguf`) |
| `download(backend, id)` | `huggingface_hub.snapshot_download` → `model/<backend>/<id>/`; the HF blob cache is pinned to `model/.hf-cache` to avoid duplication under `~/.cache` |
| `load(backend, id)` | build (or return cached) Cortex without touching the active pointer |
| `set_active(backend, id)` | evict every *other* cached Cortex **first**, then build the new one — guarantees peak GPU/Metal residency ≈ one model |
| `unload(backend, id)` | drop a single cached Cortex; clears active pointer if it matched |
| `unload_all()` | drop every cached Cortex and clear the active pointer |
| `delete(backend, id)` | `rm -rf model/<backend>/<id>/`; refuses to delete the currently-active model |
| `active()` | `{"backend", "id"}` of the active model, or `None` |

Eviction goes through `_release_cortex`, which nulls the heavy `model` /
`tokenizer` references on both the Cortex and its underlying LM wrapper,
runs `gc.collect()`, then calls `torch.cuda.empty_cache()` /
`torch.mps.empty_cache()` / `mlx.core.clear_cache()` so the framework
allocators actually return the memory. See ADR-004 for the full rationale.

All lifecycle transitions (download, load, activate, unload, delete) and
backend generation requests are logged via `hat.utils.logging.get_logger`
— set `HAT_LOG_LEVEL=DEBUG` to see per-request entries.
