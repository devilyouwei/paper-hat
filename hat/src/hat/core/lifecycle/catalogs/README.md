# `models/catalogs/` — curated model catalogs

YAML lists of model entries surfaced by `/api/models` and the **Models**
tab in the web UI. Shipped as package data inside the wheel via
`[tool.hatch.build.targets.wheel.force-include]`.

## Files

| File | Backend | Description |
| --- | --- | --- |
| `mlx.yaml` | MLX (Apple Silicon) | ~21 quantised entries, mostly 4-bit MLX builds, sized for 8 GB+ Macs. Includes the default `qwen2.5-0.5b-instruct-4bit` and recommended `qwen3.5-4b-4bit`. |
| `hf.yaml` | HuggingFace | ~15 HF baselines for CUDA / generic boxes. |

Each entry parses into a `CatalogEntry` (`id`, `repo_id`, `display`,
`size_gb`, `notes`) via [`../catalog.py`](../catalog.py).

To add a model: append an entry to the matching YAML — no code change
required. A project-local override at
`<HAT_MODEL_ROOT>/<backend>/catalog.yaml` (same shape) wins over the
bundled file if present, so private deployments can curate their own list
without forking.
