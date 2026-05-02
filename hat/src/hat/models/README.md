# `models/` — backends and trainers

Concrete implementations of `LanguageModel` and `SWSTrainer`. Each backend is
gated behind an extras group:

| Backend | Extra | Best for |
| --- | --- | --- |
| HuggingFace Transformers | `uv sync --extra hf` | CUDA / generic |
| **MLX** (Apple Metal) | `uv sync --extra mlx` | **Apple Silicon (M1/M2/M3)** |
| vLLM | `uv sync --extra vllm` | server-class GPU |
| Ollama | `uv sync --extra ollama` | local Ollama daemon |
| LoRA / EWC trainers | `uv sync --extra train` | SWS fine-tuning |
