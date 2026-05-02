"""vLLM backend (stub). Install with ``uv sync --extra vllm``."""

from __future__ import annotations

from typing import Any


def build_vllm_model(model_name: str, **kwargs: Any):  # pragma: no cover - stub
    try:
        import vllm  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install with `uv sync --extra vllm`") from e
    raise NotImplementedError("vLLM backend stub")
