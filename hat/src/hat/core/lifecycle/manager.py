"""Runtime manager for installed / active *LLM* models.

Thin subclass of :class:`~hat.core.lifecycle.base.BaseModelManager` that knows
how to build a :class:`~hat.abstract.cortex.Cortex` for each supported backend.
All download / cache / activate plumbing lives in the base class.

The manager is process-local. If you scale to a separate sleep worker, it keeps
its own cache there — that is intentional, since each process has its own
GPU/Metal context.
"""

from __future__ import annotations

from pathlib import Path

from hat.abstract.cortex import Cortex
from hat.config.settings import get_settings
from hat.core.lifecycle.base import BaseModelManager
from hat.core.lifecycle.catalog import SUPPORTED_BACKENDS
from hat.utils.logging import get_logger

log = get_logger(__name__)


class ModelManagerError(RuntimeError):
    pass


class ModelManager(BaseModelManager[Cortex]):
    supported = SUPPORTED_BACKENDS
    weight_suffixes = frozenset({".safetensors", ".bin", ".gguf"})
    error_cls = ModelManagerError
    noun = "model"
    inner_attr = "lm"

    def _build(self, backend: str, path: str) -> Cortex:
        s = get_settings()
        log.info("building cortex backend={} path={}", backend, path)
        if backend == "cloud":
            from hat.core.cortex.cloud import CloudCortex, build_cloud_model

            # Cloud has no local files; ``path`` ends in the catalog id.
            entry = self._entry(backend, Path(path).name)
            return CloudCortex(
                build_cloud_model(
                    entry.repo_id,
                    base_url=entry.base_url or "https://api.openai.com/v1",
                    api_key_env=entry.api_key_env,
                    max_tokens=s.default_max_tokens,
                    temperature=s.default_temperature,
                )
            )
        if backend == "mlx":
            from hat.core.cortex.mlx import MLXCortex, build_mlx_model

            return MLXCortex(build_mlx_model(path))
        if backend == "hf":
            from hat.core.cortex.hf import HFCortex, build_hf_model

            return HFCortex(
                build_hf_model(
                    path,
                    device=s.hf_device,
                    dtype=s.hf_dtype,
                    offload=s.hf_offload,
                    max_gpu_gb=s.hf_max_gpu_gb,
                    max_cpu_gb=s.hf_max_cpu_gb,
                    offload_dir=str(s.hf_offload_dir),
                    load_in_4bit=s.hf_load_in_4bit,
                )
            )
        raise ModelManagerError(f"backend {backend!r} cannot host a model")


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


__all__ = ["ModelManager", "ModelManagerError", "get_manager"]
