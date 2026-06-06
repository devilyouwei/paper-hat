"""Runtime manager for installed / active *embedding* models.

Thin subclass of :class:`~hat.core.lifecycle.base.BaseModelManager` that knows
how to build an :class:`~hat.core.neocortex.embeddings.managed.Embedder` for
each supported embedding backend. All download / cache / activate plumbing
lives in the base class.

Embedding checkpoints live on disk under the same ``model/<backend>/<id>/``
layout as LLM checkpoints; the backend names are ``mlx_embed`` / ``hf_embed``
(and the platform-independent ``cloud_embed``) so they don't collide with the
Cortex catalogs.
"""

from __future__ import annotations

from pathlib import Path

from hat.config.settings import get_settings
from hat.core.lifecycle.base import BaseModelManager
from hat.core.lifecycle.catalog import SUPPORTED_EMBED_BACKENDS
from hat.core.neocortex.embeddings.managed import Embedder, ManagedEmbedder
from hat.utils.logging import get_logger

log = get_logger(__name__)


class EmbeddingManagerError(RuntimeError):
    pass


class EmbeddingManager(BaseModelManager[Embedder]):
    supported = SUPPORTED_EMBED_BACKENDS
    weight_suffixes = frozenset({".safetensors", ".bin", ".gguf", ".npz"})
    error_cls = EmbeddingManagerError
    noun = "embed model"
    log_prefix = "[embed] "
    inner_attr = "inner"

    def _build(self, backend: str, path: str) -> Embedder:
        s = get_settings()
        log.info("[embed] building embedder backend={} path={}", backend, path)
        if backend == "cloud_embed":
            from hat.core.neocortex.embeddings.cloud import build_cloud_embed_model

            # Cloud has no local files; ``path`` ends in the catalog id.
            model_id = Path(path).name
            entry = self._entry(backend, model_id)
            inner = build_cloud_embed_model(
                entry.repo_id,
                base_url=entry.base_url or "https://api.openai.com/v1",
                api_key_env=entry.api_key_env,
            )
            return ManagedEmbedder(inner, backend=backend, model_id=model_id)
        if backend == "mlx_embed":
            from hat.core.neocortex.embeddings.mlx import build_mlx_embed_model

            mlx_inner = build_mlx_embed_model(path, device=s.embed_device)
            return ManagedEmbedder(mlx_inner, backend=backend, model_id=Path(path).name)
        if backend == "hf_embed":
            from hat.core.neocortex.embeddings.hf import build_hf_embed_model

            hf_inner = build_hf_embed_model(path, device=s.embed_device)
            return ManagedEmbedder(hf_inner, backend=backend, model_id=Path(path).name)
        raise EmbeddingManagerError(
            f"backend {backend!r} cannot host an embedding model"
        )


_manager: EmbeddingManager | None = None


def get_embedding_manager() -> EmbeddingManager:
    global _manager
    if _manager is None:
        _manager = EmbeddingManager()
    return _manager


__all__ = [
    "EmbeddingManager",
    "EmbeddingManagerError",
    "get_embedding_manager",
]
