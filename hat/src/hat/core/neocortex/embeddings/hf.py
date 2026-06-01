"""HuggingFace embedding backend (placeholder).

Wraps ``sentence_transformers.SentenceTransformer`` so any local
SentenceTransformer checkpoint under ``model/hf_embed/<id>/`` works as
a managed embedder. The default catalog ships empty in this iteration;
the backend is wired so the UI / API treat it as a first-class peer of
``mlx_embed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hat.utils.logging import get_logger

log = get_logger(__name__)


class HFEmbeddingModel:
    """Thin SentenceTransformer wrapper exposing the Embedder API."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "HF embed backend requires `sentence-transformers`"
            ) from e
        self.model_path = model_path
        self.name = f"hf_embed:{model_path}"
        self._device = device if device and device != "auto" else None
        log.info("[hf_embed] loading path={}", model_path)
        self.model = SentenceTransformer(model_path, device=self._device)
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self._dim = int(getter())
        log.info("[hf_embed] loaded path={} dim={}", model_path, self._dim)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32).tolist()

    @property
    def dim(self) -> int:
        return self._dim


def build_hf_embed_model(model_path: str, **kwargs: Any) -> HFEmbeddingModel:
    return HFEmbeddingModel(model_path, **kwargs)


__all__ = ["HFEmbeddingModel", "build_hf_embed_model"]
