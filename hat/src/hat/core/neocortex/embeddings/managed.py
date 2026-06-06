"""Embedding helpers used for curated-memory deduplication.

Embedders are produced exclusively by
:class:`hat.core.lifecycle.embedding_manager.EmbeddingManager` from the
catalog; this module only defines the :class:`ManagedEmbedder` adapter that
tags vectors with their ``<backend>/<id>`` source. The embedder seam itself is
the :class:`hat.abstract.neocortex.Embedder` Protocol, re-exported here for
convenience.
"""

from __future__ import annotations

from collections.abc import Sequence

from hat.abstract.neocortex import Embedder


class ManagedEmbedder:
    """Adapter wrapping a managed (catalog-driven) embedding backend.

    The underlying ``inner`` object is a backend-specific model (e.g.
    :class:`MLXEmbeddingModel`) that already implements
    ``embed(texts) -> list[list[float]]`` and exposes ``dim``. We add the
    ``backend`` / ``model_id`` tag so the loop can stamp memory rows with
    the embedder that wrote them.
    """

    def __init__(self, inner: Embedder, *, backend: str, model_id: str) -> None:
        self.inner = inner
        self.backend = backend
        self.model_id = model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.inner.embed(texts)

    @property
    def dim(self) -> int:
        return int(self.inner.dim)

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.model_id}"


__all__ = ["Embedder", "ManagedEmbedder"]
