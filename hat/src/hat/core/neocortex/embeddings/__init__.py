"""Embedder backends used by the Neocortex deduper / vector index.

The :class:`hat.abstract.neocortex.Embedder` Protocol is the seam;
``ManagedEmbedder`` wraps a real backend with hot-swap lifecycle.
"""

from .managed import ManagedEmbedder

__all__ = ["ManagedEmbedder"]
