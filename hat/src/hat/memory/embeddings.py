"""Embedding helpers used by the novelty estimator."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def hash_embed(texts: Sequence[str], dim: int = 64) -> list[list[float]]:
    """Deterministic byte-hash embedding. Replace with a real model in production
    (sentence-transformers, OpenAI embeddings, BGE, …).
    """
    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:dim]])
    return out


__all__ = ["hash_embed"]
