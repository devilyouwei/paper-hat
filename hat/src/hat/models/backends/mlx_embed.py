"""MLX-native embedding backend.

Uses `mlx-embeddings <https://github.com/Blaizzy/mlx-embeddings>`_ — the
community port of common embedding architectures (BERT, E5, BGE, Qwen3,
EmbeddingGemma) to Apple's Metal-native MLX runtime.

Install with::

    uv sync --extra mlx

Implements the :class:`hat.memory.embeddings.Embedder` shape:

* ``embed(texts) -> list[list[float]]`` (L2-normalised)
* ``dim`` — vector dimensionality (lazy, populated on first call)
* ``name`` — descriptive identifier for logs / metadata

The MLX embedding APIs are still settling, so the inference call adapts
to whichever shape the loaded model exposes:

1. ``model.encode(texts)`` — used by some adapters to do batched inference.
2. ``model(input_ids)`` returning a dict with ``text_embeds`` / ``embeddings``
   / ``last_hidden_state`` (in that preference order, with mean-pool over
   tokens applied to ``last_hidden_state``).

If neither path produces a vector the call raises ``RuntimeError`` with
the model's class name so we surface integration breaks loudly instead of
silently falling back to the hash embedder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...utils.logging import get_logger

log = get_logger(__name__)


def _l2(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class MLXEmbeddingModel:
    """Thin wrapper around ``mlx_embeddings.load`` exposing the Embedder API."""

    def __init__(self, model_path: str, device: str = "auto") -> None:  # noqa: ARG002 - device unused on MLX
        try:
            from mlx_embeddings.utils import load  # type: ignore[import-not-found]
        except ImportError:
            try:
                from mlx_embeddings import load  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "MLX embed backend requires `mlx-embeddings`: "
                    "`uv sync --extra mlx`"
                ) from e

        self.model_path = model_path
        self.name = f"mlx_embed:{model_path}"
        log.info("[mlx_embed] loading path={}", model_path)
        try:
            self.model, self.tokenizer = load(model_path)
        except Exception:
            log.exception("[mlx_embed] failed to load path={}", model_path)
            raise
        self._dim: int | None = None
        log.info("[mlx_embed] loaded path={}", model_path)

    # -- Embedder protocol ------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._encode(list(texts))
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = _l2(arr)
        if self._dim is None:
            self._dim = int(arr.shape[1])
        return arr.tolist()

    @property
    def dim(self) -> int:
        if self._dim is None:
            # eager probe so callers can size storage up-front
            self.embed(["probe"])
        assert self._dim is not None
        return self._dim

    # -- inference adapter ------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Preferred path: model exposes its own batched encoder.
        encode = getattr(self.model, "encode", None)
        if callable(encode):
            try:
                out = encode(texts)
                return _to_numpy(out)
            except Exception:  # noqa: BLE001 - try the manual path
                log.debug("[mlx_embed] model.encode raised; falling back")

        rows: list[np.ndarray] = []
        for text in texts:
            rows.append(self._encode_one(text))
        return np.vstack(rows)

    def _encode_one(self, text: str) -> np.ndarray:
        import mlx.core as mx  # type: ignore[import-not-found]

        enc = self.tokenizer(text, return_tensors="np")
        input_ids = mx.array(enc["input_ids"])
        kwargs: dict[str, Any] = {}
        if "attention_mask" in enc:
            kwargs["attention_mask"] = mx.array(enc["attention_mask"])
        out = self.model(input_ids, **kwargs)
        vec = _extract_vector(out, kwargs.get("attention_mask"))
        return np.asarray(vec, dtype=np.float32).reshape(-1)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "tolist"):
        return np.asarray(x.tolist(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _extract_vector(out: Any, attention_mask: Any) -> np.ndarray:
    """Pull a single embedding out of a model forward result."""
    # dict-like
    for key in ("text_embeds", "sentence_embeddings", "embeddings", "pooler_output"):
        v = _maybe(out, key)
        if v is not None:
            return _to_numpy(v).reshape(-1)
    # mean-pool last_hidden_state
    hidden = _maybe(out, "last_hidden_state") or _maybe(out, "hidden_states")
    if hidden is not None:
        h = _to_numpy(hidden)
        if h.ndim == 3:
            if attention_mask is not None:
                m = _to_numpy(attention_mask).astype(np.float32)
                m = m.reshape(*m.shape, 1)
                summed = (h * m).sum(axis=1)
                denom = np.clip(m.sum(axis=1), a_min=1e-6, a_max=None)
                return (summed / denom).reshape(-1)
            return h.mean(axis=1).reshape(-1)
        if h.ndim == 2:
            return h.mean(axis=0).reshape(-1)
        return h.reshape(-1)
    raise RuntimeError(
        f"mlx_embed: cannot extract embedding from {type(out).__name__}"
    )


def _maybe(out: Any, key: str) -> Any:
    if isinstance(out, dict):
        return out.get(key)
    return getattr(out, key, None)


def build_mlx_embed_model(model_path: str, **kwargs: Any) -> MLXEmbeddingModel:
    return MLXEmbeddingModel(model_path, **kwargs)


__all__ = ["MLXEmbeddingModel", "build_mlx_embed_model"]
