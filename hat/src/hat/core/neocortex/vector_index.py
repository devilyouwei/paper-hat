"""Lightweight NPZ-backed vector index for curated memory dedup.

The index keeps two arrays in memory and on disk:

* ``ids``  — ``str`` array of ``trace_id``s
* ``vecs`` — ``float32`` matrix ``[N, D]`` of L2-normalised embeddings

That keeps lookup to a single ``vecs @ q`` matmul (cosine because every
row is unit-norm). Sized for ~10⁵ rows; replace with a real ANN store if
the dataset ever outgrows that.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hat.abstract.neocortex import Match, VectorIndex


class NpzVectorIndex(VectorIndex):
    """In-memory ``(ids, vecs)`` table persisted as a single ``.npz`` file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.load()

    # ------------------------------------------------------------------ I/O
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                ids = data["ids"]
                vecs = data["vecs"]
        except Exception:
            # corrupt/empty file — start fresh
            self._ids, self._vecs = [], np.zeros((0, 0), dtype=np.float32)
            return
        self._ids = [str(x) for x in ids.tolist()]
        self._vecs = np.asarray(vecs, dtype=np.float32)
        if self._vecs.ndim == 1:
            self._vecs = self._vecs.reshape(0, 0)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            np.savez(
                tmp,
                ids=np.asarray(self._ids, dtype=object),
                vecs=self._vecs.astype(np.float32, copy=False),
            )
            os.replace(tmp, self.path)

    # ----------------------------------------------------------------- meta
    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return int(self._vecs.shape[1]) if self._vecs.size else 0

    def ids(self) -> list[str]:
        return list(self._ids)

    def has(self, trace_id: str) -> bool:
        return trace_id in self._ids

    # --------------------------------------------------------------- writes
    def append(self, trace_id: str, vec: Sequence[float]) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        with self._lock:
            if not self._ids:
                self._vecs = v
            else:
                if v.shape[1] != self._vecs.shape[1]:
                    raise ValueError(
                        f"vector dim mismatch: existing={self._vecs.shape[1]} new={v.shape[1]}"
                    )
                self._vecs = np.vstack([self._vecs, v])
            self._ids.append(trace_id)
        self.save()

    def update(self, trace_id: str, vec: Sequence[float]) -> bool:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        with self._lock:
            try:
                idx = self._ids.index(trace_id)
            except ValueError:
                return False
            if v.shape[0] != self._vecs.shape[1]:
                raise ValueError(
                    f"vector dim mismatch: existing={self._vecs.shape[1]} new={v.shape[0]}"
                )
            self._vecs[idx] = v
        self.save()
        return True

    def remove(self, trace_id: str) -> bool:
        with self._lock:
            try:
                idx = self._ids.index(trace_id)
            except ValueError:
                return False
            self._ids.pop(idx)
            mask = np.ones(self._vecs.shape[0], dtype=bool)
            mask[idx] = False
            self._vecs = self._vecs[mask]
        self.save()
        return True

    # ---------------------------------------------------------------- query
    def top1(
        self, vec: Sequence[float], *, exclude: str | None = None
    ) -> Match | None:
        if not self._ids:
            return None
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vecs.shape[1]:
            return None
        sims = self._vecs @ q  # both sides are unit-norm => cosine similarity
        order = np.argsort(-sims)
        for i in order:
            tid = self._ids[i]
            if exclude is not None and tid == exclude:
                continue
            return Match(trace_id=tid, similarity=float(sims[i]))
        return None


__all__ = ["Match", "NpzVectorIndex"]