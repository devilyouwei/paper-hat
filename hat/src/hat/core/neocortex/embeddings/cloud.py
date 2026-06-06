"""Cloud (remote API) embedding backend.

Calls any OpenAI-compatible ``/v1/embeddings`` endpoint. Like the cloud LLM
backend it runs no local weights and is platform-independent (Apple / CUDA /
CPU). The HTTP layer uses stdlib ``urllib`` only — no extra dependencies.

Exposes the :class:`hat.core.neocortex.embeddings.managed.Embedder` Protocol
surface (``embed`` / ``dim`` / ``name``). Vectors are L2-normalised on the
client so they match the contract of the local embedders even if the provider
returns unnormalised embeddings.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

import numpy as np

from hat.utils.logging import get_logger

log = get_logger(__name__)


class CloudEmbedError(RuntimeError):
    pass


def _l2(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vecs / norms


class CloudEmbeddingModel:
    """OpenAI-compatible embeddings client exposing the Embedder API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"cloud_embed:{model}"
        self._dim: int | None = None
        log.info(
            "[cloud_embed] init model={} url={} key={}",
            model, self.base_url, "set" if api_key else "none",
        )

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/embeddings"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network
            detail = ""
            with contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", "replace")
            log.error("[cloud_embed] http {} {}: {}", e.code, url, detail[:400])
            raise CloudEmbedError(
                f"cloud embed request failed: {e.code} {e.reason}"
            ) from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.error("[cloud_embed] request error {}: {}", type(e).__name__, e)
            raise CloudEmbedError(f"cloud embed request failed: {e}") from e

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post({"model": self.model, "input": list(texts)})
        try:
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs = np.asarray(
                [r["embedding"] for r in rows], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CloudEmbedError(f"unexpected embed response: {e}") from e
        vecs = _l2(vecs)
        self._dim = int(vecs.shape[1])
        return vecs.tolist()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["probe"])
        return int(self._dim or 0)


def build_cloud_embed_model(
    model: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 60.0,
    **_: Any,
) -> CloudEmbeddingModel:
    if api_key is None and api_key_env:
        api_key = os.environ.get(api_key_env)
    return CloudEmbeddingModel(
        model, base_url=base_url, api_key=api_key, timeout=timeout
    )


__all__ = ["CloudEmbeddingModel", "CloudEmbedError", "build_cloud_embed_model"]
