"""Schemas for the embedding-model management API.

Mirrors :mod:`hat.api.schemas.models` but typed for the embed kind.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EmbedBackend = Literal["mlx_embed", "hf_embed", "cloud_embed"]


class EmbeddingCatalogItem(BaseModel):
    id: str
    repo_id: str
    display: str
    backend: EmbedBackend
    size_gb: float | None = None
    notes: str | None = None
    installed: bool
    local_dir: str


class EmbeddingModelListResponse(BaseModel):
    backend: EmbedBackend
    items: list[EmbeddingCatalogItem]


class EmbeddingModelActionRequest(BaseModel):
    backend: EmbedBackend
    id: str


class EmbeddingModelDownloadResponse(BaseModel):
    backend: str
    id: str
    local_dir: str
    installed: bool = True


class ActiveEmbedder(BaseModel):
    backend: str
    id: str
    name: str | None = None
    index_path: str | None = None
