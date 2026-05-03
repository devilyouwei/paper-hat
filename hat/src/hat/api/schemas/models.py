from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CatalogItem(BaseModel):
    id: str
    repo_id: str
    display: str
    backend: Literal["mlx", "hf"]
    size_gb: float | None = None
    notes: str | None = None
    installed: bool
    local_dir: str


class ModelListResponse(BaseModel):
    backend: Literal["mlx", "hf"]
    items: list[CatalogItem]


class ModelActionRequest(BaseModel):
    backend: Literal["mlx", "hf"]
    id: str


class ModelDownloadResponse(BaseModel):
    backend: str
    id: str
    local_dir: str
    installed: bool = True


class ActiveModel(BaseModel):
    backend: str
    id: str
    name: str | None = None
