"""Embedding-model management endpoints (mounted at ``/api/embedding-models``)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..schemas.embedding_models import (
    ActiveEmbedder,
    EmbeddingCatalogItem,
    EmbeddingModelActionRequest,
    EmbeddingModelDownloadResponse,
    EmbeddingModelListResponse,
)
from ..services import embedding_models as svc

router = APIRouter()


@router.get("", response_model=EmbeddingModelListResponse)
def list_embedding_models(
    backend: str = Query(default="mlx_embed"),
) -> EmbeddingModelListResponse:
    try:
        items = svc.list_models(backend)
    except svc.EmbeddingManagerError as e:
        raise HTTPException(400, str(e)) from e
    return EmbeddingModelListResponse(
        backend=backend,  # type: ignore[arg-type]
        items=[EmbeddingCatalogItem(**i) for i in items],
    )


@router.post("/download", response_model=EmbeddingModelDownloadResponse)
def download_embedding_model(
    req: EmbeddingModelActionRequest,
) -> EmbeddingModelDownloadResponse:
    try:
        path = svc.download(req.backend, req.id)
    except svc.EmbeddingManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, f"download failed: {e}") from e
    return EmbeddingModelDownloadResponse(
        backend=req.backend, id=req.id, local_dir=path,
    )


@router.get("/download/stream")
def download_embedding_stream(
    backend: str = Query(...), id: str = Query(...),
) -> StreamingResponse:
    try:
        gen = svc.download_stream(backend, id)
    except svc.DownloadAlreadyRunningError as e:
        raise HTTPException(409, str(e)) from e
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/download/cancel")
def cancel_embedding_download(req: EmbeddingModelActionRequest) -> dict:
    try:
        svc.cancel_download(req.backend, req.id)
    except svc.NoActiveDownloadError as e:
        raise HTTPException(404, str(e)) from e
    return {"backend": req.backend, "id": req.id, "cancelled": True}


@router.post("/active", response_model=ActiveEmbedder)
def set_active_embedder(req: EmbeddingModelActionRequest) -> ActiveEmbedder:
    try:
        emb = svc.activate(req.backend, req.id)
    except svc.EmbeddingManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(
            500, f"activate failed: {type(e).__name__}: {e}",
        ) from e
    return ActiveEmbedder(
        backend=req.backend,
        id=req.id,
        name=getattr(emb, "name", None),
        index_path=svc.index_path_for(req.backend, req.id),
    )


@router.get("/active", response_model=ActiveEmbedder | None)
def get_active_embedder() -> ActiveEmbedder | None:
    a = svc.get_active()
    if a is None:
        return None
    return ActiveEmbedder(
        backend=a["backend"],
        id=a["id"],
        index_path=svc.index_path_for(a["backend"], a["id"]),
    )


@router.delete("/active")
def deactivate_active_embedder() -> dict:
    return {"unloaded": svc.deactivate()}


@router.delete("/{backend}/{model_id:path}")
def delete_embedding_model(backend: str, model_id: str) -> dict:
    try:
        removed = svc.delete(backend, model_id)
    except svc.EmbeddingManagerError as e:
        raise HTTPException(400, str(e)) from e
    return {"backend": backend, "id": model_id, "removed": removed}
