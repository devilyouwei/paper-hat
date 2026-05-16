"""Model management endpoints.

* ``GET    /api/models?backend=`` — catalog + installed status
* ``POST   /api/models/download``  — snapshot_download a catalog entry (blocking)
* ``GET    /api/models/download/stream?backend=&id=`` — SSE progress stream
* ``POST   /api/models/download/cancel`` — cancel an in-flight stream download
* ``POST   /api/models/active``    — load + set as the loop's active Cortex
* ``GET    /api/models/active``    — current active model
* ``DELETE /api/models/active``    — unload all models, free memory
* ``DELETE /api/models/{backend}/{id}`` — remove installed weights from disk
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...models.manager import ModelManagerError, get_manager
from ...utils.logging import get_logger
from ..deps import deactivate_cortex, swap_active_cortex
from ..schemas.models import (
    ActiveModel,
    CatalogItem,
    ModelActionRequest,
    ModelDownloadResponse,
    ModelListResponse,
)

log = get_logger(__name__)

router = APIRouter()


# In-flight streaming downloads keyed by (backend, id). The value is the
# ``threading.Event`` the worker polls between files; setting it triggers
# best-effort cancellation + cleanup of the partial directory.
_DOWNLOADS_LOCK = threading.Lock()
_DOWNLOADS: dict[tuple[str, str], threading.Event] = {}


@router.get("", response_model=ModelListResponse)
def list_models(backend: str = Query(default="mlx")) -> ModelListResponse:
    try:
        items = get_manager().list_models(backend)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return ModelListResponse(
        backend=backend,  # type: ignore[arg-type]
        items=[CatalogItem(**i) for i in items],
    )


@router.post("/download", response_model=ModelDownloadResponse)
def download_model(req: ModelActionRequest) -> ModelDownloadResponse:
    try:
        path = get_manager().download(req.backend, req.id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # pragma: no cover - network errors surface here
        raise HTTPException(500, f"download failed: {e}") from e
    return ModelDownloadResponse(
        backend=req.backend, id=req.id, local_dir=str(path)
    )


@router.get("/download/stream")
def download_stream(
    backend: str = Query(...), id: str = Query(...)
) -> StreamingResponse:
    """Stream download progress as Server-Sent Events.

    The browser opens this endpoint via ``EventSource``. Each event is a
    JSON payload with a ``stage`` discriminator (``start`` / ``progress`` /
    ``done`` / ``cancelled`` / ``error``). The connection closes once a
    terminal event is sent.

    Only one streaming download per ``(backend, id)`` may be in flight at
    a time; concurrent requests get HTTP 409.
    """
    key = (backend, id)
    with _DOWNLOADS_LOCK:
        if key in _DOWNLOADS:
            raise HTTPException(409, f"download already running for {backend}/{id}")
        cancel_event = threading.Event()
        _DOWNLOADS[key] = cancel_event

    def sse() -> Iterator[bytes]:
        try:
            # An initial comment forces some proxies (and the browser) to
            # flush the response headers immediately so the EventSource
            # transitions to ``open`` before the first real event.
            yield b": connected\n\n"
            for ev in get_manager().download_streaming(
                backend, id, cancel_event
            ):
                stage = ev.get("stage", "progress")
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: {stage}\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:  # noqa: BLE001 - terminal event below
            log.exception("download stream crashed backend={} id={}", backend, id)
            err = json.dumps({"stage": "error", "message": str(e)})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")
        finally:
            with _DOWNLOADS_LOCK:
                _DOWNLOADS.pop(key, None)

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx, uvicorn behind a reverse
            # proxy) so events reach the browser as they are produced.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/download/cancel")
def cancel_download(req: ModelActionRequest) -> dict:
    """Signal an in-flight streaming download to abort. The worker waits
    for the current file to finish before deleting the partial directory.
    """
    key = (req.backend, req.id)
    with _DOWNLOADS_LOCK:
        ev = _DOWNLOADS.get(key)
    if ev is None:
        raise HTTPException(404, f"no active download for {req.backend}/{req.id}")
    ev.set()
    log.info("download cancel requested backend={} id={}", req.backend, req.id)
    return {"backend": req.backend, "id": req.id, "cancelled": True}


@router.post("/active", response_model=ActiveModel)
def set_active(req: ModelActionRequest) -> ActiveModel:
    try:
        cortex = swap_active_cortex(req.backend, req.id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # backend load errors (OOM, missing extras, ...)
        # Without this, FastAPI converts the exception to a bare 500 with no
        # body, and the UI shows nothing useful. Surface the type + message
        # so the toast can tell the user *why* activation failed.
        raise HTTPException(
            500, f"activate failed: {type(e).__name__}: {e}"
        ) from e
    return ActiveModel(
        backend=req.backend, id=req.id, name=getattr(cortex, "name", None)
    )


@router.get("/active", response_model=ActiveModel | None)
def get_active() -> ActiveModel | None:
    a = get_manager().active()
    if a is None:
        return None
    return ActiveModel(backend=a["backend"], id=a["id"])


@router.delete("/active")
def deactivate() -> dict:
    """Unload the current model (and any other cached weights), freeing
    GPU/Metal memory. The loop falls back to the Noop cortex until a new
    model is activated."""
    n = deactivate_cortex()
    return {"unloaded": n}


@router.delete("/{backend}/{model_id:path}")
def delete_model(backend: str, model_id: str) -> dict:
    """Remove an installed model's weights from disk. Refuses if it's
    currently the active model."""
    try:
        removed = get_manager().delete(backend, model_id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return {"backend": backend, "id": model_id, "removed": removed}
