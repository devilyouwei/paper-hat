"""Embedding-model lifecycle service.

Mirror of :mod:`.models` for the embedder pipeline. Wraps
:class:`~hat.models.embedding_manager.EmbeddingManager` plus the deduper
swap helpers in :mod:`.container`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

from ...config.settings import embed_index_path_for
from ...memory.embeddings import Embedder
from ...models.embedding_manager import (
    EmbeddingManagerError,
    get_embedding_manager,
)
from ...utils.logging import get_logger
from .container import deactivate_embedder, swap_active_embedder

log = get_logger(__name__)

_DOWNLOADS_LOCK = threading.Lock()
_DOWNLOADS: dict[tuple[str, str], threading.Event] = {}


class DownloadAlreadyRunningError(Exception):
    pass


class NoActiveDownloadError(Exception):
    pass


def list_models(backend: str) -> list[dict]:
    return get_embedding_manager().list_models(backend)


def download(backend: str, model_id: str) -> str:
    return str(get_embedding_manager().download(backend, model_id))


def download_stream(backend: str, model_id: str) -> Iterator[bytes]:
    key = (backend, model_id)
    with _DOWNLOADS_LOCK:
        if key in _DOWNLOADS:
            raise DownloadAlreadyRunningError(
                f"download already running for {backend}/{model_id}"
            )
        cancel_event = threading.Event()
        _DOWNLOADS[key] = cancel_event

    def _gen() -> Iterator[bytes]:
        try:
            yield b": connected\n\n"
            for ev in get_embedding_manager().download_streaming(
                backend, model_id, cancel_event,
            ):
                stage = ev.get("stage", "progress")
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: {stage}\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:  # noqa: BLE001
            log.exception(
                "embed download stream crashed backend={} id={}",
                backend, model_id,
            )
            err = json.dumps({"stage": "error", "message": str(e)})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")
        finally:
            with _DOWNLOADS_LOCK:
                _DOWNLOADS.pop(key, None)

    return _gen()


def cancel_download(backend: str, model_id: str) -> None:
    key = (backend, model_id)
    with _DOWNLOADS_LOCK:
        ev = _DOWNLOADS.get(key)
    if ev is None:
        raise NoActiveDownloadError(
            f"no active download for {backend}/{model_id}"
        )
    ev.set()


def activate(backend: str, model_id: str) -> Embedder:
    return swap_active_embedder(backend, model_id)


def get_active() -> dict | None:
    return get_embedding_manager().active()


def deactivate() -> int:
    return deactivate_embedder()


def delete(backend: str, model_id: str) -> bool:
    return get_embedding_manager().delete(backend, model_id)


def index_path_for(backend: str, model_id: str) -> str:
    return str(embed_index_path_for(backend, model_id))


__all__ = [
    "EmbeddingManagerError",
    "DownloadAlreadyRunningError",
    "NoActiveDownloadError",
    "list_models",
    "download",
    "download_stream",
    "cancel_download",
    "activate",
    "get_active",
    "deactivate",
    "delete",
    "index_path_for",
]
