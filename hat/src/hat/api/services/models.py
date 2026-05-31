"""LLM (Cortex) lifecycle service.

Wraps the :class:`~hat.models.manager.ModelManager` and the loop swap
helpers in :mod:`.container` so HTTP controllers stay thin. Owns the
in-flight streaming-download registry shared across requests.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

from ...core.cortex.base import Cortex
from ...models.manager import ModelManagerError, get_manager
from ...utils.logging import get_logger
from .container import deactivate_cortex, swap_active_cortex

log = get_logger(__name__)

_DOWNLOADS_LOCK = threading.Lock()
_DOWNLOADS: dict[tuple[str, str], threading.Event] = {}


class DownloadAlreadyRunningError(Exception):
    """Raised when a streaming download is already in flight for the pair."""


class NoActiveDownloadError(Exception):
    """Raised when ``cancel_download`` cannot find a matching in-flight stream."""


def list_models(backend: str) -> list[dict]:
    return get_manager().list_models(backend)


def download(backend: str, model_id: str) -> str:
    return str(get_manager().download(backend, model_id))


def download_stream(backend: str, model_id: str) -> Iterator[bytes]:
    """SSE byte iterator. Raises :class:`DownloadAlreadyRunningError` on
    concurrent invocation for the same pair."""
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
            for ev in get_manager().download_streaming(
                backend, model_id, cancel_event
            ):
                stage = ev.get("stage", "progress")
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: {stage}\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:  # noqa: BLE001 - terminal event below
            log.exception(
                "download stream crashed backend={} id={}", backend, model_id
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
    log.info("download cancel requested backend={} id={}", backend, model_id)


def activate(backend: str, model_id: str) -> Cortex:
    return swap_active_cortex(backend, model_id)


def get_active() -> dict | None:
    return get_manager().active()


def deactivate() -> int:
    return deactivate_cortex()


def delete(backend: str, model_id: str) -> bool:
    return get_manager().delete(backend, model_id)


__all__ = [
    "ModelManagerError",
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
]
