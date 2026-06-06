"""Shared runtime-manager machinery for installed / active models.

Both the LLM :class:`~hat.core.lifecycle.manager.ModelManager` and the
embedding :class:`~hat.core.lifecycle.embedding_manager.EmbeddingManager` own
the *same* lifecycle: resolve a catalog entry to a local directory, download
weights from HuggingFace with cancellable SSE progress, cache built instances,
and track a single *active* ``(backend, id)`` pointer.

`BaseModelManager` captures all of that once. Subclasses declare a handful of
class attributes (which backends they serve, which file suffixes count as
"installed", their error type, …) and implement :meth:`_build`, the only step
that differs between a ``Cortex`` and an ``Embedder``.

The manager is process-local. If you scale to a separate sleep worker it keeps
its own cache there — intentional, since each process owns its GPU/Metal
context.
"""

from __future__ import annotations

import contextlib
import gc
import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator
from pathlib import Path
from threading import Lock
from typing import Generic, TypeVar

from hat.config.settings import get_settings
from hat.core.lifecycle.catalog import CatalogEntry, is_cloud_backend, load_catalog
from hat.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class BaseModelManager(ABC, Generic[T]):
    """Generic install / download / load / activate state machine.

    Subclasses configure the placeholders below and implement :meth:`_build`.
    """

    #: Backend names this manager accepts (e.g. ``SUPPORTED_BACKENDS``).
    supported: Collection[str]
    #: File suffixes that mark a directory as "installed".
    weight_suffixes: Collection[str]
    #: Exception type raised for caller-facing errors.
    error_cls: type[Exception]
    #: Human-readable noun used in error messages (e.g. ``"model"``).
    noun: str = "model"
    #: Prefix prepended to log lines (e.g. ``"[embed] "``).
    log_prefix: str = ""
    #: Attribute holding the heavy inner model on a built instance
    #: (``"lm"`` for a Cortex, ``"inner"`` for an Embedder).
    inner_attr: str = "lm"

    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[tuple[str, str], T] = {}
        self._active: tuple[str, str] | None = None

    # ---------- subclass hook -------------------------------------------

    @abstractmethod
    def _build(self, backend: str, path: str) -> T:
        """Construct the concrete instance for ``backend`` from ``path``.

        For cloud backends ``path`` ends in the catalog id (there are no local
        files); subclasses resolve the entry via :meth:`_entry` to recover the
        remote model name / base_url / api-key env var.
        """

    # ---------- paths ----------------------------------------------------

    def model_dir(self, backend: str, model_id: str) -> Path:
        return get_settings().model_root / backend / model_id

    def is_installed(self, backend: str, model_id: str) -> bool:
        # Cloud models have no local weights — they call a remote API, so there
        # is nothing to download. Treat them as always installed so the load /
        # activate paths don't reject them.
        if is_cloud_backend(backend):
            return True
        d = self.model_dir(backend, model_id)
        if not d.is_dir():
            return False
        # A directory with at least one weight-shaped file counts as installed.
        # Avoids importing huggingface_hub just to introspect.
        return any(child.suffix in self.weight_suffixes for child in d.iterdir())

    # ---------- listing --------------------------------------------------

    def list_models(self, backend: str) -> list[dict]:
        if backend not in self.supported:
            raise self.error_cls(f"unsupported backend {backend!r}")
        out: list[dict] = []
        for entry in load_catalog(backend):
            out.append(
                {
                    **entry.model_dump(),
                    "backend": backend,
                    "installed": self.is_installed(backend, entry.id),
                    "local_dir": str(self.model_dir(backend, entry.id)),
                }
            )
        return out

    def _entry(self, backend: str, model_id: str) -> CatalogEntry:
        for e in load_catalog(backend):
            if e.id == model_id:
                return e
        raise self.error_cls(
            f"unknown {self.noun} id {model_id!r} for backend {backend!r}"
        )

    # ---------- download -------------------------------------------------

    def download(self, backend: str, model_id: str) -> Path:
        entry = self._entry(backend, model_id)
        # Cloud models have nothing to fetch; resolving the entry validates it.
        if is_cloud_backend(backend):
            log.info(
                "{}cloud {} is remote; download is a no-op id={}",
                self.log_prefix, self.noun, model_id,
            )
            return self.model_dir(backend, model_id)
        dst = self.model_dir(backend, model_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover
            raise self.error_cls(
                "huggingface_hub is required for downloads"
            ) from e
        # Pin the HF blob cache inside the project's model root so weights are
        # not duplicated under ~/.cache/huggingface. ``local_dir`` keeps the
        # snapshot layout flat under model/<backend>/<id>/.
        cache_root = get_settings().model_root / ".hf-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "{}downloading backend={} id={} repo={} -> {}",
            self.log_prefix, backend, model_id, entry.repo_id, dst,
        )
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                local_dir=str(dst),
                cache_dir=str(cache_root),
            )
        except Exception:
            log.exception(
                "{}download failed backend={} id={} repo={}",
                self.log_prefix, backend, model_id, entry.repo_id,
            )
            raise
        log.info(
            "{}download complete backend={} id={}", self.log_prefix, backend, model_id
        )
        return dst

    # ---------- streaming download (with progress + cancel) -------------

    def download_streaming(
        self,
        backend: str,
        model_id: str,
        cancel_event: threading.Event,
        *,
        poll_interval: float = 0.5,
    ) -> Iterator[dict]:
        """Yield progress events while downloading a catalog entry.

        Events have a ``stage`` field:

        * ``start``     — total file / byte counts known
        * ``progress``  — per-poll snapshot (``bytes_done``, ``files_done``)
        * ``done``      — local_dir on disk
        * ``cancelled`` — caller set ``cancel_event``; partial dir removed
        * ``error``     — terminal error (``message`` field)

        Cancellation is *best effort*: the worker thread cannot interrupt a
        single in-flight file download cleanly, so we wait for the current file
        to finish (or fail) before tearing down the destination directory. The
        next file is never started once cancellation is observed between files.
        """
        try:
            entry = self._entry(backend, model_id)
        except self.error_cls as e:  # type: ignore[misc]
            yield {"stage": "error", "message": str(e)}
            return

        # Cloud models are remote: nothing to fetch. Emit a synthetic
        # start/done pair so the UI's progress renderer completes cleanly.
        if is_cloud_backend(backend):
            yield {
                "stage": "start",
                "backend": backend,
                "id": model_id,
                "repo_id": entry.repo_id,
                "files_total": 0,
                "bytes_total": 0,
                "local_dir": str(self.model_dir(backend, model_id)),
            }
            yield {
                "stage": "done",
                "backend": backend,
                "id": model_id,
                "local_dir": str(self.model_dir(backend, model_id)),
                "files_total": 0,
                "bytes_total": 0,
            }
            return

        dst = self.model_dir(backend, model_id)
        dst.mkdir(parents=True, exist_ok=True)
        cache_root = get_settings().model_root / ".hf-cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as e:  # pragma: no cover
            yield {"stage": "error", "message": f"huggingface_hub missing: {e}"}
            return

        log.info(
            "{}streaming download backend={} id={} repo={}",
            self.log_prefix, backend, model_id, entry.repo_id,
        )

        # Resolve the list of repo files + sizes up front so we can render a
        # meaningful progress bar instead of an indeterminate spinner.
        try:
            info = HfApi().model_info(entry.repo_id, files_metadata=True)
        except Exception as e:  # noqa: BLE001 - reported via stream
            log.exception("{}model_info failed repo={}", self.log_prefix, entry.repo_id)
            yield {"stage": "error", "message": f"model_info failed: {e}"}
            return

        siblings = list(getattr(info, "siblings", None) or [])
        files: list[tuple[str, int]] = [
            (s.rfilename, int(getattr(s, "size", 0) or 0)) for s in siblings
        ]
        files_total = len(files)
        bytes_total = sum(sz for _, sz in files)
        # Fallback when the API omits file sizes: use the catalog's advertised
        # size_gb so the bar at least has a denominator.
        if bytes_total <= 0 and entry.size_gb:
            bytes_total = int(entry.size_gb * (1024**3))

        yield {
            "stage": "start",
            "backend": backend,
            "id": model_id,
            "repo_id": entry.repo_id,
            "files_total": files_total,
            "bytes_total": bytes_total,
            "local_dir": str(dst),
        }

        bytes_done = 0
        files_done = 0

        for rel, sz in files:
            if cancel_event.is_set():
                break

            target_path = dst / rel
            # If a previous run already fetched this file, skip the worker.
            if target_path.exists() and sz and target_path.stat().st_size >= sz:
                bytes_done += sz
                files_done += 1
                yield {
                    "stage": "progress",
                    "file": rel,
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
                continue

            result: dict = {"err": None}

            def _worker(rel: str = rel, result: dict = result) -> None:
                try:
                    hf_hub_download(
                        repo_id=entry.repo_id,
                        filename=rel,
                        local_dir=str(dst),
                        cache_dir=str(cache_root),
                    )
                except Exception as e:  # noqa: BLE001 - reported via stream
                    result["err"] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while t.is_alive():
                # Poll partial file size on disk so very large shards show
                # within-file progress, not just per-file ticks.
                cur = target_path.stat().st_size if target_path.exists() else 0
                yield {
                    "stage": "progress",
                    "file": rel,
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done + cur,
                    "bytes_total": bytes_total,
                }
                if cancel_event.is_set():
                    # Let the in-flight file settle to a consistent state before
                    # we rmtree() the directory — yanking the file out from
                    # under the worker can leave HF's atomic-write tempfile
                    # dangling.
                    break
                t.join(timeout=poll_interval)

            if cancel_event.is_set():
                t.join()  # wait out the in-flight write
                shutil.rmtree(dst, ignore_errors=True)
                log.info(
                    "{}download cancelled backend={} id={}",
                    self.log_prefix, backend, model_id,
                )
                yield {
                    "stage": "cancelled",
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
                return

            if result["err"] is not None:
                log.exception(
                    "{}download error backend={} id={} file={}",
                    self.log_prefix, backend, model_id, rel,
                )
                yield {
                    "stage": "error",
                    "file": rel,
                    "message": f"{type(result['err']).__name__}: {result['err']}",
                }
                return

            bytes_done += sz or (
                target_path.stat().st_size if target_path.exists() else 0
            )
            files_done += 1
            yield {
                "stage": "progress",
                "file": rel,
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }

        # Loop exit without an explicit cancel branch: either everything
        # downloaded or the very first iteration saw ``cancel_event`` set before
        # any file was touched. Handle the latter explicitly.
        if cancel_event.is_set():
            shutil.rmtree(dst, ignore_errors=True)
            log.info(
                "{}download cancelled backend={} id={}",
                self.log_prefix, backend, model_id,
            )
            yield {
                "stage": "cancelled",
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }
            return

        log.info(
            "{}download complete backend={} id={}", self.log_prefix, backend, model_id
        )
        yield {
            "stage": "done",
            "backend": backend,
            "id": model_id,
            "local_dir": str(dst),
            "files_total": files_total,
            "bytes_total": bytes_total,
        }

    # ---------- load + activate -----------------------------------------

    def load(self, backend: str, model_id: str) -> T:
        key = (backend, model_id)
        with self._lock:
            if key in self._cache:
                log.debug(
                    "{}{} cache hit backend={} id={}",
                    self.log_prefix, self.noun, backend, model_id,
                )
                return self._cache[key]
            if not self.is_installed(backend, model_id):
                log.error(
                    "{}load aborted: {} not installed backend={} id={}",
                    self.log_prefix, self.noun, backend, model_id,
                )
                raise self.error_cls(
                    f"{self.noun} {model_id!r} not installed; download it first"
                )
            log.info(
                "{}loading {} backend={} id={}",
                self.log_prefix, self.noun, backend, model_id,
            )
            try:
                built = self._build(backend, str(self.model_dir(backend, model_id)))
            except Exception:
                log.exception(
                    "{}{} load failed backend={} id={}",
                    self.log_prefix, self.noun, backend, model_id,
                )
                raise
            self._cache[key] = built
            log.info(
                "{}{} loaded backend={} id={}",
                self.log_prefix, self.noun, backend, model_id,
            )
            return built

    def _release(self, instance: T) -> None:
        """Best-effort teardown so the previous model frees GPU/Metal memory.

        The built wrapper keeps a reference to a heavy model + tokenizer;
        dropping the wrapper alone is not enough on CUDA/MPS where the
        allocator caches blocks. We null out the heavy attrs and ask the
        framework to release its cache.
        """
        inner = getattr(instance, self.inner_attr, None)
        for obj in (inner, instance):
            for attr in ("model", "tokenizer", "_model", "_tokenizer"):
                if hasattr(obj, attr):
                    with contextlib.suppress(Exception):
                        setattr(obj, attr, None)
        gc.collect()
        with contextlib.suppress(ImportError):  # CUDA / MPS allocator
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if callable(empty):
                    empty()
        with contextlib.suppress(ImportError):  # MLX (Metal) allocator
            import mlx.core as mx

            clear = getattr(mx, "clear_cache", None) or getattr(
                getattr(mx, "metal", None), "clear_cache", None
            )
            if callable(clear):
                clear()

    def unload(self, backend: str, model_id: str) -> bool:
        """Drop a cached instance and free its memory. Returns True if removed."""
        key = (backend, model_id)
        with self._lock:
            instance = self._cache.pop(key, None)
            if self._active == key:
                self._active = None
        if instance is None:
            log.debug(
                "{}unload no-op (not cached) backend={} id={}",
                self.log_prefix, backend, model_id,
            )
            return False
        log.info(
            "{}unloading {} backend={} id={}",
            self.log_prefix, self.noun, backend, model_id,
        )
        self._release(instance)
        log.info(
            "{}{} unloaded backend={} id={}",
            self.log_prefix, self.noun, backend, model_id,
        )
        return True

    def unload_all(self) -> int:
        """Drop every cached instance and clear the active pointer."""
        with self._lock:
            evicted = list(self._cache.values())
            self._cache.clear()
            self._active = None
        if evicted:
            log.info(
                "{}unloading all {}s count={}", self.log_prefix, self.noun, len(evicted)
            )
        for old in evicted:
            self._release(old)
        return len(evicted)

    def delete(self, backend: str, model_id: str) -> bool:
        """Remove an installed model's weights from disk. Refuses to delete the
        currently-active model. Returns True if files were removed."""
        if backend not in self.supported:
            raise self.error_cls(f"unsupported backend {backend!r}")
        with self._lock:
            if self._active == (backend, model_id):
                raise self.error_cls(
                    f"cannot delete the active {self.noun}; unload it first"
                )
            # Drop any cached (but not active) instance for this id too.
            instance = self._cache.pop((backend, model_id), None)
        if instance is not None:
            self._release(instance)
        # Cloud models have no on-disk weights to remove.
        if is_cloud_backend(backend):
            log.debug(
                "{}delete no-op (cloud) backend={} id={}",
                self.log_prefix, backend, model_id,
            )
            return False
        d = self.model_dir(backend, model_id)
        if not d.exists():
            log.debug(
                "{}delete no-op (not on disk) backend={} id={}",
                self.log_prefix, backend, model_id,
            )
            return False
        log.warning(
            "{}deleting {} weights backend={} id={} dir={}",
            self.log_prefix, self.noun, backend, model_id, d,
        )
        shutil.rmtree(d, ignore_errors=False)
        return True

    def set_active(self, backend: str, model_id: str) -> T:
        new_key = (backend, model_id)

        # Fast path: already cached — just flip the pointer and evict siblings.
        with self._lock:
            if new_key in self._cache:
                instance = self._cache[new_key]
                stale = [k for k in self._cache if k != new_key]
                evicted = [self._cache.pop(k) for k in stale]
                self._active = new_key
            else:
                instance = None
                evicted = list(self._cache.values())
                self._cache.clear()
                self._active = None
        if evicted:
            log.info(
                "{}evicting {} cached {}(s) before activating {}/{}",
                self.log_prefix, len(evicted), self.noun, backend, model_id,
            )
            for old in evicted:
                self._release(old)
        if instance is not None:
            log.info(
                "{}active {} set (cached) backend={} id={}",
                self.log_prefix, self.noun, backend, model_id,
            )
            return instance

        # Slow path: build the new instance *after* freeing the old one so the
        # GPU/Metal allocator only has to host one model at a time.
        if not self.is_installed(backend, model_id):
            raise self.error_cls(
                f"{self.noun} {model_id!r} not installed; download it first"
            )
        log.info(
            "{}activating {} backend={} id={}",
            self.log_prefix, self.noun, backend, model_id,
        )
        try:
            instance = self._build(backend, str(self.model_dir(backend, model_id)))
        except Exception:
            log.exception(
                "{}activation failed backend={} id={}",
                self.log_prefix, backend, model_id,
            )
            raise
        with self._lock:
            self._cache[new_key] = instance
            self._active = new_key
        log.info(
            "{}active {} set backend={} id={}",
            self.log_prefix, self.noun, backend, model_id,
        )
        return instance

    def active(self) -> dict | None:
        if self._active is None:
            return None
        backend, model_id = self._active
        return {"backend": backend, "id": model_id}
