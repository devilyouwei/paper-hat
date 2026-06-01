"""Model lifecycle: registry, catalogs, manager, embedding manager.

These types own checkpoint download / load / unload and back the
``models`` and ``embedding_models`` API endpoints.
"""

from .catalog import (
    ALL_SUPPORTED_BACKENDS,
    SUPPORTED_BACKENDS,
    SUPPORTED_EMBED_BACKENDS,
    CatalogEntry,
    load_catalog,
)
from .embedding_manager import (
    EmbeddingManagerError,
    get_embedding_manager,
)
from .manager import ModelManagerError, get_manager

__all__ = [
    "ALL_SUPPORTED_BACKENDS",
    "CatalogEntry",
    "EmbeddingManagerError",
    "ModelManagerError",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_EMBED_BACKENDS",
    "get_embedding_manager",
    "get_manager",
    "load_catalog",
]
