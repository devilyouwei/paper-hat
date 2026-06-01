"""Composition root used by both the FastAPI service layer and the CLI.

By living in :mod:`hat.core.runtime`, this module keeps the front-ends
symmetric — neither the API nor the CLI is privileged. Every long-lived
singleton (the wake/sleep loop, session store, raw interaction log, the
active model-manager-driven Cortex / Embedder, and the model swap hooks)
is exposed from here.
"""

from .container import *  # noqa: F401,F403
from . import container as container  # re-export module for legacy access
