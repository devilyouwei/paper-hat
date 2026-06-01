"""Session and raw-interaction-log interfaces.

Raw chat history storage. Strictly separated from the Neocortex: only
the Hippocampus Agent reads here to produce traces. Concrete
implementations live in :mod:`hat.core.sessions`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .schemas import Interaction, Session


class SessionStoreError(RuntimeError):
    pass


class RawInteractionLog(ABC):
    """Append-only log of raw user interactions.

    No training pipeline touches raw logs directly.
    """

    @abstractmethod
    def append(self, interaction: Interaction) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Interaction]: ...


class SessionStore(ABC):
    """Per-session raw chat history.

    A *session* is a topic-scoped conversation, in the spirit of
    ChatGPT-style chat tools. This is the **only** writer of raw chat
    data.
    """

    @abstractmethod
    def list(self) -> list[Session]: ...

    @abstractmethod
    def latest(self) -> Session | None: ...

    @abstractmethod
    def get(self, session_id: str) -> Session: ...

    @abstractmethod
    def create(
        self, title: str | None = None, model: str | None = None
    ) -> Session: ...

    @abstractmethod
    def rename(self, session_id: str, title: str) -> Session: ...

    @abstractmethod
    def delete(self, session_id: str) -> bool: ...

    @abstractmethod
    def append(self, session_id: str, interaction: Interaction) -> Session: ...

    @abstractmethod
    def messages(self, session_id: str) -> list[Interaction]: ...

    @abstractmethod
    def update_last_hat(self, session_id: str, hat: dict | None) -> bool: ...

    @abstractmethod
    def append_anonymous(self, interaction: Interaction) -> Session: ...


__all__ = [
    "RawInteractionLog",
    "SessionStore",
    "SessionStoreError",
]
