from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from hat.abstract.schemas import Interaction
from hat.abstract.sessions import RawInteractionLog


class JsonlRawLog(RawInteractionLog):
    """Single-file JSONL backend, kept for tools and tests that don't need
    per-session organisation. Production code should use
    :class:`hat.memory.raw.sessions.JsonlSessionStore` instead."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, interaction: Interaction) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(interaction.model_dump_json() + "\n")

    def __iter__(self) -> Iterator[Interaction]:
        if not self.path.exists():
            return iter(())

        def gen() -> Iterator[Interaction]:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield Interaction.model_validate_json(line)

        return gen()


class SessionRawLog(RawInteractionLog):
    """Adapter so a :class:`JsonlSessionStore` can stand in wherever the loop
    expects a :class:`RawInteractionLog`. ``append`` writes to the supplied
    session id (or a synthetic ``default`` session if none is set).
    """

    def __init__(self, store, session_id: str | None = None) -> None:
        # Imported lazily to avoid a circular import at package load time.
        from hat.core.sessions.store import JsonlSessionStore  # noqa: F401

        self.store = store
        self.session_id = session_id

    def with_session(self, session_id: str | None) -> "SessionRawLog":
        return SessionRawLog(self.store, session_id)

    def append(self, interaction: Interaction) -> None:
        if self.session_id is None:
            self.store.append_anonymous(interaction)
        else:
            self.store.append(self.session_id, interaction)

    def __iter__(self) -> Iterator[Interaction]:
        sid = self.session_id
        if sid is None:
            # iterate every session, in updated_at order
            for s in self.store.list():
                yield from self.store.messages(s.id)
            return
        yield from self.store.messages(sid)
