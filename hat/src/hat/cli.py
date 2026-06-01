from __future__ import annotations

import typer
from rich import print as rprint

from .utils.logging import setup as setup_logging

app = typer.Typer(help="HAT command-line interface", no_args_is_help=True)


@app.callback()
def _root() -> None:
    setup_logging()


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server (which also serves the web UI at ``/``)."""
    import uvicorn

    uvicorn.run("hat.api.main:app", host=host, port=port, reload=reload)


@app.command()
def sleep(dry_run: bool = True, k: int = 32) -> None:
    """Trigger a slow-wave-sleep cycle."""
    from hat.core.runtime.container import get_loop

    loop = get_loop()
    if dry_run:
        rprint(
            f"[bold]SWS dry-run[/bold] traces={len(loop.neocortex)} "
            f"k={k} trainer={type(loop.trainer).__name__}"
        )
        return
    rprint(loop.sleep_step(k=k))


@app.command()
def train() -> None:
    """Train the Cortex from the curated Neocortex (stub)."""
    rprint("[yellow]train: not implemented in skeleton[/yellow]")


@app.command(name="eval")
def eval_(benchmark: str = "nq") -> None:
    """Run an evaluation benchmark (stub)."""
    rprint(f"[yellow]eval {benchmark}: not implemented in skeleton[/yellow]")


@app.command()
def ingest(path: str) -> None:
    """Replay a JSONL file of interactions through the wake loop (stub)."""
    rprint(f"[yellow]ingest {path}: not implemented in skeleton[/yellow]")


@app.command(name="reindex-memory")
def reindex_memory(
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Embed backend (mlx_embed | hf_embed). Defaults to the active managed embedder.",
    ),
    id: str | None = typer.Option(  # noqa: A002 - Typer expects 'id'
        None,
        "--id",
        help="Embed model id. Required together with --backend.",
    ),
) -> None:
    """Rebuild the embedding index from the curated memory store.

    Walks every row in ``train.jsonl``, embeds the canonical user query
    with the chosen embedder, and rewrites the matching NPZ from
    scratch. With no flags, the currently-active managed embedder is
    used; pass ``--backend`` and ``--id`` to target an installed
    embedder that is not currently active.
    """
    from hat.core.runtime.container import _get_vector_index_for, get_loop  # noqa: PLC0415
    from .config.settings import get_settings  # noqa: PLC0415
    from hat.core.lifecycle.embedding_manager import get_embedding_manager  # noqa: PLC0415

    settings = get_settings()
    if not settings.dedup_enabled:
        rprint("[yellow]dedup disabled — nothing to reindex[/yellow]")
        return

    if (backend is None) ^ (id is None):
        rprint("[red]--backend and --id must be passed together[/red]")
        raise typer.Exit(code=2)

    loop = get_loop()
    store = loop.neocortex
    entries_fn = getattr(store, "entries", None)
    if entries_fn is None:
        rprint("[red]active neocortex backend has no entries() — cannot reindex[/red]")
        raise typer.Exit(code=1)

    rows = entries_fn()

    mgr = get_embedding_manager()
    if backend and id:
        embed_backend, embed_id = backend, id
    else:
        active = mgr.active()
        if active is None:
            rprint(
                "[red]no embedding model active; pass --backend and --id "
                "or activate one via /api/embedding-models/active[/red]"
            )
            raise typer.Exit(code=2)
        embed_backend, embed_id = active["backend"], active["id"]

    embedder = mgr.load(embed_backend, embed_id)
    index = _get_vector_index_for(embed_backend, embed_id)
    tag = f"{embed_backend}/{embed_id}"

    import numpy as np  # noqa: PLC0415

    index._ids.clear()  # type: ignore[attr-defined]
    index._vecs = np.zeros((0, 0), dtype=np.float32)  # type: ignore[attr-defined]

    n_indexed = 0
    n_skipped = 0
    for row in rows:
        trace_id = row.get("trace_id")
        msgs = row.get("messages") or []
        query = next(
            (m.get("content") for m in msgs if m.get("role") == "user"), None,
        )
        if not trace_id or not query:
            n_skipped += 1
            continue
        vec = embedder.embed([query])[0]
        index.append(trace_id, vec)
        n_indexed += 1

    rprint(
        f"[bold green]reindex done[/bold green] "
        f"indexed={n_indexed} skipped={n_skipped} "
        f"path={index.path} dim={index.dim} tag={tag}"
    )


if __name__ == "__main__":
    app()
