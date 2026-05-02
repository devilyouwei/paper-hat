from __future__ import annotations

import typer
from rich import print as rprint

from .utils.logging import setup as setup_logging

app = typer.Typer(help="HAT command-line interface", no_args_is_help=True)


@app.callback()
def _root() -> None:
    setup_logging()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("hat.api.main:app", host=host, port=port, reload=reload)


@app.command()
def ui(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Launch the Gradio chat UI (requires the server to be running)."""
    from .ui.gradio_app import build

    build().launch(server_name=host, server_port=port)


@app.command()
def sleep(dry_run: bool = True, k: int = 32) -> None:
    """Trigger a slow-wave-sleep cycle."""
    from .api.deps import get_loop

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


if __name__ == "__main__":
    app()
