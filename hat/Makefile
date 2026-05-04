.PHONY: install serve sleep test lint typecheck format clean

install:
	uv sync

serve:
	uv run uvicorn hat.api.main:app --reload --host 127.0.0.1 --port 8000

sleep:
	uv run hat sleep --dry-run

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src/hat

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
