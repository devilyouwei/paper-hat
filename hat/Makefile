.PHONY: install serve sleep test lint typecheck format clean ui-build ui-dev

UI_DIR := src/hat/ui_vue

install:
	uv sync

ui-build:
	cd $(UI_DIR) && corepack pnpm install --frozen-lockfile && corepack pnpm build

ui-dev:
	cd $(UI_DIR) && corepack pnpm dev

serve:
	uv run uvicorn hat.api.main:app --reload --host 0.0.0.0 --port 8000

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
