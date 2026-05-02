.PHONY: install serve ui dashboard sleep test lint typecheck format clean

install:
	uv sync

serve:
	uv run uvicorn hat.api.main:app --reload --host 127.0.0.1 --port 8000

ui:
	uv run python -m hat.ui.gradio_app

dashboard:
	uv run streamlit run src/hat/ui/dashboard.py

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
