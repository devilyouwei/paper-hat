# ============================================================
# Research workflow paper agent — Makefile
#   run from the workflow/ directory:  make <target>
#
# Environment is managed by uv in this directory's own project
# (pyproject.toml + .venv), fully isolated from the HAT demo repo and conda.
# `uv run` auto-syncs the venv first, so most targets need no
# separate install step.
# ============================================================
SCRIPT := script
RUN := uv run --

.DEFAULT_GOAL := help

.PHONY: help sync install index download enrich edit missing-bibtex graph graph-nollm stats search clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Create/update the local uv environment (.venv)
	uv sync

install: sync ## Alias for `sync` (set up dependencies)

index: ## Validate paper YAML and (re)build index.json
	$(RUN) python $(SCRIPT)/paper_db.py

download: ## Download all paper PDFs into doc/ (gitignored)
	$(RUN) python $(SCRIPT)/download.py

enrich: ## Backfill metadata from Semantic Scholar
	$(RUN) python $(SCRIPT)/enrich.py

edit: ## Open a paper YAML for manual curation: make edit KEY=agenticmemory2026
	$(RUN) python $(SCRIPT)/edit.py "$(KEY)"

missing-bibtex: ## List papers missing manually curated BibTeX/cite_command
	$(RUN) python $(SCRIPT)/edit.py --missing-bibtex

graph: ## Build citation knowledge graph (uses LLM; needs config.env)
	$(RUN) python $(SCRIPT)/graph.py

graph-nollm: ## Build graph with root edges only (no LLM call)
	$(RUN) python $(SCRIPT)/graph.py --no-llm

stats: ## Print overview by relevance level and category
	$(RUN) python $(SCRIPT)/cli.py stats

search: ## Search arXiv + Semantic Scholar:  make search Q="memory llm"
	$(RUN) python $(SCRIPT)/search.py "$(Q)"

clean: ## Remove generated index/graph artifacts (keeps YAML + PDFs)
	rm -f index.json graph/citations.json graph/graph.mmd graph/graph.html
