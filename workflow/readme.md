# Workflow — Research Paper Agent

This directory is the paper repository's only Python workspace. It hosts a
lightweight **agent + human research workflow** for writing and maintaining the
HAT paper, starting with related-work management and gradually growing into a
general-purpose paper agent.

The HAT demo/reference implementation has moved to its own repository:
<https://github.com/devilyouwei/hat>. Keeping the demo code separate lets this
paper repository stay focused on writing, figures, experiments, references, and
automation around the paper itself.

Long term, the goal is to evolve this workflow into an independent CLI/library
that can help researchers manage papers, build citation graphs, revise drafts,
and run a lightweight ReAct-style agent loop during idle time: inspect the paper,
reason about possible improvements, propose edits, and hand changes back to the
human author for review.

The current implemented agent manages the related-work corpus. Each paper is
tracked as **metadata (YAML) + PDF source**, classified by **category** and
graded by **relevance level**, and woven into a **citation knowledge graph**
rooted at our HAT paper.

> PDFs under [`doc/`](doc/) are **gitignored**. On a fresh clone, re-hydrate them
> with one command: `make download`.

## Layout

| Path | Purpose |
| --- | --- |
| [`papers/`](papers/) | One editable `*.yaml` per paper (the source of truth). |
| [`doc/`](doc/) | Downloaded PDF sources (`<citekey>.pdf`), gitignored. |
| `index.json` | Auto-generated aggregate of all papers (for tooling/graph). |
| [`graph/`](graph/) | Generated `citations.json`, `graph.mmd`, `graph.html`. |
| [`script/`](script/) | Current command scripts: search, download, enrich, graph. |
| [`lib/`](lib/) | Vendored/static assets used by generated views. |
| `Makefile` | Task runner (`make help`). |

## Roadmap

The workflow should remain small and useful while leaving a clean path toward a
standalone research-agent package.

1. **Paper library agent**: search papers, download PDFs, enrich metadata, grade
	relevance, and build citation graphs.
2. **Draft-reading agent**: read `main.tex`, `sections/*.tex`, figures, and the
	bibliography; detect missing citations, weak claims, stale TODOs, and unclear
	positioning.
3. **ReAct-style writing loop**: plan -> inspect sources -> propose edits -> run
	checks -> summarize for human review. The agent should expose its reasoning
	trace as concise actions/observations rather than dumping private chain of
	thought.
4. **Idle-time assistant**: run on demand or on a schedule to suggest paper
	improvements, related-work updates, citation fixes, and experiment-reporting
	inconsistencies.
5. **Standalone CLI/library**: extract the reusable pieces into a package usable
	by other paper projects, while keeping project-specific metadata in each
	repository.

## Relevance levels (grading)

| Level | Meaning | How we use it |
| --- | --- | --- |
| **1** | Weak | Borrowed in introduction or method. |
| **2** | Related | Similar/partial; compared in Related Work. |
| **3** | High | Baseline: comparison experiments, shared evaluation, reported numbers. |
| **4** | Near-duplicate | Same task or nearly identical method — **publication risk**; ideally none. |

`category` is the topical tag(s) (e.g. `self-evolution`, `memory`, `reasoning`).

## Quick start

This directory is its **own uv project** (`pyproject.toml` + `.venv`), fully
isolated from the external HAT demo repository and any conda environment. `make` targets run through
`uv run`, which auto-syncs the environment first — no manual activation needed.

```bash
cd workflow
make sync                    # create the local .venv (uv); optional, auto-run by other targets
make download                # fetch all PDFs into doc/
make index                   # validate YAML + (re)build index.json
make stats                   # overview by level and category

cp script/config.example.env script/config.env   # add LLM + Semantic Scholar keys
make enrich                  # backfill authors/venue/year/bibtex
make graph                   # build the citation knowledge graph
open graph/graph.html        # interactive graph (root = HAT)
```

> No uv? Fall back to plain pip into any environment you choose:
> `pip install -r script/requirements.txt` then run the scripts with `python script/<name>.py`.

## Adding a paper

```bash
# Discover candidates
python script/search.py "long-term memory for LLM agents"

# Scaffold a YAML entry from an arXiv id, then edit level/category by hand
python script/search.py --arxiv 2310.08560 --add --citekey memgpt2023

make enrich      # fill metadata
make download    # fetch its PDF
make index       # revalidate + rebuild index
```

Each `papers/<citekey>.yaml` carries: `title, authors, year, date, venue,
institutions, arxiv_id, doi, url, pdf, category[], level, keywords[], summary,
bibtex, cite_command (\cite{citekey}), relations{cites,cited_by}, usage_note`.

## Knowledge graph

`make graph` runs an OpenAI-compatible LLM (configured in `script/config.env`)
that reads each paper's PDF — falling back to its abstract — and infers which
**other tracked papers** it cites. The root node is HAT; every paper links back
to the root, and inferred inter-paper citation edges are layered on top.

- `graph/citations.json` — machine-readable nodes + edges.
- `graph/graph.mmd` — Mermaid diagram (embeddable in Markdown).
- `graph/graph.html` — interactive pyvis graph.

Use `make graph-nollm` to rebuild root-only edges (and any hand-authored
`relations.cites`) without calling the LLM.

## Configuration

Copy `script/config.example.env` to `script/config.env` (gitignored) and set:

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` — any OpenAI-compatible endpoint.
- `SEMANTIC_SCHOLAR_API_KEY` — optional, raises the metadata rate limit.
