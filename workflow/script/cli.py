"""Unified entry point for the lightweight paper agent.

    python workflow/script/cli.py index           # validate YAML + build index.json
    python workflow/script/cli.py download          # fetch all PDFs into doc/
    python workflow/script/cli.py search "<query>"  # search arXiv + Semantic Scholar
    python workflow/script/cli.py enrich            # backfill metadata via Semantic Scholar
    python workflow/script/cli.py graph             # build citation knowledge graph
    python workflow/script/cli.py stats             # quick overview by level/category

Each subcommand forwards remaining args to the underlying script.
"""

from __future__ import annotations

import json
import runpy
import sys

from common import INDEX_PATH


def _run(module: str, argv: list[str]) -> int:
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def stats() -> int:
    if not INDEX_PATH.exists():
        print("index.json missing; run: cli.py index", file=sys.stderr)
        return 1
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    print(f"Papers: {data['count']}  (root: {data['root']})")
    print("\nBy level:")
    for lvl, desc in data["levels"].items():
        n = data["stats"]["by_level"].get(str(lvl), 0)
        print(f"  L{lvl} ({n:>2}): {desc}")
    print("\nBy category:")
    for cat, n in data["stats"]["by_category"].items():
        print(f"  {n:>2}  {cat}")
    return 0


COMMANDS = {
    "index": lambda a: _run("paper_db", a),
    "download": lambda a: _run("download", a),
    "search": lambda a: _run("search", a),
    "enrich": lambda a: _run("enrich", a),
    "graph": lambda a: _run("graph", a),
    "stats": lambda a: stats(),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\n", file=sys.stderr)
        print(__doc__)
        return 1
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
