"""Paper metadata database: schema, validation, and index generation.

Each paper is a YAML file under ``workflow/papers/<citekey>.yaml``. This module
loads them all, validates against the schema below, and emits an aggregate
``workflow/index.json`` consumed by the search, download, and graph tools.

Run directly to validate + (re)build the index::

    python workflow/script/paper_db.py            # validate + write index.json
    python workflow/script/paper_db.py --check     # validate only (CI-friendly)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common import INDEX_PATH, LEVELS, PAPERS_DIR, ensure_dirs

# --- Schema -----------------------------------------------------------------
# Required fields every paper YAML must define.
REQUIRED_FIELDS = ("citekey", "title", "level", "category")
# Known fields (anything else triggers a warning, not an error).
KNOWN_FIELDS = {
    "citekey",        # str  — BibTeX key / unique id, matches filename
    "title",          # str
    "authors",        # list[str]
    "year",           # int
    "date",           # str  YYYY-MM
    "venue",          # str
    "institutions",   # list[str]
    "arxiv_id",       # str  e.g. 2404.14387
    "doi",            # str
    "url",            # str
    "pdf",            # str  relative path under doc/ (default <citekey>.pdf)
    "category",       # list[str]  e.g. [self-evolution, memory]
    "level",          # int  1..4 (relevance grade)
    "keywords",       # list[str]
    "summary",        # str  one-paragraph description
    "bibtex",         # str  full @article{...} block
    "cite_command",   # str  \cite{citekey}
    "relations",      # dict {cites: [...], cited_by: [...]}
    "usage_note",     # str  how we use it (intro / compare / baseline)
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_paper(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: top-level YAML must be a mapping")
    data.setdefault("citekey", path.stem)
    data.setdefault("pdf", f"{data['citekey']}.pdf")
    return data


def validate_paper(path: Path, data: dict[str, Any]) -> ValidationResult:
    res = ValidationResult()
    for field_name in REQUIRED_FIELDS:
        if not data.get(field_name):
            res.errors.append(f"{path.name}: missing required field '{field_name}'")

    if data.get("citekey") and data["citekey"] != path.stem:
        res.errors.append(
            f"{path.name}: citekey '{data['citekey']}' does not match filename '{path.stem}'"
        )

    level = data.get("level")
    if level is not None and level not in LEVELS:
        res.errors.append(f"{path.name}: level must be one of {sorted(LEVELS)} (got {level!r})")

    category = data.get("category")
    if category is not None and not isinstance(category, list):
        res.errors.append(f"{path.name}: 'category' must be a list")

    relations = data.get("relations")
    if relations is not None and not isinstance(relations, dict):
        res.errors.append(f"{path.name}: 'relations' must be a mapping")

    for key in data:
        if key not in KNOWN_FIELDS:
            res.warnings.append(f"{path.name}: unknown field '{key}'")

    return res


def load_all(papers_dir: Path = PAPERS_DIR) -> tuple[list[dict[str, Any]], ValidationResult]:
    combined = ValidationResult()
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(papers_dir.glob("*.yaml")):
        try:
            data = load_paper(path)
        except Exception as exc:  # noqa: BLE001 - surface parse errors as validation errors
            combined.errors.append(str(exc))
            continue
        res = validate_paper(path, data)
        combined.errors.extend(res.errors)
        combined.warnings.extend(res.warnings)
        key = data.get("citekey", path.stem)
        if key in seen:
            combined.errors.append(f"{path.name}: duplicate citekey '{key}'")
        seen.add(key)
        papers.append(data)
    return papers, combined


def build_index(papers: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for p in papers:
        by_level[str(p.get("level"))] = by_level.get(str(p.get("level")), 0) + 1
        for cat in p.get("category", []) or []:
            by_category[cat] = by_category.get(cat, 0) + 1
    return {
        "root": "hat2026learning",
        "count": len(papers),
        "levels": LEVELS,
        "stats": {"by_level": by_level, "by_category": dict(sorted(by_category.items()))},
        "papers": sorted(papers, key=lambda p: (-int(p.get("level") or 0), p.get("citekey", ""))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paper YAML and build index.json")
    parser.add_argument("--check", action="store_true", help="validate only, do not write index")
    args = parser.parse_args()

    ensure_dirs()
    papers, res = load_all()

    for w in res.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    for e in res.errors:
        print(f"  ERROR: {e}", file=sys.stderr)

    if not res.ok:
        print(f"\nValidation failed: {len(res.errors)} error(s).", file=sys.stderr)
        return 1

    print(f"Validated {len(papers)} paper(s). No errors.")
    if args.check:
        return 0

    index = build_index(papers)
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {INDEX_PATH.relative_to(INDEX_PATH.parent.parent)}  ({index['count']} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
