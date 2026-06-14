"""Manual paper metadata editing helpers.

BibTeX and cite commands are intentionally human-curated. This CLI keeps that
workflow light: open a paper YAML, list missing citation fields, and validate a
pasted BibTeX entry against the YAML citekey.

Examples::

    python workflow/script/edit.py agenticmemory2026
    python workflow/script/edit.py --missing-bibtex
    python workflow/script/edit.py agenticmemory2026 --print-cite
    python workflow/script/edit.py agenticmemory2026 --validate-bibtex
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from common import PAPERS_DIR, ensure_dirs
from paper_db import load_all, validate_paper

BIBTEX_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")


def paper_path(citekey: str) -> Path:
    return PAPERS_DIR / f"{citekey}.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def open_in_editor(path: Path) -> int:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        print(path)
        print("Set EDITOR to open this file directly, e.g. EDITOR=code")
        return 0
    return subprocess.call([*shlex.split(editor), str(path)])


def missing_bibtex() -> int:
    papers, res = load_all()
    for warning in res.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    if not res.ok:
        for error in res.errors:
            print(f"  ERROR: {error}", file=sys.stderr)
        return 1

    missing = [
        p for p in papers
        if not p.get("bibtex") or not p.get("cite_command")
    ]
    if not missing:
        print("All papers have bibtex and cite_command.")
        return 0

    for p in missing:
        fields = []
        if not p.get("bibtex"):
            fields.append("bibtex")
        if not p.get("cite_command"):
            fields.append("cite_command")
        print(f"{p['citekey']}: missing {', '.join(fields)}")
    print(f"\nMissing citation fields: {len(missing)}/{len(papers)} paper(s).")
    return 0


def validate_bibtex(citekey: str) -> int:
    path = paper_path(citekey)
    if not path.exists():
        print(f"Paper not found: {path}", file=sys.stderr)
        return 1
    data = load_yaml(path)
    res = validate_paper(path, data)
    bibtex = data.get("bibtex")
    if not bibtex:
        print(f"{citekey}: bibtex is empty")
        return 1
    match = BIBTEX_KEY_RE.search(str(bibtex))
    if not match:
        print(f"{citekey}: bibtex does not look like a BibTeX entry", file=sys.stderr)
        return 1
    entry_key = match.group(1)
    if entry_key != citekey:
        print(f"{citekey}: BibTeX key is '{entry_key}', expected '{citekey}'", file=sys.stderr)
        return 1
    if data.get("cite_command") and citekey not in str(data["cite_command"]):
        print(f"{citekey}: cite_command does not contain '{citekey}'", file=sys.stderr)
        return 1
    for warning in res.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    print(f"{citekey}: BibTeX key matches citekey")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Open and validate manually curated paper YAML")
    parser.add_argument("citekey", nargs="?", help="paper citekey, e.g. agenticmemory2026")
    parser.add_argument("--missing-bibtex", action="store_true", help="list papers missing bibtex/cite_command")
    parser.add_argument("--print-cite", action="store_true", help="print suggested cite command without writing")
    parser.add_argument("--validate-bibtex", action="store_true", help="validate BibTeX key against citekey")
    args = parser.parse_args()

    ensure_dirs()

    if args.missing_bibtex:
        return missing_bibtex()
    if not args.citekey:
        parser.error("provide a citekey, or use --missing-bibtex")
    if args.print_cite:
        print(f"\\cite{{{args.citekey}}}")
        return 0
    if args.validate_bibtex:
        return validate_bibtex(args.citekey)

    path = paper_path(args.citekey)
    if not path.exists():
        print(f"Paper not found: {path}", file=sys.stderr)
        return 1
    return open_in_editor(path)


if __name__ == "__main__":
    raise SystemExit(main())
