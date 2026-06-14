"""Backfill paper metadata from Semantic Scholar (authors, venue, year, DOI, BibTeX).

Only fills fields that are currently empty/placeholder; never overwrites curated
values such as ``level``, ``category``, ``summary``, or ``usage_note``.

    python workflow/script/enrich.py                  # enrich all
    python workflow/script/enrich.py --only star2022   # one paper
    python workflow/script/enrich.py --overwrite        # refresh even if present
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
import yaml

from common import PAPERS_DIR, load_env
from paper_db import load_all

SS_PAPER = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
FIELDS = "title,year,venue,authors.name,externalIds,citationStyles,publicationVenue"
USER_AGENT = "hat-paper-agent/0.1 (research)"


def fetch(arxiv_id: str, api_key: str | None) -> dict | None:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(
            SS_PAPER.format(arxiv_id=arxiv_id),
            params={"fields": FIELDS},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"    HTTP {resp.status_code} for arXiv:{arxiv_id}", file=sys.stderr)
    except requests.RequestException as exc:
        print(f"    error: {exc}", file=sys.stderr)
    return None


def is_empty(value) -> bool:
    return value in (None, "", [], {}) or value == ["uncategorized"]


def enrich_one(data: dict, meta: dict, overwrite: bool) -> bool:
    changed = False

    def set_if(key, value):
        nonlocal changed
        if value and (overwrite or is_empty(data.get(key))):
            if data.get(key) != value:
                data[key] = value
                changed = True

    set_if("title", meta.get("title"))
    set_if("year", meta.get("year"))
    set_if("venue", meta.get("venue") or (meta.get("publicationVenue") or {}).get("name"))
    authors = [a.get("name") for a in meta.get("authors", []) if a.get("name")]
    set_if("authors", authors)
    ext = meta.get("externalIds") or {}
    set_if("doi", ext.get("DOI"))

    bib = (meta.get("citationStyles") or {}).get("bibtex")
    if bib and (overwrite or is_empty(data.get("bibtex")) or "author={TBD}" in (data.get("bibtex") or "")):
        # Re-key the bibtex to our citekey so \cite matches.
        first_brace = bib.find("{")
        first_comma = bib.find(",")
        if first_brace != -1 and first_comma != -1:
            bib = bib[: first_brace + 1] + data["citekey"] + bib[first_comma:]
        if data.get("bibtex") != bib:
            data["bibtex"] = bib
            changed = True

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich paper metadata via Semantic Scholar")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_env()
    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

    papers, res = load_all()
    if not res.ok:
        print("Fix paper YAML validation errors first (run paper_db.py).", file=sys.stderr)
        return 1

    updated = 0
    for data in papers:
        citekey = data["citekey"]
        if args.only and citekey not in args.only:
            continue
        arxiv_id = data.get("arxiv_id")
        if not arxiv_id:
            continue
        meta = fetch(arxiv_id, ss_key)
        time.sleep(1)
        if not meta:
            continue
        if enrich_one(data, meta, args.overwrite):
            (PAPERS_DIR / f"{citekey}.yaml").write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
                encoding="utf-8",
            )
            print(f"  updated {citekey}")
            updated += 1

    print(f"\nEnriched {updated} paper(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
