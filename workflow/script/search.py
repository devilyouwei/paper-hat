"""Search for papers (arXiv + Semantic Scholar) and scaffold a new YAML entry.

Examples::

    # Free-text search across both sources
    python workflow/script/search.py "hippocampal memory consolidation LLM"

    # Look up a specific arXiv id and create workflow/papers/<citekey>.yaml
    python workflow/script/search.py --arxiv 2404.14387 --add --citekey selfevolutionsurvey2024

New entries default to level 1 / category [uncategorized] for human review.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import requests
import yaml

from common import PAPERS_DIR, ensure_dirs, load_env, slugify

ARXIV_API = "http://export.arxiv.org/api/query"
SS_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_PAPER = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
ATOM = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "hat-paper-agent/0.1 (research)"


def arxiv_search(query: str, limit: int) -> list[dict]:
    resp = requests.get(
        ARXIV_API,
        params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        aid = (entry.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
        out.append({
            "arxiv_id": aid.split("v")[0],
            "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
            "authors": [a.findtext(f"{ATOM}name") for a in entry.findall(f"{ATOM}author")],
            "date": (entry.findtext(f"{ATOM}published") or "")[:7],
            "summary": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
            "url": entry.findtext(f"{ATOM}id"),
        })
    return out


def arxiv_lookup(arxiv_id: str) -> dict | None:
    results = arxiv_search(f"id:{arxiv_id}", 1)
    if results:
        return results[0]
    # Fallback: query by id_list
    resp = requests.get(
        ARXIV_API,
        params={"id_list": arxiv_id},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        return None
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
        "authors": [a.findtext(f"{ATOM}name") for a in entry.findall(f"{ATOM}author")],
        "date": (entry.findtext(f"{ATOM}published") or "")[:7],
        "summary": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
        "url": entry.findtext(f"{ATOM}id"),
    }


def ss_search(query: str, limit: int, api_key: str | None) -> list[dict]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    resp = requests.get(
        SS_SEARCH,
        params={"query": query, "limit": limit, "fields": "title,year,authors,abstract,externalIds,url"},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    out = []
    for item in (resp.json() or {}).get("data", []):
        ext = item.get("externalIds") or {}
        out.append({
            "arxiv_id": ext.get("ArXiv"),
            "title": item.get("title"),
            "authors": [a.get("name") for a in item.get("authors", [])],
            "date": str(item.get("year") or ""),
            "summary": item.get("abstract") or "",
            "url": item.get("url"),
        })
    return out


def make_yaml(meta: dict, citekey: str) -> dict:
    arxiv_id = meta.get("arxiv_id")
    year = int((meta.get("date") or "0")[:4] or 0) or None
    return {
        "citekey": citekey,
        "title": meta.get("title"),
        "authors": meta.get("authors") or [],
        "year": year,
        "date": meta.get("date") or None,
        "venue": None,
        "institutions": [],
        "arxiv_id": arxiv_id,
        "doi": None,
        "url": meta.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
        "pdf": f"{citekey}.pdf" if arxiv_id else None,
        "category": ["uncategorized"],
        "level": 1,
        "keywords": [],
        "summary": (meta.get("summary") or "")[:600],
        "bibtex": None,
        "cite_command": f"\\cite{{{citekey}}}",
        "relations": {"cites": [], "cited_by": []},
        "usage_note": "TODO: classify and grade after review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search papers and scaffold YAML entries")
    parser.add_argument("query", nargs="?", help="free-text search query")
    parser.add_argument("--arxiv", help="look up a specific arXiv id")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--add", action="store_true", help="write a YAML entry for the result")
    parser.add_argument("--citekey", help="citekey for --add (default: slug of title)")
    args = parser.parse_args()

    ensure_dirs()
    load_env()
    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

    if args.arxiv:
        meta = arxiv_lookup(args.arxiv)
        if not meta:
            print(f"arXiv {args.arxiv} not found.", file=sys.stderr)
            return 1
        print(f"{meta['title']}\n  {meta['date']}  arXiv:{meta['arxiv_id']}")
        if args.add:
            citekey = args.citekey or slugify(meta["title"])[:40].replace("-", "")
            path = PAPERS_DIR / f"{citekey}.yaml"
            if path.exists():
                print(f"  {path.name} already exists; not overwriting.", file=sys.stderr)
                return 1
            path.write_text(
                yaml.safe_dump(make_yaml(meta, citekey), sort_keys=False, allow_unicode=True, width=100),
                encoding="utf-8",
            )
            print(f"  wrote {path.relative_to(PAPERS_DIR.parent)} (review level/category)")
        return 0

    if not args.query:
        parser.error("provide a query or --arxiv")

    print("== arXiv ==")
    for r in arxiv_search(args.query, args.limit):
        print(f"  [{r['date']}] {r['title']}  (arXiv:{r['arxiv_id']})")
    print("== Semantic Scholar ==")
    for r in ss_search(args.query, args.limit, ss_key):
        aid = f"  (arXiv:{r['arxiv_id']})" if r.get("arxiv_id") else ""
        print(f"  [{r['date']}] {r['title']}{aid}")
    print("\nTo add one:  python search.py --arxiv <id> --add --citekey <key>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
