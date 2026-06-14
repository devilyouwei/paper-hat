"""Download PDF sources for every paper in workflow/papers/ into workflow/doc/.

PDFs are gitignored; this re-hydrates them on a fresh clone::

    python workflow/script/download.py            # download all missing
    python workflow/script/download.py --force     # re-download everything
    python workflow/script/download.py --only star2022 reflexion2023

Sources, in order: explicit arxiv_id -> arXiv PDF; otherwise Semantic Scholar
open-access PDF (if available); papers without a usable source are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from common import DOC_DIR, ensure_dirs, load_env
from paper_db import load_all

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
SS_PAPER = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
USER_AGENT = "hat-paper-agent/0.1 (research; contact: maintainer)"


def _ss_open_pdf(arxiv_id: str, api_key: str | None) -> str | None:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(
            SS_PAPER.format(arxiv_id=arxiv_id),
            params={"fields": "openAccessPdf"},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            pdf = (resp.json() or {}).get("openAccessPdf") or {}
            return pdf.get("url")
    except requests.RequestException:
        return None
    return None


def _download(url: str, dest: Path) -> bool:
    headers = {"User-Agent": USER_AGENT}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} for {url}", file=sys.stderr)
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 15):
                    fh.write(chunk)
            tmp.replace(dest)
            return True
    except requests.RequestException as exc:
        print(f"    error: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download paper PDFs into workflow/doc/")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--only", nargs="*", default=None, help="limit to these citekeys")
    args = parser.parse_args()

    ensure_dirs()
    load_env()
    import os

    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

    papers, res = load_all()
    if not res.ok:
        print("Fix paper YAML validation errors first (run paper_db.py).", file=sys.stderr)
        return 1

    ok = skipped = failed = 0
    for p in papers:
        citekey = p["citekey"]
        if args.only and citekey not in args.only:
            continue
        pdf_name = p.get("pdf")
        if not pdf_name:
            continue  # resources without a PDF
        dest = DOC_DIR / pdf_name
        if dest.exists() and not args.force:
            skipped += 1
            continue

        arxiv_id = p.get("arxiv_id")
        url = None
        if arxiv_id:
            url = ARXIV_PDF.format(arxiv_id=arxiv_id)
        if not url and arxiv_id:
            url = _ss_open_pdf(arxiv_id, ss_key)
        if not url:
            print(f"  {citekey}: no downloadable source, skipped", file=sys.stderr)
            skipped += 1
            continue

        print(f"  {citekey}: {url}")
        if _download(url, dest):
            ok += 1
        else:
            # Fallback to Semantic Scholar open-access PDF if arXiv failed.
            if arxiv_id and url.startswith("https://arxiv.org"):
                alt = _ss_open_pdf(arxiv_id, ss_key)
                if alt and _download(alt, dest):
                    ok += 1
                    time.sleep(1)
                    continue
            failed += 1
        time.sleep(1)  # be polite to hosts

    print(f"\nDownloaded {ok}, skipped {skipped}, failed {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
