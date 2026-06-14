"""Build a citation knowledge graph rooted at our HAT paper.

An OpenAI-compatible LLM reads each tracked paper (its downloaded PDF, or its
abstract/summary as fallback) and infers which *other tracked papers* it cites.
The root node is HAT (``hat2026learning``); every tracked paper is connected to
the root, and inferred inter-paper edges are added on top.

Outputs (under workflow/graph/):
  * citations.json  — nodes + edges (machine-readable)
  * graph.mmd       — Mermaid diagram (for docs/readme)
  * graph.html      — interactive pyvis graph

Usage::

    cp script/config.example.env script/config.env   # fill LLM_* keys
    python workflow/script/graph.py                    # full build (calls LLM)
    python workflow/script/graph.py --no-llm            # root edges only, no LLM
    python workflow/script/graph.py --only star2022 reflexion2023
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

from common import DOC_DIR, GRAPH_DIR, ensure_dirs, load_env
from paper_db import load_all

ROOT = "hat2026learning"
ROOT_TITLE = "HAT: Hippocampal Memory Consolidation for Continual Model Adaptation"

LEVEL_COLOR = {1: "#cbd5e1", 2: "#93c5fd", 3: "#fbbf24", 4: "#f87171"}


# --- PDF / text extraction --------------------------------------------------
def extract_text(pdf_path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    if not pdf_path.exists():
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception as exc:  # noqa: BLE001
        print(f"    PDF parse error ({pdf_path.name}): {exc}", file=sys.stderr)
        return ""


# --- LLM call ---------------------------------------------------------------
def llm_extract_citations(source_title: str, source_text: str, candidates: dict[str, str]) -> list[str]:
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set (see script/config.example.env)")

    catalog = "\n".join(f"- {ck}: {title}" for ck, title in candidates.items())
    system = (
        "You are a precise research assistant building a citation graph. "
        "Given a source paper's text and a catalog of candidate papers, identify "
        "which candidates the source paper CITES. Use only the catalog; do not invent. "
        "Respond with strict JSON: {\"cites\": [\"citekey\", ...]}."
    )
    user = (
        f"SOURCE PAPER: {source_title}\n\n"
        f"CANDIDATE PAPERS (citekey: title):\n{catalog}\n\n"
        f"SOURCE TEXT (may be truncated):\n{source_text or '(no full text available; rely on title only)'}\n\n"
        "Return the citekeys of candidates that the source paper cites."
    )
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        cites = json.loads(content).get("cites", [])
    except json.JSONDecodeError:
        return []
    return [c for c in cites if c in candidates]


# --- Graph rendering --------------------------------------------------------
def render_mermaid(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["flowchart LR", f'  {ROOT}["HAT (our paper)"]']
    for n in nodes:
        if n["id"] == ROOT:
            continue
        label = n["title"].replace('"', "'")[:48]
        lines.append(f'  {n["id"]}["L{n["level"]} {label}"]')
    for e in edges:
        arrow = "==>" if e["source"] == ROOT else "-->"
        lines.append(f'  {e["source"]} {arrow} {e["target"]}')
    return "\n".join(lines) + "\n"


def render_html(nodes: list[dict], edges: list[dict], out: Path) -> bool:
    try:
        from pyvis.network import Network
    except ImportError:
        print("    pyvis not installed; skipping graph.html", file=sys.stderr)
        return False
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff")
    net.barnes_hut()
    for n in nodes:
        is_root = n["id"] == ROOT
        net.add_node(
            n["id"],
            label=n["title"][:40],
            title=f"{n['title']} (L{n['level']}, {', '.join(n.get('category', []))})",
            color="#111827" if is_root else LEVEL_COLOR.get(n["level"], "#cbd5e1"),
            shape="star" if is_root else "dot",
            size=40 if is_root else 12 + 5 * n["level"],
        )
    for e in edges:
        net.add_edge(e["source"], e["target"], color="#9ca3af")
    net.write_html(str(out), notebook=False, open_browser=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build citation knowledge graph rooted at HAT")
    parser.add_argument("--no-llm", action="store_true", help="root edges only; skip LLM extraction")
    parser.add_argument("--only", nargs="*", default=None, help="limit LLM extraction to these citekeys")
    args = parser.parse_args()

    ensure_dirs()
    load_env()
    max_chars = int(os.environ.get("LLM_MAX_CHARS", "24000"))

    papers, res = load_all()
    if not res.ok:
        print("Fix paper YAML validation errors first (run paper_db.py).", file=sys.stderr)
        return 1

    candidates = {p["citekey"]: p["title"] for p in papers}
    nodes = [{"id": ROOT, "title": ROOT_TITLE, "level": 0, "category": ["ours"]}]
    for p in papers:
        nodes.append({
            "id": p["citekey"],
            "title": p["title"],
            "level": int(p.get("level") or 1),
            "category": p.get("category", []),
        })

    # Root edges: our paper relates to every tracked paper.
    edges = [{"source": ROOT, "target": p["citekey"], "kind": "root"} for p in papers]

    # Inferred inter-paper edges via LLM (or pre-existing relations in YAML).
    if args.no_llm:
        for p in papers:
            for tgt in (p.get("relations") or {}).get("cites", []) or []:
                if tgt in candidates:
                    edges.append({"source": p["citekey"], "target": tgt, "kind": "cites"})
    else:
        for p in papers:
            ck = p["citekey"]
            if args.only and ck not in args.only:
                continue
            others = {k: v for k, v in candidates.items() if k != ck}
            text = extract_text(DOC_DIR / (p.get("pdf") or ""), max_chars) if p.get("pdf") else ""
            if not text:
                text = p.get("summary", "")
            print(f"  extracting citations for {ck} ...")
            try:
                cites = llm_extract_citations(p["title"], text, others)
            except Exception as exc:  # noqa: BLE001
                print(f"    LLM error for {ck}: {exc}", file=sys.stderr)
                continue
            for tgt in cites:
                edges.append({"source": ck, "target": tgt, "kind": "cites"})

    # De-duplicate edges.
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["kind"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    graph = {"root": ROOT, "nodes": nodes, "edges": unique_edges}
    (GRAPH_DIR / "citations.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (GRAPH_DIR / "graph.mmd").write_text(render_mermaid(nodes, unique_edges), encoding="utf-8")
    html_ok = render_html(nodes, unique_edges, GRAPH_DIR / "graph.html")

    print(f"\nGraph: {len(nodes)} nodes, {len(unique_edges)} edges.")
    print(f"  wrote graph/citations.json, graph/graph.mmd" + (", graph/graph.html" if html_ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
