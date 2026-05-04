# ADR-001 — Directory layout

## Status
Accepted.

## Context
The implementation must mirror the paper closely so reviewers can map every algorithmic claim to code, while still being deployable as a real service (FastAPI + Gradio + a long-running SWS worker).

## Decision
Use a **src layout** with one importable package, ``hat``, split into:

* ``core/`` — paper algorithms only. No I/O, no torch, no FastAPI. Every pluggable seam is a ``Protocol`` or ABC.
* ``models/`` — model lifecycle and inference backends (HF, vLLM, Ollama). Each backend is gated behind an extras group.
* ``memory/`` — storage backends. Two strictly disjoint subpackages: ``raw/`` (chat history) and ``curated/`` (Neocortex). See ADR-002.
* ``api/`` — FastAPI surface, organised as ``routers → controllers → injected protocols``. Controllers are pure Python.
* ``services/`` — long-running orchestration (SWS scheduler, replay worker, job queue).
* ``ui/`` — vanilla HTML/CSS/JS web app served by FastAPI from ``ui/static/``. The UI calls the same REST surface as any external client.
* ``data/`` and ``eval/`` — benchmark adapters and metrics for paper §4.

## Consequences
* The wake side (api + ``core/cortex`` + ``core/hippocampus``) and the sleep side (``services/replay_worker`` + ``core/sws`` + ``models/training``) can be deployed as separate processes that communicate only through ``memory/`` and the job queue.
* Heavy ML deps stay optional; ``pip install hat`` gives you the protocols + reference Cortex + JSONL stores. ``hat[hf]``, ``hat[train]`` opt in.
* Every paper section has exactly one home in the tree (see the mapping table in the top-level README).
