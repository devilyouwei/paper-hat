# `core/` — paper algorithms

Pure Python. No FastAPI, no torch, no I/O. Every heavy dependency is a Protocol
or ABC defined here and implemented elsewhere (`hat.models`, `hat.memory`, …).

| File | Paper section |
| --- | --- |
| `schemas.py` | §3.1, §3.4, §3.7 data types |
| `protocols.py` | All pluggable seams |
| `cortex/` | §3.3 |
| `hippocampus/` | §3.4 (abstraction, selection, replay) + scoring channels |
| `neocortex/` | §3.6 — write-token contract enforced here |
| `oracle/` | §3.5 |
| `sws/` | §3.7 |
| `loop.py` | §3.8 — wake/sleep orchestrator |
