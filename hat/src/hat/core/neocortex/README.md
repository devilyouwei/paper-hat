# `core/neocortex/` — long-term curated memory (paper §3.6)

The Neocortex is the only legal source of training data. The ABC enforces
the **write-token contract**: `write(trace, decision)` rejects any
`WriteDecision` that is missing, mismatched, or not accepted. This is the
type-level boundary between raw chat history and SFT corpus (ADR-002).

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `NeocortexStore`, `InMemoryNeocortex`, `NeocortexWriteError`. |
| `store.py` | `NeocortexStore` ABC with the write-token check + `InMemoryNeocortex` (heap-ordered, for tests and small experiments). The JSONL on-disk implementation lives in [`../../memory/curated/jsonl_store.py`](../../memory/curated/jsonl_store.py). |
