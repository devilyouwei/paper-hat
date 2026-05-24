# `memory/curated/` — Neocortex on-disk store

Concrete JSONL implementation of `core.neocortex.NeocortexStore`. The
file format is aligned with the OpenAI / HuggingFace SFT convention so it
can be fed directly into a fine-tuning pipeline.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `jsonl_store.py` | `JsonlNeocortex` — appends one SFT row per accepted trace to `runs/neocortex/train.jsonl`, mirrors the full `MemoryTrace` into a sidecar `traces.jsonl`. Thread-safe (single `Lock`). |

Row shape:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "trace_id": "...", "interaction_id": "...",
 "score": 0.83, "signals": {"uncertainty": 0.6, ...},
 "metadata": {"timestamp": "...", "source": "user", "extras": {...}}}
```

Writes go through the inherited `NeocortexStore.write(trace, decision)`,
which rejects calls missing a valid accepted `WriteDecision` (ADR-002).
