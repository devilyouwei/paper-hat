# `memory/` — storage

Strictly two subpackages, with no shared state:

* `raw/` — append-only chat history. Append on every wake step.
* `curated/` — Neocortex. Written **only** through `NeocortexStore.write(trace, decision)`,
  and the store rejects writes without a valid `WriteDecision`. See ADR-002.

The training pipeline must never import from `raw/`.

## Disk layout

```
runs/
├── raw/
│   ├── index.json                 # list[Session]
│   └── sessions/
│       └── <session_id>.jsonl     # one Interaction per line, per chat
└── neocortex/
    ├── train.jsonl                # SFT-format training rows (HF/OpenAI)
    └── traces.jsonl               # full MemoryTrace records (sidecar)
```

### `raw/`
`JsonlSessionStore` ([raw/sessions.py](raw/sessions.py)) owns the chat-history
tree. Each session is a topic-scoped conversation à la ChatGPT (id, title,
created/updated timestamps, message count). The session title is auto-summarised
by the Cortex on the first turn. Public methods: `list / latest / get / create
/ rename / delete / append / messages`. The legacy `JsonlRawLog` (single-file
JSONL) is retained for tools and tests; production code uses
`SessionRawLog(store, session_id=...)` so the wake-step writer stays the same.

### `curated/`
`JsonlNeocortex` ([curated/jsonl_store.py](curated/jsonl_store.py)) writes one
SFT row per accepted trace:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "trace_id": "...", "interaction_id": "...",
 "score": 0.83, "signals": {"uncertainty": 0.6, ...},
 "metadata": {...}}
```

This is the format consumed by HuggingFace `datasets.load_dataset("json", ...)`
and OpenAI fine-tuning pipelines, so the Neocortex file can be fed directly
into the SWS trainer without any post-processing. Full `MemoryTrace` records
are mirrored into a sidecar `traces.jsonl` for inspection.
