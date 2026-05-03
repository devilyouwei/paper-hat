<p align="center">
  <img src="logo.png" alt="HAT logo" width="320" />
</p>

# HAT — Hippocampus-Augmented Transformer

Reference implementation of the wake–sleep selective-memory-consolidation system from
*Learning What to Learn: Hippocampal Memory Consolidation for Continual Model Adaptation*.

This is a **skeleton**: protocols, ABCs, and a no-op end-to-end path. Real model loading,
training, and benchmark code land in subsequent iterations.

## Quickstart

```bash
uv sync
make serve         # FastAPI on :8000
make ui            # Gradio chat + uncertainty/novelty inspector
make sleep         # run a slow-wave-sleep cycle (dry-run by default)
make test
```

The default Cortex is a no-op echo so the loop runs without ML deps. To chat
with a real local model, see **Local model chat** below — pick the backend
that matches your machine.

## Local model chat — Apple Silicon (M1/M2/M3, recommended on Mac)

Use the **MLX** backend (Apple's official Metal-native runtime). On an 8 GB
M1, a 4-bit Qwen 1.5B runs comfortably (~1.5 GB resident, ~30–60 tok/s).

```bash
uv sync --extra mlx
```

Set in `.env`:

```env
HAT_CORTEX_BACKEND=mlx
```

Models are downloaded from the **Models** tab in the Gradio UI (or via
`POST /api/models/download`) into `model/mlx/<id>/`. The catalog ships with
8 GB-friendly defaults including `qwen2.5-1.5b-instruct-4bit` (≈1.0 GB,
default), `qwen3.5-2b-optiq-4bit` (≈1.5 GB), `qwen3.5-4b-optiq-4bit`
(≈3.0 GB, recommended quality/footprint pick).

Then in two terminals:

```bash
make serve
make ui      # → http://127.0.0.1:7860
```

## Local model chat — HuggingFace (CUDA / generic)

Use the **HF** backend if you have a CUDA box or want full PyTorch control.

1. Install the HF extra (downloads PyTorch + transformers):

   ```bash
   uv sync --extra hf
   ```

2. Configure the backend. Copy `.env.example` to `.env` and set:

   ```env
   HAT_CORTEX_BACKEND=hf
   HAT_HF_DEVICE=auto         # auto | cpu | cuda | mps
   HAT_HF_DTYPE=auto          # auto | float16 | bfloat16 | float32
   ```

3. Start the server and the chat UI, then download a model from the Models
   tab. Weights land under `model/hf/<id>/`.

   ```bash
   make serve
   make ui   # → http://127.0.0.1:7860
   ```

Per-call generation parameters (temperature, max tokens) are exposed in the
chat UI under "Generation settings" and forwarded to the OpenAI-compatible
endpoint, so you can tune them without touching env vars.

## OpenAI-compatible API

The server exposes an OpenAI-compatible endpoint, so you can also point any
OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="hat-local")
resp = client.chat.completions.create(
    model="hat-cortex",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

Each call still flows through the wake step, so the Hippocampus Agent will
score the turn and (selectively) consolidate it into the Neocortex.

## Architecture

```
                ┌──────────┐  query   ┌──────────────────┐
   user  ──────▶│  Cortex  │─────────▶│  Hippocampus     │
                └──────────┘  resp    │  abs / sel /     │
                                      │  replay          │
                                      └──────┬───────────┘
                                             │ WriteDecision
                                             ▼
                       ┌──────────────┐         ┌──────────────┐
                       │ Raw chat log │ (TTL)   │  Neocortex   │
                       │ (append-only)│         │  (curated)   │
                       └──────────────┘         └──────┬───────┘
                                                       │ replay batch
                                                       ▼
                                              ┌────────────────────┐
                                              │   SWS Trainer      │
                                              │ L_sup+λ_KD+λ_stab  │
                                              └─────────┬──────────┘
                                                        │ θ^{t+1}
                                                        ▼
                                                    (Cortex)
```

## Paper → module map

| Paper section                     | Module                                            |
| --------------------------------- | ------------------------------------------------- |
| §3.3 Cortex                       | `hat.core.cortex`                                 |
| §3.4 Hippocampus Agent            | `hat.core.hippocampus`                            |
| §3.4.1 Abstraction                | `hat.core.hippocampus.abstraction`                |
| §3.4.2 Selection (αU+βF+γN)       | `hat.core.hippocampus.selection` + `scoring/`     |
| §3.4.3 Replay construction        | `hat.core.hippocampus.replay`                     |
| §3.5 Oracle (on-demand teacher)   | `hat.core.oracle`                                 |
| §3.6 Neocortex (curated memory)   | `hat.core.neocortex` + `hat.memory.curated`       |
| §3.7 SWS trainer                  | `hat.core.sws` + `hat.models.training`            |
| §3.8 Iterative wake–sleep loop    | `hat.core.loop`                                   |
| §4 Experiments                    | `hat.data` + `hat.eval` + `experiments/`          |

## Layout

- `src/hat/core/` — paper algorithms only. No I/O, no model loading.
- `src/hat/models/` — model lifecycle and backends (HF / vLLM / Ollama / OpenAI-compatible).
- `src/hat/memory/` — storage layer. **Raw** chat history vs **curated** Neocortex are
  enforced as separate stores; only the Hippocampus Agent can move data across the gap.
- `src/hat/api/` — FastAPI: `routers/` → `controllers/` → injected protocols.
- `src/hat/services/` — long-running orchestration (SWS scheduler, replay worker, job queue).
- `src/hat/ui/` — Gradio chat UI and Streamlit operator dashboard.
- `src/hat/data/`, `src/hat/eval/` — benchmark loaders and evaluation harness.
- `experiments/` — YAML configs and run manifests.
- `docs/` — ADRs and architecture notes.

## Design contracts

1. **Wake/Sleep are deployment-separable.** `api/` + `core/cortex` + `core/hippocampus`
   serve users synchronously. `services/replay_worker` + `core/sws` + `models/training`
   run offline. They communicate **only** through `memory/` and `services/job_queue`.
2. **Raw vs curated separation is type-enforced.** `NeocortexStore.write` requires an
   accepted `WriteDecision` produced by a `WritePolicy`; you cannot accidentally promote
   a raw log entry into training data.
3. **Backends are protocols.** `LanguageModel`, `Trainer`, `OracleClient`, `NoveltyEstimator`,
   etc. are `typing.Protocol`s; HF/vLLM/Ollama/PEFT implement them; the registry swaps them.
4. **Controllers are pure Python.** Routers are thin FastAPI handlers; controllers depend
   on protocols, not FastAPI. The Gradio UI reuses the same controllers.

See `docs/adr-001-directory-layout.md`, `docs/adr-002-raw-vs-curated.md`,
`docs/adr-003-backend-protocol.md`.
