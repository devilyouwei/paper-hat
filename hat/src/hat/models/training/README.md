# `models/training/` — SWS trainer implementations

Parameter-efficient fine-tuners that satisfy `core.sws.trainer.SWSTrainer`.
Stub today; concrete LoRA / EWC implementations land under this package
and are wired by the dependency container.

Install the optional extra:

```bash
uv sync --extra train
```

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. Concrete trainer modules will live alongside it (e.g. `lora.py`, `ewc.py`). |

The default trainer in the no-op loop is `core.sws.trainer.DryRunTrainer`,
used by `hat sleep --dry-run`.
