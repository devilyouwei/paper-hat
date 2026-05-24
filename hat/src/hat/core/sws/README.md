# `core/sws/` — slow-wave-sleep trainer (paper §3.7)

Defines the trainer interface used during the sleep phase. Concrete
parameter-efficient implementations (LoRA, EWC, …) live under
[`../../models/training/`](../../models/training/) so they can be swapped
without touching the loop.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `SWSTrainer`, `DryRunTrainer`. |
| `trainer.py` | `SWSTrainer` ABC (`fit(batch, objective) -> SWSStats`) + `DryRunTrainer` no-op used by `hat sleep --dry-run` and plumbing tests. |
