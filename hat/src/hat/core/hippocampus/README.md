# `core/hippocampus/` — Hippocampus Agent (paper §3.4)

Selective memory consolidation. Three composable stages plus a scoring
package, all pure Python.

## Files

| File | Paper § | Purpose |
| --- | --- | --- |
| `__init__.py` | — | Re-exports `Abstractor`, `IdentityAbstractor`, `LLMAbstractor`, `WritePolicy`, `UncertaintyGatePolicy`, `ReplayBuilder`, `SupervisedReplayBuilder`. |
| `abstraction.py` | §3.4.1 | `m = H_abs(c, x, y, f)`. `IdentityAbstractor` (verbatim copy) and `LLMAbstractor` (two-step **triage → route** prompt workflow: drop trivial turns first, then CREATE vs REVISE against prior session traces). |
| `selection.py` | §3.4.2 | `WritePolicy` ABC + `UncertaintyGatePolicy` — single-signal uncertainty gate that emits `WriteDecision`s the Neocortex requires before accepting a write. |
| `replay.py` | §3.4.3 | `ReplayBuilder` ABC + `SupervisedReplayBuilder` — turns retained traces into training-ready `ReplayExample` pairs. |

## Subpackages

| Directory | Role |
| --- | --- |
| [`scoring/`](scoring/README.md) | Per-signal scorers (currently uncertainty only) + shared LLM-as-judge plumbing. |
| [`prompts/`](prompts/README.md) | Markdown prompt templates used by `LLMAbstractor`. |
