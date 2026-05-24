# `core/hippocampus/prompts/` — abstractor prompt templates

Markdown templates loaded at runtime by `scoring/llm_judge.load_prompt` and
rendered into chat messages for the active Cortex. Kept as plain `.md`
files so they can be edited without touching code.

## Files

| File | Used by | Purpose |
| --- | --- | --- |
| `abstraction_triage.md` | `LLMAbstractor` (step 1) | Decides whether the current `(query, response)` turn carries a knowledge point worth remembering. Trivial small-talk gets dropped here without ever touching the routing step. |
| `abstraction_route.md` | `LLMAbstractor` (step 2) | Only invoked after triage says *keep*. Receives the session's existing traces and the new turn, decides CREATE vs REVISE, and emits the canonical `(query, target)` pair to store. |

Both templates are cached after first read (small, read-mostly).
