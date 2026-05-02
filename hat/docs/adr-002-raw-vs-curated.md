# ADR-002 — Raw vs curated memory: a type-enforced boundary

## Status
Accepted.

## Context
A central claim of the paper is that the Hippocampus *selectively* consolidates interactions — it is not a chat-history dump. If the training pipeline accidentally reads the raw log, the experiment becomes meaningless. The user explicitly asked for a strict separation between chat history and the algorithm-filtered memory.

## Decision
Two storage abstractions, each with its own subpackage and ABC:

* ``hat.memory.raw.RawInteractionLog`` — append-only chat history. Only the wake-time controllers append. The Hippocampus reads from it.
* ``hat.core.neocortex.NeocortexStore`` — curated long-term memory. Its ``write(trace, decision: WriteDecision)`` method **raises** ``NeocortexWriteError`` unless the supplied ``WriteDecision``:
  1. has ``trace_id == trace.id``, and
  2. has ``accepted=True``.

The only component allowed to mint a ``WriteDecision`` is a ``WritePolicy``, which is itself part of the Hippocampus pipeline. The training pipeline never sees ``RawInteractionLog``; it iterates the Neocortex via ``sample(k)``.

## Consequences
* The boundary is enforced at the type level rather than by convention. A reviewer can verify the property by reading one method.
* Tests in ``tests/unit/test_smoke.py`` cover both rejection paths (mismatched id, ``accepted=False``).
* Implementations of either store can be swapped (in-memory, JSONL, SQLite, vector-DB) without touching the loop.
