"""Storage layer.

Two strictly separated stores:

* ``raw``     — append-only chat history (``RawInteractionLog``).
* ``curated`` — Neocortex traces written *only* by the Hippocampus Agent.

The training pipeline must never read ``raw``; the Hippocampus is the only
component that can promote a raw interaction into a curated trace. See
``docs/adr-002-raw-vs-curated.md``.
"""
