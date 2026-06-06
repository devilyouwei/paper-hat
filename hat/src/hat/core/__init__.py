"""HAT core: paper algorithms and runtime backends.

Each subpackage maps to a section of the method. Interfaces (Protocols / ABCs /
schemas) live in :mod:`hat.abstract`; this package holds their *implementations*
in two tiers:

* **Pure algorithm plumbing** (``loop`` + ``hippocampus`` / ``neocortex`` /
  ``oracle`` / ``sws``) — depends only on the abstract seams, no web or
  heavy-ML concerns.
* **Concrete model backends & lifecycle** (``cortex`` / ``neocortex.embeddings``
  / ``lifecycle`` / ``runtime`` / ``sessions``) — import heavy deps
  (``mlx-lm``, ``transformers``, ``huggingface_hub``) and touch disk / network,
  but always lazily behind an abstract seam.

Layout:

* ``cortex`` — online interaction model (paper §3.3).
* ``hippocampus`` — abstraction → selection → replay (paper §3.4).
* ``oracle`` — on-demand external teacher (paper §3.5).
* ``neocortex`` — long-term curated store + vector index + embedders (paper §3.6).
* ``sws`` — slow-wave-sleep trainer (paper §3.7).
* ``loop`` — wake–sleep orchestrator that ties everything together (paper §3.8).
* ``lifecycle`` / ``runtime`` / ``sessions`` — model catalog & management,
  composition root, and raw chat-history persistence.

``core`` never imports from :mod:`hat.api`.
"""

from . import loop

__all__ = ["loop"]
