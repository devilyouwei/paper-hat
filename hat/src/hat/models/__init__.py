"""Model lifecycle and backends.

The ``core`` package owns the *algorithms*; this package owns *machinery*:
loading checkpoints, calling inference servers, running PEFT trainers. Each
backend is gated behind an extras group (``hf``, ``vllm``, ``ollama``) so the
default install stays small.
"""

from .registry import available, create, register

__all__ = ["register", "create", "available"]
