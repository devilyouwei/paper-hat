"""HAT core algorithms.

This package is the *paper library*: every module corresponds to a section of the
method. It contains pure logic — no FastAPI, no torch, no I/O. Heavy backends are
plugged in via the ``protocols`` defined here.

Structure:

* ``schemas`` — pydantic data classes for interactions, traces, decisions, etc.
* ``protocols`` — ``typing.Protocol`` interfaces for every pluggable seam.
* ``cortex`` — online interaction model (paper §3.3).
* ``hippocampus`` — abstraction → selection → replay (paper §3.4).
* ``neocortex`` — long-term curated store (paper §3.6).
* ``oracle`` — on-demand external teacher (paper §3.5).
* ``sws`` — slow-wave-sleep trainer abstractions (paper §3.7).
* ``loop`` — wake–sleep orchestrator that ties everything together (paper §3.8).
"""

from . import loop

__all__ = ["loop"]
