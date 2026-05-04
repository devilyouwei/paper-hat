"""JSONL-backed Neocortex.

The on-disk format is aligned with the OpenAI / HuggingFace SFT convention,
so the file can be fed directly into a fine-tuning pipeline:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}],
     "trace_id": "...",
     "interaction_id": "...",
     "score": 0.83,
     "signals": {"uncertainty": 0.6, "feedback": 1.0, "novelty": 0.4},
     "metadata": {"timestamp": "...", "source": "user", "extras": {...}}}

A second sidecar file (``traces.jsonl``) keeps the full :class:`MemoryTrace`
records for inspection / replay; it is optional. The training file is the
single source of truth for the SWS trainer.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from ...core.neocortex.store import NeocortexStore
from ...core.schemas import MemoryTrace, TraceMetadata, WriteDecision


def _trace_to_sft(trace: MemoryTrace, decision: WriteDecision) -> dict:
    """Map a :class:`MemoryTrace` to one HF/OpenAI SFT training row."""
    target = trace.target_response or trace.cortex_response or ""
    messages = [
        {"role": "user", "content": trace.query},
        {"role": "assistant", "content": target},
    ]
    return {
        "messages": messages,
        "trace_id": trace.id,
        "interaction_id": trace.interaction_id,
        "score": float(decision.score),
        "signals": decision.signals.model_dump(mode="json"),
        "metadata": trace.metadata.model_dump(mode="json"),
    }


def _sft_to_trace(row: dict) -> tuple[MemoryTrace, float]:
    msgs = row.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    assistant = next(
        (m["content"] for m in msgs if m.get("role") == "assistant"), None
    )
    md = row.get("metadata") or {}
    trace = MemoryTrace(
        id=row.get("trace_id") or "",
        interaction_id=row.get("interaction_id") or "",
        query=user,
        target_response=assistant,
        metadata=TraceMetadata.model_validate(md) if md else TraceMetadata(),
    )
    return trace, float(row.get("score") or 0.0)


class JsonlNeocortex(NeocortexStore):
    """Reference Neocortex backed by an SFT-format JSONL file."""

    def __init__(
        self,
        path: str | Path,
        traces_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.traces_path = Path(traces_path) if traces_path else None
        if self.traces_path is not None:
            self.traces_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _persist(self, trace: MemoryTrace, decision: WriteDecision) -> None:
        # Primary: SFT-format training row.
        row = _trace_to_sft(trace, decision)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            # Sidecar: full MemoryTrace record for inspection. Optional; the
            # trainer only consumes the SFT file above.
            if self.traces_path is not None:
                full = {
                    "trace": trace.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                }
                with self.traces_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(full, default=str) + "\n")

    # -- read helpers ----------------------------------------------------

    def _read_rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _read_trace_rows(self) -> list[dict]:
        if self.traces_path is None or not self.traces_path.exists():
            return []
        rows: list[dict] = []
        with self.traces_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load(self) -> list[tuple[MemoryTrace, float]]:
        return [_sft_to_trace(r) for r in self._read_rows()]

    def __iter__(self) -> Iterator[MemoryTrace]:
        for tr, _ in self._load():
            yield tr

    def __len__(self) -> int:
        return len(self._read_rows())

    def sample(self, k: int) -> Iterable[MemoryTrace]:
        rows = self._load()
        rows.sort(key=lambda x: x[1], reverse=True)
        return [tr for tr, _ in rows[:k]]

    # -- management API (manual curation) -------------------------------

    def entries(self) -> list[dict]:
        """Return all stored SFT rows verbatim, newest last (insertion order)."""
        with self._lock:
            return self._read_rows()

    def get_entry(self, trace_id: str) -> dict | None:
        with self._lock:
            for r in self._read_rows():
                if r.get("trace_id") == trace_id:
                    return r
        return None

    def _atomic_write(self, path: Path, rows: list[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(tmp, path)

    def delete(self, trace_id: str) -> bool:
        """Remove the entry from both the SFT file and the trace sidecar."""
        with self._lock:
            rows = self._read_rows()
            kept = [r for r in rows if r.get("trace_id") != trace_id]
            if len(kept) == len(rows):
                return False
            self._atomic_write(self.path, kept)
            if self.traces_path is not None:
                trace_rows = self._read_trace_rows()
                kept_traces = [
                    r for r in trace_rows
                    if (r.get("trace") or {}).get("id") != trace_id
                ]
                self._atomic_write(self.traces_path, kept_traces)
            return True

    def update(
        self,
        trace_id: str,
        *,
        query: str | None = None,
        response: str | None = None,
        score: float | None = None,
    ) -> dict | None:
        """Edit an existing entry in place. Returns the updated row or ``None``."""
        with self._lock:
            rows = self._read_rows()
            updated: dict | None = None
            for i, r in enumerate(rows):
                if r.get("trace_id") != trace_id:
                    continue
                msgs = list(r.get("messages") or [])
                if query is not None:
                    for m in msgs:
                        if m.get("role") == "user":
                            m["content"] = query
                            break
                    else:
                        msgs.insert(0, {"role": "user", "content": query})
                if response is not None:
                    for m in msgs:
                        if m.get("role") == "assistant":
                            m["content"] = response
                            break
                    else:
                        msgs.append({"role": "assistant", "content": response})
                r["messages"] = msgs
                if score is not None:
                    r["score"] = float(score)
                rows[i] = r
                updated = r
                break
            if updated is None:
                return None
            self._atomic_write(self.path, rows)
            # Mirror edits into the sidecar so the two files stay consistent.
            if self.traces_path is not None:
                trace_rows = self._read_trace_rows()
                for i, tr in enumerate(trace_rows):
                    trace_obj = tr.get("trace") or {}
                    if trace_obj.get("id") != trace_id:
                        continue
                    if query is not None:
                        trace_obj["query"] = query
                    if response is not None:
                        trace_obj["target_response"] = response
                    tr["trace"] = trace_obj
                    if score is not None:
                        decision = tr.get("decision") or {}
                        decision["score"] = float(score)
                        tr["decision"] = decision
                    trace_rows[i] = tr
                    break
                self._atomic_write(self.traces_path, trace_rows)
            return updated
