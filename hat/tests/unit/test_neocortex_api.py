from __future__ import annotations

from fastapi.testclient import TestClient

from hat.api.deps import get_loop
from hat.api.main import app
from hat.core.schemas import (
    MemoryTrace,
    ScoreSignals,
    TraceMetadata,
    WriteDecision,
)


def _seed_trace(query: str = "what is hat", response: str = "a hat is a hat") -> str:
    """Push one accepted trace through the public NeocortexStore.write contract."""
    trace = MemoryTrace(
        interaction_id="i-test",
        query=query,
        target_response=response,
        metadata=TraceMetadata(source="user"),
    )
    decision = WriteDecision(
        trace_id=trace.id,
        score=0.7,
        threshold=0.5,
        signals=ScoreSignals(uncertainty=0.6, feedback=1.0, novelty=0.4),
        accepted=True,
    )
    get_loop().neocortex.write(trace, decision)
    return trace.id


def test_neocortex_list_and_get() -> None:
    client = TestClient(app)
    tid = _seed_trace()

    listed = client.get("/api/neocortex").json()
    assert listed["object"] == "list"
    assert any(e["trace_id"] == tid for e in listed["data"])

    detail = client.get(f"/api/neocortex/{tid}").json()
    assert detail["query"] == "what is hat"
    assert detail["response"] == "a hat is a hat"
    assert detail["score"] == 0.7


def test_neocortex_patch_updates_messages_and_keeps_score() -> None:
    """Score is intentionally immutable via the API; PATCH should ignore it
    and persist only the editable query / response fields."""
    client = TestClient(app)
    tid = _seed_trace()

    patched = client.patch(
        f"/api/neocortex/{tid}",
        json={"response": "edited answer", "score": 0.9},
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["response"] == "edited answer"
    # Score from seed is preserved, the requested 0.9 is silently dropped.
    assert body["score"] == 0.7
    # query untouched
    assert body["query"] == "what is hat"


def test_neocortex_delete_removes_row() -> None:
    client = TestClient(app)
    tid = _seed_trace()

    assert client.delete(f"/api/neocortex/{tid}").status_code == 200
    assert client.get(f"/api/neocortex/{tid}").status_code == 404
    listed = client.get("/api/neocortex").json()["data"]
    assert all(e["trace_id"] != tid for e in listed)


def test_neocortex_unknown_id_returns_404() -> None:
    client = TestClient(app)
    assert client.get("/api/neocortex/does-not-exist").status_code == 404
    assert (
        client.patch(
            "/api/neocortex/does-not-exist", json={"response": "x"}
        ).status_code
        == 404
    )
    assert client.delete("/api/neocortex/does-not-exist").status_code == 404
