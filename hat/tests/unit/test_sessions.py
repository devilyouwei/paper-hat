from __future__ import annotations

from fastapi.testclient import TestClient

from hat.api.main import app


def test_sessions_lifecycle() -> None:
    client = TestClient(app)
    # initially empty
    assert client.get("/api/sessions").json() == {"object": "list", "data": []}

    # explicit create
    created = client.post("/api/sessions", json={"title": "first"}).json()
    assert created["title"] == "first"
    assert created["message_count"] == 0

    # rename
    renamed = client.patch(
        f"/api/sessions/{created['id']}", json={"title": "after"}
    ).json()
    assert renamed["title"] == "after"

    # listing returns the renamed session
    listed = client.get("/api/sessions").json()["data"]
    assert listed and listed[0]["id"] == created["id"]
    assert listed[0]["title"] == "after"

    # delete
    assert client.delete(f"/api/sessions/{created['id']}").status_code == 200
    assert client.get(f"/api/sessions/{created['id']}").status_code == 404


def test_chat_creates_and_appends_to_session() -> None:
    client = TestClient(app)
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "hat-cortex",
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    sid = data.get("hat_session_id")
    assert sid, "controller should auto-create a session"

    # session is listed
    listed = client.get("/api/sessions").json()["data"]
    assert any(s["id"] == sid for s in listed)

    # transcript has exactly one Interaction
    detail = client.get(f"/api/sessions/{sid}").json()
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["query"] == "hello there"
    assert detail["session"]["message_count"] == 1


def test_chat_streaming_carries_session_id() -> None:
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "hat-cortex",
            "messages": [{"role": "user", "content": "stream hi"}],
            "stream": True,
        },
    ) as res:
        assert res.status_code == 200, res.read()
        body = b"".join(res.iter_bytes()).decode("utf-8")
    # the session id is broadcast on at least one chunk
    assert "hat_session_id" in body
