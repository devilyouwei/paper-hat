from __future__ import annotations

from fastapi.testclient import TestClient

from hat.api.main import app


def test_openai_chat_completions_smoke() -> None:
    client = TestClient(app)
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "hat-cortex",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert "hat_consolidated" in data


def test_openai_chat_streaming_smoke() -> None:
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "hat-cortex",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as res:
        assert res.status_code == 200, res.read()
        assert res.headers["content-type"].startswith("text/event-stream")
        body = b"".join(res.iter_bytes()).decode("utf-8")
    assert "data: " in body
    assert "[DONE]" in body
    # at least one chunk should carry assistant content
    assert '"delta"' in body


def test_openai_models_list() -> None:
    client = TestClient(app)
    res = client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"]
