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


def test_openai_chat_rejects_streaming() -> None:
    client = TestClient(app)
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "hat-cortex",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert res.status_code == 400


def test_openai_models_list() -> None:
    client = TestClient(app)
    res = client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"]
