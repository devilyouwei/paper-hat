from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hat.api.main import app
from hat.config.settings import get_settings
from hat.models.catalog import load_catalog
from hat.models.manager import get_manager


@pytest.fixture(autouse=True)
def isolated_model_root(tmp_path, monkeypatch):
    """Point ``model_root`` at a fresh tmp dir so installed-state tests are
    deterministic regardless of what the developer has cached locally."""
    s = get_settings()
    monkeypatch.setattr(s, "model_root", tmp_path)
    get_manager()._cache.clear()
    get_manager()._active = None
    yield


def test_default_catalogs_load() -> None:
    mlx = load_catalog("mlx")
    hf = load_catalog("hf")
    assert mlx and hf
    assert all(e.repo_id and e.id and e.display for e in mlx + hf)


def test_models_router_lists_catalog() -> None:
    client = TestClient(app)
    r = client.get("/api/models", params={"backend": "mlx"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "mlx"
    ids = {i["id"] for i in body["items"]}
    assert "qwen2.5-1.5b-instruct-4bit" in ids
    assert all(i["installed"] is False for i in body["items"])


def test_models_router_rejects_unknown_backend() -> None:
    client = TestClient(app)
    r = client.get("/api/models", params={"backend": "tpu"})
    assert r.status_code == 400


def test_set_active_rejects_uninstalled() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/models/active",
        json={"backend": "mlx", "id": "qwen2.5-1.5b-instruct-4bit"},
    )
    assert r.status_code == 400
    assert "not installed" in r.json()["detail"].lower()


def test_set_active_rejects_unknown_id() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/models/active", json={"backend": "mlx", "id": "no-such-model"}
    )
    assert r.status_code == 400


def test_active_starts_empty() -> None:
    client = TestClient(app)
    r = client.get("/api/models/active")
    assert r.status_code == 200
    assert r.json() is None
