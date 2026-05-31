"""Tests for the embedding-model REST API and manager."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hat.api.main import app
from hat.config.settings import embed_index_path_for, get_settings
from hat.models.catalog import load_catalog
from hat.models.embedding_manager import EmbeddingManagerError, get_embedding_manager


@pytest.fixture(autouse=True)
def isolated_model_root(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "model_root", tmp_path)
    monkeypatch.setattr(s, "embed_index_root", tmp_path / "emb-idx")
    mgr = get_embedding_manager()
    mgr._cache.clear()  # type: ignore[attr-defined]
    mgr._active = None  # type: ignore[attr-defined]
    yield


def test_embed_catalogs_load() -> None:
    mlx = load_catalog("mlx_embed")
    hf = load_catalog("hf_embed")
    assert mlx, "mlx_embed catalog should not be empty"
    assert hf == [] or all(e.repo_id and e.id for e in hf)


def test_embed_catalogs_present_in_api() -> None:
    client = TestClient(app)
    r = client.get("/api/embedding-models", params={"backend": "mlx_embed"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "mlx_embed"
    ids = {i["id"] for i in body["items"]}
    assert "embeddinggemma-300m-4bit" in ids
    assert all(i["installed"] is False for i in body["items"])


def test_embed_set_active_rejects_uninstalled() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/embedding-models/active",
        json={"backend": "mlx_embed", "id": "embeddinggemma-300m-4bit"},
    )
    assert r.status_code == 400
    assert "not installed" in r.json()["detail"].lower()


def test_embed_active_starts_null_then_unloaded() -> None:
    client = TestClient(app)
    r = client.get("/api/embedding-models/active")
    assert r.status_code == 200
    assert r.json() is None
    r = client.delete("/api/embedding-models/active")
    assert r.status_code == 200
    assert r.json() == {"unloaded": 0}


def test_embed_index_path_partitioned_per_model(tmp_path) -> None:
    s = get_settings()
    s.embed_index_root = tmp_path / "idx"
    a = embed_index_path_for("mlx_embed", "embeddinggemma-300m-4bit")
    b = embed_index_path_for("mlx_embed", "qwen3-embedding-0.6b-8bit")
    c = embed_index_path_for("hf_embed", "some/repo")
    assert a != b != c
    assert a.suffix == ".npz"
    assert "/" not in c.name


def test_embed_manager_delete_unknown_returns_false() -> None:
    """Delete returns False for a catalog entry whose weights aren't on disk."""
    mgr = get_embedding_manager()
    assert mgr.delete("mlx_embed", "embeddinggemma-300m-4bit") is False


def test_embed_manager_rejects_bad_backend() -> None:
    mgr = get_embedding_manager()
    with pytest.raises(EmbeddingManagerError):
        mgr.delete("mlx", "qwen2.5-0.5b-instruct-4bit")


def test_policy_includes_active_embedder_block() -> None:
    client = TestClient(app)
    r = client.get("/api/policy")
    assert r.status_code == 200
    dedup = r.json()["dedup"]
    assert "active_embedder" in dedup
    assert "index_path" in dedup
    # Nothing activated → null active and legacy path.
    assert dedup["active_embedder"] is None
