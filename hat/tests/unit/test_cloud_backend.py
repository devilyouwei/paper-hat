from __future__ import annotations

import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from hat.api.main import app
from hat.config.settings import get_settings
from hat.core.lifecycle.catalog import is_cloud_backend, load_catalog
from hat.core.lifecycle.embedding_manager import get_embedding_manager
from hat.core.lifecycle.manager import get_manager


@pytest.fixture(autouse=True)
def isolated_managers(tmp_path, monkeypatch):
    """Fresh model_root + cleared manager caches. Cloud needs no files, but
    we still isolate so local checkpoints don't leak into assertions."""
    s = get_settings()
    monkeypatch.setattr(s, "model_root", tmp_path)
    get_manager()._cache.clear()
    get_manager()._active = None
    get_embedding_manager()._cache.clear()
    get_embedding_manager()._active = None
    yield
    get_manager()._cache.clear()
    get_manager()._active = None
    get_embedding_manager()._cache.clear()
    get_embedding_manager()._active = None


# ---------- catalog --------------------------------------------------------


def test_cloud_catalogs_load() -> None:
    cloud = load_catalog("cloud")
    cloud_embed = load_catalog("cloud_embed")
    assert cloud and cloud_embed
    # Cloud entries carry a base_url + api_key_env instead of HF weights.
    assert all(e.base_url and e.api_key_env for e in cloud + cloud_embed)


def test_is_cloud_backend_helper() -> None:
    assert is_cloud_backend("cloud")
    assert is_cloud_backend("cloud_embed")
    assert not is_cloud_backend("mlx")
    assert not is_cloud_backend("hf_embed")


def test_cloud_models_listed_as_installed() -> None:
    client = TestClient(app)
    r = client.get("/api/models", params={"backend": "cloud"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "cloud"
    assert body["items"]
    # No local weights, but cloud models are always usable.
    assert all(i["installed"] is True for i in body["items"])


def test_cloud_embed_models_listed_as_installed() -> None:
    client = TestClient(app)
    r = client.get("/api/embedding-models", params={"backend": "cloud_embed"})
    assert r.status_code == 200
    body = r.json()
    assert all(i["installed"] is True for i in body["items"])


# ---------- activation without local files ---------------------------------


def test_cloud_set_active_succeeds_without_files() -> None:
    """Inverse of ``test_set_active_rejects_uninstalled``: a cloud model
    activates with nothing on disk and never makes a network call."""
    client = TestClient(app)
    entry = load_catalog("cloud")[0]
    r = client.post(
        "/api/models/active", json={"backend": "cloud", "id": entry.id}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "cloud"
    assert body["id"] == entry.id

    active = client.get("/api/models/active")
    assert active.json()["id"] == entry.id


def test_cloud_set_active_rejects_unknown_id() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/models/active", json={"backend": "cloud", "id": "no-such-model"}
    )
    assert r.status_code == 400


def test_cloud_delete_is_noop() -> None:
    mgr = get_manager()
    entry = load_catalog("cloud")[0]
    # Nothing on disk → delete returns False rather than raising.
    assert mgr.delete("cloud", entry.id) is False


# ---------- backend HTTP behaviour (mocked) --------------------------------


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_cloud_language_model_chat_parses_response(monkeypatch) -> None:
    from hat.core.cortex import cloud as cloud_mod

    payload = {"choices": [{"message": {"content": " hello there "}}]}

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(cloud_mod.urllib.request, "urlopen", fake_urlopen)
    lm = cloud_mod.CloudLanguageModel("gpt-4o-mini", api_key="sk-test")
    out = lm.chat([{"role": "user", "content": "hi"}])
    assert out == "hello there"


def test_cloud_language_model_token_logprobs(monkeypatch) -> None:
    from hat.core.cortex import cloud as cloud_mod

    payload = {
        "choices": [
            {
                "message": {"content": "ok"},
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.5}]},
            }
        ]
    }

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(cloud_mod.urllib.request, "urlopen", fake_urlopen)
    lm = cloud_mod.CloudLanguageModel("gpt-4o-mini", api_key="sk-test")
    lps = lm.token_logprobs("prompt", "ok")
    assert lps == [-0.1, -0.5]


def test_cloud_cortex_uncertainty_from_logprobs(monkeypatch) -> None:
    from hat.abstract.schemas import Interaction
    from hat.core.cortex import cloud as cloud_mod

    payload = {
        "choices": [
            {
                "message": {"content": "ok"},
                "logprobs": {"content": [{"logprob": 0.0}]},
            }
        ]
    }
    monkeypatch.setattr(
        cloud_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(json.dumps(payload).encode()),
    )
    cortex = cloud_mod.CloudCortex(
        cloud_mod.CloudLanguageModel("gpt-4o-mini", api_key="sk-test")
    )
    iax = Interaction(id="i1", query="q", response="ok")
    # logprob 0.0 => confidence exp(0)=1 => uncertainty 0.
    assert cortex.uncertainty(iax) == pytest.approx(0.0)


def test_cloud_embedding_model_embed_normalised(monkeypatch) -> None:
    from hat.core.neocortex.embeddings import cloud as cloud_embed_mod

    payload = {
        "data": [
            {"index": 0, "embedding": [3.0, 4.0]},
            {"index": 1, "embedding": [0.0, 2.0]},
        ]
    }
    monkeypatch.setattr(
        cloud_embed_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(json.dumps(payload).encode()),
    )
    emb = cloud_embed_mod.CloudEmbeddingModel(
        "text-embedding-3-small", api_key="sk-test"
    )
    vecs = emb.embed(["a", "b"])
    assert emb.dim == 2
    # L2-normalised on the client.
    assert vecs[0] == pytest.approx([0.6, 0.8])
    assert vecs[1] == pytest.approx([0.0, 1.0])
