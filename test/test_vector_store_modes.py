"""Tests for selecting where the vector store lives.

A deployment that silently falls back to an on-disk store is the bug this
whole change exists to fix: on Cloud Run that store is wiped on restart and is
not shared between instances, so uploads vanish and two instances disagree
about what is indexed. These tests check that the configuration says what it
means and fails loudly when it cannot.

No Chroma client is constructed. The clients are patched out: what is under
test is the choice, not Chroma.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import Settings
from src.ingestion.database import VectorDatabase


class FakeCollection:
    metadata: dict[str, Any] = {}

    def count(self) -> int:
        return 0


class FakeClient:
    """Records how it was constructed so the test can assert on it."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def get_or_create_collection(self, **_kwargs: Any) -> FakeCollection:
        return FakeCollection()


@pytest.fixture
def chroma(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch all three Chroma constructors and record which one was used."""
    import chromadb

    used: dict[str, Any] = {}

    def recorder(name: str) -> Any:
        def build(**kwargs: Any) -> FakeClient:
            used["client"] = name
            used["kwargs"] = kwargs
            return FakeClient(**kwargs)

        return build

    monkeypatch.setattr(chromadb, "PersistentClient", recorder("persistent"))
    monkeypatch.setattr(chromadb, "HttpClient", recorder("http"))
    monkeypatch.setattr(chromadb, "CloudClient", recorder("cloud"))
    return used


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Settings:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
    for module in ("src.config", "src.ingestion.database"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings, raising=False)
    return settings


# --- client selection -------------------------------------------------------


def test_local_mode_opens_a_directory(
    monkeypatch: pytest.MonkeyPatch, chroma: dict[str, Any], tmp_path: Any
) -> None:
    _settings(monkeypatch, chroma_mode="local", chroma_db_path=tmp_path / "db")

    database = VectorDatabase()

    assert chroma["client"] == "persistent"
    assert database.mode == "local"


def test_http_mode_connects_to_a_server(
    monkeypatch: pytest.MonkeyPatch, chroma: dict[str, Any]
) -> None:
    _settings(monkeypatch, chroma_mode="http", chroma_host="chroma.internal", chroma_port=9000)

    database = VectorDatabase()

    assert chroma["client"] == "http"
    assert chroma["kwargs"]["host"] == "chroma.internal"
    assert chroma["kwargs"]["port"] == 9000
    assert database.location == "http://chroma.internal:9000"


def test_cloud_mode_passes_the_credentials(
    monkeypatch: pytest.MonkeyPatch, chroma: dict[str, Any]
) -> None:
    _settings(
        monkeypatch,
        chroma_mode="cloud",
        chroma_api_key="ck-secret",
        chroma_tenant="tenant-1",
        chroma_database="researchgpt",
    )

    database = VectorDatabase()

    assert chroma["client"] == "cloud"
    assert chroma["kwargs"]["api_key"] == "ck-secret"
    assert chroma["kwargs"]["tenant"] == "tenant-1"
    assert database.location == "chroma-cloud/tenant-1/researchgpt"


def test_an_explicit_path_forces_a_local_store(
    monkeypatch: pytest.MonkeyPatch, chroma: dict[str, Any], tmp_path: Any
) -> None:
    """The evaluation scripts keep their own store beside the application's.

    Without this, running an evaluation with a cloud configuration in the
    environment would write its scratch collections into the live store.
    """
    _settings(
        monkeypatch,
        chroma_mode="cloud",
        chroma_api_key="ck-secret",
        chroma_tenant="tenant-1",
        chroma_database="researchgpt",
    )

    database = VectorDatabase(db_path=tmp_path / "eval_chroma")

    assert chroma["client"] == "persistent"
    assert database.mode == "local"


# --- configuration validation ----------------------------------------------


def test_http_mode_requires_a_host() -> None:
    with pytest.raises(ValueError, match="CHROMA_HOST"):
        Settings(_env_file=None, chroma_mode="http")  # type: ignore[call-arg]


def test_cloud_mode_names_every_missing_credential() -> None:
    """One error listing all three beats three deploys finding them one at a time."""
    with pytest.raises(ValueError) as caught:
        Settings(_env_file=None, chroma_mode="cloud")  # type: ignore[call-arg]

    message = str(caught.value)
    assert "CHROMA_API_KEY" in message
    assert "CHROMA_TENANT" in message
    assert "CHROMA_DATABASE" in message


def test_cloud_mode_accepts_complete_credentials() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        chroma_mode="cloud",
        chroma_api_key="ck-secret",
        chroma_tenant="tenant-1",
        chroma_database="researchgpt",
    )

    assert settings.chroma_mode == "cloud"


def test_the_default_is_local() -> None:
    """A fresh clone should run with no configuration at all."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.chroma_mode == "local"
    assert settings.upload_bucket is None


def test_the_api_key_is_not_printed_by_accident() -> None:
    """Settings get logged and dumped. The key must not ride along in clear."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        chroma_mode="cloud",
        chroma_api_key="ck-secret",
        chroma_tenant="tenant-1",
        chroma_database="researchgpt",
    )

    assert "ck-secret" not in str(settings)
