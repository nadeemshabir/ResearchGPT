"""Tests for the uploaded-PDF store and the paper ids that name its objects.

The local backend is exercised against a real temporary directory. The GCS
backend is exercised against a fake client: the point of these tests is the
code in `src/ingestion/storage.py`, not Google's, and a test that needs a
bucket is a test nobody runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.exceptions import PaperStorageError
from src.ingestion.ids import clean_paper_id
from src.ingestion.storage import PaperStorage


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4 pretend")
    return source


@pytest.fixture
def storage(tmp_path: Path) -> PaperStorage:
    return PaperStorage(bucket=None, directory=tmp_path / "uploads")


# --- paper ids --------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("attention_2017", "attention_2017"),
        (
            "NIPS-2017-attention-is-all-you-need-Paper (1)",
            "nips-2017-attention-is-all-you-need-paper",
        ),
        ("Attention Is All You Need", "attention_is_all_you_need"),
        ("  spaced  out  ", "spaced_out"),
        ("Dotted.Name.v2", "dotted_name_v2"),
    ],
)
def test_ids_are_readable(stem: str, expected: str) -> None:
    """The id is the citation key a reader sees, not an internal handle."""
    assert clean_paper_id(stem) == expected


def test_an_id_is_never_empty() -> None:
    """An empty id would collide with the next empty one and name nothing."""
    assert clean_paper_id("***") == "paper"
    assert clean_paper_id("") == "paper"


def test_an_id_cannot_escape_its_directory() -> None:
    """The id becomes a filename and an object name. Separators must not survive."""
    cleaned = clean_paper_id("../../etc/passwd")

    assert "/" not in cleaned
    assert ".." not in cleaned


def test_a_long_id_is_truncated() -> None:
    cleaned = clean_paper_id("x" * 400)

    assert 0 < len(cleaned) <= 96


# --- local backend ----------------------------------------------------------


def test_local_round_trip(storage: PaperStorage, pdf: Path, tmp_path: Path) -> None:
    storage.save("attention_2017", pdf)

    fetched = storage.download("attention_2017", tmp_path / "out.pdf")

    assert fetched.read_bytes() == b"%PDF-1.4 pretend"


def test_save_leaves_the_source_in_place(storage: PaperStorage, pdf: Path) -> None:
    """The caller owns the temp file and deletes it in a finally block.

    Moving it instead of copying turned that cleanup into a confusing
    "file not found".
    """
    storage.save("attention_2017", pdf)

    assert pdf.is_file()


def test_exists_and_delete(storage: PaperStorage, pdf: Path) -> None:
    assert storage.exists("attention_2017") is False

    storage.save("attention_2017", pdf)
    assert storage.exists("attention_2017") is True

    assert storage.delete("attention_2017") is True
    assert storage.exists("attention_2017") is False


def test_deleting_what_is_not_there_is_not_an_error(storage: PaperStorage) -> None:
    """Deleting a paper whose PDF predates this store must still succeed."""
    assert storage.delete("never_uploaded") is False


def test_downloading_what_is_not_there_raises(storage: PaperStorage, tmp_path: Path) -> None:
    with pytest.raises(PaperStorageError, match="No stored PDF"):
        storage.download("never_uploaded", tmp_path / "out.pdf")


def test_list_ids(storage: PaperStorage, pdf: Path) -> None:
    storage.save("bert_2019", pdf)
    storage.save("attention_2017", pdf)

    assert storage.list_ids() == ["attention_2017", "bert_2019"]


# --- GCS backend ------------------------------------------------------------


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str):
        self.bucket = bucket
        self.name = name

    def upload_from_filename(self, path: str, content_type: str | None = None) -> None:
        self.bucket.objects[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self.bucket.objects[self.name])

    def exists(self) -> bool:
        return self.name in self.bucket.objects

    def delete(self) -> None:
        del self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def list_blobs(self, prefix: str | None = None) -> list[FakeBlob]:
        return [
            FakeBlob(self, name)
            for name in sorted(self.objects)
            if prefix is None or name.startswith(prefix)
        ]


@pytest.fixture
def gcs(monkeypatch: pytest.MonkeyPatch) -> tuple[PaperStorage, FakeBucket]:
    bucket = FakeBucket()
    storage = PaperStorage(bucket="research-papers", prefix="papers")
    monkeypatch.setattr(storage, "_get_bucket", lambda: bucket)
    return storage, bucket


def test_gcs_object_names_carry_the_prefix(gcs: tuple[PaperStorage, FakeBucket], pdf: Path) -> None:
    """One bucket should be able to serve more than one environment."""
    storage, bucket = gcs

    storage.save("attention_2017", pdf)

    assert list(bucket.objects) == ["papers/attention_2017.pdf"]


def test_gcs_save_returns_a_gs_uri(gcs: tuple[PaperStorage, FakeBucket], pdf: Path) -> None:
    storage, _ = gcs

    uri = storage.save("attention_2017", pdf)

    assert uri == "gs://research-papers/papers/attention_2017.pdf"


def test_gcs_round_trip(gcs: tuple[PaperStorage, FakeBucket], pdf: Path, tmp_path: Path) -> None:
    storage, _ = gcs
    storage.save("attention_2017", pdf)

    fetched = storage.download("attention_2017", tmp_path / "out.pdf")

    assert fetched.read_bytes() == b"%PDF-1.4 pretend"


def test_gcs_delete_and_list(gcs: tuple[PaperStorage, FakeBucket], pdf: Path) -> None:
    storage, _ = gcs
    storage.save("attention_2017", pdf)
    storage.save("bert_2019", pdf)

    assert storage.list_ids() == ["attention_2017", "bert_2019"]
    assert storage.delete("bert_2019") is True
    assert storage.list_ids() == ["attention_2017"]


def test_a_bucket_failure_is_reported_as_storage_error(
    gcs: tuple[PaperStorage, FakeBucket], pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers map PaperStorageError to a 502. A raw google exception is a 500."""
    storage, bucket = gcs

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("network is unreachable")

    monkeypatch.setattr(FakeBlob, "upload_from_filename", boom)

    with pytest.raises(PaperStorageError, match="attention_2017"):
        storage.save("attention_2017", pdf)


def test_the_backend_is_chosen_by_whether_a_bucket_is_configured(tmp_path: Path) -> None:
    assert PaperStorage(bucket=None, directory=tmp_path).backend == "local"
    assert PaperStorage(bucket="some-bucket").backend == "gcs"
