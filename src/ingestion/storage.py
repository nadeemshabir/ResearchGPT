"""Where an uploaded PDF is kept after it has been indexed.

The vector store holds the chunks; this holds the file they came from. Both
are needed: re-embedding after a model change, re-chunking with different
settings, and letting a reader open the source all require the original PDF,
and none of them can reconstruct it from the chunks.

Two backends, chosen by configuration rather than by code:

``gcs``
    A Google Cloud Storage bucket, selected by setting ``UPLOAD_BUCKET``. This
    is what a Cloud Run deployment needs. Cloud Run gives each instance its own
    container filesystem, wiped on every restart and not shared with the
    instance beside it, so a PDF written to local disk there is invisible to
    the next request and gone by the next deploy.

``local``
    A directory on disk, used when ``UPLOAD_BUCKET`` is unset. Correct for
    development, and the reason the GCS dependency is optional rather than
    required.

The two are deliberately the same interface. Nothing above this module should
have to ask which one is running.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from src.config import get_settings
from src.exceptions import PaperStorageError
from src.utils.logging import get_logger

logger = get_logger(__name__)

StorageBackend = Literal["gcs", "local"]

#: Stored object names are ``<prefix>/<paper_id>.pdf``.
_SUFFIX = ".pdf"


class PaperStorage:
    """Keeps the uploaded PDF belonging to a ``paper_id``."""

    def __init__(
        self,
        bucket: str | None = None,
        directory: str | Path | None = None,
        prefix: str | None = None,
    ):
        """
        Args:
            bucket: GCS bucket name. Defaults to ``Settings.upload_bucket``.
            directory: Local directory, used when no bucket is configured.
                Defaults to ``Settings.upload_dir``.
            prefix: Object-name prefix inside the bucket. Defaults to
                ``Settings.upload_prefix``.
        """
        settings = get_settings()
        self.bucket_name = bucket if bucket is not None else settings.upload_bucket
        self.prefix = (prefix if prefix is not None else settings.upload_prefix).strip("/")
        self.directory = Path(directory) if directory else settings.upload_dir
        self.backend: StorageBackend = "gcs" if self.bucket_name else "local"

        self._bucket: Any | None = None
        if self.backend == "local":
            self.directory.mkdir(parents=True, exist_ok=True)

        logger.info("Paper storage ready (%s, %s)", self.backend, self.location)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    @property
    def location(self) -> str:
        """Human-readable description of where PDFs are kept."""
        if self.backend == "gcs":
            return f"gs://{self.bucket_name}/{self.prefix}"
        return str(self.directory)

    def _get_bucket(self) -> Any:
        """The GCS bucket handle, created on first use.

        Imported here rather than at module scope so that a local install
        without ``google-cloud-storage`` still works -- it is only needed by
        the deployment that sets ``UPLOAD_BUCKET``.
        """
        if self._bucket is not None:
            return self._bucket
        try:
            from google.cloud import storage  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - import guard
            raise PaperStorageError(
                "UPLOAD_BUCKET is set but google-cloud-storage is not installed. "
                "Install it with: pip install google-cloud-storage"
            ) from exc

        try:
            self._bucket = storage.Client().bucket(self.bucket_name)
        except Exception as exc:  # noqa: BLE001 - typed error for callers
            raise PaperStorageError(f"Could not reach bucket {self.bucket_name!r}: {exc}") from exc
        return self._bucket

    def _object_name(self, paper_id: str) -> str:
        return f"{self.prefix}/{paper_id}{_SUFFIX}" if self.prefix else f"{paper_id}{_SUFFIX}"

    def _local_path(self, paper_id: str) -> Path:
        return self.directory / f"{paper_id}{_SUFFIX}"

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def save(self, paper_id: str, source: str | Path) -> str:
        """Store the PDF at ``source`` under ``paper_id``. Returns its URI.

        Raises:
            PaperStorageError: The file could not be written.
        """
        source = Path(source)
        try:
            if self.backend == "gcs":
                blob = self._get_bucket().blob(self._object_name(paper_id))
                blob.upload_from_filename(str(source), content_type="application/pdf")
                uri = f"gs://{self.bucket_name}/{self._object_name(paper_id)}"
            else:
                destination = self._local_path(paper_id)
                # copyfile, not move: the caller owns the temp file and deletes
                # it in a finally block. Moving it out from under that is how
                # you get a confusing "file not found" on the cleanup path.
                shutil.copyfile(source, destination)
                uri = str(destination)
        except PaperStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperStorageError(f"Could not store the PDF for {paper_id!r}: {exc}") from exc

        logger.info("Stored PDF for %r at %s", paper_id, uri)
        return uri

    def exists(self, paper_id: str) -> bool:
        """Whether a PDF is stored for ``paper_id``."""
        try:
            if self.backend == "gcs":
                return bool(self._get_bucket().blob(self._object_name(paper_id)).exists())
            return self._local_path(paper_id).is_file()
        except PaperStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperStorageError(f"Could not check storage for {paper_id!r}: {exc}") from exc

    def download(self, paper_id: str, destination: str | Path) -> Path:
        """Fetch the stored PDF to a local path. Returns that path.

        Raises:
            PaperStorageError: Nothing is stored under this id, or the read
                failed.
        """
        destination = Path(destination)
        try:
            if self.backend == "gcs":
                blob = self._get_bucket().blob(self._object_name(paper_id))
                if not blob.exists():
                    raise PaperStorageError(f"No stored PDF for {paper_id!r}")
                blob.download_to_filename(str(destination))
            else:
                source = self._local_path(paper_id)
                if not source.is_file():
                    raise PaperStorageError(f"No stored PDF for {paper_id!r}")
                shutil.copyfile(source, destination)
        except PaperStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperStorageError(f"Could not read the PDF for {paper_id!r}: {exc}") from exc
        return destination

    def delete(self, paper_id: str) -> bool:
        """Remove the stored PDF. Returns whether a file was deleted.

        A missing file is not an error: deleting a paper whose PDF predates
        this storage layer, or whose upload failed midway, should still leave
        the system in the state the caller asked for.
        """
        try:
            if self.backend == "gcs":
                blob = self._get_bucket().blob(self._object_name(paper_id))
                if not blob.exists():
                    return False
                blob.delete()
            else:
                path = self._local_path(paper_id)
                if not path.is_file():
                    return False
                path.unlink()
        except PaperStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperStorageError(f"Could not delete the PDF for {paper_id!r}: {exc}") from exc

        logger.info("Deleted stored PDF for %r", paper_id)
        return True

    def list_ids(self) -> list[str]:
        """Every ``paper_id`` with a stored PDF, sorted."""
        try:
            if self.backend == "gcs":
                blobs = self._get_bucket().list_blobs(
                    prefix=f"{self.prefix}/" if self.prefix else None
                )
                names = [Path(blob.name).name for blob in blobs]
            else:
                names = [path.name for path in self.directory.glob(f"*{_SUFFIX}")]
        except PaperStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperStorageError(f"Could not list stored PDFs: {exc}") from exc

        return sorted(name[: -len(_SUFFIX)] for name in names if name.endswith(_SUFFIX))
