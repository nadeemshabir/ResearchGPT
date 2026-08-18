"""Embedding generation via sentence-transformers."""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.exceptions import IngestionError
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Turn text into dense vectors.

    The model is loaded eagerly in ``__init__`` (a few seconds on first run
    while weights download) so that a bad model name fails at startup rather
    than on the first query.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        """
        Args:
            model_name: sentence-transformers model id. Defaults to
                ``Settings.embedding_model``.
            device: ``"cpu"``, ``"cuda"``, or ``"mps"``. Auto-detected if omitted.
            batch_size: Texts encoded per forward pass.

        Raises:
            IngestionError: The model could not be loaded.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.device = resolve_device(device or settings.embedding_device)

        logger.info("Loading embedding model %s on %s", self.model_name, self.device)

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001 - surface as a typed error
            raise IngestionError(
                f"Could not load embedding model {self.model_name!r} on device "
                f"{self.device!r}: {exc}"
            ) from exc

        embedding_dim = self.model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            raise IngestionError(
                f"Model {self.model_name!r} did not report an embedding dimension"
            )
        self.embedding_dim: int = embedding_dim
        logger.info("Embedding model ready (%d dimensions)", self.embedding_dim)

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: Texts to embed. An empty list returns an empty list.
            batch_size: Overrides the instance default.
            show_progress: Render a progress bar. Off by default; it writes to
                stdout, which library code should not do unattended.

        Returns:
            One vector per input text, in the same order.
        """
        if not texts:
            return []

        logger.debug("Embedding %d texts", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def generate_single_embedding(self, text: str) -> list[float]:
        """Embed one string.

        Raises:
            ValueError: ``text`` is empty or whitespace only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Attach an ``embedding`` field to each chunk in place."""
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding"] = embedding
        return chunks

    def get_model_info(self) -> dict[str, Any]:
        """Return model identity and shape, for logging and DB provenance."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, "max_seq_length", None),
        }
