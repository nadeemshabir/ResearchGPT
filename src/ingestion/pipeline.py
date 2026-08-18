"""End-to-end ingestion: PDF file to indexed, queryable chunks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import PDFMethod, get_settings
from src.exceptions import IngestionError, PDFParseError, ResearchGPTError
from src.ingestion.chunker import TextChunker
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.pdf_parser import PDFParser
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Parse, chunk, embed, and store research papers."""

    def __init__(
        self,
        pdf_method: PDFMethod | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str | None = None,
        database: VectorDatabase | None = None,
        embedder: EmbeddingGenerator | None = None,
    ):
        """
        Args:
            pdf_method: PDF backend. Defaults to ``Settings.pdf_method``.
            chunk_size: Tokens per chunk. Defaults to ``Settings.chunk_size``.
            chunk_overlap: Token overlap. Defaults to ``Settings.chunk_overlap``.
            embedding_model: Defaults to ``Settings.embedding_model``.
            database: Reuse an existing store instead of opening another.
            embedder: Reuse a loaded model instead of loading it twice.
        """
        logger.info("Initialising ingestion pipeline")

        self.pdf_parser = PDFParser(method=pdf_method)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder or EmbeddingGenerator(model_name=embedding_model)
        self.database = database or VectorDatabase()

        # Fail now, with a clear message, rather than on the first query.
        self.database.verify_embedding_model(self.embedder.model_name, self.embedder.embedding_dim)

        logger.info("Ingestion pipeline ready")

    def process_paper(
        self,
        pdf_path: str | Path,
        paper_id: str | None = None,
        use_sections: bool = True,
    ) -> dict[str, Any]:
        """Run one PDF through the full pipeline.

        Args:
            pdf_path: Path to the PDF.
            paper_id: Unique id. Defaults to the filename stem.
            use_sections: Chunk section by section rather than as flat text.

        Returns:
            Ingestion statistics for the paper.

        Raises:
            PDFParseError: The file could not be read.
            IngestionError: Chunking, embedding, or storage failed.
        """
        start = time.perf_counter()
        path = Path(pdf_path)
        paper_id = paper_id or path.stem

        logger.info("Ingesting %r from %s", paper_id, path.name)

        # 1. Parse. Section detection needs the original line breaks, so the
        #    raw text is chunked first and each chunk cleaned afterwards.
        parsed = self.pdf_parser.extract_text(path)
        raw_text = parsed["text"]

        # 2. Chunk.
        if use_sections:
            chunks = self.chunker.chunk_with_context(raw_text, metadata=parsed["metadata"])
        else:
            chunks = self.chunker.chunk_text(raw_text, metadata=parsed["metadata"])

        for chunk in chunks:
            chunk["text"] = self.pdf_parser.clean_text(chunk["text"])
        chunks = [c for c in chunks if c["text"].strip()]

        if not chunks:
            raise IngestionError(f"{path.name} produced no usable chunks after cleaning.")

        chunk_stats = self.chunker.summarise(chunks)
        logger.info(
            "%r: %d chunks across %d sections (avg %.0f tokens)",
            paper_id,
            chunk_stats["num_chunks"],
            chunk_stats["num_sections"],
            chunk_stats["avg_tokens"],
        )

        # 3. Embed.
        chunks = self.embedder.embed_chunks(chunks)

        # 4. Store.
        paper_metadata = {
            "title": parsed["metadata"].get("title") or path.stem,
            "author": parsed["metadata"].get("author") or "Unknown",
            "file_name": parsed["file_name"],
            "num_pages": parsed["num_pages"],
        }
        num_stored = self.database.add_chunks(
            chunks=chunks, paper_id=paper_id, paper_metadata=paper_metadata
        )

        elapsed = time.perf_counter() - start
        logger.info("Ingested %r: %d chunks stored in %.2fs", paper_id, num_stored, elapsed)

        return {
            "paper_id": paper_id,
            "file_name": parsed["file_name"],
            "num_pages": parsed["num_pages"],
            "num_chunks": len(chunks),
            "num_stored": num_stored,
            "num_sections": chunk_stats["num_sections"],
            "avg_tokens_per_chunk": chunk_stats["avg_tokens"],
            "processing_time_seconds": round(elapsed, 2),
            "metadata": paper_metadata,
            "status": "success",
        }

    def process_directory(
        self,
        directory_path: str | Path,
        file_pattern: str = "*.pdf",
        continue_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """Ingest every PDF in a directory.

        Args:
            directory_path: Directory to scan.
            file_pattern: Glob pattern.
            continue_on_error: Record per-file failures and carry on. When
                ``False``, the first failure aborts the batch.

        Returns:
            One result record per file; failures carry ``status="failed"``
            and an ``error`` message.

        Raises:
            IngestionError: The directory does not exist.
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        pdf_files = sorted(directory.glob(file_pattern))
        if not pdf_files:
            logger.warning("No files matching %r in %s", file_pattern, directory)
            return []

        logger.info("Batch ingesting %d files from %s", len(pdf_files), directory)

        results: list[dict[str, Any]] = []
        succeeded = 0

        for index, pdf_file in enumerate(pdf_files, 1):
            logger.info("[%d/%d] %s", index, len(pdf_files), pdf_file.name)
            try:
                results.append(self.process_paper(pdf_file))
                succeeded += 1
            except (PDFParseError, ResearchGPTError) as exc:
                logger.error("Failed to ingest %s: %s", pdf_file.name, exc)
                if not continue_on_error:
                    raise
                results.append(
                    {
                        "paper_id": pdf_file.stem,
                        "file_name": pdf_file.name,
                        "status": "failed",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        logger.info(
            "Batch complete: %d succeeded, %d failed", succeeded, len(pdf_files) - succeeded
        )
        return results

    def get_stats(self) -> dict[str, Any]:
        """Current pipeline and store configuration, for the UI and health checks."""
        settings = get_settings()
        return {
            "database": self.database.get_stats(),
            "embedding_model": self.embedder.get_model_info(),
            "chunker": {
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "encoding": self.chunker.encoding_name,
            },
            "pdf_method": self.pdf_parser.method,
            "max_pdf_size_mb": settings.max_pdf_size_mb,
        }
