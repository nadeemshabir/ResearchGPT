"""Token-aware text chunking for academic papers.

Chunk boundaries are measured in tokens rather than characters so that a chunk
never silently overflows the embedding model's context. Splitting prefers
section breaks, then paragraphs, then sentences, then words.
"""

from __future__ import annotations

import re
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.exceptions import ChunkingError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Section headings found in most papers, used by :meth:`TextChunker.split_sections`.
SECTION_HEADERS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "methodology",
    "methods",
    "method",
    "approach",
    "system",
    "model",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "analysis",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
)

#: Matches "3", "3.", "3.1", "IV." and similar leading section numbers.
_SECTION_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s*", re.IGNORECASE)

#: A heading line is short; anything longer is prose that happens to start
#: with a heading word.
_MAX_HEADER_LINE_LENGTH = 60


class TextChunker:
    """Split paper text into overlapping, token-bounded chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str | None = None,
    ):
        """
        Args:
            chunk_size: Target tokens per chunk. Defaults to ``Settings.chunk_size``.
            chunk_overlap: Tokens shared between neighbours. Defaults to
                ``Settings.chunk_overlap``.
            encoding_name: tiktoken encoding used for counting. Defaults to
                ``Settings.tokenizer_encoding``.

        Raises:
            ValueError: ``chunk_overlap`` is not smaller than ``chunk_size``.
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.encoding_name = encoding_name or settings.tokenizer_encoding

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking cannot advance."
            )

        self.encoding = tiktoken.get_encoding(self.encoding_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        )

        logger.debug(
            "Chunker ready (size=%d tokens, overlap=%d tokens, encoding=%s)",
            self.chunk_size,
            self.chunk_overlap,
            self.encoding_name,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the configured encoding."""
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        section_title: str | None = None,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        """Split ``text`` into chunk records.

        Args:
            text: Text to split.
            metadata: Copied onto every chunk produced.
            section_title: Recorded on each chunk when known.
            start_index: First ``chunk_id`` to assign, so callers chunking
                section-by-section can keep ids unique across a document.

        Returns:
            Chunk dicts with ``text``, ``chunk_id``, ``num_tokens``,
            ``num_characters``, and ``section_title``.

        Raises:
            ChunkingError: ``text`` is empty or whitespace only.
        """
        if not text or not text.strip():
            raise ChunkingError("Cannot chunk empty text.")

        raw_chunks = [c for c in self.splitter.split_text(text) if c.strip()]
        if not raw_chunks:
            raise ChunkingError(f"Splitting produced no usable chunks from {len(text)} characters.")

        chunks: list[dict[str, Any]] = []
        for offset, chunk_text in enumerate(raw_chunks):
            chunk_id = start_index + offset
            chunk: dict[str, Any] = {
                "text": chunk_text,
                "chunk_id": chunk_id,
                "num_tokens": self.count_tokens(chunk_text),
                "num_characters": len(chunk_text),
                "section_title": section_title or "Full Text",
            }
            if metadata:
                chunk["metadata"] = {**metadata, "chunk_id": chunk_id}
            chunks.append(chunk)

        return chunks

    def chunk_with_context(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk section by section so no chunk straddles a section boundary.

        Falls back to flat chunking when no sections are detected.

        Important: ``text`` must retain its newlines. Section headings are
        identified per line, so passing text through
        :meth:`PDFParser.clean_text` first makes every document look like a
        single unstructured section.
        """
        sections = self.split_sections(text)

        if len(sections) <= 1:
            logger.debug("No section structure detected; chunking as one block")
            return self.chunk_text(text, metadata)

        logger.debug("Detected %d sections", len(sections))

        all_chunks: list[dict[str, Any]] = []
        for section in sections:
            if not section["text"].strip():
                continue
            try:
                section_chunks = self.chunk_text(
                    section["text"],
                    metadata,
                    section_title=section["title"],
                    start_index=len(all_chunks),
                )
            except ChunkingError:
                # A heading with no body under it is normal; skip it.
                continue
            all_chunks.extend(section_chunks)

        if not all_chunks:
            raise ChunkingError("Section-aware chunking produced no chunks.")

        return all_chunks

    def split_sections(self, text: str) -> list[dict[str, str]]:
        """Split paper text into ``{title, text}`` sections by heading lines.

        A line is treated as a heading when, after stripping any leading
        section number, it begins with a known heading word and is short enough
        to be a title rather than prose.
        """
        lines = text.split("\n")
        sections: list[dict[str, str]] = []
        current = {"title": "Preamble", "body": []}  # type: dict[str, Any]

        for line in lines:
            heading = self._match_heading(line)
            if heading is not None:
                if any(part.strip() for part in current["body"]):
                    sections.append({"title": current["title"], "text": "\n".join(current["body"])})
                current = {"title": heading, "body": []}
            else:
                current["body"].append(line)

        if any(part.strip() for part in current["body"]):
            sections.append({"title": current["title"], "text": "\n".join(current["body"])})

        if not sections:
            return [{"title": "Full Text", "text": text}]
        return sections

    @staticmethod
    def _match_heading(line: str) -> str | None:
        """Return the normalised heading title, or ``None`` if not a heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > _MAX_HEADER_LINE_LENGTH:
            return None

        # Trailing punctuation is stripped for both matching *and* the returned
        # title. Normalising only for the match would make "Introduction" and
        # "Introduction:" two distinct section titles for the same section,
        # fragmenting labels across papers that punctuate differently.
        without_number = _SECTION_NUMBER.sub("", stripped).rstrip(":.").strip()
        candidate = without_number.lower()

        for header in SECTION_HEADERS:
            if candidate == header or candidate.startswith(header + " "):
                return without_number or header.title()
        return None

    @staticmethod
    def summarise(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Return aggregate statistics for a chunk list.

        Safe on an empty list, which is why callers use this instead of
        computing averages inline.
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0.0,
                "min_tokens": 0,
                "max_tokens": 0,
                "num_sections": 0,
            }

        token_counts = [c["num_tokens"] for c in chunks]
        return {
            "num_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "num_sections": len({c.get("section_title", "Full Text") for c in chunks}),
        }
