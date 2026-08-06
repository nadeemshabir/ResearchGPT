"""PDF text and metadata extraction.

Three backends are supported. PyMuPDF is the default: it is the fastest and
preserves reading order best on two-column academic layouts. pdfplumber is more
faithful on tables at roughly 5x the cost; PyPDF2 is a last-resort fallback.

All backend-specific failures are translated into :mod:`src.exceptions` types so
callers can distinguish "encrypted", "no text layer", and "corrupt" without
inspecting error strings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.config import PDFMethod, get_settings
from src.exceptions import (
    CorruptPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    PDFParseError,
    PDFTooLargeError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


#: arXiv stamps, venue banners, and page numbers that sit above the title.
_PRE_TITLE_NOISE = re.compile(
    r"^(arxiv[:\s]|\d+\s*$|[\d.]+v\d|preprint|under review|published as|"
    r"proceedings of|to appear|draft|\[?cs\.[a-z]{2}\]?)",
    re.IGNORECASE,
)

#: Function words. Title lines nearly always contain one; author lines do not.
#: This is what separates a title continuation such as "Language Models: A
#: Survey" from an author line such as "Ashish Vaswani Noam Shazeer".
_FUNCTION_WORDS = re.compile(
    r"\b(a|an|the|of|for|with|in|on|to|and|or|via|from|by|as|at|into|"
    r"using|towards?|through|without|beyond)\b",
    re.IGNORECASE,
)

#: Footnote daggers and superscripts attached to author names.
_AUTHOR_MARKS = re.compile(r"[∗*†‡§¶]|\d\s*,|\b[A-Z]\.\s")

_AFFILIATION = re.compile(
    r"\b(universit|institute|research|laborator|labs?|inc\.|llc|college|"
    r"school of|department|google|microsoft|openai|meta ai|deepmind)\b",
    re.IGNORECASE,
)


def _ends_title(line: str) -> bool:
    """True when ``line`` begins the author block rather than continuing a title."""
    if "@" in line or _AFFILIATION.search(line) or line.lower().startswith("abstract"):
        return True
    # A line with no function words is a candidate author list, but only call it
    # one when a name marker or comma separation confirms it -- otherwise short
    # title continuations like "Reinforcement Learning" would be cut off.
    if not _FUNCTION_WORDS.search(line):
        return bool(_AUTHOR_MARKS.search(line) or line.count(",") >= 2)
    return False


class PDFParser:
    """Extract text and metadata from a PDF."""

    def __init__(self, method: PDFMethod | None = None, min_extracted_chars: int | None = None):
        """
        Args:
            method: Backend to use. Defaults to ``Settings.pdf_method``.
            min_extracted_chars: Below this, the document is treated as having
                no text layer. Defaults to ``Settings.min_extracted_chars``.
        """
        settings = get_settings()
        self.method: str = method or settings.pdf_method
        self.min_extracted_chars = (
            min_extracted_chars if min_extracted_chars is not None else settings.min_extracted_chars
        )
        self.max_size_mb = settings.max_pdf_size_mb

        if self.method not in ("pymupdf", "pdfplumber", "pypdf2"):
            raise ValueError(
                f"Unknown PDF method {self.method!r}; expected one of "
                "'pymupdf', 'pdfplumber', 'pypdf2'"
            )

    def extract_text(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract text and metadata from a single PDF.

        Args:
            pdf_path: Path to the file.

        Returns:
            Mapping with ``text``, ``num_pages``, ``metadata``, ``file_name``,
            ``file_path``, and ``file_size_mb``.

        Raises:
            PDFParseError: The file is missing, empty, oversized, encrypted,
                corrupt, or carries no extractable text layer.
        """
        path = Path(pdf_path)
        self._validate_file(path)

        logger.info("Parsing %s with %s", path.name, self.method)

        extractors = {
            "pypdf2": self._extract_pypdf2,
            "pdfplumber": self._extract_pdfplumber,
            "pymupdf": self._extract_pymupdf,
        }

        try:
            result = extractors[self.method](path)
        except PDFParseError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a typed error below
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise EncryptedPDFError(
                    f"{path.name} is password protected and cannot be read.", path=str(path)
                ) from exc
            raise CorruptPDFError(
                f"{path.name} could not be parsed with {self.method}: {exc}", path=str(path)
            ) from exc

        text = result["text"]
        if len(text.strip()) < self.min_extracted_chars:
            raise EmptyPDFError(
                f"{path.name} yielded only {len(text.strip())} characters "
                f"(minimum {self.min_extracted_chars}). It is most likely a scanned "
                f"document with no text layer; OCR is required and is not supported.",
                path=str(path),
            )

        result["file_name"] = path.name
        result["file_path"] = str(path.absolute())
        result["file_size_mb"] = path.stat().st_size / (1024 * 1024)

        metadata = result.setdefault("metadata", {})
        if not str(metadata.get("title", "")).strip():
            recovered = self._title_from_first_page(text)
            if recovered:
                metadata["title"] = recovered
                logger.info("Recovered title from page 1 of %s: %r", path.name, recovered)
            else:
                logger.warning(
                    "%s has no /Title and no usable heading on page 1; "
                    "citations for it will name the paper id instead.",
                    path.name,
                )

        logger.info(
            "Extracted %d characters from %d pages of %s",
            len(text),
            result["num_pages"],
            path.name,
        )
        return result

    def _validate_file(self, path: Path) -> None:
        """Reject unusable files before spending time on parsing."""
        if not path.exists():
            raise PDFParseError(f"PDF not found: {path}", path=str(path))
        if not path.is_file():
            raise PDFParseError(f"Not a file: {path}", path=str(path))

        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise CorruptPDFError(f"{path.name} is empty (0 bytes).", path=str(path))

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.max_size_mb:
            raise PDFTooLargeError(
                f"{path.name} is {size_mb:.1f} MB, over the {self.max_size_mb} MB limit.",
                path=str(path),
            )

        # A real PDF starts with the %PDF- magic bytes. Checking here turns a
        # confusing library traceback into a clear message for the common case
        # of a mislabelled or truncated upload.
        try:
            with open(path, "rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise CorruptPDFError(
                        f"{path.name} does not begin with the %PDF- header; "
                        "it is not a valid PDF.",
                        path=str(path),
                    )
        except OSError as exc:
            raise PDFParseError(f"Could not read {path.name}: {exc}", path=str(path)) from exc

    def _extract_pypdf2(self, path: Path) -> dict[str, Any]:
        """Extract with PyPDF2 (basic, fast, weakest layout handling)."""
        import PyPDF2

        text_parts: list[str] = []
        with open(path, "rb") as handle:
            reader = PyPDF2.PdfReader(handle)

            if reader.is_encrypted:
                # An empty user password is common and decrypts silently.
                try:
                    if reader.decrypt("") == 0:
                        raise EncryptedPDFError(
                            f"{path.name} is password protected.", path=str(path)
                        )
                except EncryptedPDFError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise EncryptedPDFError(
                        f"{path.name} is password protected.", path=str(path)
                    ) from exc

            num_pages = len(reader.pages)
            metadata = reader.metadata

            for page_num in range(num_pages):
                page_text = reader.pages[page_num].extract_text() or ""
                text_parts.append(page_text)

        return {
            "text": "\n\n".join(text_parts),
            "num_pages": num_pages,
            "metadata": {
                "title": str(metadata.get("/Title", "") or ""),
                "author": str(metadata.get("/Author", "") or ""),
                "subject": str(metadata.get("/Subject", "") or ""),
                "creator": str(metadata.get("/Creator", "") or ""),
            }
            if metadata
            else {},
        }

    def _extract_pdfplumber(self, path: Path) -> dict[str, Any]:
        """Extract with pdfplumber (best on tables, noticeably slower)."""
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            num_pages = len(pdf.pages)
            raw_metadata = pdf.metadata or {}

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return {
            "text": "\n\n".join(text_parts),
            "num_pages": num_pages,
            "metadata": {
                "title": str(raw_metadata.get("Title", "") or ""),
                "author": str(raw_metadata.get("Author", "") or ""),
                "subject": str(raw_metadata.get("Subject", "") or ""),
                "creator": str(raw_metadata.get("Creator", "") or ""),
            },
        }

    def _extract_pymupdf(self, path: Path) -> dict[str, Any]:
        """Extract with PyMuPDF/fitz (default: fastest, best reading order)."""
        import fitz

        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                raise EncryptedPDFError(f"{path.name} is password protected.", path=str(path))

            num_pages = len(doc)
            raw_metadata = doc.metadata or {}
            text_parts = [doc[page_num].get_text() for page_num in range(num_pages)]
        finally:
            doc.close()

        return {
            "text": "\n\n".join(text_parts),
            "num_pages": num_pages,
            "metadata": {
                "title": raw_metadata.get("title", "") or "",
                "author": raw_metadata.get("author", "") or "",
                "subject": raw_metadata.get("subject", "") or "",
                "creator": raw_metadata.get("creator", "") or "",
                "producer": raw_metadata.get("producer", "") or "",
                "creation_date": raw_metadata.get("creationDate", "") or "",
            },
        }

    @staticmethod
    def _title_from_first_page(text: str) -> str:
        """Recover a paper title from the opening lines when ``/Title`` is blank.

        LaTeX writes no ``/Title`` unless the author loads ``hyperref`` with
        ``pdftitle``, so most arXiv preprints carry empty PDF metadata. Measured
        on this corpus: 5 of 8 papers had no embedded title, which left the
        context header reading ``[Source 1: Unknown | ...]`` and gave the model
        nothing meaningful to cite.

        The title is taken as the first run of lines that reads like a heading,
        skipping arXiv stamps and other pre-title furniture.
        """
        lines: list[str] = []
        for raw in text.split("\n")[:40]:
            line = raw.strip()
            if not line or _PRE_TITLE_NOISE.match(line):
                # Furniture before the title is skipped; furniture after it ends
                # the title.
                if lines:
                    break
                continue
            if lines and _ends_title(line):
                break
            lines.append(line)
            if len(" ".join(lines)) > 120:
                break

        title = re.sub(r"\s+", " ", " ".join(lines)).strip(" .,-")
        # Reject fragments too short to identify a paper, and runaway captures
        # that clearly swallowed body text.
        if not 10 <= len(title) <= 200:
            return ""
        return title

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise whitespace and strip common PDF extraction artifacts.

        Note that this collapses all newlines, so it must run *after* any
        structure detection that depends on line breaks.
        """
        text = text.replace("\x00", "")  # null bytes
        text = text.replace("", "")  # private-use bullet glyph
        text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
        # Ligatures survive extraction and silently break keyword matching:
        # "workflow" extracted with an fl-ligature never matches a BM25 query
        # for "workflow".
        for ligature, replacement in (
            ("ﬀ", "ff"),
            ("ﬁ", "fi"),
            ("ﬂ", "fl"),
            ("ﬃ", "ffi"),
            ("ﬄ", "ffl"),
        ):
            text = text.replace(ligature, replacement)
        text = " ".join(text.split())
        text = text.replace(" .", ".").replace(" ,", ",")
        return text
