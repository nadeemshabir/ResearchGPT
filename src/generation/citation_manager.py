"""Citation formatting, extraction, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.config import CitationStyle, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Patterns matching the citation formats this system emits.
#:
#: The ``[Source N: ...]`` entry is not a style the prompt asks for -- it is the
#: header ``RetrievalSystem.build_context`` puts above each chunk. Models copy
#: that header far more often than they follow the requested ``[Title, Year]``,
#: so an audit that ignores it undercounts real citations. Measured on 43
#: answers: 8 used the header form, 2 used the requested form.
_SOURCE_TAG_PATTERN = re.compile(
    r"\[Source\s+\d+\s*:\s*([^\]|]+?)\s*(?:\|[^\]]*)?\]", re.IGNORECASE
)
_INLINE_PATTERN = re.compile(r"\[([^\]]+?,\s*(?:\d{4}|n\.d\.))\]")  # [Title, 2017]
_NUMBERED_PATTERN = re.compile(r"\[(\d+)\]")  # [1]
_APA_PATTERN = re.compile(r"\(([^)]+?,\s*(?:\d{4}|n\.d\.))\)")  # (Author, 2017)

_CITATION_PATTERNS = (
    _SOURCE_TAG_PATTERN,  # [Source 1: Title | Section: Intro]
    _INLINE_PATTERN,
    _NUMBERED_PATTERN,
    _APA_PATTERN,
)

#: Which pattern belongs to which configured style. Auditing must not apply all
#: of them at once: papers quote their own references, so an APA scan of an
#: answer built from paper text matches things like "(Lu et al., 2023)" that the
#: model copied out of a source rather than emitted as a citation. Scoring those
#: as fabricated references was inflating `unknown_citations`.
_STYLE_PATTERNS = {
    "inline": _INLINE_PATTERN,
    "numbered": _NUMBERED_PATTERN,
    "apa": _APA_PATTERN,
}


#: Words carrying no topical signal. Overlap is measured on content words only,
#: otherwise every paragraph matches every chunk on "the", "of", and "is".
_STOPWORDS = frozenset(
    # noqa: SIM905 -- a 100-element list literal is unreadable and this list is
    # edited by hand often enough that legibility wins.
    """
    the a an and or but if then than that this these those of to in on at for with
    from by as is are was were be been being it its their they them we our you your
    which who whom what when where how why not no nor can could may might will would
    should shall do does did done has have had having there here also such other
    more most some any each both few many much own same so too very just only over
    under between into through during before after above below up down out off
    """.split()  # noqa: SIM905
)


def _content_words(text: str) -> set[str]:
    """Topical vocabulary of ``text``, lowercased.

    Tokens shorter than three characters are dropped along with stopwords; they
    are almost always articles, symbols, or fragments of split words.
    """
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _source_of(chunk: dict[str, Any]) -> dict[str, Any]:
    """Citation fields for a retrieved chunk, taken from its stored metadata.

    ``chunk_id`` comes from the chunk's own ``id``, not from metadata: it is
    what a client uses to look the passage back up, so it has to survive even
    when metadata is sparse.
    """
    metadata = chunk.get("metadata", {}) or {}
    paper_id = str(metadata.get("paper_id", ""))
    return {
        "chunk_id": str(chunk.get("id") or metadata.get("chunk_id") or ""),
        "paper_id": paper_id,
        "title": metadata.get("title") or paper_id or "Unknown",
        "author": metadata.get("author") or "Unknown",
        "year": metadata.get("year") or _year_from_paper_id(paper_id) or "n.d.",
        "section": metadata.get("section_title") or "Unknown",
    }


@dataclass(frozen=True)
class CitedSource:
    """One chunk credited for a paragraph."""

    chunk_id: str
    paper_id: str
    title: str
    year: str
    section: str
    #: Content-word overlap with the paragraph, 0-1. Exposed so a client can
    #: show how strong an attribution is rather than treating all as equal.
    score: float


@dataclass(frozen=True)
class ParagraphAttribution:
    """One paragraph of an answer and the chunks it draws on.

    ``sources`` is empty for a paragraph that matched nothing above the
    threshold, and for headings and lists (``structural``). Both are kept in
    the list so the answer can be rebuilt in order from the attributions alone.
    """

    text: str
    sources: list[CitedSource]
    structural: bool = False


def _year_from_paper_id(paper_id: str) -> str:
    """Pull a publication year out of an id like ``attention_2017``.

    PDF metadata rarely carries a usable year, but the ids in
    ``data/raw/paper_ids.json`` end in one, and ``[Title, 2017]`` reads far
    better than ``[Title, n.d.]``.
    """
    match = re.search(r"(?:^|[_\-])((?:19|20)\d{2})(?:$|[_\-])", paper_id)
    return match.group(1) if match else ""


#: A single token of word characters -- no spaces. "attention_2017_chunk_5",
#: "vaswani2017", "12". Distinguishes a citation key from bracketed prose.
_IDENTIFIER_BRACKET = re.compile(r"[\w][\w\-.]*")


def _strip_citations(text: str, titles: set[str]) -> str:
    """Remove citations the model wrote, preserving line structure.

    Models invent citation formats freely. On one run the same model produced
    ``[Attention is All you Need, Undated]`` and
    ``[Training language models... | Section: Model, Figure 2]`` -- neither
    matches the formats this system emits. Chasing formats with regexes is a
    losing game.

    So the test is semantic rather than syntactic: **a bracketed span is a
    citation if it names a source that was retrieved.** The titles are known,
    which makes this reliable regardless of how the model chose to format them.

    Unlike :meth:`CitationManager.remove_citations` this does not collapse
    whitespace, because paragraph breaks are what the citer works on.
    """

    def drop_if_source(match: re.Match[str]) -> str:
        inner = match.group(1).strip().lower()
        if not inner:
            return match.group(0)
        for title in titles:
            # Either direction: the model may quote the full title or shorten it.
            if title in inner or (len(inner) >= 10 and inner in title):
                return ""
        # A bracket holding a single identifier-like token is a citation key,
        # not prose. Observed: a model asked *not* to cite still emitted
        # "[attention_2017_chunk_5]" -- an id it had never been shown, matching
        # this project's internal format by coincidence. Left in, it reads to a
        # user as a real reference.
        #
        # Prose in brackets is untouched: "[see the appendix]" has spaces, so it
        # fails this test and survives.
        if _IDENTIFIER_BRACKET.fullmatch(inner):
            return ""
        return match.group(0)

    cleaned = _SOURCE_TAG_PATTERN.sub("", text)
    cleaned = re.sub(r"\[([^\[\]]*)\]", drop_if_source, cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # A removed citation that sat after a full stop leaves a stranded one.
    cleaned = re.sub(r"(?<=[.!?])[ \t]*\.", "", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines())


def _is_structural(paragraph: str) -> bool:
    """True for headings, bullet lists, and code blocks.

    Citing these reads badly: a citation after a heading attaches to nothing,
    and one after a bullet list attaches only to the final bullet.
    """
    stripped = paragraph.strip()
    if stripped.startswith(("#", "```", ">")):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(re.match(r"^([-*+]|\d+[.)])\s", line) for line in lines):
        return True
    # A single short line with no sentence-ending punctuation is a heading.
    return len(lines) == 1 and len(stripped) < 60 and not stripped.endswith((".", "!", "?"))


class CitationManager:
    """Format and audit citations in generated text."""

    def __init__(self, citation_style: CitationStyle | None = None):
        """
        Args:
            citation_style: ``"inline"`` (``[Title, Year]``), ``"numbered"``
                (``[1]``), or ``"apa"`` (``(Author, Year)``). Defaults to
                ``Settings.citation_style``.
        """
        self.citation_style = citation_style or get_settings().citation_style
        #: Sources seen so far, in first-cited order. Backs numbered style.
        self.citations_used: list[dict[str, Any]] = []

    def format_citation(self, source: dict[str, Any]) -> str:
        """Render one citation in the configured style.

        Numbered style assigns each distinct source a stable index on first
        use, matched on ``(title, year)`` rather than dict identity so that
        equivalent records from different chunks share a number.
        """
        title = source.get("title", "Unknown")
        year = source.get("year", "n.d.")
        author = source.get("author", "Unknown")

        if self.citation_style == "apa":
            return f"({author}, {year})"

        if self.citation_style == "numbered":
            key = (str(title), str(year))
            for index, seen in enumerate(self.citations_used, 1):
                if (str(seen.get("title")), str(seen.get("year"))) == key:
                    return f"[{index}]"
            self.citations_used.append(source)
            return f"[{len(self.citations_used)}]"

        return f"[{title}, {year}]"

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Return every citation string found in ``text``, in order."""
        found: list[str] = []
        for pattern in _CITATION_PATTERNS:
            found.extend(pattern.findall(text))
        return found

    @staticmethod
    def remove_citations(text: str) -> str:
        """Strip citations, for comparing cited and uncited text."""
        cleaned = text
        for pattern in _CITATION_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
        return " ".join(cleaned.split())

    @staticmethod
    def merge_duplicate_citations(text: str) -> str:
        """Collapse adjacent citations: ``[A, 2023] [B, 2024]`` to ``[A, 2023; B, 2024]``."""
        pattern = re.compile(r"\[([^\]]+)\]\s*\[([^\]]+)\]")
        merged = text
        # Repeat until stable, so runs of three or more also collapse.
        for _ in range(10):
            new = pattern.sub(lambda m: f"[{m.group(1)}; {m.group(2)}]", merged)
            if new == merged:
                break
            merged = new
        return merged

    def add_citations_to_text(self, text: str, sources: list[dict[str, Any]]) -> str:
        """Append one citation per sentence, in source order.

        This is a positional fallback used when no LLM is available. It assumes
        sentence *i* came from source *i*, which is frequently wrong.

        Prefer :meth:`add_paragraph_citations`, which matches text to sources by
        content instead of position and is what the pipeline uses by default.
        """
        if not sources:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", text)
        cited: list[str] = []
        for index, sentence in enumerate(sentences):
            if index < len(sources) and sentence.strip():
                cited.append(f"{sentence} {self.format_citation(sources[index])}")
            else:
                cited.append(sentence)
        return " ".join(cited)

    def attribute_paragraphs(
        self,
        text: str,
        chunks: list[dict[str, Any]],
        min_similarity: float | None = None,
        max_per_paragraph: int | None = None,
    ) -> list[ParagraphAttribution]:
        """Map each paragraph of ``text`` to the chunks it draws on.

        This is where the work happens; :meth:`add_paragraph_citations` renders
        the result as text. Callers that need to *link* a claim to its source --
        a UI highlighting the passage behind a sentence, for instance -- want
        this structure, not a string they would have to parse back.

        Attribution is by content-word overlap between the paragraph and each
        chunk, so a source can never be invented: every attribution names a
        chunk that was actually retrieved. A paragraph matching nothing above
        ``min_similarity`` is left unattributed rather than given a guess.

        Matching is **chunk-level, not sentence-level**. A lone sentence carries
        too little vocabulary to match confidently; sentence-level attribution
        would look more precise and be less correct.

        Args:
            text: The generated answer.
            chunks: Retrieved chunks, each with ``id``, ``text`` and ``metadata``.
            min_similarity: Overlap floor. Defaults to
                ``Settings.citation_min_similarity``.
            max_per_paragraph: Cap on sources per paragraph. Defaults to
                ``Settings.citation_max_per_paragraph``.

        Returns:
            One :class:`ParagraphAttribution` per paragraph, in order, including
            structural and unattributed ones so the answer can be rebuilt from
            the list without consulting the original text.
        """
        if not text.strip():
            return []

        settings = get_settings()
        floor = min_similarity if min_similarity is not None else settings.citation_min_similarity
        cap = (
            max_per_paragraph
            if max_per_paragraph is not None
            else settings.citation_max_per_paragraph
        )

        scored_chunks = [
            (_content_words(chunk.get("text", "")), _source_of(chunk)) for chunk in chunks
        ]
        known_titles = {
            str(source["title"]).lower() for _, source in scored_chunks if source.get("title")
        }

        attributions: list[ParagraphAttribution] = []
        for raw_paragraph in re.split(r"\n\s*\n", _strip_citations(text, known_titles)):
            body = raw_paragraph.rstrip()

            if not body.strip():
                continue
            if _is_structural(body):
                attributions.append(ParagraphAttribution(text=body, sources=[], structural=True))
                continue

            words = _content_words(body)
            if not words or not scored_chunks:
                attributions.append(ParagraphAttribution(text=body, sources=[]))
                continue

            # Score every chunk, keep the best per paper: two chunks from one
            # paper should produce one citation, not two.
            best_by_paper: dict[str, tuple[float, dict[str, Any]]] = {}
            for chunk_words, source in scored_chunks:
                if not chunk_words:
                    continue
                overlap = len(words & chunk_words) / len(words)
                if overlap < floor:
                    continue
                key = str(source.get("paper_id") or source.get("title"))
                if key not in best_by_paper or overlap > best_by_paper[key][0]:
                    best_by_paper[key] = (overlap, source)

            ranked = sorted(best_by_paper.values(), key=lambda pair: -pair[0])[:cap]
            attributions.append(
                ParagraphAttribution(
                    text=body,
                    sources=[
                        CitedSource(
                            chunk_id=str(source.get("chunk_id", "")),
                            paper_id=str(source.get("paper_id", "")),
                            title=str(source.get("title", "Unknown")),
                            year=str(source.get("year", "n.d.")),
                            section=str(source.get("section", "Unknown")),
                            score=round(score, 4),
                        )
                        for score, source in ranked
                    ],
                )
            )

        return attributions

    def add_paragraph_citations(
        self,
        text: str,
        chunks: list[dict[str, Any]],
        min_similarity: float | None = None,
        max_per_paragraph: int | None = None,
    ) -> str:
        """Render :meth:`attribute_paragraphs` as text with inline citations.

        Citations are placed **at the end of each paragraph**, never after
        individual sentences. When a paragraph draws on several papers, all of
        them appear together in one group: ``[A, 2017; B, 2019]``.

        Per-sentence citation is the obvious first instinct and it is worse on
        both counts -- it makes answers unreadable, and a lone sentence carries
        too little vocabulary to match a chunk confidently.

        Returns:
            The answer with citation groups appended to matching paragraphs.
        """
        if not text.strip() or not chunks:
            return text

        rendered: list[str] = []
        for paragraph in self.attribute_paragraphs(text, chunks, min_similarity, max_per_paragraph):
            if not paragraph.sources:
                rendered.append(paragraph.text)
                continue
            group = " ".join(
                self.format_citation(
                    {"title": source.title, "year": source.year, "author": "Unknown"}
                )
                for source in paragraph.sources
            )
            rendered.append(f"{paragraph.text} {self.merge_duplicate_citations(group)}")

        return "\n\n".join(rendered)

    def generate_bibliography(self, sources: list[dict[str, Any]]) -> str:
        """Render a deduplicated reference list in Markdown."""
        if not sources:
            return ""

        seen: set[tuple[str, str]] = set()
        unique: list[dict[str, Any]] = []
        for source in sources:
            key = (str(source.get("title", "")), str(source.get("year", "")))
            if key not in seen:
                seen.add(key)
                unique.append(source)

        lines = ["## References", ""]
        for index, source in enumerate(unique, 1):
            title = source.get("title", "Unknown")
            author = source.get("author", "Unknown")
            year = source.get("year", "n.d.")
            if self.citation_style == "numbered":
                lines.append(f"{index}. {author} ({year}). {title}.")
            else:
                lines.append(f"- {author} ({year}). {title}.")
        return "\n".join(lines)

    def validate_citations(self, text: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Audit which supplied sources the text actually cites.

        Only two patterns are applied: the header format models copy from the
        context block, and the one matching the configured style. Scanning for
        every style at once produces false fabrications -- see
        :data:`_STYLE_PATTERNS`.

        Returns:
            Counts plus ``uncited_sources`` and ``unknown_citations`` --
            citations naming a source that was never supplied, which is the
            signature of a fabricated reference.
        """
        patterns = [_SOURCE_TAG_PATTERN]
        style_pattern = _STYLE_PATTERNS.get(self.citation_style)
        if style_pattern is not None:
            patterns.append(style_pattern)

        extracted: list[str] = []
        for pattern in patterns:
            extracted.extend(pattern.findall(text))

        cited_titles: set[str] = set()
        for citation in extracted:
            # A paragraph citing several papers merges them into one group:
            # "[A, 2017; B, 2019]". Splitting on ";" first is what makes both
            # titles visible -- without it only the first was ever counted, and
            # coverage read 0.5 on answers that had cited everything.
            for part in citation.split(";"):
                # "[Title, 2017]" carries a year to strip; "[Source 1: Title]"
                # does not, and a bare "[1]" carries no title at all.
                title = (part.split(",")[0] if "," in part else part).strip("[]() ").lower()
                if title and not title.isdigit():
                    cited_titles.add(title)

        available = {
            str(source.get("title", "")).lower() for source in sources if source.get("title")
        }

        return {
            "total_citations": len(extracted),
            "unique_sources_cited": len(cited_titles),
            "available_sources": len(available),
            "uncited_sources": sorted(available - cited_titles),
            "unknown_citations": sorted(cited_titles - available),
            "citation_coverage": (
                round(len(cited_titles & available) / len(available), 3) if available else 0.0
            ),
        }

    def reset(self) -> None:
        """Clear numbering state between independent answers."""
        self.citations_used.clear()
