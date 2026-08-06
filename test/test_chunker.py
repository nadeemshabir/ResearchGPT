"""Tests for token-aware chunking and section detection.

`chunk_with_context` is the highest-risk function in ingestion: it decides what
a retrieved chunk contains, and a silent failure there degrades every downstream
metric without raising anything. One such failure shipped -- the pipeline passed
text through `clean_text()` (which collapses newlines) before chunking, and
since headings are matched per line, every document became a single section.
The regression test for that is `test_section_detection_needs_newlines`.
"""

import pytest

from src.exceptions import ChunkingError
from src.ingestion.chunker import TextChunker

PAPER = """Abstract
We present a method for doing the thing.

1. Introduction
Prior work did the thing badly. We do it well.

2. Methods
We use a transformer with 12 layers.

3. Results
Accuracy reached 94.2 percent on the benchmark.

References
[1] Someone et al.
"""


@pytest.fixture
def chunker() -> TextChunker:
    """Small chunks so tests exercise splitting without huge fixtures."""
    return TextChunker(chunk_size=50, chunk_overlap=10)


# --- construction -----------------------------------------------------------


def test_overlap_equal_to_size_is_rejected() -> None:
    """Chunking could not advance: each chunk would start where the last did."""
    with pytest.raises(ValueError, match="smaller than"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_overlap_larger_than_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_zero_chunk_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=0)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)


def test_zero_overlap_is_allowed() -> None:
    """Valid, if unusual: adjacent chunks share no context."""
    assert TextChunker(chunk_size=100, chunk_overlap=0).chunk_overlap == 0


# --- chunk_text -------------------------------------------------------------


def test_empty_text_raises(chunker: TextChunker) -> None:
    with pytest.raises(ChunkingError, match="empty"):
        chunker.chunk_text("")


def test_whitespace_only_text_raises(chunker: TextChunker) -> None:
    with pytest.raises(ChunkingError, match="empty"):
        chunker.chunk_text("   \n\n \t ")


def test_short_text_yields_exactly_one_chunk(chunker: TextChunker) -> None:
    chunks = chunker.chunk_text("A short sentence about transformers.")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["num_tokens"] > 0


def test_long_text_is_split(chunker: TextChunker) -> None:
    chunks = chunker.chunk_text(" ".join(["token"] * 400))

    assert len(chunks) > 1


def test_no_chunk_greatly_exceeds_the_size_limit(chunker: TextChunker) -> None:
    """The point of counting tokens is that chunks fit the embedding window.

    A small tolerance is allowed: the splitter will not break an atomic
    separator-free run, so a single long word can overshoot.
    """
    chunks = chunker.chunk_text(PAPER * 6)

    assert max(c["num_tokens"] for c in chunks) <= chunker.chunk_size * 1.5


def test_chunk_ids_are_sequential_from_start_index(chunker: TextChunker) -> None:
    """Callers chunk section by section and rely on ids staying unique."""
    chunks = chunker.chunk_text(" ".join(["token"] * 300), start_index=100)

    assert [c["chunk_id"] for c in chunks] == list(range(100, 100 + len(chunks)))


def test_metadata_is_copied_onto_every_chunk(chunker: TextChunker) -> None:
    chunks = chunker.chunk_text(" ".join(["token"] * 300), metadata={"paper_id": "x"})

    assert all(c["metadata"]["paper_id"] == "x" for c in chunks)


def test_metadata_chunk_id_matches_the_chunk(chunker: TextChunker) -> None:
    """A stale chunk_id inside metadata would misattribute retrieved text."""
    chunks = chunker.chunk_text(" ".join(["token"] * 300), metadata={"paper_id": "x"})

    assert all(c["metadata"]["chunk_id"] == c["chunk_id"] for c in chunks)


def test_section_title_defaults_when_not_supplied(chunker: TextChunker) -> None:
    assert chunker.chunk_text("Some text here.")[0]["section_title"] == "Full Text"


def test_unicode_survives_chunking(chunker: TextChunker) -> None:
    """Papers are full of maths and Greek; a mangled chunk is unretrievable."""
    text = "The scaling factor is 1/√dk where α ≥ 0.5 and β ∈ [0, 1]. " * 20

    chunks = chunker.chunk_text(text)

    assert "√" in "".join(c["text"] for c in chunks)
    assert "α" in "".join(c["text"] for c in chunks)


def test_token_count_is_not_a_character_count(chunker: TextChunker) -> None:
    """Guards against a refactor that swaps the length function for len()."""
    text = "internationalisation " * 10

    assert chunker.count_tokens(text) < len(text)


# --- section detection ------------------------------------------------------


def test_sections_are_detected_from_headings(chunker: TextChunker) -> None:
    titles = [s["title"] for s in chunker.split_sections(PAPER)]

    assert any("Introduction" in t for t in titles)
    assert any("Methods" in t for t in titles)
    assert any("Results" in t for t in titles)


def test_leading_section_numbers_are_stripped_from_titles(
    chunker: TextChunker,
) -> None:
    """"1. Introduction" and "Introduction" must produce the same title.

    Otherwise the same section in two papers gets two different labels and
    section-based filtering splits on formatting rather than meaning.
    """
    titles = [s["title"] for s in chunker.split_sections(PAPER)]

    assert "Introduction" in titles
    assert "1. Introduction" not in titles


@pytest.mark.parametrize(
    "line",
    ["Introduction", "1. Introduction", "1 Introduction", "IV. Introduction", "Introduction:"],
)
def test_heading_variants_normalise_to_one_title(
    chunker: TextChunker, line: str
) -> None:
    assert chunker._match_heading(line) == "Introduction"


def test_text_before_the_first_heading_is_kept(chunker: TextChunker) -> None:
    """Title and author blocks live above the first heading and are real content."""
    sections = chunker.split_sections("Some title text\n\nAbstract\nBody here.")

    assert sections[0]["title"] == "Preamble"
    assert "Some title text" in sections[0]["text"]


def test_prose_starting_with_a_heading_word_is_not_a_heading(
    chunker: TextChunker,
) -> None:
    """"Results were mixed across all twelve..." is a sentence, not a section."""
    long_line = (
        "Results from the experiments were mixed across all twelve benchmarks "
        "that we evaluated during the study."
    )

    assert chunker._match_heading(long_line) is None


def test_heading_word_inside_a_sentence_is_not_a_heading(chunker: TextChunker) -> None:
    assert chunker._match_heading("We discuss the results below.") is None


def test_section_detection_needs_newlines(chunker: TextChunker) -> None:
    """Regression: collapsing newlines before chunking hid every section.

    The pipeline used to call `clean_text()` -- which joins lines -- before
    chunking. Headings are matched per line, so every paper collapsed into one
    section and `section_title` was useless for the whole corpus.
    """
    with_newlines = chunker.split_sections(PAPER)
    collapsed = chunker.split_sections(" ".join(PAPER.split()))

    assert len(with_newlines) > 1
    assert len(collapsed) == 1, "collapsed text must not appear to have sections"


def test_chunk_with_context_tags_chunks_with_their_section(
    chunker: TextChunker,
) -> None:
    chunks = chunker.chunk_with_context(PAPER)
    titles = {c["section_title"] for c in chunks}

    assert len(titles) > 1
    assert "Full Text" not in titles


def test_chunk_with_context_ids_stay_unique_across_sections(
    chunker: TextChunker,
) -> None:
    """Each section restarts its own numbering unless start_index is threaded."""
    ids = [c["chunk_id"] for c in chunker.chunk_with_context(PAPER)]

    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_chunk_with_context_falls_back_when_no_sections(
    chunker: TextChunker,
) -> None:
    chunks = chunker.chunk_with_context("Just some prose with no headings at all.")

    assert len(chunks) == 1
    assert chunks[0]["section_title"] == "Full Text"


def test_a_heading_with_no_body_does_not_break_chunking(
    chunker: TextChunker,
) -> None:
    chunks = chunker.chunk_with_context("Abstract\n\nIntroduction\nReal content here.")

    assert chunks
    assert any("Real content" in c["text"] for c in chunks)


# --- summarise --------------------------------------------------------------


def test_summarise_is_safe_on_an_empty_list() -> None:
    """Called on failed ingestions, where dividing by zero would mask the error."""
    stats = TextChunker.summarise([])

    assert stats["num_chunks"] == 0
    assert stats["avg_tokens"] == 0.0


def test_summarise_counts_distinct_sections(chunker: TextChunker) -> None:
    stats = TextChunker.summarise(chunker.chunk_with_context(PAPER))

    assert stats["num_sections"] > 1
    assert stats["num_chunks"] == stats["num_chunks"]
    assert stats["min_tokens"] <= stats["avg_tokens"] <= stats["max_tokens"]
