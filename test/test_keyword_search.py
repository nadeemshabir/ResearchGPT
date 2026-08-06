"""Tests for BM25 tokenisation and index staleness.

Two things here have bitten this project:

- **Tokenisation.** BM25 matches literal tokens, so how a term is split decides
  whether "self-attention" can be found at all. Dense retrieval blurs exactly
  these terms together, which is the whole reason BM25 is in the pipeline.
- **Staleness.** The index was built once at construction. Papers uploaded
  during a session were invisible to keyword search until restart, so hybrid
  search silently degraded to dense-only on new documents.

The store is stubbed; no ChromaDB instance is created.
"""

from typing import Any

import pytest

from src.retrieval.keyword_search import KeywordSearcher, tokenize

DOCS = [
    "The Transformer uses multi-head self-attention instead of recurrence.",
    "BERT is pre-trained with a masked language model objective.",
    "AlexNet reached a top-5 error rate of 15.3 percent on ImageNet.",
]


class StubDatabase:
    """A vector store that serves a fixed corpus and can grow on demand."""

    def __init__(self, documents: list[str]):
        self.documents = list(documents)

    def count(self) -> int:
        return len(self.documents)

    def get_all(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "metadatas": [{"paper_id": f"p{i}"} for i in range(len(self.documents))],
            "ids": [f"c{i}" for i in range(len(self.documents))],
        }

    def add(self, text: str) -> None:
        self.documents.append(text)


@pytest.fixture
def searcher() -> KeywordSearcher:
    """Uses the shipped `min_bm25_score`.

    Setting it to 0.0 would keep documents BM25 scored at exactly zero, since
    the filter is `score < min_score`. The floor is what makes "no lexical
    match" return nothing instead of the whole corpus.
    """
    return KeywordSearcher(database=StubDatabase(DOCS), min_score=0.01)  # type: ignore[arg-type]


# --- tokenisation -----------------------------------------------------------


def test_hyphenated_terms_stay_whole() -> None:
    """ "self-attention" split into two tokens would stop matching the phrase.

    These are precisely the terms BM25 is in the pipeline to catch, since dense
    embeddings place "self-attention" and "attention" close together.
    """
    assert tokenize("self-attention") == ["self-attention"]
    assert tokenize("multi-head attention") == ["multi-head", "attention"]


def test_tokenisation_is_lowercased() -> None:
    assert tokenize("BERT Transformer") == ["bert", "transformer"]


def test_numbers_and_versions_survive() -> None:
    """Model names carry digits; dropping them makes "GPT-3" unfindable.

    Note that "Llama-3.1" becomes two tokens: the pattern joins on hyphens, not
    on dots. Acceptable, since "llama-3" still matches the useful part.
    """
    assert tokenize("GPT-3 and Llama-3.1") == ["gpt-3", "and", "llama-3", "1"]


def test_punctuation_is_dropped() -> None:
    assert tokenize("Hello, world! (again)") == ["hello", "world", "again"]


def test_a_trailing_hyphen_does_not_produce_an_empty_token() -> None:
    assert "" not in tokenize("state-of-the-art -- really")


def test_empty_text_yields_no_tokens() -> None:
    assert tokenize("") == []
    assert tokenize("!!! ???") == []


def test_unicode_maths_does_not_crash_tokenisation() -> None:
    assert tokenize("scaling by 1/√dk where α ≥ 0.5") == [
        "scaling",
        "by",
        "1",
        "dk",
        "where",
        "0",
        "5",
    ]


# --- search -----------------------------------------------------------------


def test_search_finds_the_lexically_matching_document(
    searcher: KeywordSearcher,
) -> None:
    results = searcher.search("self-attention recurrence", top_k=1)

    assert results
    assert "Transformer" in results[0]["text"]


def test_search_returns_at_most_top_k(searcher: KeywordSearcher) -> None:
    assert len(searcher.search("the", top_k=2)) <= 2


def test_results_carry_a_bm25_score(searcher: KeywordSearcher) -> None:
    """`_combine_weighted` reads this key; a rename would silently zero fusion."""
    results = searcher.search("masked language model", top_k=1)

    assert "bm25_score" in results[0]
    assert results[0]["bm25_score"] > 0


def test_results_are_ordered_best_first(searcher: KeywordSearcher) -> None:
    results = searcher.search("ImageNet error rate", top_k=3)
    scores = [r["bm25_score"] for r in results]

    assert scores == sorted(scores, reverse=True)


def test_a_query_matching_nothing_returns_no_results(
    searcher: KeywordSearcher,
) -> None:
    assert searcher.search("zebra quilting bassoon", top_k=5) == []


def test_an_empty_query_does_not_crash(searcher: KeywordSearcher) -> None:
    assert searcher.search("", top_k=5) == []


def test_results_carry_identifiers_for_fusion(searcher: KeywordSearcher) -> None:
    """Fusion merges the two result lists on `id`."""
    results = searcher.search("BERT", top_k=1)

    assert results[0]["id"]


# --- staleness --------------------------------------------------------------


def test_an_empty_corpus_searches_without_error() -> None:
    empty = KeywordSearcher(database=StubDatabase([]), min_score=0.01)  # type: ignore[arg-type]

    assert empty.search("anything", top_k=5) == []


def test_refresh_picks_up_documents_added_after_construction() -> None:
    """Regression: the index was built once, so uploads were invisible.

    Hybrid search then quietly became dense-only for any paper added during a
    session -- no error, just worse results.
    """
    database = StubDatabase(DOCS)
    searcher = KeywordSearcher(database=database, min_score=0.01)  # type: ignore[arg-type]

    assert searcher.search("photosynthesis chloroplast", top_k=5) == []

    database.add("Photosynthesis occurs in the chloroplast of plant cells.")
    searcher.refresh()

    assert searcher.search("photosynthesis chloroplast", top_k=5)


def test_refresh_is_a_no_op_when_the_count_is_unchanged() -> None:
    """Rebuilding on every query would make BM25 the slowest part of retrieval."""
    searcher = KeywordSearcher(database=StubDatabase(DOCS), min_score=0.01)  # type: ignore[arg-type]

    assert searcher.refresh() is False


def test_forced_refresh_rebuilds_regardless() -> None:
    searcher = KeywordSearcher(database=StubDatabase(DOCS), min_score=0.01)  # type: ignore[arg-type]

    assert searcher.refresh(force=True) is True
