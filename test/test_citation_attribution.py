"""Tests for structured paragraph-to-chunk attribution.

`add_paragraph_citations` renders text. `attribute_paragraphs` returns the
mapping the renderer is built from, which is what a UI needs to link a claim
back to the passage behind it -- parsing citations out of the rendered string
would be guesswork.

Both must stay consistent: the text is produced *from* the structure, so any
divergence is a bug in one of them.
"""

from src.generation.citation_manager import CitationManager


def chunk(chunk_id: str, paper_id: str, title: str, year: str, text: str) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "section_title": "Model Architecture",
        },
    }


ATTENTION = chunk(
    "attention_2017_3",
    "attention_2017",
    "Attention is All you Need",
    "2017",
    "The Transformer uses multi-head self-attention instead of recurrence. "
    "Scaled dot-product attention divides by the square root of the key dimension.",
)
ALEXNET = chunk(
    "alexnet_2012_7",
    "alexnet_2012",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "2012",
    "The convolutional network reached a top-5 error rate of 15.3 percent on ImageNet "
    "using ReLU nonlinearities and dropout regularisation.",
)


def test_attribution_names_the_chunk_not_just_the_paper() -> None:
    """The chunk id is what makes highlighting possible."""
    manager = CitationManager()
    answer = "The Transformer replaces recurrence with multi-head self-attention."

    paragraphs = manager.attribute_paragraphs(answer, [ATTENTION])

    assert paragraphs[0].sources[0].chunk_id == "attention_2017_3"
    assert paragraphs[0].sources[0].paper_id == "attention_2017"


def test_attribution_carries_a_score() -> None:
    """So a client can show a weak attribution differently from a strong one."""
    manager = CitationManager()
    answer = "The Transformer replaces recurrence with multi-head self-attention."

    source = manager.attribute_paragraphs(answer, [ATTENTION])[0].sources[0]

    assert 0.0 < source.score <= 1.0


def test_each_paragraph_is_attributed_independently() -> None:
    manager = CitationManager()
    answer = (
        "Multi-head self-attention replaced recurrence in the Transformer.\n\n"
        "Convolutional networks used ReLU nonlinearities and dropout regularisation."
    )

    paragraphs = manager.attribute_paragraphs(answer, [ATTENTION, ALEXNET])

    assert paragraphs[0].sources[0].paper_id == "attention_2017"
    assert paragraphs[1].sources[0].paper_id == "alexnet_2012"


def test_sources_are_ordered_by_score() -> None:
    """A client rendering "and 2 others" should drop the weakest, not the last."""
    manager = CitationManager()
    answer = (
        "Multi-head self-attention replaced recurrence, while convolutional networks "
        "used ReLU nonlinearities and dropout to cut the top-5 error rate."
    )

    sources = manager.attribute_paragraphs(answer, [ATTENTION, ALEXNET])[0].sources
    scores = [s.score for s in sources]

    assert scores == sorted(scores, reverse=True)


def test_an_unmatched_paragraph_has_no_sources() -> None:
    manager = CitationManager()
    answer = "Rainfall patterns in coastal wetlands vary seasonally with monsoon cycles."

    assert manager.attribute_paragraphs(answer, [ATTENTION])[0].sources == []


def test_structural_paragraphs_are_marked_and_kept() -> None:
    """Kept in the list so the answer can be rebuilt in order from it alone."""
    manager = CitationManager()
    answer = (
        "## Summary\n\nThe Transformer replaces recurrence with multi-head self-attention entirely."
    )

    paragraphs = manager.attribute_paragraphs(answer, [ATTENTION])

    assert paragraphs[0].structural is True
    assert paragraphs[0].sources == []
    assert paragraphs[1].structural is False
    assert paragraphs[1].sources


def test_rendered_text_matches_the_structure() -> None:
    """The string is produced from the structure, so they cannot disagree."""
    manager = CitationManager()
    answer = (
        "Multi-head self-attention replaced recurrence in the Transformer.\n\n"
        "Convolutional networks used ReLU nonlinearities and dropout regularisation."
    )

    paragraphs = manager.attribute_paragraphs(answer, [ATTENTION, ALEXNET])
    rendered = manager.add_paragraph_citations(answer, [ATTENTION, ALEXNET])

    for paragraph in paragraphs:
        for source in paragraph.sources:
            assert source.title in rendered


def test_attribution_of_empty_text_is_empty() -> None:
    assert CitationManager().attribute_paragraphs("", [ATTENTION]) == []


def test_attribution_without_chunks_still_returns_paragraphs() -> None:
    """A client rendering from this list must not lose the answer body."""
    manager = CitationManager()

    paragraphs = manager.attribute_paragraphs("Some answer text here.", [])

    assert len(paragraphs) == 1
    assert paragraphs[0].text == "Some answer text here."
    assert paragraphs[0].sources == []


def test_model_written_citations_are_stripped_from_the_structure() -> None:
    """The paragraph text must be clean, or the UI shows a doubled citation."""
    manager = CitationManager()
    answer = (
        "The Transformer replaces recurrence with multi-head self-attention "
        "[Source 1: Attention is All you Need | Section: Model Architecture]."
    )

    text = manager.attribute_paragraphs(answer, [ATTENTION])[0].text

    assert "Source 1:" not in text
    assert "[" not in text


def test_a_hallucinated_citation_key_is_stripped() -> None:
    """Real output, and the reason this rule exists.

    Told *not* to cite, a model still emitted "[attention_2017_chunk_5]" -- an
    identifier it had never been shown, which happened to match this project's
    internal chunk-id format. Displayed verbatim it reads as a real reference,
    which is exactly what deterministic citing exists to prevent.
    """
    manager = CitationManager()
    answer = (
        "The attention function is computed as softmax of QK over the square "
        "root of d_k [attention_2017_chunk_5]."
    )

    text = manager.attribute_paragraphs(answer, [ATTENTION])[0].text

    assert "attention_2017_chunk_5" not in text
    assert "[" not in text


def test_bracketed_prose_survives_the_identifier_rule() -> None:
    """Only single tokens are treated as keys; prose has spaces and stays."""
    manager = CitationManager()
    answer = (
        "The Transformer uses multi-head self-attention [see the appendix for "
        "the derivation] rather than recurrence."
    )

    text = manager.attribute_paragraphs(answer, [ATTENTION])[0].text

    assert "[see the appendix for the derivation]" in text


def test_one_paper_across_two_chunks_is_attributed_once() -> None:
    manager = CitationManager()
    second = chunk(
        "attention_2017_9",
        "attention_2017",
        "Attention is All you Need",
        "2017",
        "Multi-head self-attention lets the model attend to several representation subspaces.",
    )
    answer = "The Transformer uses multi-head self-attention instead of recurrence."

    sources = manager.attribute_paragraphs(answer, [ATTENTION, second])[0].sources

    assert len(sources) == 1


def test_the_section_is_carried_through() -> None:
    """ "According to the Model Architecture section" is only possible with this."""
    manager = CitationManager()
    answer = "The Transformer replaces recurrence with multi-head self-attention."

    source = manager.attribute_paragraphs(answer, [ATTENTION])[0].sources[0]

    assert source.section == "Model Architecture"
