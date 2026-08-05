"""Tests for deterministic paragraph-level citation placement.

The rules under test:

- citations go at the end of a paragraph, never after individual sentences
- a paragraph drawing on several papers gets all of them in one group
- a paragraph matching nothing well enough is left uncited, not guessed
"""

import re

from src.generation.citation_manager import CitationManager


def chunk(paper_id: str, title: str, year: str, text: str) -> dict:
    return {
        "text": text,
        "metadata": {"paper_id": paper_id, "title": title, "year": year, "author": "Someone"},
    }


ATTENTION = chunk(
    "attention_2017",
    "Attention is All you Need",
    "2017",
    "The Transformer uses multi-head self-attention instead of recurrence. "
    "Scaled dot-product attention divides by the square root of the key dimension.",
)
ALEXNET = chunk(
    "alexnet_2012",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "2012",
    "The convolutional network reached a top-5 error rate of 15.3 percent on ImageNet "
    "using ReLU nonlinearities and dropout regularisation.",
)


def citations_in(text: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", text)


def test_citation_lands_at_the_end_of_the_paragraph() -> None:
    manager = CitationManager("inline")
    answer = (
        "The Transformer replaces recurrence with multi-head self-attention. "
        "Scaled dot-product attention divides by the square root of the key dimension."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])

    assert cited.endswith("]")
    assert len(citations_in(cited)) == 1


def test_no_citation_appears_mid_paragraph() -> None:
    """Every sentence but the last must be followed by prose, not a bracket."""
    manager = CitationManager("inline")
    answer = (
        "The Transformer replaces recurrence with multi-head self-attention. "
        "Scaled dot-product attention divides by the square root of the key dimension."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])
    before_last_sentence = cited[: cited.rindex(".")]

    assert "[" not in before_last_sentence


def test_two_sources_are_grouped_into_one_bracket() -> None:
    manager = CitationManager("inline")
    answer = (
        "Multi-head self-attention replaced recurrence, while convolutional networks "
        "used ReLU nonlinearities and dropout regularisation to cut the top-5 error rate."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION, ALEXNET])
    groups = citations_in(cited)

    assert len(groups) == 1, "sources must share one bracket, not sit in separate ones"
    assert ";" in groups[0]
    assert "Attention is All you Need" in groups[0]
    assert "ImageNet Classification" in groups[0]


def test_each_paragraph_is_cited_separately() -> None:
    manager = CitationManager("inline")
    answer = (
        "Multi-head self-attention replaced recurrence in the Transformer.\n\n"
        "Convolutional networks used ReLU nonlinearities and dropout regularisation."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION, ALEXNET])
    first, second = cited.split("\n\n")

    assert "Attention is All you Need" in citations_in(first)[0]
    assert "ImageNet Classification" in citations_in(second)[0]


def test_unmatched_paragraph_is_left_uncited() -> None:
    manager = CitationManager("inline")
    answer = "Rainfall patterns in coastal wetlands vary seasonally with monsoon cycles."

    cited = manager.add_paragraph_citations(answer, [ATTENTION, ALEXNET])

    assert citations_in(cited) == []


def test_one_paper_split_across_chunks_is_cited_once() -> None:
    manager = CitationManager("inline")
    second_chunk = chunk(
        "attention_2017",
        "Attention is All you Need",
        "2017",
        "Multi-head self-attention lets the model attend to several representation subspaces.",
    )
    answer = "The Transformer uses multi-head self-attention instead of recurrence."

    cited = manager.add_paragraph_citations(answer, [ATTENTION, second_chunk])

    assert len(citations_in(cited)) == 1
    assert ";" not in citations_in(cited)[0]


def test_headings_and_bullet_lists_are_not_cited() -> None:
    manager = CitationManager("inline")
    answer = (
        "## Summary\n\n"
        "- multi-head self-attention\n"
        "- scaled dot-product attention\n\n"
        "The Transformer replaces recurrence with multi-head self-attention entirely."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])
    heading, bullets, prose = cited.split("\n\n")

    assert citations_in(heading) == []
    assert citations_in(bullets) == []
    assert len(citations_in(prose)) == 1


def test_source_cap_is_respected() -> None:
    manager = CitationManager("inline")
    answer = (
        "Multi-head self-attention, ReLU nonlinearities, dropout regularisation, "
        "and scaled dot-product attention all appear in the top-5 error discussion."
    )
    many = [ATTENTION, ALEXNET, chunk("c", "Third Paper", "2020", answer)]

    cited = manager.add_paragraph_citations(answer, many, max_per_paragraph=2)

    assert citations_in(cited)[0].count(";") == 1


def test_citations_written_by_the_model_are_replaced_not_duplicated() -> None:
    """Models copy the ``[Source N: ...]`` header from the context block.

    Leaving those in produced visible duplicates:
    ``... values [Attention is All you Need]. [Attention is All you Need, 2017]``
    """
    manager = CitationManager("inline")
    answer = (
        "The Transformer replaces recurrence with multi-head self-attention "
        "and scaled dot-product attention "
        "[Source 1: Attention is All you Need | Section: Model Architecture]."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])

    assert len(citations_in(cited)) == 1
    assert "Source 1:" not in cited


def test_invented_citation_formats_are_still_stripped() -> None:
    """Models do not stick to the format they are given.

    One run produced both "[Attention is All you Need, Undated]" and
    "[... | Section: Model, Figure 2]". Matching on the known title catches
    every variant; matching on format would not.
    """
    manager = CitationManager("inline")
    answer = (
        "The Transformer replaces recurrence with multi-head self-attention "
        "[Attention is All you Need, Undated]. Scaled dot-product attention divides "
        "by the square root of the key dimension "
        "[Attention is All you Need, Section: Model Architecture, Figure 2]."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])

    assert len(citations_in(cited)) == 1
    assert "Undated" not in cited
    assert "Figure 2" not in cited


def test_non_citation_brackets_are_preserved() -> None:
    """Bracketed text that names no retrieved source is ordinary content."""
    manager = CitationManager("inline")
    answer = (
        "The Transformer uses multi-head self-attention [see the appendix for the "
        "full derivation] instead of recurrence."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION])

    assert "[see the appendix for the full derivation]" in cited


def test_year_is_recovered_from_the_paper_id() -> None:
    """PDF metadata rarely has a year, but the paper id ends in one."""
    manager = CitationManager("inline")
    answer = "The Transformer uses multi-head self-attention instead of recurrence."

    cited = manager.add_paragraph_citations(answer, [ATTENTION])

    assert "2017" in cited
    assert "n.d." not in cited


def test_audit_counts_every_source_in_a_merged_group() -> None:
    """"[A, 2017; B, 2012]" cites two papers, not one.

    Splitting only on "," saw the first title and reported coverage 0.5 on
    answers that had in fact cited everything retrieved.
    """
    manager = CitationManager("inline")
    answer = (
        "Multi-head self-attention replaced recurrence, while convolutional networks "
        "used ReLU nonlinearities and dropout regularisation to cut the top-5 error rate."
    )

    cited = manager.add_paragraph_citations(answer, [ATTENTION, ALEXNET])
    audit = manager.validate_citations(
        cited,
        [{"title": ATTENTION["metadata"]["title"]}, {"title": ALEXNET["metadata"]["title"]}],
    )

    assert audit["unique_sources_cited"] == 2
    assert audit["citation_coverage"] == 1.0
    assert audit["uncited_sources"] == []


def test_returns_text_unchanged_when_no_chunks_supplied() -> None:
    manager = CitationManager("inline")

    assert manager.add_paragraph_citations("Some answer.", []) == "Some answer."


def test_citations_only_name_retrieved_sources() -> None:
    """The whole point of doing this in code: fabrication is impossible."""
    manager = CitationManager("inline")
    answer = "The Transformer uses multi-head self-attention instead of recurrence."

    cited = manager.add_paragraph_citations(answer, [ATTENTION])
    audit = manager.validate_citations(cited, [{"title": ATTENTION["metadata"]["title"]}])

    assert audit["unknown_citations"] == []
    assert audit["citation_coverage"] == 1.0
