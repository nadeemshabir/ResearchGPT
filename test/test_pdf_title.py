"""Tests for title recovery when a PDF carries no ``/Title`` metadata.

LaTeX omits ``/Title`` unless the author loads ``hyperref`` with ``pdftitle``,
so most arXiv preprints have none. Measured on this project's corpus, 5 of 8
papers were affected, which left every citation of them naming "Unknown".

The cases below are the real first lines of those PDFs.
"""

from src.ingestion.pdf_parser import PDFParser

recover = PDFParser._title_from_first_page


def test_skips_arxiv_stamp_and_joins_a_wrapped_title() -> None:
    text = "\n".join(
        [
            "arXiv:2312.10997v5 [cs.CL] 27 Mar 2024",
            "Retrieval-Augmented Generation for Large",
            "Language Models: A Survey",
            "Yunfan Gao, Yun Xiong, Xinyu Gao",
            "Shanghai Research Institute",
        ]
    )

    assert recover(text) == "Retrieval-Augmented Generation for Large Language Models: A Survey"


def test_short_continuation_line_is_not_mistaken_for_authors() -> None:
    """ "Reinforcement Learning" has no function words but is still the title.

    An earlier heuristic treated any line without function words as the author
    block, which truncated this title at "...in LLMs via".
    """
    text = "\n".join(
        [
            "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via",
            "Reinforcement Learning",
            "research@deepseek.com",
        ]
    )

    assert recover(text).startswith(
        "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
    )


def test_stops_at_an_email_address() -> None:
    text = "The Llama 3 Herd of Models\nLlama Team, AI @ Meta\n"

    assert recover(text) == "The Llama 3 Herd of Models"


def test_stops_at_an_author_line_carrying_footnote_marks() -> None:
    text = "\n".join(
        [
            "Language Models are Few-Shot Learners",
            "Tom B. Brown∗ Benjamin Mann∗ Nick Ryder∗",
            "OpenAI",
        ]
    )

    assert recover(text) == "Language Models are Few-Shot Learners"


def test_stops_at_the_abstract_heading() -> None:
    text = "Attention Is All You Need\nAbstract\nThe dominant sequence transduction models..."

    assert recover(text) == "Attention Is All You Need"


def test_rejects_a_fragment_too_short_to_identify_a_paper() -> None:
    assert recover("Intro\n\nbody text follows") == ""


def test_rejects_empty_input() -> None:
    assert recover("") == ""


def test_does_not_run_away_into_body_text() -> None:
    """A page with no title-like heading must not return a paragraph."""
    text = "\n".join(f"body line {n} with plenty of words in it" for n in range(20))

    assert len(recover(text)) <= 200
