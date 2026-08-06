"""Tests for detecting refusals written in prose.

The system refuses in two places. Retrieval raises; the model writes a
sentence. Only the first was ever recorded, so correct prose refusals were
counted as fabricated answers -- 5 out of 5 on the first real run.

The strings below are real model output from that run.
"""

from src.generation.refusal_detection import looks_like_refusal


def test_detects_the_standard_refusal() -> None:
    assert looks_like_refusal(
        "The provided excerpts do not contain information about the accuracy "
        "GPT-6 reached on the MMLU benchmark."
    )


def test_detects_refusal_carrying_citations() -> None:
    """Citations must not push a one-line refusal over the length limit."""
    assert looks_like_refusal(
        "The provided excerpts do not contain information about the carbon "
        "footprint of training AlexNet in kilograms. "
        "[ImageNet Classification with Deep Convolutional Neural Networks, 2012; "
        "The Llama 3 Herd of Models, 2024; BERT: Pre-training of Deep Bidirectional "
        "Transformers for Language Understanding, 2019]"
    )


def test_detects_not_mentioned_phrasing() -> None:
    assert looks_like_refusal(
        "The specific learning rate is not explicitly mentioned in the excerpts."
    )


def test_detects_no_information_phrasing() -> None:
    assert looks_like_refusal("There is no information in the provided papers about this.")


def test_detects_cannot_answer_phrasing() -> None:
    assert looks_like_refusal(
        "I cannot determine the answer from the provided excerpts."
    )


def test_detects_the_harness_marker() -> None:
    assert looks_like_refusal("REFUSED: no relevant context found.")


def test_empty_answer_counts_as_refusal() -> None:
    assert looks_like_refusal("")
    assert looks_like_refusal("   \n  ")


def test_detects_refusal_where_a_pronoun_stands_for_the_sources() -> None:
    """Real output that the first version of the detector missed.

    The refusal verb attaches to "they", not to "the excerpts", so a pattern
    requiring the source word adjacent to the verb never fired.
    """
    assert looks_like_refusal(
        "The provided excerpts discuss BERT's performance on SuperGLUE, noting that "
        "the BERT-Large reference model and BERT++ were fine-tuned on the SuperGLUE "
        "training set. However, they do not provide a specific score for BERT on the "
        "SuperGLUE leaderboard. The excerpts mention BERT-Large's score on the GLUE "
        "leaderboard (80.5), but not SuperGLUE."
    )


def test_a_real_answer_is_not_a_refusal() -> None:
    assert not looks_like_refusal(
        "Scaled dot-product attention divides the dot products by the square root "
        "of d_k. Without this scaling, for large d_k the dot products grow large "
        "in magnitude and push softmax into regions with small gradients."
    )


def test_a_scientific_negative_finding_is_not_a_refusal() -> None:
    """"No evidence that X causes Y" is a result, not a refusal.

    This is why every pattern requires a source word: without it, papers
    reporting null results would be misread as declining to answer.
    """
    assert not looks_like_refusal(
        "The authors found no evidence that increasing model depth beyond 16 "
        "layers improves accuracy, and report that gains do not appear at any "
        "scale they tested."
    )


def test_a_partial_answer_is_not_a_refusal() -> None:
    """A disclaimer followed by a real answer is an answer.

    Real output: the model said SuperGLUE scores were absent, then supplied the
    GLUE score and the surrounding detail. Scoring that as a refusal would hide
    a genuine hallucination risk.
    """
    assert not looks_like_refusal(
        "The provided excerpts confirm that BERT models were fine-tuned and "
        "evaluated on SuperGLUE. However, they do not explicitly state BERT's "
        "specific score on the SuperGLUE leaderboard. A score of 80.5 is "
        "mentioned for BERTLARGE, but this is for the GLUE leaderboard, not "
        "SuperGLUE. The GLUE benchmark covers nine tasks including MNLI, QQP, "
        "QNLI, SST-2, CoLA, STS-B, MRPC and RTE, and BERTLARGE improved on the "
        "prior state of the art across all of them by a substantial margin."
    )


def test_hedging_inside_a_real_answer_is_not_a_refusal() -> None:
    assert not looks_like_refusal(
        "The model does not contain a recurrence mechanism; it relies entirely "
        "on self-attention to draw global dependencies between input and output."
    )
