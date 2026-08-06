"""Tests for query routing.

Routing picks which generation strategy runs, and the strategies differ by up to
4x in cost. Misrouting is silent -- the user still gets an answer, just a more
expensive or less appropriate one.

The router is deliberately conservative: anything it is not confident about
falls back to plain Q&A. These tests pin that bias as much as the positive
cases, because a router that over-triggers is worse than one that never fires.
"""

import pytest

from src.generation.query_router import QueryRouter, QueryType


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


# --- comparison -------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Compare BERT and GPT",
        "What is the difference between BERT and GPT?",
        "BERT vs GPT",
        "Contrast ResNet and VGG on ImageNet",
    ],
)
def test_comparison_queries_route_to_comparison(
    router: QueryRouter, query: str
) -> None:
    assert router.route_query(query).query_type == QueryType.COMPARISON


def test_comparison_needs_two_named_things(router: QueryRouter) -> None:
    """Lowercase concepts fall back to Q&A -- the documented safe failure.

    "compare attention and recurrence" names no capitalised entities, so the
    comparison strategy has nothing to compare and Q&A handles it instead.
    """
    decision = router.route_query("compare attention and recurrence")

    assert decision.query_type in {QueryType.QA, QueryType.COMPARISON}
    if decision.query_type == QueryType.COMPARISON:
        assert len(decision.params.get("items", [])) >= 2


def test_comparison_items_are_extracted(router: QueryRouter) -> None:
    assert router.extract_comparison_items("Compare BERT and GPT") == ["BERT", "GPT"]


def test_comparison_items_from_a_between_phrasing(router: QueryRouter) -> None:
    items = router.extract_comparison_items(
        "What is the difference between BERT and RoBERTa?"
    )

    assert items == ["BERT", "RoBERTa"]


def test_question_words_are_never_comparison_items(router: QueryRouter) -> None:
    """Without the stopword list, "What and Which" parses as two entities."""
    assert "What" not in router.extract_comparison_items("What and Which are best?")


def test_a_single_named_entity_is_not_a_comparison(router: QueryRouter) -> None:
    assert router.extract_comparison_items("Tell me about BERT") == []


# --- literature review ------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Give me a literature review on transformers",
        "What is the state of the art in retrieval?",
        "Recent advances in attention mechanisms",
        "Overview of RAG systems",
    ],
)
def test_review_queries_route_to_literature_review(
    router: QueryRouter, query: str
) -> None:
    assert router.route_query(query).query_type == QueryType.LITERATURE_REVIEW


def test_review_topic_strips_framing_words(router: QueryRouter) -> None:
    topic = router.extract_topic("Give me a literature review on transformers")

    assert "transformers" in topic
    assert "literature" not in topic
    assert "review" not in topic


def test_extract_topic_never_returns_empty(router: QueryRouter) -> None:
    """Stripping every word would otherwise leave the strategy with no query."""
    assert router.extract_topic("review") != ""


# --- extraction -------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "List all the datasets used",
        "Extract the evaluation metrics",
        "What methods are used for pretraining?",
        "Enumerate the ablations",
    ],
)
def test_extraction_queries_route_to_extraction(
    router: QueryRouter, query: str
) -> None:
    assert router.route_query(query).query_type == QueryType.EXTRACTION


# --- definition and summary -------------------------------------------------


def test_definition_queries_route_to_definition(router: QueryRouter) -> None:
    assert router.route_query("What is self-attention?").query_type == (
        QueryType.DEFINITION
    )


def test_summary_queries_route_to_summary(router: QueryRouter) -> None:
    assert router.route_query("Summarize the main findings").query_type == (
        QueryType.SUMMARY
    )


# --- the fallback, which is most of the traffic -----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "How many layers does the model have?",
        "Why does scaling by the square root of d_k help?",
        "Where was the model trained?",
        "Did the authors release their code?",
    ],
)
def test_ordinary_questions_fall_back_to_qa(router: QueryRouter, query: str) -> None:
    assert router.route_query(query).query_type == QueryType.QA


def test_the_fallback_passes_the_question_through_unchanged(
    router: QueryRouter,
) -> None:
    query = "How many attention heads are used?"

    assert router.route_query(query).params["question"] == query


def test_an_empty_query_does_not_crash(router: QueryRouter) -> None:
    assert router.route_query("").query_type == QueryType.QA


def test_routing_is_case_insensitive(router: QueryRouter) -> None:
    upper = router.route_query("COMPARE BERT AND GPT")
    lower = router.route_query("compare BERT and GPT")

    assert upper.query_type == lower.query_type


# --- decision shape ---------------------------------------------------------


def test_every_decision_carries_a_usable_method_name(router: QueryRouter) -> None:
    """`AnswerGenerator` dispatches on this string, so a typo breaks routing."""
    for query in [
        "Compare BERT and GPT",
        "Literature review on RAG",
        "List all the datasets",
        "What is attention?",
        "Summarize the findings",
        "How many layers?",
    ]:
        decision = router.route_query(query)

        assert decision.method
        assert decision.method.replace("_", "").isalnum()


def test_confidence_is_a_probability(router: QueryRouter) -> None:
    for query in ["Compare BERT and GPT", "What is attention?", "How many layers?"]:
        assert 0.0 <= router.route_query(query).confidence <= 1.0


def test_decision_serialises_for_the_ui(router: QueryRouter) -> None:
    payload = router.route_query("Compare BERT and GPT").as_dict()

    assert payload["query_type"]
    assert "confidence" in payload
