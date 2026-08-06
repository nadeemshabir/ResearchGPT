"""End-to-end test of the answer path with a stubbed LLM.

No API is called and no model is downloaded beyond the embedder the retrieval
stack already needs. The point is to exercise the wiring between retrieval,
generation, refusal detection and citation attachment -- the seams where four of
this project's bugs lived:

- `top_k` was accepted by the UI and never passed through to retrieval
- `adaptive_search` mutated shared weights, so one query changed the next
- `metadata["refused"]` was missing on some paths, raising KeyError
- citations were attached to refusals

Each of those is a wiring failure that unit tests on the individual modules
would not have caught.
"""

from typing import Any

import pytest

from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_manager import CitationManager

CHUNKS = [
    {
        "id": "c1",
        "text": (
            "The Transformer uses multi-head self-attention instead of recurrence. "
            "Scaled dot-product attention divides by the square root of the key "
            "dimension to keep softmax gradients from vanishing."
        ),
        "metadata": {
            "paper_id": "attention_2017",
            "title": "Attention is All you Need",
            "section_title": "Model Architecture",
        },
        "similarity_score": 0.71,
        "hybrid_score": 0.88,
    },
    {
        "id": "c2",
        "text": (
            "The convolutional network reached a top-5 error rate of 15.3 percent "
            "on ImageNet using ReLU nonlinearities and dropout regularisation."
        ),
        "metadata": {
            "paper_id": "alexnet_2012",
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "section_title": "Results",
        },
        "similarity_score": 0.52,
        "hybrid_score": 0.61,
    },
]


class StubRetrieval:
    """Retrieval that records what it was asked for and returns fixed chunks."""

    use_reranking = False

    def __init__(self, chunks: list[dict[str, Any]] | None = None):
        self.chunks = CHUNKS if chunks is None else chunks
        self.calls: list[dict[str, Any]] = []

    def get_relevant_chunks(
        self, query: str, max_tokens: int | None = None, top_k: int = 5
    ) -> dict[str, Any]:
        self.calls.append({"query": query, "top_k": top_k, "max_tokens": max_tokens})
        chunks = self.chunks[:top_k]
        return {
            "chunks": chunks,
            "context": "\n\n".join(c["text"] for c in chunks),
            "metadata": {"retrieval": {"total_ms": 12}, "total_tokens": 120},
        }

    def get_stats(self) -> dict[str, Any]:
        return {}


class StubLLM:
    """LLM that returns a scripted reply and records the prompts it received."""

    provider = "stub"
    model = "stub-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        self.prompts.append(prompt)
        return self.reply

    def get_model_info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model}


def build(reply: str, chunks: list[dict[str, Any]] | None = None) -> AnswerGenerator:
    return AnswerGenerator(
        use_multi_agent=False,
        use_smart_routing=False,
        retrieval_system=StubRetrieval(chunks),  # type: ignore[arg-type]
        llm_client=StubLLM(reply),  # type: ignore[arg-type]
    )


ANSWER = (
    "The Transformer replaces recurrence with multi-head self-attention. "
    "Scaled dot-product attention divides by the square root of the key dimension."
)


def test_a_question_produces_an_answer_with_sources() -> None:
    response = build(ANSWER).answer_question("How does attention work?")

    assert response["answer"]
    assert response["sources"]
    assert response["metadata"]["refused"] is False


def test_top_k_reaches_retrieval() -> None:
    """Regression: the UI slider was accepted and then silently ignored."""
    generator = build(ANSWER)

    generator.answer_question("How does attention work?", top_k=1)

    assert generator.retrieval_system.calls[0]["top_k"] == 1  # type: ignore[attr-defined]


def test_top_k_actually_limits_the_sources_used() -> None:
    generator = build(ANSWER)

    response = generator.answer_question("How does attention work?", top_k=1)

    assert response["metadata"]["num_sources"] == 1


def test_citations_are_attached_from_metadata() -> None:
    response = build(ANSWER).answer_question("How does attention work?")

    assert "Attention is All you Need" in response["answer"]


def test_citations_name_only_retrieved_papers() -> None:
    """The guarantee that makes deterministic citing worth having."""
    response = build(ANSWER).answer_question("How does attention work?")
    audit = CitationManager().validate_citations(
        response["answer"], [{"title": c["metadata"]["title"]} for c in CHUNKS]
    )

    assert audit["unknown_citations"] == []


def test_metadata_always_carries_the_refused_key() -> None:
    """Regression: a missing key raised KeyError on the comparison path."""
    response = build(ANSWER).answer_question("How does attention work?")

    assert "refused" in response["metadata"]


def test_a_prose_refusal_is_recorded_as_a_refusal() -> None:
    """Regression: only retrieval-raised refusals were counted."""
    refusal = "The provided excerpts do not contain information about that."

    response = build(refusal).answer_question("What is the capital of France?")

    assert response["metadata"]["refused"] is True


def test_a_refusal_is_not_given_citations() -> None:
    """Citing three papers for "these papers say nothing" is nonsense."""
    refusal = "The provided excerpts do not contain information about that."

    response = build(refusal).answer_question("What is the capital of France?")

    assert "[" not in response["answer"]


def test_the_prompt_carries_the_retrieved_context() -> None:
    generator = build(ANSWER)

    generator.answer_question("How does attention work?")
    prompt = generator.llm_client.prompts[0]  # type: ignore[attr-defined]

    assert "multi-head self-attention" in prompt
    assert "How does attention work?" in prompt


def test_repeated_questions_do_not_affect_each_other() -> None:
    """Regression: adaptive_search mutated shared weight settings in place."""
    generator = build(ANSWER)

    first = generator.answer_question("How does attention work?", top_k=2)
    generator.answer_question("Something else entirely", top_k=1)
    third = generator.answer_question("How does attention work?", top_k=2)

    assert first["metadata"]["num_sources"] == third["metadata"]["num_sources"]
    assert first["answer"] == third["answer"]


def test_timings_are_recorded() -> None:
    metadata = build(ANSWER).answer_question("How does attention work?")["metadata"]

    assert metadata["processing_time"] >= 0
    assert metadata["generation_time"] >= 0
    assert metadata["model"] == "stub-model"


def test_stats_report_the_active_configuration() -> None:
    stats = build(ANSWER).get_stats()

    assert stats["use_multi_agent"] is False
    assert "use_deterministic_citations" in stats


@pytest.mark.parametrize("top_k", [1, 2])
def test_source_list_has_one_entry_per_paper(top_k: int) -> None:
    response = build(ANSWER).answer_question("How does attention work?", top_k=top_k)
    paper_ids = [s["paper_id"] for s in response["sources"]]

    assert len(paper_ids) == len(set(paper_ids))
