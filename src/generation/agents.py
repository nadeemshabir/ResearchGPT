"""Multi-agent answer generation.

Four specialised stages run in sequence: extract from each chunk, synthesise
into one answer, attach citations, then critique and revise.

The pipeline issues roughly ``len(chunks) + 3`` LLM calls per question, so it
is several times slower and costlier than single-shot generation. Whether that
buys enough quality to justify the cost is exactly what the evaluation harness
in Milestone 2 is meant to settle; until then it stays configurable.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.exceptions import LLMAuthenticationError, LLMError
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import PromptTemplates
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """Common behaviour for pipeline stages."""

    role: str = "base"

    def __init__(self, llm_client: LLMClient, temperature: float | None = None):
        self.llm = llm_client
        self.temperature = temperature
        self.system_prompt = PromptTemplates.SYSTEM_PROMPTS.get(self.role, "")

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process()")

    def _generate(self, prompt: str) -> str:
        """Call the LLM with this agent's role and temperature."""
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )


class AnalyzerAgent(BaseAgent):
    """Extract the salient claims from each retrieved chunk."""

    role = "analyzer"

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Summarise each chunk down to its key findings.

        A chunk that fails to process is skipped, since the remaining chunks
        can still support an answer. Authentication failures are re-raised
        immediately: they will affect every chunk, so continuing wastes time
        and produces a misleadingly empty result.
        """
        chunks: list[dict[str, Any]] = input_data.get("chunks", [])
        query: str = input_data.get("query", "")

        extractions: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            if not text.strip():
                continue

            prompt = PromptTemplates.build_extraction_prompt(
                text=text, extract_type="key_findings"
            )
            try:
                content = self._generate(prompt)
            except LLMAuthenticationError:
                raise
            except LLMError as exc:
                logger.warning("Skipping chunk %d: %s", index, exc)
                continue

            metadata = chunk.get("metadata", {})
            extractions.append(
                {
                    "chunk_id": chunk.get("id"),
                    "source": metadata.get("title") or f"Source {index}",
                    "author": metadata.get("author", "Unknown"),
                    "year": metadata.get("year", "n.d."),
                    "section": metadata.get("section_title", "Unknown"),
                    "content": content,
                    "original_text": text,
                }
            )

        logger.info("Analyzer extracted from %d/%d chunks", len(extractions), len(chunks))
        return {"extractions": extractions, "query": query}


class SynthesizerAgent(BaseAgent):
    """Merge per-chunk extractions into one coherent answer."""

    role = "synthesizer"

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        extractions: list[dict[str, Any]] = input_data.get("extractions", [])
        query: str = input_data.get("query", "")

        if not extractions:
            return {
                "answer": "",
                "query": query,
                "extractions": [],
                "error": "no extractions to synthesise",
            }

        prompt = PromptTemplates.build_synthesis_prompt(
            question=query, extractions=extractions
        )
        answer = self._generate(prompt)

        logger.info("Synthesised answer (%d chars)", len(answer))
        return {"answer": answer, "query": query, "extractions": extractions}


class CitationAgent(BaseAgent):
    """Attach source citations to the synthesised answer."""

    role = "citation_agent"

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Add citations, falling back to the uncited answer on failure.

        Losing citations degrades the answer but does not invalidate it, so
        this stage never fails the request.
        """
        answer: str = input_data.get("answer", "")
        extractions: list[dict[str, Any]] = input_data.get("extractions", [])

        if not answer.strip() or not extractions:
            return {"cited_answer": answer, "sources": []}

        sources = [
            {
                "title": extraction.get("source", "Unknown"),
                "author": extraction.get("author", "Unknown"),
                "year": extraction.get("year", "n.d."),
            }
            for extraction in extractions
        ]

        try:
            cited = self._generate(
                PromptTemplates.build_citation_prompt(text=answer, sources=sources)
            )
        except LLMError as exc:
            logger.warning("Citation stage failed, returning uncited answer: %s", exc)
            return {"cited_answer": answer, "sources": sources}

        return {"cited_answer": cited, "sources": sources}


class CriticAgent(BaseAgent):
    """Review the answer and return a revised version."""

    role = "critic"

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Critique and revise, falling back to the input answer on failure."""
        answer: str = input_data.get("answer", "")
        query: str = input_data.get("query", "")
        sources: Any = input_data.get("sources", "")

        if not answer.strip():
            return {"improved_answer": answer, "critique": ""}

        try:
            response = self._generate(
                PromptTemplates.build_critique_prompt(
                    question=query, answer=answer, sources=str(sources)
                )
            )
        except LLMError as exc:
            logger.warning("Critique stage failed, keeping original answer: %s", exc)
            return {"improved_answer": answer, "critique": ""}

        improved, critique = self._split_critique(response, fallback=answer)
        return {"improved_answer": improved, "critique": critique, "original_answer": answer}

    @staticmethod
    def _split_critique(response: str, fallback: str) -> tuple[str, str]:
        """Separate the revised answer from the critique notes.

        The prompt asks for an ``IMPROVED ANSWER:`` marker. Models comply
        inconsistently, so a missing marker keeps the original answer rather
        than passing the whole critique off as the answer.
        """
        for marker in ("IMPROVED ANSWER:", "IMPROVED VERSION:", "IMPROVED:"):
            if marker in response:
                critique, _, improved = response.partition(marker)
                improved = improved.strip()
                if improved:
                    return improved, critique.strip()
        return fallback, response.strip()


class AgentOrchestrator:
    """Run the four-stage generation pipeline."""

    def __init__(self, llm_client: LLMClient | None = None):
        settings = get_settings()
        self.llm = llm_client or LLMClient()

        self.analyzer = AnalyzerAgent(self.llm, settings.analyzer_temperature)
        self.synthesizer = SynthesizerAgent(self.llm, settings.synthesizer_temperature)
        self.citation = CitationAgent(self.llm, settings.citation_temperature)
        self.critic = CriticAgent(self.llm, settings.critic_temperature)

        logger.debug("Multi-agent orchestrator ready")

    def generate_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        use_citations: bool = True,
        use_critique: bool = True,
    ) -> dict[str, Any]:
        """Produce an answer from retrieved chunks.

        Args:
            query: The user's question.
            chunks: Retrieved chunks, each with ``text`` and ``metadata``.
            use_citations: Run the citation stage.
            use_critique: Run the critique stage.

        Returns:
            The answer plus per-stage bookkeeping under ``stages``.
        """
        stages: list[str] = []

        analysis = self.analyzer.process({"chunks": chunks, "query": query})
        stages.append("analyze")
        extractions = analysis["extractions"]

        if not extractions:
            return {
                "answer": (
                    "I could not extract usable information from the retrieved "
                    "passages. This usually means the LLM provider is failing; "
                    "check the logs for details."
                ),
                "query": query,
                "sources": [],
                "num_chunks": len(chunks),
                "stages": stages,
                "error": "analyzer produced no extractions",
            }

        synthesis = self.synthesizer.process({"extractions": extractions, "query": query})
        stages.append("synthesize")
        answer = synthesis["answer"]

        if use_citations:
            citation_result = self.citation.process(
                {"answer": answer, "extractions": extractions}
            )
            answer = citation_result.get("cited_answer") or answer
            stages.append("cite")

        critique = ""
        if use_critique:
            critique_result = self.critic.process(
                {"answer": answer, "query": query, "sources": extractions}
            )
            answer = critique_result.get("improved_answer") or answer
            critique = critique_result.get("critique", "")
            stages.append("critique")

        logger.info("Multi-agent generation complete (stages: %s)", ", ".join(stages))
        return {
            "answer": answer,
            "query": query,
            "sources": extractions,
            "num_chunks": len(chunks),
            "stages": stages,
            "critique": critique,
        }

    def simple_generate(self, query: str, context: str) -> str:
        """Single-shot generation: one LLM call over the whole context.

        The baseline the multi-agent pipeline must beat to justify its cost.
        """
        return self.llm.generate(
            prompt=PromptTemplates.build_qa_prompt(query, context),
            system_prompt=PromptTemplates.SYSTEM_PROMPTS["qa"],
            temperature=get_settings().synthesizer_temperature,
        )
