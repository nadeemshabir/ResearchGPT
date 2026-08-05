"""Prompt templates for every generation task.

All prompts share one rule: answer only from the supplied excerpts, and say so
when they are insufficient. Grounding is enforced here rather than left to the
model's judgement, because ungrounded answers are the failure mode this whole
system exists to prevent.
"""

from __future__ import annotations

from typing import Any

#: Marker the critic must emit before its revised answer. Parsed by
#: :meth:`~src.generation.agents.CriticAgent._split_critique`.
IMPROVED_ANSWER_MARKER = "IMPROVED ANSWER:"

_GROUNDING_RULE = (
    "Answer only from the provided excerpts. If they do not contain enough "
    "information, say so explicitly instead of filling the gap from general "
    "knowledge."
)


class PromptTemplates:
    """Prompt builders, grouped as static methods."""

    SYSTEM_PROMPTS: dict[str, str] = {
        "qa": (
            "You are a research assistant answering questions about academic "
            "papers.\n\n" + _GROUNDING_RULE
        ),
        "analyzer": (
            "You are a research paper analyzer. Extract key information from "
            "paper excerpts.\n\n"
            "Focus on:\n"
            "- Main findings and conclusions\n"
            "- Methodologies used\n"
            "- Key technical terms and concepts\n"
            "- Quantitative results and metrics\n\n"
            "Be concise and factual. Extract only what is present in the text; "
            "never infer or embellish."
        ),
        "synthesizer": (
            "You are a research synthesis expert. Combine information from "
            "multiple sources into one coherent answer.\n\n"
            "Guidelines:\n"
            "- Organise ideas logically and remove redundancy\n"
            "- Connect related concepts across sources\n"
            "- Preserve specific numbers, metrics, and terminology exactly\n\n"
            "Never introduce information absent from the sources."
        ),
        "citation_agent": (
            "You are a citation expert. Attach citations to claims.\n\n"
            "Rules:\n"
            "- Use the format [Paper Title, Year]\n"
            "- Cite specific claims, not general statements\n"
            "- Place the citation at the end of the sentence it supports\n"
            "- Never invent a source that is not in the supplied list\n\n"
            "Return the full text with citations added and nothing else."
        ),
        "critic": (
            "You are a quality critic improving draft answers.\n\n"
            "Check for:\n"
            "- Accuracy: every claim traceable to the sources\n"
            "- Completeness: the question is fully answered\n"
            "- Clarity and logical flow\n"
            "- Correctly formatted citations\n\n"
            f"Reply with your review, then the marker {IMPROVED_ANSWER_MARKER} "
            "followed by the full revised answer."
        ),
    }

    @staticmethod
    def build_qa_prompt(question: str, context: str, write_citations: bool = True) -> str:
        """Single-shot question answering over retrieved context.

        Args:
            question: The user's question.
            context: Retrieved excerpts, already formatted.
            write_citations: Ask the model to cite. Set ``False`` when citations
                are attached afterwards from chunk metadata -- the model then
                spends no tokens on citations that would only be stripped, and
                cannot invent a format the stripper has to chase.
        """
        citation_rule = (
            "4. Cite claims as [Paper Title, Year]."
            if write_citations
            else (
                "4. Do NOT write citations, source names, or bracketed references. "
                "Sources are attached automatically afterwards."
            )
        )
        return f"""Answer the question using the research paper excerpts below.

RESEARCH EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer only from the excerpts above.
2. Be comprehensive but concise.
3. Include specific findings, numbers, and metrics where present.
{citation_rule}
5. If the excerpts do not answer the question, say so plainly and stop.

ANSWER:"""

    @staticmethod
    def build_extraction_prompt(text: str, extract_type: str = "key_findings") -> str:
        """Pull one category of information out of a single passage."""
        guides = {
            "key_findings": "extract the main findings and conclusions.",
            "methodology": "extract the methodology, approach, or techniques used.",
            "results": "extract experimental results, metrics, and outcomes.",
            "contributions": "extract the key contributions and innovations.",
        }
        guide = guides.get(extract_type, "extract the key information.")

        return f"""From the text below, {guide}

Report only what the text states. If it contains nothing relevant, reply
exactly: NOTHING RELEVANT

TEXT:
{text}

EXTRACTED INFORMATION:"""

    @staticmethod
    def build_synthesis_prompt(
        question: str,
        extractions: list[dict[str, Any]],
        max_length: int = 300,
    ) -> str:
        """Merge per-source extractions into one answer."""
        blocks = [
            f"Source {index} ({item.get('source', f'Source {index}')}):\n"
            f"{item.get('content', '')}"
            for index, item in enumerate(extractions, 1)
        ]
        sources_text = "\n\n".join(blocks)

        return f"""Synthesise the information below into a single answer.

QUESTION: {question}

INFORMATION FROM SOURCES:
{sources_text}

INSTRUCTIONS:
1. Combine the sources into one coherent answer.
2. Remove redundancy but preserve specific numbers and terminology.
3. Stay under {max_length} words.
4. Use only the information above.
5. If the sources do not answer the question, say so plainly.

SYNTHESISED ANSWER:"""

    @staticmethod
    def build_citation_prompt(text: str, sources: list[dict[str, Any]]) -> str:
        """Add citations to an already-written answer."""
        sources_text = "\n".join(
            f"{index}. {source.get('title', 'Unknown')} "
            f"({source.get('author', 'Unknown')}, {source.get('year', 'n.d.')})"
            for index, source in enumerate(sources, 1)
        )

        return f"""Add citations to the text below using only the listed sources.

TEXT TO CITE:
{text}

AVAILABLE SOURCES:
{sources_text}

INSTRUCTIONS:
1. Cite specific claims as [Title, Year].
2. Do not cite general knowledge.
3. Never cite a source that is not listed above.
4. Leave the wording otherwise unchanged.

TEXT WITH CITATIONS:"""

    @staticmethod
    def build_critique_prompt(question: str, answer: str, sources: str) -> str:
        """Review a draft answer and produce a revision."""
        return f"""Review and improve the answer below.

QUESTION: {question}

CURRENT ANSWER:
{answer}

SOURCES:
{sources}

REVIEW CHECKLIST:
1. Accuracy: does every claim match the sources?
2. Completeness: is the question fully answered?
3. Clarity: is it easy to follow?
4. Citations: correctly formatted and placed?

Write your review first. Then write the marker {IMPROVED_ANSWER_MARKER} on its
own line, followed by the complete revised answer. Always include the marker,
even when the answer needs no changes.

REVIEW:"""

    @staticmethod
    def build_literature_review_prompt(
        topic: str,
        papers_summary: list[dict[str, Any]],
        max_length: int = 500,
    ) -> str:
        """Write a literature review across several paper summaries."""
        papers_text = "\n\n".join(
            f"{index}. {paper.get('title', 'Unknown')} "
            f"({paper.get('author', 'Unknown')}, {paper.get('year', 'n.d.')})\n"
            f"{paper.get('summary', '')}"
            for index, paper in enumerate(papers_summary, 1)
        )

        return f"""Write a literature review on: {topic}

PAPERS:
{papers_text}

INSTRUCTIONS:
1. Open with an overview of the research area.
2. Discuss the main approaches and methodologies.
3. Compare and contrast the papers.
4. Identify trends and open gaps.
5. Cite claims as [Title, Year].
6. Stay under {max_length} words.
7. Base the review only on the summaries above.

LITERATURE REVIEW:"""

    @staticmethod
    def build_comparison_prompt(
        items: list[str],
        context: str,
        aspects: list[str] | None = None,
    ) -> str:
        """Compare two or more concepts using retrieved context."""
        aspects_line = (
            f"\nCompare specifically on: {', '.join(aspects)}" if aspects else ""
        )

        return f"""Compare: {" vs ".join(items)}

CONTEXT:
{context}
{aspects_line}

INSTRUCTIONS:
1. Cover both similarities and differences.
2. Use specific details from the context.
3. Organise the comparison clearly.
4. Cite claims as [Title, Year].
5. If the context lacks information on one item, say so rather than guessing.

COMPARISON:"""
