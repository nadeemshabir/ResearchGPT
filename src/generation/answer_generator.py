"""End-to-end answer generation: retrieve, generate, cite."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from src.config import get_settings
from src.exceptions import LLMError, NoRelevantContextError
from src.generation.agents import AgentOrchestrator
from src.generation.citation_manager import CitationManager, ParagraphAttribution
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import PromptTemplates
from src.generation.query_router import QueryRouter, QueryType
from src.generation.refusal_detection import looks_like_refusal
from src.retrieval.retrieval_system import RetrievalSystem
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Returned when retrieval finds nothing relevant. Phrased as a refusal so the
#: UI can surface it verbatim without implying the corpus contains an answer.
NO_CONTEXT_MESSAGE = (
    "I could not find anything relevant to that question in the papers you have "
    "uploaded. Try rephrasing it, or upload a paper that covers the topic."
)


class AnswerGenerator:
    """Answer questions over the indexed corpus."""

    def __init__(
        self,
        use_multi_agent: bool | None = None,
        use_citations: bool | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        use_smart_routing: bool | None = None,
        retrieval_system: RetrievalSystem | None = None,
        llm_client: LLMClient | None = None,
        use_deterministic_citations: bool | None = None,
    ):
        """
        Args:
            use_multi_agent: Run the four-stage pipeline instead of a single
                LLM call. Defaults to ``Settings.use_multi_agent``.
            use_citations: Attach citations. Defaults to ``Settings.use_citations``.
            llm_provider: Overrides ``Settings.llm_provider``.
            llm_model: Overrides the provider default model.
            use_smart_routing: Classify queries before answering.
            retrieval_system: Reuse a built retrieval stack.
            llm_client: Reuse a built LLM client.
            use_deterministic_citations: Attach citations in code from chunk
                metadata instead of relying on the model to write them.
                Defaults to ``Settings.use_deterministic_citations``.
        """
        settings = get_settings()
        self.use_multi_agent = (
            use_multi_agent if use_multi_agent is not None else settings.use_multi_agent
        )
        self.use_citations = use_citations if use_citations is not None else settings.use_citations
        self.use_smart_routing = (
            use_smart_routing if use_smart_routing is not None else settings.use_smart_routing
        )
        self.use_deterministic_citations = (
            use_deterministic_citations
            if use_deterministic_citations is not None
            else settings.use_deterministic_citations
        )

        logger.info(
            "Initialising answer generator (multi_agent=%s, citations=%s, routing=%s)",
            self.use_multi_agent,
            self.use_citations,
            self.use_smart_routing,
        )

        self.retrieval_system = retrieval_system or RetrievalSystem()
        self.llm_client = llm_client or LLMClient(provider=llm_provider, model=llm_model)

        self.query_router = QueryRouter() if self.use_smart_routing else None
        self.agent_orchestrator = (
            AgentOrchestrator(self.llm_client) if self.use_multi_agent else None
        )
        self.citation_manager = (
            CitationManager(settings.citation_style) if self.use_citations else None
        )

        logger.info("Answer generator ready")

    # ------------------------------------------------------------------
    # Core question answering
    # ------------------------------------------------------------------
    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """Answer one question from the corpus.

        Args:
            question: The user's question.
            top_k: Chunks to retrieve.
            max_context_tokens: Context budget for the LLM.
            include_sources: Attach formatted source records to the response.

        Returns:
            ``answer``, ``question``, ``sources``, and ``metadata``. When
            nothing relevant is indexed, ``answer`` is :data:`NO_CONTEXT_MESSAGE`
            and ``metadata["refused"]`` is ``True``.
        """
        started = time.perf_counter()

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            retrieval = self.retrieval_system.get_relevant_chunks(
                query=question, max_tokens=max_context_tokens, top_k=top_k
            )
        except NoRelevantContextError as exc:
            logger.info("Refusing to answer %r: %s", question[:60], exc)
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "question": question,
                "sources": [],
                "metadata": {
                    "refused": True,
                    "reason": "no_relevant_context",
                    # Named rather than left absent: callers reading
                    # generation_method would otherwise see "unknown" and be
                    # unable to tell a retrieval refusal from a missing field.
                    "generation_method": "refused_at_retrieval",
                    "candidates_considered": exc.candidates_considered,
                    "threshold": exc.threshold,
                    "num_sources": 0,
                    "processing_time": round(time.perf_counter() - started, 2),
                    "model": self.llm_client.model,
                },
            }

        chunks = retrieval["chunks"]
        context = retrieval["context"]

        generation_started = time.perf_counter()
        try:
            if self.use_multi_agent and self.agent_orchestrator:
                result = self.agent_orchestrator.generate_answer(
                    query=question,
                    chunks=chunks,
                    # The LLM citation stage is redundant when citations are
                    # attached deterministically afterwards, and running both
                    # double-cites every paragraph.
                    use_citations=self.use_citations and not self.use_deterministic_citations,
                    use_critique=True,
                )
                answer = result["answer"]
                stages = result.get("stages", [])
            else:
                answer = self.llm_client.generate(
                    prompt=PromptTemplates.build_qa_prompt(
                        question,
                        context,
                        write_citations=not self.use_deterministic_citations,
                    ),
                    system_prompt=PromptTemplates.SYSTEM_PROMPTS["qa"],
                )
                stages = ["single_shot"]
        except LLMError as exc:
            logger.error("Generation failed for %r: %s", question[:60], exc)
            raise

        generation_seconds = time.perf_counter() - generation_started

        # Retrieval refuses by raising; the model refuses in prose. Both are
        # correct refusals, but only the first used to be recorded, so silent
        # refusals were counted as fabricated answers.
        model_refused = looks_like_refusal(answer)

        attributions: list[ParagraphAttribution] = []
        if self.citation_manager and self.use_deterministic_citations and not model_refused:
            # Attach citations from chunk metadata rather than trusting the
            # model to write them. Sources come from a fixed list, so this
            # cannot fabricate a reference.
            #
            # Attribution runs once and both outputs derive from it: the
            # rendered string for readers, and the structure for clients that
            # need to link a claim back to the passage behind it.
            attributions = self.citation_manager.attribute_paragraphs(answer, chunks)
            answer = self._render_attributions(attributions)
            stages = [*stages, "paragraph_citations"]

        total_seconds = time.perf_counter() - started

        metadata: dict[str, Any] = {
            "refused": model_refused,
            "num_sources": len(chunks),
            "processing_time": round(total_seconds, 2),
            "retrieval_time": round(
                retrieval["metadata"]["retrieval"].get("total_ms", 0) / 1000, 2
            ),
            "generation_time": round(generation_seconds, 2),
            "total_tokens": retrieval["metadata"]["total_tokens"],
            "retrieval_method": "hybrid + rerank"
            if self.retrieval_system.use_reranking
            else "hybrid",
            "generation_method": "multi-agent" if self.use_multi_agent else "single-shot",
            "stages": stages,
            "used_citations": self.use_citations,
            "model": self.llm_client.model,
        }

        if self.citation_manager:
            metadata["citation_audit"] = self.citation_manager.validate_citations(
                answer, self._source_records(chunks)
            )

        logger.info(
            "Answered %r in %.2fs (%d sources, %s)",
            question[:60],
            total_seconds,
            len(chunks),
            metadata["generation_method"],
        )

        return {
            "answer": answer,
            "question": question,
            "sources": self._format_sources(chunks) if include_sources else [],
            # Paragraph-to-chunk links, so a client can highlight the passage
            # behind a claim without parsing the rendered citations back out.
            "paragraphs": [
                {
                    "text": item.text,
                    "structural": item.structural,
                    "sources": [asdict(source) for source in item.sources],
                }
                for item in attributions
            ],
            "chunks": self._chunk_records(chunks) if include_sources else [],
            "metadata": metadata,
        }

    @staticmethod
    def _render_attributions(attributions: list[ParagraphAttribution]) -> str:
        """Rebuild the answer text with one citation group per paragraph."""
        manager = CitationManager()
        rendered: list[str] = []
        for paragraph in attributions:
            if not paragraph.sources:
                rendered.append(paragraph.text)
                continue
            group = " ".join(
                manager.format_citation(
                    {"title": source.title, "year": source.year, "author": "Unknown"}
                )
                for source in paragraph.sources
            )
            rendered.append(f"{paragraph.text} {manager.merge_duplicate_citations(group)}")
        return "\n\n".join(rendered)

    @staticmethod
    def _chunk_records(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Full passages, keyed by the ids that appear in ``paragraphs``.

        The whole chunk text is returned, not a preview: a client highlighting
        a source needs the passage a reader will actually read.
        """
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            records.append(
                {
                    "chunk_id": str(chunk.get("id", "")),
                    "paper_id": metadata.get("paper_id", "unknown"),
                    "title": metadata.get("title") or metadata.get("paper_id") or "Unknown",
                    "section": metadata.get("section_title", "Unknown"),
                    "text": chunk.get("text", ""),
                    "relevance_score": round(
                        float(chunk.get("rerank_score", chunk.get("hybrid_score", 0.0))), 4
                    ),
                }
            )
        return records

    def smart_answer(self, user_query: str, **kwargs: Any) -> dict[str, Any]:
        """Route the query, then answer with the matching strategy.

        Falls back to standard Q&A when routing is disabled, or when a
        comparison route fires but the items could not be extracted.
        """
        if not self.use_smart_routing or not self.query_router:
            return self.answer_question(user_query, **kwargs)

        decision = self.query_router.route_query(user_query)
        logger.info(
            "Routed %r to %s (confidence %.2f)",
            user_query[:60],
            decision.label,
            decision.confidence,
        )

        if decision.query_type is QueryType.COMPARISON:
            items = decision.params.get("items", [])
            if len(items) < 2:
                logger.info("Comparison route lacked two items; falling back to Q&A")
                return self._with_routing_metadata(
                    self.answer_question(user_query, **kwargs), decision
                )
            response = self.compare_papers(items, aspects=kwargs.get("aspects"))
            coverage = response.get("coverage", {})
            return self._with_routing_metadata(
                {
                    "answer": response["comparison"],
                    "question": user_query,
                    "sources": response.get("sources", []),
                    "metadata": {
                        "num_sources": len(response.get("sources", [])),
                        # Nothing was found for any item, so no comparison was
                        # actually made.
                        "refused": bool(coverage) and not any(coverage.values()),
                        "coverage": coverage,
                    },
                },
                decision,
            )

        if decision.query_type is QueryType.LITERATURE_REVIEW:
            response = self.generate_literature_review(
                topic=decision.params.get("topic", user_query),
                max_papers=kwargs.get("max_papers", 10),
            )
            return self._with_routing_metadata(
                {
                    "answer": response["review"],
                    "question": user_query,
                    "sources": response["papers"],
                    "metadata": {
                        "num_sources": response["num_papers"],
                        "refused": response["num_papers"] == 0,
                    },
                },
                decision,
            )

        return self._with_routing_metadata(self.answer_question(user_query, **kwargs), decision)

    @staticmethod
    def _with_routing_metadata(response: dict[str, Any], decision: Any) -> dict[str, Any]:
        """Record the routing decision on a response.

        Also guarantees ``refused`` is present. Every branch of ``smart_answer``
        must return the same metadata shape, or callers have to guess which
        keys exist for which query type.
        """
        response.setdefault("metadata", {})
        response["metadata"].setdefault("refused", False)
        response["metadata"]["query_type"] = decision.query_type.value
        response["metadata"]["query_type_label"] = decision.label
        response["metadata"]["routing_confidence"] = round(decision.confidence, 3)
        return response

    # ------------------------------------------------------------------
    # Specialised strategies
    # ------------------------------------------------------------------
    def generate_literature_review(
        self,
        topic: str,
        max_papers: int = 10,
        max_length: int = 500,
    ) -> dict[str, Any]:
        """Summarise each relevant paper, then synthesise a review.

        Costs one LLM call per paper plus one for the review, so ``max_papers``
        directly sets the price of the request.
        """
        started = time.perf_counter()
        logger.info("Generating literature review on %r", topic)

        search = self.retrieval_system.search(topic, top_k=max_papers * 3)
        results = search["results"]

        if not results:
            return {
                "review": f"No indexed papers appear to cover '{topic}'.",
                "papers": [],
                "topic": topic,
                "num_papers": 0,
            }

        grouped: dict[str, dict[str, Any]] = {}
        for result in results:
            metadata = result["metadata"]
            paper_id = metadata.get("paper_id", "unknown")
            paper = grouped.setdefault(
                paper_id,
                {
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "year": metadata.get("year", "n.d."),
                    "chunks": [],
                },
            )
            paper["chunks"].append(result["text"])

        papers = list(grouped.values())[:max_papers]

        summaries: list[dict[str, Any]] = []
        for paper in papers:
            try:
                summary = self.llm_client.generate(
                    prompt=PromptTemplates.build_extraction_prompt(
                        text=" ".join(paper["chunks"][:3]),
                        extract_type="key_findings",
                    ),
                    system_prompt=PromptTemplates.SYSTEM_PROMPTS["analyzer"],
                    temperature=get_settings().analyzer_temperature,
                )
            except LLMError as exc:
                logger.warning("Skipping %r in review: %s", paper["title"], exc)
                continue
            summaries.append(
                {**{k: paper[k] for k in ("title", "author", "year")}, "summary": summary}
            )

        if not summaries:
            return {
                "review": f"Could not summarise any papers on '{topic}'; see logs.",
                "papers": [],
                "topic": topic,
                "num_papers": 0,
            }

        review = self.llm_client.generate(
            prompt=PromptTemplates.build_literature_review_prompt(
                topic=topic, papers_summary=summaries, max_length=max_length
            ),
            system_prompt=PromptTemplates.SYSTEM_PROMPTS["synthesizer"],
            temperature=get_settings().synthesizer_temperature,
        )

        logger.info(
            "Literature review over %d papers in %.2fs",
            len(summaries),
            time.perf_counter() - started,
        )
        return {
            "review": review,
            "papers": summaries,
            "topic": topic,
            "num_papers": len(summaries),
        }

    def compare_papers(
        self,
        items: list[str],
        aspects: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare concepts by retrieving context for each separately.

        Retrieving per item matters: a single query for "BERT vs GPT" tends to
        return chunks about whichever term dominates the corpus, leaving the
        other side of the comparison unsupported.
        """
        logger.info("Comparing %s", " vs ".join(items))

        all_chunks: list[dict[str, Any]] = []
        per_item_counts: dict[str, int] = {}
        for item in items:
            results = self.retrieval_system.search(item, top_k=5)["results"]
            per_item_counts[item] = len(results)
            all_chunks.extend(results)

        missing = [item for item, count in per_item_counts.items() if count == 0]
        if len(missing) == len(items):
            return {
                "comparison": (f"The indexed papers contain nothing about {' or '.join(items)}."),
                "items": items,
                "aspects": aspects,
                "sources": [],
                "coverage": per_item_counts,
            }

        context = "\n\n".join(
            f"[Source: {chunk['metadata'].get('title', 'Unknown')}]\n{chunk['text']}"
            for chunk in all_chunks
        )

        comparison = self.llm_client.generate(
            prompt=PromptTemplates.build_comparison_prompt(
                items=items, context=context, aspects=aspects
            ),
            system_prompt=PromptTemplates.SYSTEM_PROMPTS["synthesizer"],
        )

        if missing:
            comparison = (
                f"_Note: no indexed content was found for {', '.join(missing)}, "
                f"so that side of the comparison is unsupported._\n\n{comparison}"
            )

        return {
            "comparison": comparison,
            "items": items,
            "aspects": aspects,
            "sources": self._format_sources(all_chunks),
            "coverage": per_item_counts,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _source_records(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Minimal source dicts for citation auditing."""
        return [
            {
                "title": chunk.get("metadata", {}).get("title", "Unknown"),
                "author": chunk.get("metadata", {}).get("author", "Unknown"),
                "year": chunk.get("metadata", {}).get("year", "n.d."),
            }
            for chunk in chunks
        ]

    @staticmethod
    def _format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One record per distinct paper, keeping its best-scoring chunk."""
        by_paper: dict[str, dict[str, Any]] = {}

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            paper_id = metadata.get("paper_id", "unknown")
            score = chunk.get("rerank_score", chunk.get("hybrid_score", 0.0))

            existing = by_paper.get(paper_id)
            if existing is None or score > existing["relevance_score"]:
                by_paper[paper_id] = {
                    "paper_id": paper_id,
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "section": metadata.get("section_title", "Unknown"),
                    "chunk_preview": chunk.get("text", "")[:200].rstrip() + "...",
                    "relevance_score": round(float(score), 4),
                }

        return sorted(by_paper.values(), key=lambda s: s["relevance_score"], reverse=True)

    def get_stats(self) -> dict[str, Any]:
        """System configuration, for the UI and health checks."""
        return {
            "retrieval": self.retrieval_system.get_stats(),
            "llm": self.llm_client.get_model_info(),
            "use_multi_agent": self.use_multi_agent,
            "use_citations": self.use_citations,
            "use_deterministic_citations": self.use_deterministic_citations,
            "use_smart_routing": self.use_smart_routing,
        }
