"""Query routing.

Classifies a question so the generator can pick a strategy: a comparison needs
context on both items, a literature review needs breadth across papers, and a
plain question needs depth on one topic.

Routing is rule-based. That is a deliberate trade-off: it is instant and free
where an LLM classifier would add a round trip to every request, but it will
misfire on phrasings the patterns do not cover. Each decision carries a
confidence score, and low-confidence routes fall through to standard Q&A.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Query categories the system handles."""

    QA = "qa"
    COMPARISON = "comparison"
    LITERATURE_REVIEW = "review"
    EXTRACTION = "extraction"
    DEFINITION = "definition"
    SUMMARY = "summary"


#: Human-readable labels for the UI.
QUERY_TYPE_LABELS: dict[QueryType, str] = {
    QueryType.QA: "Question & Answer",
    QueryType.COMPARISON: "Comparison",
    QueryType.LITERATURE_REVIEW: "Literature Review",
    QueryType.EXTRACTION: "Information Extraction",
    QueryType.DEFINITION: "Definition",
    QueryType.SUMMARY: "Summary",
}


@dataclass
class RoutingDecision:
    """The outcome of routing one query."""

    query_type: QueryType
    method: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable name of the detected type."""
        return QUERY_TYPE_LABELS.get(self.query_type, "Unknown")

    def as_dict(self) -> dict[str, Any]:
        """Serialise for logging and API responses."""
        return {
            "query_type": self.query_type.value,
            "label": self.label,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "params": self.params,
        }


#: Confidence a route must reach to be selected, per type. Comparison and
#: review trigger expensive multi-search paths, so they are held to a higher
#: bar than definition, which is nearly free.
ROUTE_THRESHOLDS: dict[QueryType, float] = {
    QueryType.COMPARISON: 0.7,
    QueryType.LITERATURE_REVIEW: 0.7,
    QueryType.EXTRACTION: 0.7,
    QueryType.SUMMARY: 0.6,
    QueryType.DEFINITION: 0.5,
}

_COMPARISON_PATTERNS = tuple(
    re.compile(p) for p in (
        r"\b(compare|comparison|versus|vs\.?)\b",
        r"\b(difference|differ|different)\b.*\b(between|from)\b",
        r"\b(similar|similarity)\b.*\b(to|with)\b",
        r"\b(contrast|contrasting)\b",
        r"\bwhich is better\b",
    )
)

_REVIEW_PATTERNS = tuple(
    re.compile(p) for p in (
        r"\b(review|overview|survey)\b",
        r"\b(summarize|summarise|summary)\s+(all|the)\b",
        r"\bliterature review\b",
        r"\bstate of (the )?art\b",
        r"\bcurrent research\b",
        r"\brecent (work|research|advances|developments)\b",
        r"\bprogress in\b",
    )
)

_EXTRACTION_PATTERNS = tuple(
    re.compile(p) for p in (
        r"\bextract\b",
        r"\blist (all|the)\b",
        r"\bfind (all|the)\b.*\b(methods|techniques|approaches)\b",
        r"\bwhat (methods|techniques|approaches|algorithms)\b",
        r"\benumerate\b",
        r"\bidentify (all|the)\b",
    )
)

_DEFINITION_PATTERNS = tuple(
    re.compile(p) for p in (
        r"^\s*what (is|are)\b",
        r"^\s*define\b",
        r"^\s*definition of\b",
        r"\bmeaning of\b",
        r"^\s*explain\s+\w+\s*\??\s*$",
    )
)

_SUMMARY_PATTERNS = tuple(
    re.compile(p) for p in (
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bsummary of\b",
        r"\bmain (points|findings|contributions)\b",
        r"\bkey (points|findings|takeaways)\b",
        r"\bin (brief|short)\b",
    )
)

_COMPARISON_KEYWORDS = frozenset(
    {"compare", "comparison", "versus", "vs", "difference", "different",
     "differ", "contrast", "similar", "similarity", "better", "worse"}
)

_REVIEW_KEYWORDS = frozenset(
    {"review", "overview", "survey", "literature", "recent", "current",
     "progress", "advances", "developments"}
)

#: Words that look like proper nouns but are never the subject of a comparison.
_ITEM_STOPWORDS = frozenset(
    {"What", "Which", "How", "Why", "When", "Where", "Who", "Compare",
     "Contrast", "Explain", "Describe", "The", "This", "That", "These"}
)

_TOPIC_NOISE = (
    "literature review", "state of the art", "tell me about", "give me",
    "show me", "provide", "review", "overview", "survey", "summarize",
    "summarise", "summary", "recent", "current", "what is", "what are",
)


class QueryRouter:
    """Classify queries and choose a generation strategy."""

    def route_query(self, query: str) -> RoutingDecision:
        """Route ``query`` to the best-matching strategy.

        Candidates are scored independently and evaluated most-specific first;
        the first to clear its threshold wins. Anything else becomes Q&A.
        """
        normalised = query.lower().strip()

        candidates = (
            self._score_comparison(query, normalised),
            self._score_review(query, normalised),
            self._score_extraction(query, normalised),
            self._score_summary(query, normalised),
            self._score_definition(query, normalised),
        )

        for decision in candidates:
            if decision.confidence >= ROUTE_THRESHOLDS[decision.query_type]:
                return decision

        return RoutingDecision(
            query_type=QueryType.QA,
            method="answer_question",
            confidence=0.8,
            params={"question": query},
        )

    def _score_comparison(self, query: str, normalised: str) -> RoutingDecision:
        confidence = sum(0.3 for p in _COMPARISON_PATTERNS if p.search(normalised))

        items = self.extract_comparison_items(query)
        if len(items) >= 2:
            confidence += 0.5

        matched_keywords = sum(1 for kw in _COMPARISON_KEYWORDS if kw in normalised.split())
        confidence += min(matched_keywords * 0.1, 0.3)

        return RoutingDecision(
            query_type=QueryType.COMPARISON,
            method="compare_papers",
            confidence=min(confidence, 1.0),
            params={"items": items, "query": query},
        )

    def _score_review(self, query: str, normalised: str) -> RoutingDecision:
        confidence = sum(0.4 for p in _REVIEW_PATTERNS if p.search(normalised))
        matched = sum(1 for kw in _REVIEW_KEYWORDS if kw in normalised.split())
        confidence += min(matched * 0.15, 0.4)

        return RoutingDecision(
            query_type=QueryType.LITERATURE_REVIEW,
            method="generate_literature_review",
            confidence=min(confidence, 1.0),
            params={"topic": self.extract_topic(query), "query": query},
        )

    @staticmethod
    def _score_extraction(query: str, normalised: str) -> RoutingDecision:
        confidence = sum(0.5 for p in _EXTRACTION_PATTERNS if p.search(normalised))
        return RoutingDecision(
            query_type=QueryType.EXTRACTION,
            method="answer_question",
            confidence=min(confidence, 1.0),
            params={"question": query, "extract_mode": True},
        )

    @staticmethod
    def _score_summary(query: str, normalised: str) -> RoutingDecision:
        confidence = sum(0.4 for p in _SUMMARY_PATTERNS if p.search(normalised))
        return RoutingDecision(
            query_type=QueryType.SUMMARY,
            method="answer_question",
            confidence=min(confidence, 1.0),
            params={"question": query, "summary_mode": True},
        )

    @staticmethod
    def _score_definition(query: str, normalised: str) -> RoutingDecision:
        confidence = sum(0.3 for p in _DEFINITION_PATTERNS if p.search(normalised))
        # Short queries are usually term lookups ("BERT?", "self-attention").
        if len(query.split()) <= 5:
            confidence += 0.2
        return RoutingDecision(
            query_type=QueryType.DEFINITION,
            method="answer_question",
            confidence=min(confidence, 1.0),
            params={"question": query, "definition_mode": True},
        )

    @staticmethod
    def extract_comparison_items(query: str) -> list[str]:
        """Pull the two things being compared out of a query.

        Relies on capitalisation, which works for model and dataset names
        ("Compare BERT and GPT") but not for lowercase concepts
        ("compare attention and recurrence"). Those fall back to Q&A, which is
        the safe failure.
        """
        patterns = (
            r"\bbetween\s+([A-Z][A-Za-z0-9\-]*)\s+and\s+([A-Z][A-Za-z0-9\-]*)\b",
            r"\b([A-Z][A-Za-z0-9\-]*)\s+(?:vs\.?|versus)\s+([A-Z][A-Za-z0-9\-]*)\b",
            r"\b([A-Z][A-Za-z0-9\-]*)\s+and\s+([A-Z][A-Za-z0-9\-]*)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                first, second = match.group(1), match.group(2)
                if first not in _ITEM_STOPWORDS and second not in _ITEM_STOPWORDS:
                    return [first, second]

        candidates = [
            word
            for word in re.findall(r"\b[A-Z][A-Za-z0-9\-]+\b", query)
            if word not in _ITEM_STOPWORDS
        ]
        return candidates[:2] if len(candidates) >= 2 else []

    @staticmethod
    def extract_topic(query: str) -> str:
        """Strip framing words to leave the subject of a review request."""
        topic = query.lower()
        for phrase in _TOPIC_NOISE:
            topic = topic.replace(phrase, " ")
        topic = re.sub(r"\b(?:on|about|of|the|a|an|all|papers?)\b", " ", topic)
        topic = " ".join(topic.split()).strip(" ?.,")
        return topic or query

    @staticmethod
    def get_query_type_description(query_type: QueryType) -> str:
        """Human-readable label for a query type."""
        return QUERY_TYPE_LABELS.get(query_type, "Unknown")
