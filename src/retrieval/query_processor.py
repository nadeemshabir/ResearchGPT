"""Query cleaning, expansion, and intent detection.

Runs before retrieval to make queries more robust: strips punctuation that
would confuse BM25, optionally expands academic synonyms, and classifies the
question so downstream components can adapt.
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Function words that carry no retrieval signal.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }
)

#: Near-synonyms common in academic writing, used for query expansion.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "method": ("approach", "technique", "methodology"),
    "model": ("architecture", "framework", "system"),
    "result": ("outcome", "finding", "conclusion"),
    "improve": ("enhance", "optimize", "boost"),
    "use": ("utilize", "employ", "apply"),
    "show": ("demonstrate", "indicate", "reveal"),
    "propose": ("introduce", "present", "suggest"),
}

#: Extra terms appended when a query opens with a question word.
QUESTION_EXPANSIONS: dict[str, str] = {
    "what": "definition explanation",
    "how": "method process mechanism",
    "why": "reason cause rationale",
    "when": "time timeline history",
    "where": "location context application",
}

_QUESTION_WORDS = ("what", "how", "why", "when", "where", "who", "which")
_TECHNICAL_INDICATORS = ("algorithm", "model", "architecture", "function", "method")
_PHRASE_PLACEHOLDER = "\x00PHRASE\x00"


class QueryProcessor:
    """Clean and enrich user queries before they reach the retrievers."""

    @staticmethod
    def clean_query(query: str) -> str:
        """Strip punctuation and collapse whitespace, keeping hyphens.

        Hyphens are preserved so "self-attention" is not split into two terms.
        """
        if not query:
            return ""
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", query)
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def remove_stopwords(query: str, preserve_phrases: bool = True) -> str:
        """Drop stopwords, optionally leaving quoted phrases intact."""
        if not preserve_phrases or '"' not in query:
            return " ".join(w for w in query.split() if w.lower() not in STOPWORDS)

        phrases = re.findall(r'"([^"]*)"', query)
        masked = re.sub(r'"[^"]*"', _PHRASE_PLACEHOLDER, query)
        kept = [
            word
            for word in masked.split()
            if word == _PHRASE_PLACEHOLDER or word.lower() not in STOPWORDS
        ]
        result = " ".join(kept)
        for phrase in phrases:
            result = result.replace(_PHRASE_PLACEHOLDER, f'"{phrase}"', 1)
        return result

    @staticmethod
    def expand_query(query: str, max_expansions: int = 2) -> str:
        """Append synonyms for recognised academic terms."""
        expanded: list[str] = []
        for word in query.lower().split():
            expanded.append(word)
            expanded.extend(SYNONYMS.get(word, ())[:max_expansions])
        return " ".join(expanded)

    @staticmethod
    def extract_key_terms(query: str, top_n: int = 5) -> list[str]:
        """Return the most distinctive terms, longest first.

        Length is a crude proxy for specificity, but it reliably surfaces
        domain terms over function words in academic queries.
        """
        cleaned = QueryProcessor.clean_query(query)
        words = [w for w in cleaned.lower().split() if w not in STOPWORDS]
        # Sort by length descending, breaking ties by original order so the
        # result is deterministic.
        ordered = sorted(enumerate(words), key=lambda pair: (-len(pair[1]), pair[0]))
        return [word for _, word in ordered[:top_n]]

    @staticmethod
    def enhance_question(query: str) -> str:
        """Replace a leading question word with terms likely to appear in text.

        Papers rarely contain the word "why"; they contain "because" and
        "rationale". Swapping improves lexical recall.
        """
        lowered = query.lower()
        for question_word, expansion in QUESTION_EXPANSIONS.items():
            if lowered.startswith(question_word):
                remainder = query[len(question_word) :].strip()
                # Drop the copula left behind by removing the question word,
                # so "What is X" yields "X ..." rather than "is the X ...".
                remainder = re.sub(
                    r"^(?:is|are|was|were|does|do|did)\b\s*", "", remainder, flags=re.IGNORECASE
                )
                return f"{remainder} {expansion}".strip()
        return query

    @staticmethod
    def detect_intent(query: str) -> dict[str, Any]:
        """Classify what the user is asking for."""
        lowered = query.lower()

        intent: dict[str, Any] = {
            "type": "general",
            "is_question": False,
            "is_definition": False,
            "is_comparison": False,
            "is_howto": False,
            "has_technical_terms": any(t in lowered for t in _TECHNICAL_INDICATORS),
        }

        if lowered.startswith(_QUESTION_WORDS):
            intent["is_question"] = True
            intent["type"] = "question"

        if any(t in lowered for t in ("what is", "define", "definition of", "explain")):
            intent["is_definition"] = True
            intent["type"] = "definition"

        if lowered.startswith("how to") or "step" in lowered or "process" in lowered:
            intent["is_howto"] = True
            intent["type"] = "howto"

        # Checked last so it wins: a comparison is more specific than a
        # definition, and "what is the difference between X and Y" is both.
        if any(t in lowered for t in ("compare", "difference", "versus", " vs", "better")):
            intent["is_comparison"] = True
            intent["type"] = "comparison"

        return intent

    def generate_variations(self, query: str, num_variations: int = 3) -> list[str]:
        """Produce distinct rephrasings for multi-query retrieval."""
        variations = [query]

        for candidate in (
            self.remove_stopwords(query),
            self.expand_query(query),
            " ".join(self.extract_key_terms(query, top_n=3)),
        ):
            if len(variations) >= num_variations:
                break
            if candidate and candidate not in variations:
                variations.append(candidate)

        return variations[:num_variations]

    def process_query(
        self,
        query: str,
        remove_stops: bool = False,
        expand: bool = True,
        enhance_questions: bool = True,
    ) -> dict[str, Any]:
        """Run the full processing pipeline over a query.

        Returns:
            Mapping with ``original``, ``cleaned``, ``expanded``,
            ``variations``, ``key_terms``, and ``intent``.
        """
        cleaned = self.clean_query(query)
        intent = self.detect_intent(cleaned)

        if remove_stops:
            cleaned = self.remove_stopwords(cleaned)
        if enhance_questions and intent["is_question"]:
            cleaned = self.enhance_question(cleaned)

        expanded = self.expand_query(cleaned) if expand else cleaned

        result = {
            "original": query,
            "cleaned": cleaned,
            "expanded": expanded,
            "variations": self.generate_variations(cleaned),
            "key_terms": self.extract_key_terms(cleaned),
            "intent": intent,
        }

        logger.debug("Processed query %r -> intent=%s", query[:60], intent["type"])
        return result
