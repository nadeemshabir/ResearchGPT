"""Detect when an answer is really a refusal.

The system refuses in two different places, and only one of them was being
counted:

1. **Retrieval** raises :class:`~src.exceptions.NoRelevantContextError` when
   nothing clears ``min_semantic_similarity``. Loud, easy to detect.
2. **The model** writes "the provided excerpts do not contain information
   about ..." when retrieval passed weak context through. Silent.

Case 2 is a correct refusal, but nothing recorded it: ``metadata["refused"]``
stayed ``False``, so evaluation counted these as fabricated answers. Measured on
five unanswerable questions, all five were refused in prose and all five were
scored as failures.

Detection is deliberately narrow. A phrase must say the *sources* lack the
information -- "there is no evidence that X causes Y" is a finding, not a
refusal, and must not be swept up.
"""

from __future__ import annotations

import re

#: Phrases that mean "the supplied sources do not answer this".
#:
#: Each requires a source word (excerpt, context, passage, paper, document,
#: text, source) so that ordinary hedging inside a real answer is not matched.
_SOURCE = r"(?:excerpts?|contexts?|passages?|papers?|documents?|texts?|sources?)"

_REFUSAL_PATTERNS = (
    # "the provided excerpts do not contain / mention / discuss / provide ..."
    re.compile(
        rf"\b(?:the\s+)?(?:provided\s+|given\s+|supplied\s+|available\s+)?{_SOURCE}\b"
        rf"[^.!?]{{0,80}}?\b(?:do|does)\s+not\s+"
        rf"(?:contain|mention|discuss|provide|include|specify|state|report|address)",
        re.IGNORECASE,
    ),
    # "there is no information in the excerpts ..."
    re.compile(
        rf"\bno\s+(?:information|data|details?|mention|reference)\b[^.!?]{{0,60}}?\b{_SOURCE}\b",
        re.IGNORECASE,
    ),
    # "... is not mentioned / stated / found in the provided excerpts"
    re.compile(
        rf"\b(?:not|never)\s+(?:explicitly\s+)?"
        rf"(?:mentioned|stated|specified|discussed|found|reported|provided|included)\b"
        rf"[^.!?]{{0,60}}?\b{_SOURCE}\b",
        re.IGNORECASE,
    ),
    # "I cannot answer this from the excerpts"
    re.compile(
        rf"\b(?:cannot|can't|unable to)\s+(?:be\s+)?(?:answer|determine|find|establish)"
        rf"[^.!?]{{0,60}}?\b{_SOURCE}\b",
        re.IGNORECASE,
    ),
    # The harness's own marker, so eval records round-trip.
    re.compile(r"^\s*REFUSED:", re.IGNORECASE),
)

#: Same refusal, but with a pronoun standing in for the sources: "The excerpts
#: discuss X. However, **they** do not provide a specific score." Only trusted
#: when a source word appears somewhere in the text, so that "it does not
#: contain a recurrence mechanism" -- a claim about an architecture -- is left
#: alone.
_PRONOUN_REFUSAL = re.compile(
    r"\b(?:they|these|those)\s+(?:do|does)\s+not\s+"
    r"(?:contain|mention|discuss|provide|include|specify|state|report)",
    re.IGNORECASE,
)
_MENTIONS_SOURCE = re.compile(rf"\b{_SOURCE}\b", re.IGNORECASE)

#: A refusal that then answers anyway is not a refusal. When the text runs on
#: well past the disclaimer, it is a partial answer and is treated as one.
#:
#: Set from observed output rather than picked round: the longest genuine
#: refusal seen was 431 characters ("the excerpts discuss X ... however they do
#: not provide Y ... they mention Z instead"), which explains the headroom over
#: the one-line case.
_MAX_REFUSAL_CHARS = 500


def looks_like_refusal(text: str) -> bool:
    """True when ``text`` declines to answer because the sources fall short.

    Args:
        text: A generated answer.

    Returns:
        Whether the answer is a refusal rather than an attempt.
    """
    stripped = text.strip()
    if not stripped:
        return True

    matched = any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS) or (
        bool(_PRONOUN_REFUSAL.search(stripped)) and bool(_MENTIONS_SOURCE.search(stripped))
    )
    if not matched:
        return False

    # Position discriminates better than length. A refusal *leads* with the
    # disclaimer and then explains what the sources cover instead:
    #
    #     "The provided sources do not contain any information about chocolate
    #      cake. Source 1 discusses attention mechanisms..."
    #
    # A partial answer gives information first and hedges afterwards:
    #
    #     "The excerpts confirm BERT was evaluated on SuperGLUE. However, they
    #      do not state the score. A score of 80.5 is mentioned for GLUE..."
    #
    # The length rule below could not tell those apart -- the first is a refusal
    # at any length, and it was being scored as an answer whenever the model
    # was talkative about what the corpus *does* cover.
    first_sentence = re.split(r"(?<=[.!?])\s", stripped, maxsplit=1)[0]
    if any(pattern.search(first_sentence) for pattern in _REFUSAL_PATTERNS):
        return True

    # Strip citation groups before measuring length: a one-line refusal
    # carrying three long paper titles is still a one-line refusal.
    without_citations = re.sub(r"\[[^\]]*\]", "", stripped).strip()
    return len(without_citations) <= _MAX_REFUSAL_CHARS
