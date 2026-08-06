"""Retry policy for transient LLM provider failures.

Groq's free tier rate-limits aggressively and the multi-agent pipeline issues
four calls per question, so an un-retried 429 fails the whole request. This
module retries only errors classified as transient; authentication and
malformed-request failures propagate immediately.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from src.exceptions import RETRYABLE_LLM_ERRORS
from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    description: str = "LLM call",
) -> T:
    """Invoke ``func``, retrying retryable LLM errors with exponential backoff.

    Backoff is ``initial_backoff * 2**attempt`` capped at ``max_backoff``, with
    full jitter so that concurrent agents do not retry in lockstep.

    Args:
        func: Zero-argument callable performing the request.
        max_retries: Additional attempts after the first. ``0`` disables retrying.
        initial_backoff: Base delay in seconds.
        max_backoff: Ceiling on any single delay.
        description: Used in log messages to identify the call site.

    Returns:
        Whatever ``func`` returns.

    Raises:
        LLMError: The last error seen, once attempts are exhausted. Errors
            outside :data:`RETRYABLE_LLM_ERRORS` propagate on the first failure.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RETRYABLE_LLM_ERRORS as exc:
            last_error = exc
            if attempt == max_retries:
                logger.error("%s failed after %d attempts: %s", description, attempt + 1, exc)
                raise
            delay = min(initial_backoff * (2**attempt), max_backoff)
            delay = random.uniform(0, delay)  # noqa: S311 - jitter, not crypto
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description,
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error  # type: ignore[misc]  # pragma: no cover
