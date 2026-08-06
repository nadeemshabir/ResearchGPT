"""Tests for the LLM retry policy.

No real API is called, and no real sleeping happens -- `time.sleep` is patched,
so the exponential backoff is asserted rather than waited out. A test suite that
actually slept through this policy would take about a minute per case.

The behaviour matters: Groq's free tier rate-limits hard enough that this code
fires routinely in normal use.
"""

from typing import Any

import pytest

from src.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.utils.retry import call_with_retry

POLICY: dict[str, Any] = {
    "max_retries": 4,
    "initial_backoff": 1.0,
    "max_backoff": 30.0,
}


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture backoff delays instead of sleeping them."""
    delays: list[float] = []
    monkeypatch.setattr("src.utils.retry.time.sleep", delays.append)
    return delays


@pytest.fixture
def no_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make jitter deterministic by always drawing the maximum."""
    monkeypatch.setattr("src.utils.retry.random.uniform", lambda _low, high: high)


def failing(times: int, error: Exception, result: str = "ok") -> Any:
    """A callable that raises ``error`` ``times`` times, then succeeds."""
    state = {"calls": 0}

    def call() -> str:
        state["calls"] += 1
        if state["calls"] <= times:
            raise error
        return result

    call.state = state  # type: ignore[attr-defined]
    return call


def test_a_successful_call_is_not_retried(no_sleep: list[float]) -> None:
    call = failing(0, LLMRateLimitError("never raised"))

    assert call_with_retry(call, **POLICY) == "ok"
    assert call.state["calls"] == 1  # type: ignore[attr-defined]
    assert no_sleep == []


def test_a_transient_failure_is_retried_then_succeeds(no_sleep: list[float]) -> None:
    call = failing(2, LLMRateLimitError("429"))

    assert call_with_retry(call, **POLICY) == "ok"
    assert call.state["calls"] == 3  # type: ignore[attr-defined]


def test_timeouts_are_retried(no_sleep: list[float]) -> None:
    call = failing(1, LLMTimeoutError("timed out"))

    assert call_with_retry(call, **POLICY) == "ok"


def test_authentication_errors_are_not_retried(no_sleep: list[float]) -> None:
    """A bad key will still be bad in one second. Retrying wastes a minute."""
    call = failing(1, LLMAuthenticationError("bad key"))

    with pytest.raises(LLMAuthenticationError):
        call_with_retry(call, **POLICY)

    assert call.state["calls"] == 1  # type: ignore[attr-defined]
    assert no_sleep == []


def test_malformed_response_errors_are_not_retried(no_sleep: list[float]) -> None:
    call = failing(1, LLMResponseError("bad json"))

    with pytest.raises(LLMResponseError):
        call_with_retry(call, **POLICY)

    assert call.state["calls"] == 1  # type: ignore[attr-defined]


def test_the_original_error_surfaces_once_attempts_run_out(
    no_sleep: list[float],
) -> None:
    """Callers switch on the exception type, so it must not be wrapped."""
    call = failing(99, LLMRateLimitError("Limit 100000, Used 98251"))

    with pytest.raises(LLMRateLimitError, match="98251"):
        call_with_retry(call, **POLICY)


def test_attempt_count_is_retries_plus_one(no_sleep: list[float]) -> None:
    call = failing(99, LLMRateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        call_with_retry(call, max_retries=3, initial_backoff=1.0, max_backoff=30.0)

    assert call.state["calls"] == 4  # type: ignore[attr-defined]


def test_zero_retries_means_one_attempt(no_sleep: list[float]) -> None:
    call = failing(99, LLMRateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        call_with_retry(call, max_retries=0, initial_backoff=1.0, max_backoff=30.0)

    assert call.state["calls"] == 1  # type: ignore[attr-defined]
    assert no_sleep == []


def test_backoff_doubles_each_attempt(no_sleep: list[float], no_jitter: None) -> None:
    call = failing(99, LLMRateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        call_with_retry(call, max_retries=4, initial_backoff=1.0, max_backoff=30.0)

    assert no_sleep == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped(no_sleep: list[float], no_jitter: None) -> None:
    """Without a cap, attempt 10 would wait about seventeen minutes."""
    call = failing(99, LLMRateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        call_with_retry(call, max_retries=6, initial_backoff=1.0, max_backoff=5.0)

    assert no_sleep == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    assert max(no_sleep) <= 5.0


def test_jitter_keeps_delays_within_the_computed_bound(
    no_sleep: list[float],
) -> None:
    """Full jitter draws uniformly from [0, delay] so concurrent agents spread out.

    Without it, four agents rate-limited by the same request retry in lockstep
    and rate-limit each other again.
    """
    call = failing(99, LLMRateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        call_with_retry(call, max_retries=4, initial_backoff=1.0, max_backoff=30.0)

    for delay, bound in zip(no_sleep, [1.0, 2.0, 4.0, 8.0], strict=True):
        assert 0.0 <= delay <= bound


def test_a_non_llm_exception_propagates_immediately(no_sleep: list[float]) -> None:
    """Retrying a ValueError would hide a bug behind five identical failures."""
    call = failing(1, ValueError("programming error"))

    with pytest.raises(ValueError):
        call_with_retry(call, **POLICY)

    assert call.state["calls"] == 1  # type: ignore[attr-defined]
