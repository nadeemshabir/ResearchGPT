"""Tests for mapping provider errors onto this project's exception types.

Classification decides whether a failure is retried. Get it wrong in one
direction and a bad API key is retried five times over a minute; wrong in the
other and a transient 429 fails the request outright.

Provider SDKs raise different exception types but converge on the same
vocabulary in their messages, so matching is on substrings. The strings below
are real messages seen from Groq, OpenAI and Gemini during this project.

`LLMClient.__init__` loads SDKs and requires an API key, so `_translate_error`
is tested directly on an object built without running it.
"""

import pytest

from src.exceptions import (
    RETRYABLE_LLM_ERRORS,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.generation.llm_client import LLMClient


@pytest.fixture
def client() -> LLMClient:
    """An LLMClient with only the attributes `_translate_error` reads."""
    instance = object.__new__(LLMClient)
    instance.provider = "groq"
    instance.model = "llama-3.3-70b-versatile"
    instance.timeout = 60.0
    return instance


def translate(client: LLMClient, message: str) -> Exception:
    return client._translate_error(RuntimeError(message))


# --- rate limits ------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - rate_limit_exceeded",
        "Rate limit reached for model `llama-3.3-70b-versatile`",
        "429 Too Many Requests",
        "RESOURCE_EXHAUSTED: quota exceeded",
    ],
)
def test_rate_limit_messages_are_classified_as_rate_limits(
    client: LLMClient, message: str
) -> None:
    assert isinstance(translate(client, message), LLMRateLimitError)


def test_the_real_groq_daily_limit_message_is_retryable(client: LLMClient) -> None:
    """The exact message that blocked three evaluation runs."""
    message = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`llama-3.3-70b-versatile` in organization `org_x` service tier `on_demand` "
        "on tokens per day (TPD): Limit 100000, Used 98251, Requested 2932.'}}"
    )

    error = translate(client, message)

    assert isinstance(error, LLMRateLimitError)
    assert isinstance(error, RETRYABLE_LLM_ERRORS)


# --- timeouts ---------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Timeout of 60.0s exceeded",
        "The request timed out",
        "504 Deadline Exceeded",
        "Connection reset by peer",
        "connection aborted",
    ],
)
def test_timeout_messages_are_classified_as_timeouts(
    client: LLMClient, message: str
) -> None:
    assert isinstance(translate(client, message), LLMTimeoutError)


def test_the_real_gemini_outage_message_is_retryable(client: LLMClient) -> None:
    """The 503 that corrupted a refusal run before errors were tracked."""
    message = (
        "Timeout of 60.0s exceeded, last exception: 503 failed to connect to all "
        "addresses; last error: UNKNOWN: ipv4:172.217.115.4:443: tcp handshaker shutdown"
    )

    assert isinstance(translate(client, message), RETRYABLE_LLM_ERRORS)


# --- authentication ---------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Incorrect API key provided",
        "401 Unauthorized",
        "invalid_api_key",
        "Authentication failed",
        "403 Forbidden",
        "Permission denied on resource",
    ],
)
def test_auth_messages_are_classified_as_authentication_errors(
    client: LLMClient, message: str
) -> None:
    assert isinstance(translate(client, message), LLMAuthenticationError)


def test_authentication_errors_are_not_retryable(client: LLMClient) -> None:
    """A bad key will still be bad in eight seconds. Retrying wastes a minute."""
    error = translate(client, "401 Unauthorized: invalid api key")

    assert not isinstance(error, RETRYABLE_LLM_ERRORS)


def test_auth_errors_name_the_environment_variable_to_fix(client: LLMClient) -> None:
    """The message is the whole remedy for a misconfigured install."""
    error = translate(client, "invalid_api_key")

    assert "GROQ_API_KEY" in str(error)


# --- everything else --------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "model `gpt-5-turbo` does not exist",
        "Error code: 404 - The model was not found",
        "context length exceeded",
        "something entirely unexpected",
    ],
)
def test_unrecognised_failures_become_provider_errors(
    client: LLMClient, message: str
) -> None:
    assert isinstance(translate(client, message), LLMProviderError)


def test_unrecognised_failures_are_not_retried(client: LLMClient) -> None:
    """Regression guard: a 404 for a wrong model name is permanent.

    Retrying it five times is what turned a one-line config mistake into a
    four-minute run of 43 identical failures.
    """
    error = translate(client, "Error code: 404 - The model `gemini-2.5-flash` does not exist")

    assert not isinstance(error, RETRYABLE_LLM_ERRORS)


def test_classification_is_case_insensitive(client: LLMClient) -> None:
    assert isinstance(translate(client, "RATE LIMIT EXCEEDED"), LLMRateLimitError)


def test_the_original_message_is_preserved(client: LLMClient) -> None:
    """Debugging a provider failure needs the provider's own wording."""
    error = translate(client, "Error code: 429 - Used 98251 of 100000")

    assert "98251" in str(error)


def test_the_error_records_provider_and_model(client: LLMClient) -> None:
    """Multi-provider runs are unreadable without knowing which one failed."""
    error = translate(client, "Error code: 429 - rate limit")

    assert error.provider == "groq"  # type: ignore[attr-defined]
    assert error.model == "llama-3.3-70b-versatile"  # type: ignore[attr-defined]


def test_a_bare_token_budget_message_is_not_assumed_to_be_a_rate_limit() -> None:
    """Classification keys on explicit markers, not on the word "limit".

    Groq's real 429 always carries "Rate limit reached" and "429" alongside the
    token counts, so nothing is lost. Matching "limit" on its own would sweep in
    "context length limit exceeded", which retrying cannot fix.
    """
    instance = object.__new__(LLMClient)
    instance.provider = "groq"
    instance.model = "m"
    instance.timeout = 60.0

    error = instance._translate_error(RuntimeError("Limit 100000, Used 98251"))

    assert not isinstance(error, RETRYABLE_LLM_ERRORS)
