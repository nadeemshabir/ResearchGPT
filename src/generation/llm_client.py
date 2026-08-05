"""Unified LLM client across Groq, OpenAI, and Gemini.

Provider SDKs raise unrelated exception types for the same conditions, so every
call is funnelled through :meth:`LLMClient._translate_error`, which maps them
onto the :mod:`src.exceptions` hierarchy. Only transient failures are retried;
an invalid API key fails immediately rather than after four backoffs.
"""

from __future__ import annotations

from typing import Any

from src.config import API_KEY_ENV_VARS, DEFAULT_MODELS, get_settings
from src.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.utils.logging import get_logger
from src.utils.retry import call_with_retry

logger = get_logger(__name__)

#: Substrings identifying each failure class in provider error messages. SDKs
#: differ in exception types but converge on this vocabulary.
_AUTH_MARKERS = ("api key", "unauthorized", "authentication", "invalid_api_key", "permission denied", "forbidden", "401", "403")
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "too many requests", "quota", "429", "resource_exhausted")
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline exceeded", "connection reset", "connection aborted")


class LLMClient:
    """Text generation against a configured provider."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Args:
            provider: ``"groq"``, ``"openai"``, or ``"gemini"``. Defaults to
                ``Settings.llm_provider``.
            model: Model id. Defaults to the provider's entry in
                :data:`~src.config.DEFAULT_MODELS`.
            api_key: Overrides the key from the environment.
            temperature: Default sampling temperature.
            max_tokens: Default response cap.

        Raises:
            ConfigurationError: Unknown provider, missing API key, or the
                provider SDK is not installed.
        """
        settings = get_settings()
        self.provider = (provider or settings.llm_provider).lower()

        if self.provider not in DEFAULT_MODELS:
            raise ConfigurationError(
                f"Unsupported provider {self.provider!r}. "
                f"Expected one of {sorted(DEFAULT_MODELS)}."
            )

        self.model = model or settings.llm_model or DEFAULT_MODELS[self.provider]
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._initial_backoff = settings.llm_retry_initial_backoff
        self._max_backoff = settings.llm_retry_max_backoff

        self.api_key = api_key or settings.api_key_for(self.provider)
        if not self.api_key:
            raise ConfigurationError(
                f"No API key for {self.provider}. Set "
                f"{API_KEY_ENV_VARS[self.provider]} in your .env file. "
                f"Groq keys are free at https://console.groq.com"
            )

        self._client = self._build_client()
        logger.info("LLM client ready (%s / %s)", self.provider, self.model)

    def _build_client(self) -> Any:
        """Instantiate the provider SDK client.

        Raises:
            ConfigurationError: The SDK is not installed. Previously this
                shelled out to ``pip install`` at import time, which silently
                mutated the user's environment; now it fails with instructions.
        """
        try:
            if self.provider == "groq":
                from groq import Groq

                return Groq(api_key=self.api_key, timeout=self.timeout)

            if self.provider == "openai":
                from openai import OpenAI

                return OpenAI(api_key=self.api_key, timeout=self.timeout)

            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model)

        except ImportError as exc:
            package = {
                "groq": "groq",
                "openai": "openai",
                "gemini": "google-generativeai",
            }[self.provider]
            raise ConfigurationError(
                f"The {self.provider} provider needs the {package!r} package. "
                f"Install it with: pip install {package}"
            ) from exc

    # ------------------------------------------------------------------
    # Error translation
    # ------------------------------------------------------------------
    def _translate_error(self, exc: Exception) -> Exception:
        """Map a provider exception onto this project's exception hierarchy."""
        message = str(exc).lower()
        context = {"provider": self.provider, "model": self.model}

        if any(marker in message for marker in _AUTH_MARKERS):
            return LLMAuthenticationError(
                f"{self.provider} rejected the API key. Check "
                f"{API_KEY_ENV_VARS[self.provider]} in your .env. ({exc})",
                **context,
            )
        if any(marker in message for marker in _RATE_LIMIT_MARKERS):
            return LLMRateLimitError(f"{self.provider} rate limit hit: {exc}", **context)
        if any(marker in message for marker in _TIMEOUT_MARKERS):
            return LLMTimeoutError(
                f"{self.provider} request timed out after {self.timeout}s: {exc}", **context
            )
        return LLMProviderError(f"{self.provider} request failed: {exc}", **context)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: User message.
            system_prompt: Instructions shaping the model's behaviour.
            temperature: Overrides the client default.
            max_tokens: Overrides the client default.

        Returns:
            The generated text.

        Raises:
            LLMAuthenticationError: The key was rejected.
            LLMRateLimitError: Rate limited after exhausting retries.
            LLMTimeoutError: Timed out after exhausting retries.
            LLMResponseError: The provider returned nothing usable.
            LLMProviderError: Any other provider failure.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Cannot generate from an empty prompt.")

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._complete(messages, temperature, max_tokens)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a reply to a multi-turn conversation.

        Args:
            messages: ``{"role": ..., "content": ...}`` records, oldest first.

        Raises:
            ValueError: ``messages`` is empty.
        """
        if not messages:
            raise ValueError("chat() requires at least one message.")
        return self._complete(
            messages,
            temperature if temperature is not None else self.temperature,
            max_tokens or self.max_tokens,
        )

    def _complete(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Dispatch to the provider, with retries on transient failures."""

        def attempt() -> str:
            try:
                if self.provider in ("groq", "openai"):
                    return self._complete_openai_compatible(messages, temperature, max_tokens)
                return self._complete_gemini(messages, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, (LLMResponseError, ValueError)):
                    raise
                raise self._translate_error(exc) from exc

        return call_with_retry(
            attempt,
            max_retries=self._max_retries,
            initial_backoff=self._initial_backoff,
            max_backoff=self._max_backoff,
            description=f"{self.provider}:{self.model}",
        )

    def _complete_openai_compatible(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Groq and OpenAI share the chat-completions interface."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise LLMResponseError(
                f"{self.provider} returned no choices.",
                provider=self.provider,
                model=self.model,
            )

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMResponseError(
                f"{self.provider} returned an empty completion "
                f"(finish_reason={getattr(response.choices[0], 'finish_reason', 'unknown')}).",
                provider=self.provider,
                model=self.model,
            )
        return content

    def _complete_gemini(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Gemini has no system role, so system text is prepended."""
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [
            f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
        ]
        prompt = "\n\n".join([*system_parts, *conversation])

        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise LLMResponseError(
                "Gemini returned an empty response, most likely due to a safety filter.",
                provider=self.provider,
                model=self.model,
            )
        return text

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate at ~4 characters per token.

        Adequate for budgeting; not exact, and not a substitute for the
        provider's own accounting when measuring cost.
        """
        return max(1, len(text) // 4)

    def get_model_info(self) -> dict[str, Any]:
        """Client configuration, for logging and the UI."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "max_retries": self._max_retries,
        }
