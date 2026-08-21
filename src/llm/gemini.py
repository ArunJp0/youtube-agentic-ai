# Gemini LLM provider (Google Generative Language API)
from __future__ import annotations

import random
import time
from typing import Optional

import httpx

from src.llm.provider import LLMProvider

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash-lite"

# Transient failures worth retrying: rate limiting, request timeout, server-side.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_JITTER_SECONDS = 0.5


class GeminiProviderError(Exception):
    """Raised when the Gemini LLM provider fails to produce a response.

    Covers non-retryable failures (bad request, auth, malformed/blocked
    response) as well as the final failure after all retries/fallback
    attempts have been exhausted.
    """


class _GeminiTransientError(Exception):
    """Internal: a single Gemini call failed with a retryable condition."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiLLMProvider(LLMProvider):
    """LLM provider backed by Google's Gemini API.

    Talks to the Generative Language REST API directly over ``httpx`` so no
    extra SDK dependency is required. Retry/backoff and fallback-model logic
    are fully contained here; callers only see the plain-text
    ``generate_text`` interface shared with every other LLMProvider.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
        timeout_seconds: float = 30.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
        max_jitter_seconds: float = DEFAULT_MAX_JITTER_SECONDS,
    ) -> None:
        if not api_key:
            raise GeminiProviderError(
                "GEMINI_API_KEY is not set. Set it in the environment or .env file "
                "to use the Gemini LLM provider."
            )
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_jitter_seconds = max_jitter_seconds

        # Diagnostics from the most recent generate_text() call.
        self.last_model_used: Optional[str] = None
        self.last_attempt_count: int = 0
        self.last_used_fallback: bool = False

    def generate_text(self, prompt: str) -> str:
        """Generate a response for the given prompt via the Gemini API.

        Retries transient failures (429/408/5xx, timeouts, network errors)
        with exponential backoff + jitter on the primary model. If the
        primary model is still unavailable after exhausting retries and a
        fallback model is configured, the same retry policy is applied to
        the fallback model.

        Raises:
            GeminiProviderError: On a non-retryable error, or if all
                configured models fail after retries.
        """
        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        last_error: Optional[Exception] = None
        for index, model in enumerate(models_to_try):
            is_fallback_attempt = index > 0
            try:
                text, attempts = self._call_with_retry(model, prompt)
            except _GeminiTransientError as e:
                last_error = e
                has_more_models = index + 1 < len(models_to_try)
                if has_more_models:
                    print(
                        f"[Gemini] Model '{model}' still unavailable after "
                        f"{self.max_attempts} attempt(s); falling back to "
                        f"'{models_to_try[index + 1]}'."
                    )
                    continue
                break
            else:
                self.last_model_used = model
                self.last_attempt_count = attempts
                self.last_used_fallback = is_fallback_attempt
                if is_fallback_attempt:
                    print(
                        f"[Gemini] Succeeded using fallback model '{model}' "
                        f"after {attempts} attempt(s)."
                    )
                return text

        raise GeminiProviderError(
            f"Gemini request failed after retries on all configured models "
            f"({', '.join(models_to_try)}): {last_error}"
        )

    def _call_with_retry(self, model: str, prompt: str) -> tuple[str, int]:
        """Call one Gemini model with retry/backoff.

        Returns:
            (response_text, attempts_used)

        Raises:
            _GeminiTransientError: If every attempt fails with a retryable error.
            GeminiProviderError: Immediately, for a non-retryable error.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                text = self._call_once(model, prompt)
                return text, attempt
            except _GeminiTransientError as e:
                if attempt >= self.max_attempts:
                    raise
                delay = self._compute_delay(attempt)
                print(
                    f"[Gemini] '{model}' attempt {attempt}/{self.max_attempts} failed "
                    f"({e}); retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    def _compute_delay(self, attempt: int) -> float:
        """Exponential backoff (~1s, 2s, 4s, 8s, ...) plus small random jitter."""
        base = self.base_delay_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.max_jitter_seconds)
        return base + jitter

    def _call_once(self, model: str, prompt: str) -> str:
        """Make a single Gemini API call.

        Raises:
            _GeminiTransientError: For 429/408/5xx responses, timeouts, or
                other network-level failures.
            GeminiProviderError: For non-retryable HTTP errors (e.g.
                400/401/403) or a malformed/empty/blocked response.
        """
        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = httpx.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as e:
            raise _GeminiTransientError(f"request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise _GeminiTransientError(f"request failed: {e}") from e

        if response.status_code in RETRYABLE_STATUS_CODES:
            raise _GeminiTransientError(
                f"transient HTTP {response.status_code}", status_code=response.status_code
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GeminiProviderError(
                f"Gemini API returned an error: {e.response.status_code} {e.response.text}"
            ) from e

        data = response.json()

        candidates = data.get("candidates")
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise GeminiProviderError(f"Gemini blocked the prompt: {block_reason}")
            raise GeminiProviderError(f"Malformed Gemini response (no candidates): {data}")

        try:
            parts = candidates[0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiProviderError(f"Malformed Gemini response: {data}") from e

        if not text.strip():
            raise GeminiProviderError("Gemini returned an empty response")

        return text
