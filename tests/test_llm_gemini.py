# Tests for GeminiLLMProvider (all network calls and sleeps are mocked)

from __future__ import annotations

import httpx
import pytest

from src.llm import gemini as gemini_module
from src.llm.gemini import GeminiLLMProvider, GeminiProviderError


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com/fake")
            raise httpx.HTTPStatusError(
                f"HTTP error {self.status_code}", request=request, response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._json_data


def _success_payload(text: str = "Generated response text") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Every test in this module runs with retries; never actually sleep."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(gemini_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(gemini_module.random, "uniform", lambda a, b: 0.0)
    return sleep_calls


class TestGeminiLLMProviderInit:
    """Tests for provider construction / missing API key handling."""

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(GeminiProviderError, match="GEMINI_API_KEY"):
            GeminiLLMProvider(api_key=None)

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(GeminiProviderError, match="GEMINI_API_KEY"):
            GeminiLLMProvider(api_key="")

    def test_valid_api_key_constructs_with_defaults(self) -> None:
        provider = GeminiLLMProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gemini-3.6-flash"
        assert provider.fallback_model == "gemini-3.5-flash-lite"
        assert provider.max_attempts == 5


class TestGeminiLLMProviderGenerateText:
    """Tests for generate_text with the network call mocked out."""

    def test_successful_generation_first_try(self, monkeypatch, no_real_sleep) -> None:
        calls = []

        def fake_post(url, params=None, json=None, timeout=None):
            calls.append(url)
            return FakeResponse(_success_payload("Dreams help consolidate memories."))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        result = provider.generate_text("Why do humans dream?")

        assert result == "Dreams help consolidate memories."
        assert len(calls) == 1
        assert provider.last_model_used == "gemini-3.6-flash"
        assert provider.last_attempt_count == 1
        assert provider.last_used_fallback is False
        assert no_real_sleep == []  # no retry -> no sleep

    def test_joins_multiple_parts(self, monkeypatch) -> None:
        payload = {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}}]}

        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse(payload)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        assert provider.generate_text("prompt") == "Hello world"


class TestGeminiRetryBehavior:
    """Tests for retry-on-transient-error behavior with exponential backoff."""

    def test_retries_on_503_then_succeeds(self, monkeypatch, no_real_sleep) -> None:
        responses = [FakeResponse({}, status_code=503), FakeResponse({}, status_code=503), FakeResponse(_success_payload("ok"))]
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        result = provider.generate_text("prompt")

        assert result == "ok"
        assert call_count["n"] == 3
        assert provider.last_attempt_count == 3
        # Backoff should double each time: ~1s, ~2s (jitter mocked to 0)
        assert no_real_sleep == [1.0, 2.0]

    def test_retries_on_429_then_succeeds(self, monkeypatch, no_real_sleep) -> None:
        responses = [FakeResponse({}, status_code=429), FakeResponse(_success_payload("ok"))]
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        result = provider.generate_text("prompt")

        assert result == "ok"
        assert call_count["n"] == 2
        assert no_real_sleep == [1.0]

    def test_retries_on_408_and_5xx_family(self, monkeypatch, no_real_sleep) -> None:
        statuses = [408, 500, 502, 504]
        for status in statuses:
            no_real_sleep.clear()
            responses = [FakeResponse({}, status_code=status), FakeResponse(_success_payload("ok"))]
            call_count = {"n": 0}

            def fake_post(url, params=None, json=None, timeout=None, _responses=responses, _cc=call_count):
                resp = _responses[_cc["n"]]
                _cc["n"] += 1
                return resp

            monkeypatch.setattr(httpx, "post", fake_post)
            provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
            assert provider.generate_text("prompt") == "ok"
            assert call_count["n"] == 2

    def test_no_retry_on_400(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            return FakeResponse({}, status_code=400)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)

        with pytest.raises(GeminiProviderError, match="400"):
            provider.generate_text("prompt")

        assert call_count["n"] == 1
        assert no_real_sleep == []

    def test_no_retry_on_401(self, monkeypatch) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse({}, status_code=401)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="bad-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="401"):
            provider.generate_text("prompt")

    def test_no_retry_on_403(self, monkeypatch) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse({}, status_code=403)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="403"):
            provider.generate_text("prompt")

    def test_exhausts_max_attempts_and_raises(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            return FakeResponse({}, status_code=503)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None, max_attempts=4)

        with pytest.raises(GeminiProviderError):
            provider.generate_text("prompt")

        assert call_count["n"] == 4
        assert len(no_real_sleep) == 3  # sleeps between attempts, not after the last

    def test_timeout_is_retried(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.TimeoutException("timed out")
            return FakeResponse(_success_payload("ok"))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        assert provider.generate_text("prompt") == "ok"
        assert call_count["n"] == 2

    def test_network_error_is_retried(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("connection refused")
            return FakeResponse(_success_payload("ok"))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        assert provider.generate_text("prompt") == "ok"
        assert call_count["n"] == 2


class TestGeminiFallbackModel:
    """Tests for fallback-model behavior when the primary is unavailable."""

    def test_falls_back_after_primary_exhausts_retries(self, monkeypatch, no_real_sleep) -> None:
        call_log = []

        def fake_post(url, params=None, json=None, timeout=None):
            call_log.append(url)
            if "gemini-3.6-flash" in url:
                return FakeResponse({}, status_code=503)
            return FakeResponse(_success_payload("fallback answer"))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="gemini-3.6-flash",
            fallback_model="gemini-3.5-flash-lite",
            max_attempts=3,
        )
        result = provider.generate_text("prompt")

        assert result == "fallback answer"
        assert provider.last_model_used == "gemini-3.5-flash-lite"
        assert provider.last_used_fallback is True
        # 3 failed attempts on primary + 1 successful attempt on fallback
        primary_calls = [c for c in call_log if "gemini-3.6-flash" in c]
        fallback_calls = [c for c in call_log if "gemini-3.5-flash-lite" in c]
        assert len(primary_calls) == 3
        assert len(fallback_calls) == 1

    def test_fallback_also_retries_transient_errors(self, monkeypatch, no_real_sleep) -> None:
        primary_attempts = {"n": 0}
        fallback_attempts = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            if "gemini-3.6-flash" in url:
                primary_attempts["n"] += 1
                return FakeResponse({}, status_code=503)
            fallback_attempts["n"] += 1
            if fallback_attempts["n"] < 2:
                return FakeResponse({}, status_code=429)
            return FakeResponse(_success_payload("fallback ok after retry"))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="gemini-3.6-flash",
            fallback_model="gemini-3.5-flash-lite",
            max_attempts=2,
        )
        result = provider.generate_text("prompt")

        assert result == "fallback ok after retry"
        assert primary_attempts["n"] == 2
        assert fallback_attempts["n"] == 2
        assert provider.last_used_fallback is True

    def test_no_fallback_configured_raises_after_exhausting_primary(self, monkeypatch, no_real_sleep) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse({}, status_code=503)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None, max_attempts=2)

        with pytest.raises(GeminiProviderError):
            provider.generate_text("prompt")

    def test_non_retryable_error_does_not_trigger_fallback(self, monkeypatch, no_real_sleep) -> None:
        call_log = []

        def fake_post(url, params=None, json=None, timeout=None):
            call_log.append(url)
            return FakeResponse({}, status_code=400)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="gemini-3.6-flash",
            fallback_model="gemini-3.5-flash-lite",
        )
        with pytest.raises(GeminiProviderError, match="400"):
            provider.generate_text("prompt")

        # Only the primary model should have been called; a 400 is not transient.
        assert all("gemini-3.6-flash" in url for url in call_log)
        assert len(call_log) == 1

    def test_fallback_equal_to_primary_is_not_duplicated(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            return FakeResponse({}, status_code=503)

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="gemini-3.6-flash",
            fallback_model="gemini-3.6-flash",
            max_attempts=2,
        )
        with pytest.raises(GeminiProviderError):
            provider.generate_text("prompt")

        assert call_count["n"] == 2  # only the primary's retry budget, no duplicate fallback pass


class TestGeminiMalformedAndBlockedResponses:
    """Malformed/blocked responses are not retryable."""

    def test_malformed_response_no_candidates_raises_without_retry(self, monkeypatch, no_real_sleep) -> None:
        call_count = {"n": 0}

        def fake_post(url, params=None, json=None, timeout=None):
            call_count["n"] += 1
            return FakeResponse({"unexpected": "shape"})

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="Malformed"):
            provider.generate_text("prompt")
        assert call_count["n"] == 1

    def test_blocked_prompt_raises_clear_error(self, monkeypatch) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}})

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="blocked"):
            provider.generate_text("prompt")

    def test_malformed_candidate_shape_raises(self, monkeypatch) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse({"candidates": [{"content": {}}]})

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="Malformed"):
            provider.generate_text("prompt")

    def test_empty_text_response_raises(self, monkeypatch) -> None:
        def fake_post(url, params=None, json=None, timeout=None):
            return FakeResponse(_success_payload(""))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = GeminiLLMProvider(api_key="test-key", fallback_model=None)
        with pytest.raises(GeminiProviderError, match="empty"):
            provider.generate_text("prompt")
