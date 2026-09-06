# Tests for the standalone Gemini model health-check tool. All network
# calls are mocked - no real Gemini/network calls.
from __future__ import annotations

import httpx
import pytest

from src.gemini_health_check import (
    ModelHealthResult,
    list_available_models,
    probe_model,
    run_health_check,
)


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://generativelanguage.googleapis.com/fake")
            raise httpx.HTTPStatusError(
                f"HTTP error {self.status_code}", request=request, response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._json_data


def _success_payload(text: str = "OK") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class TestListAvailableModels:
    def test_returns_models_supporting_generate_content(self, monkeypatch) -> None:
        payload = {
            "models": [
                {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            ]
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(payload))

        models = list_available_models("test-key")

        assert models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    def test_listing_failure_returns_empty_list_not_exception(self, monkeypatch) -> None:
        def fake_get(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", fake_get)

        assert list_available_models("test-key") == []

    def test_non_2xx_response_returns_empty_list(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse({}, status_code=403))

        assert list_available_models("test-key") == []


class TestProbeModel:
    def test_successful_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("OK")))

        result = probe_model("test-key", "gemini-3.5-flash-lite")

        assert isinstance(result, ModelHealthResult)
        assert result.success is True
        assert result.category == "ok"
        assert result.model == "gemini-3.5-flash-lite"
        assert result.latency_seconds >= 0.0

    def test_429_categorized_as_rate_limited(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({}, status_code=429))

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "429"

    def test_503_family_categorized_as_503(self, monkeypatch) -> None:
        for status in (500, 502, 503, 504):
            monkeypatch.setattr(httpx, "post", lambda *a, status=status, **k: FakeResponse({}, status_code=status))
            result = probe_model("test-key", "some-model")
            assert result.success is False
            assert result.category == "503"

    def test_timeout_categorized_as_timeout(self, monkeypatch) -> None:
        def fake_post(*a, **k):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx, "post", fake_post)

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "timeout"

    def test_other_http_error_categorized_as_error(self, monkeypatch) -> None:
        def fake_post(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "post", fake_post)

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "error"

    def test_4xx_other_than_429_categorized_as_error(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({}, status_code=404))

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "error"

    def test_malformed_response_no_candidates(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({"unexpected": "shape"}))

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "malformed"

    def test_empty_text_categorized_as_malformed(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("")))

        result = probe_model("test-key", "some-model")

        assert result.success is False
        assert result.category == "malformed"

    def test_no_retries_single_call_per_probe(self, monkeypatch) -> None:
        calls = {"n": 0}

        def fake_post(*a, **k):
            calls["n"] += 1
            return FakeResponse({}, status_code=503)

        monkeypatch.setattr(httpx, "post", fake_post)
        probe_model("test-key", "some-model")

        assert calls["n"] == 1


class TestRunHealthCheck:
    def test_missing_api_key_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            run_health_check(candidates=["some-model"])

    def test_explicit_candidates_skip_list_models(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        list_calls = {"n": 0}

        def fake_get(*a, **k):
            list_calls["n"] += 1
            return FakeResponse({"models": []})

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("OK")))

        results = run_health_check(candidates=["model-a", "model-b"])

        assert list_calls["n"] == 0
        assert [r.model for r in results] == ["model-a", "model-b"]

    def test_falls_back_to_configured_pair_when_listing_unavailable(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "configured-primary")
        monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "configured-fallback")

        def fake_get(*a, **k):
            raise httpx.ConnectError("listing unavailable")

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("OK")))

        results = run_health_check()

        assert [r.model for r in results] == ["configured-primary", "configured-fallback"]

    def test_uses_listed_models_when_available(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        list_payload = {
            "models": [{"name": "models/listed-model", "supportedGenerationMethods": ["generateContent"]}]
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(list_payload))
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("OK")))

        results = run_health_check()

        assert [r.model for r in results] == ["listed-model"]

    def test_never_probes_an_invented_model_name(self, monkeypatch) -> None:
        """If listing fails and no fallback model is configured, only the
        configured primary is probed - never a name this tool made up."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "configured-primary")
        # Settings.gemini_fallback_model falls back to its own built-in
        # default when the env var is simply unset - an empty string is
        # the actual "no fallback configured" signal (see settings.py).
        monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")

        def fake_get(*a, **k):
            raise httpx.ConnectError("listing unavailable")

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(_success_payload("OK")))

        results = run_health_check()

        assert [r.model for r in results] == ["configured-primary"]
