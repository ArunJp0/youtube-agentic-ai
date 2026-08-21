# Tests for provider selection/configuration (src.config.providers)

from __future__ import annotations

import pytest

from src.config.providers import ProviderConfigError, get_llm_provider, get_search_provider
from src.config.settings import Settings
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import MockSearchProvider


class TestGetLLMProvider:
    """Tests for LLM provider selection."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(llm_provider="mock")
        provider = get_llm_provider(settings)
        assert isinstance(provider, MockLLMProvider)

    def test_mock_provider_case_insensitive(self) -> None:
        settings = Settings(llm_provider="MOCK")
        provider = get_llm_provider(settings)
        assert isinstance(provider, MockLLMProvider)

    def test_gemini_provider_selected(self) -> None:
        from src.llm.gemini import GeminiLLMProvider

        settings = Settings(llm_provider="gemini", gemini_api_key="fake-key")
        provider = get_llm_provider(settings)
        assert isinstance(provider, GeminiLLMProvider)
        assert provider.api_key == "fake-key"

    def test_gemini_provider_missing_api_key_raises(self) -> None:
        from src.llm.gemini import GeminiProviderError

        settings = Settings(llm_provider="gemini", gemini_api_key=None)
        with pytest.raises(GeminiProviderError):
            get_llm_provider(settings)

    def test_unknown_llm_provider_raises_config_error(self) -> None:
        settings = Settings(llm_provider="not-a-real-provider")
        with pytest.raises(ProviderConfigError):
            get_llm_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)


class TestGetSearchProvider:
    """Tests for search provider selection."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(search_provider="mock")
        provider = get_search_provider(settings)
        assert isinstance(provider, MockSearchProvider)

    def test_wikipedia_provider_selected(self) -> None:
        from src.tools.wikipedia_provider import WikipediaSearchProvider

        settings = Settings(search_provider="wikipedia")
        provider = get_search_provider(settings)
        assert isinstance(provider, WikipediaSearchProvider)

    def test_unknown_search_provider_raises_config_error(self) -> None:
        settings = Settings(search_provider="not-a-real-provider")
        with pytest.raises(ProviderConfigError):
            get_search_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("SEARCH_PROVIDER", "mock")
        provider = get_search_provider()
        assert isinstance(provider, MockSearchProvider)
