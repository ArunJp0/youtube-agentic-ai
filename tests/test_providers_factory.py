# Tests for provider selection/configuration (src.config.providers)

from __future__ import annotations

import pytest

from src.config.providers import (
    ProviderConfigError,
    get_llm_provider,
    get_media_provider,
    get_search_provider,
    get_transcription_provider,
    get_voice_provider,
)
from src.config.settings import Settings
from src.llm.mock import MockLLMProvider
from src.tools.media_provider import MockMediaProvider
from src.tools.search_provider import MockSearchProvider
from src.tools.transcription_provider import MockTranscriptionProvider
from src.tools.voice_provider import MockVoiceProvider


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


class TestGetVoiceProvider:
    """Tests for voice provider selection."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(voice_provider="mock")
        provider = get_voice_provider(settings)
        assert isinstance(provider, MockVoiceProvider)

    def test_mock_provider_case_insensitive(self) -> None:
        settings = Settings(voice_provider="MOCK")
        provider = get_voice_provider(settings)
        assert isinstance(provider, MockVoiceProvider)

    def test_edge_provider_selected(self) -> None:
        from src.tools.edge_voice_provider import EdgeVoiceProvider

        settings = Settings(voice_provider="edge")
        provider = get_voice_provider(settings)
        assert isinstance(provider, EdgeVoiceProvider)

    def test_unknown_voice_provider_raises_config_error(self) -> None:
        settings = Settings(voice_provider="not-a-real-provider")
        with pytest.raises(ProviderConfigError):
            get_voice_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("VOICE_PROVIDER", "mock")
        provider = get_voice_provider()
        assert isinstance(provider, MockVoiceProvider)


class TestGetMediaProvider:
    """Tests for media provider selection."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(media_provider="mock")
        provider = get_media_provider(settings)
        assert isinstance(provider, MockMediaProvider)

    def test_mock_provider_case_insensitive(self) -> None:
        settings = Settings(media_provider="MOCK")
        provider = get_media_provider(settings)
        assert isinstance(provider, MockMediaProvider)

    def test_pexels_provider_selected(self) -> None:
        from src.tools.pexels_media_provider import PexelsMediaProvider

        settings = Settings(media_provider="pexels", pexels_api_key="fake-key")
        provider = get_media_provider(settings)
        assert isinstance(provider, PexelsMediaProvider)
        assert provider.api_key == "fake-key"

    def test_pexels_provider_missing_api_key_raises(self) -> None:
        from src.tools.media_provider import MediaProviderError

        settings = Settings(media_provider="pexels", pexels_api_key=None)
        with pytest.raises(MediaProviderError):
            get_media_provider(settings)

    def test_unknown_media_provider_raises_config_error(self) -> None:
        settings = Settings(media_provider="not-a-real-provider")
        with pytest.raises(ProviderConfigError):
            get_media_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("MEDIA_PROVIDER", "mock")
        provider = get_media_provider()
        assert isinstance(provider, MockMediaProvider)


class TestGetTranscriptionProvider:
    """Tests for transcription provider selection."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(transcription_provider="mock")
        provider = get_transcription_provider(settings)
        assert isinstance(provider, MockTranscriptionProvider)

    def test_mock_provider_case_insensitive(self) -> None:
        settings = Settings(transcription_provider="MOCK")
        provider = get_transcription_provider(settings)
        assert isinstance(provider, MockTranscriptionProvider)

    def test_whisper_provider_selected(self) -> None:
        from src.tools.whisper_transcription_provider import WhisperTranscriptionProvider

        settings = Settings(transcription_provider="whisper", whisper_model_size="tiny")
        provider = get_transcription_provider(settings)
        assert isinstance(provider, WhisperTranscriptionProvider)
        assert provider.model_size == "tiny"

    def test_unknown_transcription_provider_raises_config_error(self) -> None:
        settings = Settings(transcription_provider="not-a-real-provider")
        with pytest.raises(ProviderConfigError):
            get_transcription_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "mock")
        provider = get_transcription_provider()
        assert isinstance(provider, MockTranscriptionProvider)
