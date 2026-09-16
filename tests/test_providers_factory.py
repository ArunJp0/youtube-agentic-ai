# Tests for provider selection/configuration (src.config.providers)

from __future__ import annotations

import pytest

from src.config.providers import (
    ProviderConfigError,
    get_ai_video_provider,
    get_current_news_topic_source_provider,
    get_llm_provider,
    get_media_provider,
    get_search_provider,
    get_topic_planner_source,
    get_topic_source_provider,
    get_transcription_provider,
    get_voice_provider,
)
from src.config.settings import Settings
from src.llm.mock import MockLLMProvider
from src.tools.media_provider import MockMediaProvider
from src.tools.search_provider import MockSearchProvider
from src.tools.topic_source_provider import CompositeTopicSourceProvider, MockTopicSourceProvider
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


class TestGetTopicSourceProvider:
    """Tests for the evergreen topic-source factory (TOPIC_PLANNER_SOURCE)."""

    def test_mock_provider_selected(self) -> None:
        settings = Settings(topic_planner_source="mock")
        provider = get_topic_source_provider(settings)
        assert isinstance(provider, MockTopicSourceProvider)

    def test_mock_provider_case_insensitive(self) -> None:
        settings = Settings(topic_planner_source="MOCK")
        provider = get_topic_source_provider(settings)
        assert isinstance(provider, MockTopicSourceProvider)

    def test_youtube_provider_selected(self) -> None:
        from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider

        settings = Settings(topic_planner_source="youtube", youtube_api_key="fake-key")
        provider = get_topic_source_provider(settings)
        assert isinstance(provider, YouTubeTopicSourceProvider)
        assert provider.api_key == "fake-key"

    def test_unknown_topic_source_raises_config_error(self) -> None:
        settings = Settings(topic_planner_source="not-a-real-source")
        with pytest.raises(ProviderConfigError):
            get_topic_source_provider(settings)


class TestGetCurrentNewsTopicSourceProvider:
    """Tests for the current/trending-news topic-source factory - reuses
    TOPIC_PLANNER_SOURCE's mock/real toggle rather than a second setting."""

    def test_mock_when_topic_planner_source_is_mock(self) -> None:
        settings = Settings(topic_planner_source="mock")
        provider = get_current_news_topic_source_provider(settings)
        assert isinstance(provider, MockTopicSourceProvider)
        assert provider.name == "current_news"

    def test_real_provider_when_topic_planner_source_is_non_mock(self) -> None:
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider

        settings = Settings(topic_planner_source="youtube")
        provider = get_current_news_topic_source_provider(settings)
        assert isinstance(provider, CurrentNewsTopicSourceProvider)

    def test_no_api_key_required_for_real_news_provider(self) -> None:
        """Google News RSS needs no credential - a real news provider must
        be constructible even with no YOUTUBE_API_KEY configured at all."""
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider

        settings = Settings(topic_planner_source="youtube", youtube_api_key=None)
        provider = get_current_news_topic_source_provider(settings)
        assert isinstance(provider, CurrentNewsTopicSourceProvider)


class TestGetTopicPlannerSource:
    """Tests for TOPIC_MODE composition - the exact seam AutonomousRunController
    depends on for real autonomous topic-source wiring."""

    def test_evergreen_mode_returns_plain_evergreen_source(self) -> None:
        settings = Settings(topic_mode="evergreen", topic_planner_source="mock")
        provider = get_topic_planner_source(settings)
        assert isinstance(provider, MockTopicSourceProvider)
        assert not isinstance(provider, CompositeTopicSourceProvider)

    def test_trending_mode_requires_news_sources_enabled(self) -> None:
        settings = Settings(topic_mode="trending", topic_news_sources_enabled=False)
        with pytest.raises(ProviderConfigError, match="TOPIC_NEWS_SOURCES_ENABLED"):
            get_topic_planner_source(settings)

    def test_trending_mode_returns_real_news_source_when_enabled(self) -> None:
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider

        settings = Settings(topic_mode="trending", topic_news_sources_enabled=True, topic_planner_source="youtube")
        provider = get_topic_planner_source(settings)
        assert isinstance(provider, CurrentNewsTopicSourceProvider)

    def test_mixed_mode_without_news_enabled_degrades_to_evergreen_only(self) -> None:
        settings = Settings(topic_mode="mixed", topic_news_sources_enabled=False, topic_planner_source="mock")
        provider = get_topic_planner_source(settings)
        assert isinstance(provider, MockTopicSourceProvider)
        assert not isinstance(provider, CompositeTopicSourceProvider)

    def test_mixed_mode_with_news_enabled_composes_both_real_sources(self) -> None:
        """The exact configuration this milestone recommends for autonomous
        execution: real evergreen + real current news, merged."""
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider
        from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider

        settings = Settings(topic_mode="mixed", topic_news_sources_enabled=True, topic_planner_source="youtube")
        provider = get_topic_planner_source(settings)
        assert isinstance(provider, CompositeTopicSourceProvider)
        assert any(isinstance(p, YouTubeTopicSourceProvider) for p in provider.providers)
        assert any(isinstance(p, CurrentNewsTopicSourceProvider) for p in provider.providers)

    def test_unknown_topic_mode_raises_config_error(self) -> None:
        settings = Settings(topic_mode="not-a-real-mode")
        with pytest.raises(ProviderConfigError):
            get_topic_planner_source(settings)

    def test_default_env_configuration_produces_mixed_real_sources(self, monkeypatch) -> None:
        """Regression guard for the evergreen-only root cause: the
        recommended .env configuration (TOPIC_MODE=mixed,
        TOPIC_NEWS_SOURCES_ENABLED=true, TOPIC_PLANNER_SOURCE=youtube) must
        actually produce a composite of two real sources, not silently fall
        back to the mock catalog."""
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider
        from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider

        monkeypatch.setenv("TOPIC_MODE", "mixed")
        monkeypatch.setenv("TOPIC_NEWS_SOURCES_ENABLED", "true")
        monkeypatch.setenv("TOPIC_PLANNER_SOURCE", "youtube")

        provider = get_topic_planner_source()

        assert isinstance(provider, CompositeTopicSourceProvider)
        assert any(isinstance(p, YouTubeTopicSourceProvider) for p in provider.providers)
        assert any(isinstance(p, CurrentNewsTopicSourceProvider) for p in provider.providers)


class TestGetAIVideoProvider:
    """Tests for the AI Video Generation provider factory (AI_VIDEO_*)."""

    def test_disabled_by_default_returns_none(self) -> None:
        settings = Settings(ai_video_enabled=False)
        assert get_ai_video_provider(settings) is None

    def test_enabled_with_mock_provider(self) -> None:
        from src.tools.ai_video_provider import MockAIVideoProvider

        settings = Settings(ai_video_enabled=True, ai_video_provider="mock")
        provider = get_ai_video_provider(settings)
        assert isinstance(provider, MockAIVideoProvider)

    def test_enabled_with_local_provider(self, tmp_path) -> None:
        from src.tools.ai_video_provider import LocalAIVideoProvider

        settings = Settings(
            ai_video_enabled=True, ai_video_provider="local", ai_video_local_clips_dir=str(tmp_path)
        )
        provider = get_ai_video_provider(settings)
        assert isinstance(provider, LocalAIVideoProvider)
        assert provider.clips_dir == str(tmp_path)

    def test_local_provider_without_clips_dir_raises(self) -> None:
        settings = Settings(ai_video_enabled=True, ai_video_provider="local", ai_video_local_clips_dir=None)
        with pytest.raises(ProviderConfigError):
            get_ai_video_provider(settings)

    def test_unknown_provider_raises_config_error(self) -> None:
        settings = Settings(ai_video_enabled=True, ai_video_provider="fal_kling")
        with pytest.raises(ProviderConfigError):
            get_ai_video_provider(settings)

    def test_default_settings_used_when_none_passed(self, monkeypatch) -> None:
        monkeypatch.setenv("AI_VIDEO_ENABLED", "false")
        assert get_ai_video_provider() is None
