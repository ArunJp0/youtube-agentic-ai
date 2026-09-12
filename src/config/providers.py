# Provider selection: builds LLM/search providers from Settings
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.llm.provider import LLMProvider
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import SearchProvider, MockSearchProvider
from src.tools.voice_provider import VoiceProvider, MockVoiceProvider
from src.tools.media_provider import MediaProvider, MockMediaProvider
from src.tools.transcription_provider import MockTranscriptionProvider, TranscriptionProvider
from src.tools.visual_relevance_evaluator import MockVisualRelevanceEvaluator, VisualRelevanceEvaluator
from src.tools.topic_source_provider import CompositeTopicSourceProvider, MockTopicSourceProvider, TopicSourceProvider


class ProviderConfigError(Exception):
    """Raised when LLM_PROVIDER/SEARCH_PROVIDER configuration is invalid."""


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Build the configured LLMProvider.

    Gemini-specific imports are done lazily so the mock path never depends
    on it, and so ResearchAgent/workflow code never has to know which
    concrete provider is in use.

    Raises:
        ProviderConfigError: If LLM_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.llm_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockLLMProvider()
    if provider_name == "gemini":
        from src.llm.gemini import GeminiLLMProvider

        return GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            fallback_model=settings.gemini_fallback_model,
        )

    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER: '{settings.llm_provider}'. Expected 'mock' or 'gemini'."
    )


def get_search_provider(settings: Optional[Settings] = None) -> SearchProvider:
    """Build the configured SearchProvider.

    Raises:
        ProviderConfigError: If SEARCH_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.search_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockSearchProvider()
    if provider_name == "wikipedia":
        from src.tools.wikipedia_provider import WikipediaSearchProvider

        return WikipediaSearchProvider()

    raise ProviderConfigError(
        f"Unknown SEARCH_PROVIDER: '{settings.search_provider}'. Expected 'mock' or 'wikipedia'."
    )


def get_voice_provider(settings: Optional[Settings] = None) -> VoiceProvider:
    """Build the configured VoiceProvider.

    Edge-tts-specific imports are done lazily so the mock path never depends
    on it, and so VoiceService never has to know which concrete TTS engine
    is in use.

    Raises:
        ProviderConfigError: If VOICE_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.voice_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockVoiceProvider()
    if provider_name == "edge":
        from src.tools.edge_voice_provider import EdgeVoiceProvider

        return EdgeVoiceProvider()

    raise ProviderConfigError(
        f"Unknown VOICE_PROVIDER: '{settings.voice_provider}'. Expected 'mock' or 'edge'."
    )


def get_media_provider(settings: Optional[Settings] = None) -> MediaProvider:
    """Build the configured MediaProvider.

    Pexels-specific imports are done lazily so the mock path never depends
    on it, and so VisualMediaService never has to know which concrete stock
    media API is in use.

    Raises:
        ProviderConfigError: If MEDIA_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.media_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockMediaProvider()
    if provider_name == "pexels":
        from src.tools.pexels_media_provider import PexelsMediaProvider

        return PexelsMediaProvider(api_key=settings.pexels_api_key)

    raise ProviderConfigError(
        f"Unknown MEDIA_PROVIDER: '{settings.media_provider}'. Expected 'mock' or 'pexels'."
    )


def get_visual_relevance_evaluator(settings: Optional[Settings] = None) -> VisualRelevanceEvaluator:
    """Build the configured VisualRelevanceEvaluator (Visual QC's vision model).

    Reuses LLM_PROVIDER rather than a separate setting: vision QC is the
    same underlying LLM capability (Gemini) as Research/Script, just with
    image input, so a mock LLM setup gets a mock evaluator and a real
    Gemini setup gets the real one, with no extra configuration required.

    Gemini-specific imports are done lazily so the mock path never depends
    on it, matching every other provider factory here.

    Raises:
        ProviderConfigError: If LLM_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.llm_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockVisualRelevanceEvaluator()
    if provider_name == "gemini":
        from src.tools.gemini_visual_relevance_evaluator import GeminiVisualRelevanceEvaluator

        return GeminiVisualRelevanceEvaluator(api_key=settings.gemini_api_key, model=settings.gemini_model)

    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER: '{settings.llm_provider}'. Expected 'mock' or 'gemini'."
    )


def get_transcription_provider(settings: Optional[Settings] = None) -> TranscriptionProvider:
    """Build the configured TranscriptionProvider (Caption Service's speech-to-text).

    faster-whisper-specific imports are done lazily so the mock path never
    depends on it, matching every other provider factory here.

    Raises:
        ProviderConfigError: If TRANSCRIPTION_PROVIDER is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.transcription_provider or "mock").strip().lower()

    if provider_name == "mock":
        return MockTranscriptionProvider()
    if provider_name == "whisper":
        from src.tools.whisper_transcription_provider import WhisperTranscriptionProvider

        return WhisperTranscriptionProvider(model_size=settings.whisper_model_size)

    raise ProviderConfigError(
        f"Unknown TRANSCRIPTION_PROVIDER: '{settings.transcription_provider}'. Expected 'mock' or 'whisper'."
    )


def get_topic_source_provider(settings: Optional[Settings] = None) -> TopicSourceProvider:
    """Build the configured TopicSourceProvider (Topic Planner Agent's
    candidate-discovery source).

    YouTube-specific imports are done lazily so the mock path never
    depends on it, matching every other provider factory here.

    Raises:
        ProviderConfigError: If TOPIC_PLANNER_SOURCE is not a recognized value.
    """
    settings = settings or Settings()
    provider_name = (settings.topic_planner_source or "mock").strip().lower()

    if provider_name == "mock":
        return MockTopicSourceProvider()
    if provider_name == "youtube":
        from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider

        return YouTubeTopicSourceProvider(api_key=settings.youtube_api_key)

    raise ProviderConfigError(
        f"Unknown TOPIC_PLANNER_SOURCE: '{settings.topic_planner_source}'. Expected 'mock' or 'youtube'."
    )


def get_current_news_topic_source_provider(settings: Optional[Settings] = None) -> TopicSourceProvider:
    """Build the configured current/trending-news TopicSourceProvider.

    Reuses the same TOPIC_PLANNER_SOURCE mock/real toggle as
    ``get_topic_source_provider`` (a "mock" configuration mocks every
    topic source uniformly; a real configuration uses real sources
    throughout) rather than adding a second, parallel mock/real setting.

    Google-News-RSS-specific imports are done lazily so the mock path
    never depends on it, matching every other provider factory here.
    """
    settings = settings or Settings()
    provider_name = (settings.topic_planner_source or "mock").strip().lower()

    if provider_name == "mock":
        return MockTopicSourceProvider(provider_name="current_news")

    from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider

    return CurrentNewsTopicSourceProvider(language=settings.topic_language)


def get_topic_planner_source(settings: Optional[Settings] = None) -> TopicSourceProvider:
    """Compose the configured topic source(s) for TOPIC_MODE.

    - "evergreen" (default): returns exactly what ``get_topic_source_provider``
      already returned before the current/trending-news extension existed -
      byte-for-byte the original single-source behavior, unchanged.
    - "trending": the current-news source only (requires
      TOPIC_NEWS_SOURCES_ENABLED=true).
    - "mixed": both the evergreen source and (if enabled) the news source,
      merged via CompositeTopicSourceProvider - if news sources aren't
      enabled, degrades to the evergreen source alone rather than failing.

    Raises:
        ProviderConfigError: If TOPIC_MODE is not a recognized value, or
            TOPIC_MODE=trending is requested without TOPIC_NEWS_SOURCES_ENABLED.
    """
    settings = settings or Settings()
    mode = (settings.topic_mode or "evergreen").strip().lower()

    if mode == "evergreen":
        return get_topic_source_provider(settings)

    if mode == "trending":
        if not settings.topic_news_sources_enabled:
            raise ProviderConfigError("TOPIC_MODE=trending requires TOPIC_NEWS_SOURCES_ENABLED=true")
        return get_current_news_topic_source_provider(settings)

    if mode == "mixed":
        evergreen = get_topic_source_provider(settings)
        if not settings.topic_news_sources_enabled:
            return evergreen
        news = get_current_news_topic_source_provider(settings)
        return CompositeTopicSourceProvider([evergreen, news])

    raise ProviderConfigError(f"Unknown TOPIC_MODE: '{settings.topic_mode}'. Expected 'evergreen', 'trending', or 'mixed'.")
