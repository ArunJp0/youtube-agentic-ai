# Provider selection: builds LLM/search providers from Settings
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.llm.provider import LLMProvider
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import SearchProvider, MockSearchProvider
from src.tools.voice_provider import VoiceProvider, MockVoiceProvider
from src.tools.media_provider import MediaProvider, MockMediaProvider
from src.tools.visual_relevance_evaluator import MockVisualRelevanceEvaluator, VisualRelevanceEvaluator


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
