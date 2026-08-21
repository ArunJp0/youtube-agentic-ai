# Provider selection: builds LLM/search providers from Settings
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.llm.provider import LLMProvider
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import SearchProvider, MockSearchProvider


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
