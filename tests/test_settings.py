# Tests for Settings environment-variable configuration

from __future__ import annotations

from src.config.settings import Settings


class TestSettings:
    """Tests for Settings reading provider configuration from the environment."""

    def test_defaults_to_mock_providers(self, monkeypatch) -> None:
        """With no env vars set, both providers should default to mock."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
        settings = Settings()
        assert settings.llm_provider == "mock"
        assert settings.search_provider == "mock"

    def test_env_vars_select_real_providers(self, monkeypatch) -> None:
        """Env vars should be able to select the real providers."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("SEARCH_PROVIDER", "wikipedia")
        settings = Settings()
        assert settings.llm_provider == "gemini"
        assert settings.search_provider == "wikipedia"

    def test_gemini_settings_from_env(self, monkeypatch) -> None:
        """Gemini API key and model should be read from the environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
        settings = Settings()
        assert settings.gemini_api_key == "test-key-123"
        assert settings.gemini_model == "gemini-test-model"

    def test_gemini_api_key_defaults_to_none(self, monkeypatch) -> None:
        """Missing GEMINI_API_KEY should surface as None, not a placeholder."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        settings = Settings()
        assert settings.gemini_api_key is None

    def test_each_instantiation_rereads_environment(self, monkeypatch) -> None:
        """Settings() should reflect env changes made between instantiations."""
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert Settings().llm_provider == "mock"
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert Settings().llm_provider == "gemini"
