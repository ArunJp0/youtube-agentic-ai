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

    def test_voice_defaults_to_mock_with_default_voice_name(self, monkeypatch) -> None:
        monkeypatch.delenv("VOICE_PROVIDER", raising=False)
        monkeypatch.delenv("VOICE_NAME", raising=False)
        settings = Settings()
        assert settings.voice_provider == "mock"
        assert settings.voice_name == "en-US-AriaNeural"

    def test_voice_env_vars_select_real_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("VOICE_PROVIDER", "edge")
        monkeypatch.setenv("VOICE_NAME", "en-US-GuyNeural")
        settings = Settings()
        assert settings.voice_provider == "edge"
        assert settings.voice_name == "en-US-GuyNeural"

    def test_media_defaults_to_mock_with_no_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("MEDIA_PROVIDER", raising=False)
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        settings = Settings()
        assert settings.media_provider == "mock"
        assert settings.pexels_api_key is None

    def test_media_env_vars_select_real_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("MEDIA_PROVIDER", "pexels")
        monkeypatch.setenv("PEXELS_API_KEY", "test-pexels-key")
        settings = Settings()
        assert settings.media_provider == "pexels"
        assert settings.pexels_api_key == "test-pexels-key"

    def test_autonomous_defaults_are_disabled_and_never_auto_publish(self, monkeypatch) -> None:
        """A fresh deployment must never accidentally start autonomous
        execution or publish publicly."""
        for var in (
            "AUTONOMOUS_ENABLED",
            "AUTONOMOUS_SCHEDULE",
            "AUTONOMOUS_TIMEZONE",
            "AUTONOMOUS_PUBLISHING_MODE",
            "AUTONOMOUS_MAX_RUN_DURATION",
            "AUTONOMOUS_LOCK_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)
        settings = Settings()
        assert settings.autonomous_enabled is False
        assert settings.autonomous_publishing_mode == "disabled"
        assert settings.autonomous_timezone == "UTC"
        assert settings.autonomous_max_run_duration > 0
        assert settings.autonomous_lock_timeout > 0

    def test_autonomous_env_vars_are_read(self, monkeypatch) -> None:
        monkeypatch.setenv("AUTONOMOUS_ENABLED", "true")
        monkeypatch.setenv("AUTONOMOUS_SCHEDULE", "0 */6 * * *")
        monkeypatch.setenv("AUTONOMOUS_TIMEZONE", "Asia/Kolkata")
        monkeypatch.setenv("AUTONOMOUS_PUBLISHING_MODE", "private")
        monkeypatch.setenv("AUTONOMOUS_MAX_RUN_DURATION", "3600")
        monkeypatch.setenv("AUTONOMOUS_LOCK_TIMEOUT", "600")
        settings = Settings()
        assert settings.autonomous_enabled is True
        assert settings.autonomous_schedule == "0 */6 * * *"
        assert settings.autonomous_timezone == "Asia/Kolkata"
        assert settings.autonomous_publishing_mode == "private"
        assert settings.autonomous_max_run_duration == 3600
        assert settings.autonomous_lock_timeout == 600

    def test_script_duration_defaults_to_standard_profile_no_override(self, monkeypatch) -> None:
        monkeypatch.delenv("SCRIPT_DURATION_PROFILE", raising=False)
        monkeypatch.delenv("SCRIPT_TARGET_DURATION_MINUTES", raising=False)
        settings = Settings()
        assert settings.script_duration_profile == "standard"
        assert settings.script_target_duration_minutes is None

    def test_script_duration_env_vars_are_read(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRIPT_DURATION_PROFILE", "long")
        monkeypatch.setenv("SCRIPT_TARGET_DURATION_MINUTES", "9.5")
        settings = Settings()
        assert settings.script_duration_profile == "long"
        assert settings.script_target_duration_minutes == 9.5

    def test_script_target_duration_minutes_empty_string_is_none(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRIPT_TARGET_DURATION_MINUTES", "")
        settings = Settings()
        assert settings.script_target_duration_minutes is None

    def test_ai_video_defaults_are_disabled_with_stock_fallback_on(self, monkeypatch) -> None:
        for var in (
            "AI_VIDEO_ENABLED",
            "AI_VIDEO_PROVIDER",
            "AI_VIDEO_MODEL",
            "AI_VIDEO_LOCAL_CLIPS_DIR",
            "AI_VIDEO_MAX_CLIPS_PER_RUN",
            "AI_VIDEO_MAX_RETRIES",
            "AI_VIDEO_MAX_COST_USD",
            "STOCK_FALLBACK_ENABLED",
        ):
            monkeypatch.delenv(var, raising=False)
        settings = Settings()
        assert settings.ai_video_enabled is False
        assert settings.ai_video_provider == "mock"
        assert settings.ai_video_model is None
        assert settings.ai_video_local_clips_dir is None
        assert settings.ai_video_max_clips_per_run > 0
        assert settings.ai_video_max_retries >= 0
        assert settings.ai_video_max_cost_usd is None
        assert settings.stock_fallback_enabled is True

    def test_ai_video_env_vars_are_read(self, monkeypatch) -> None:
        monkeypatch.setenv("AI_VIDEO_ENABLED", "true")
        monkeypatch.setenv("AI_VIDEO_PROVIDER", "local")
        monkeypatch.setenv("AI_VIDEO_MODEL", "kling-v2")
        monkeypatch.setenv("AI_VIDEO_LOCAL_CLIPS_DIR", "assets/ai_video_demo")
        monkeypatch.setenv("AI_VIDEO_MAX_CLIPS_PER_RUN", "3")
        monkeypatch.setenv("AI_VIDEO_MAX_RETRIES", "1")
        monkeypatch.setenv("AI_VIDEO_MAX_COST_USD", "2.5")
        monkeypatch.setenv("STOCK_FALLBACK_ENABLED", "false")
        settings = Settings()
        assert settings.ai_video_enabled is True
        assert settings.ai_video_provider == "local"
        assert settings.ai_video_model == "kling-v2"
        assert settings.ai_video_local_clips_dir == "assets/ai_video_demo"
        assert settings.ai_video_max_clips_per_run == 3
        assert settings.ai_video_max_retries == 1
        assert settings.ai_video_max_cost_usd == 2.5
        assert settings.stock_fallback_enabled is False
