# Simple configuration using environment variables
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables.

    Fields use ``default_factory`` (instead of a bare default) so that each
    ``Settings()`` instantiation re-reads the environment, rather than
    freezing values at import time.
    """

    # LLM Provider: "mock" or "gemini"
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock"))

    # Search Provider: "mock" or "wikipedia"
    search_provider: str = field(default_factory=lambda: os.environ.get("SEARCH_PROVIDER", "mock"))

    # Voice Provider: "mock" or "edge"
    voice_provider: str = field(default_factory=lambda: os.environ.get("VOICE_PROVIDER", "mock"))
    voice_name: str = field(default_factory=lambda: os.environ.get("VOICE_NAME", "en-US-AriaNeural"))

    # Media Provider: "mock" or "pexels"
    media_provider: str = field(default_factory=lambda: os.environ.get("MEDIA_PROVIDER", "mock"))
    pexels_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("PEXELS_API_KEY") or None)

    # Transcription Provider (for captions): "mock" or "whisper"
    transcription_provider: str = field(
        default_factory=lambda: os.environ.get("TRANSCRIPTION_PROVIDER", "mock")
    )
    whisper_model_size: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL_SIZE", "base"))

    # Gemini. Defaults mirror src/llm/gemini.py's DEFAULT_GEMINI_MODEL/
    # DEFAULT_FALLBACK_MODEL (chosen via a real health check during the
    # LLM/Gemini Reliability Audit milestone - not duplicated by accident,
    # kept in sync deliberately since Settings has no import dependency on
    # the llm/ layer).
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or None)
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    gemini_fallback_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite") or None
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/youtube_ai"
        )
    )

    # FastAPI
    api_host: str = field(default_factory=lambda: os.environ.get("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("API_PORT", "8000")))

    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    # YouTube Data API v3 OAuth (installed-app/Desktop flow) - upload requires
    # user-authorized OAuth credentials, never a plain API key, so this is
    # separate from the unused youtube_api_key placeholder below.
    youtube_oauth_client_secret_path: str = field(
        default_factory=lambda: os.environ.get(
            "YOUTUBE_OAUTH_CLIENT_SECRET_PATH", os.path.join("secrets", "youtube_client_secret.json")
        )
    )
    youtube_oauth_token_path: str = field(
        default_factory=lambda: os.environ.get(
            "YOUTUBE_OAUTH_TOKEN_PATH", os.path.join("secrets", "youtube_token.json")
        )
    )

    # External API keys (placeholders, not used in MVP)
    youtube_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("YOUTUBE_API_KEY"))
    elevenlabs_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY"))
    serper_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("SERPER_API_KEY"))
    brave_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY"))
