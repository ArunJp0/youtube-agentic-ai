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

    # Gemini
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or None)
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    gemini_fallback_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite") or None
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

    # External API keys (placeholders, not used in MVP)
    youtube_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("YOUTUBE_API_KEY"))
    elevenlabs_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY"))
    serper_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("SERPER_API_KEY"))
    brave_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY"))
