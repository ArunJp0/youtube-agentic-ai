# Simple configuration using environment variables
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # LLM Provider
    llm_provider: str = os.environ.get("LLM_PROVIDER", "dummy")

    # Database
    database_url: str = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/youtube_ai")

    # FastAPI
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")

    # External API keys (placeholders, not used in MVP)
    youtube_api_key: Optional[str] = os.environ.get("YOUTUBE_API_KEY")
    elevenlabs_api_key: Optional[str] = os.environ.get("ELEVENLABS_API_KEY")
    serper_api_key: Optional[str] = os.environ.get("SERPER_API_KEY")
    brave_api_key: Optional[str] = os.environ.get("BRAVE_API_KEY")