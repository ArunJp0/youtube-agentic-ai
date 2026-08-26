# Voice generation data model
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VoiceResult(BaseModel):
    """Structured output of narration audio generation (Voice Service)."""

    audio_file_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to the generated audio file, if successful",
    )
    provider: str = Field(description="Voice provider used, e.g. 'mock' or 'edge'", min_length=1)
    voice_name: str = Field(description="Provider-specific voice identifier used", min_length=1)
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Narration audio duration in seconds, if determinable",
    )
    format: str = Field(description="Audio file format/extension, e.g. 'mp3'", min_length=1)
    success: bool = Field(description="Whether audio generation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
