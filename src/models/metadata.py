# YouTube upload metadata data models (Metadata Agent)
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """One YouTube chapter marker.

    ``timestamp_seconds`` is always deterministically derived from real
    section-timing data (see MetadataAgent) - never invented by the LLM.
    Only ``title`` may come from the LLM (or fall back to the section's own
    heading).
    """

    timestamp_seconds: float = Field(ge=0.0)
    timestamp_text: str = Field(
        min_length=1, description="YouTube-format timestamp, e.g. '0:00', '2:15', '1:05:30'"
    )
    title: str = Field(min_length=1, description="Chapter label reflecting the actual section content")


class MetadataResult(BaseModel):
    """Structured output of the Metadata Agent."""

    success: bool = Field(description="Whether usable YouTube metadata was produced")
    topic: str = Field(default="")
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    seo_summary: Optional[str] = Field(default=None, description="Short SEO-friendly summary")
    tags: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    chapters_available: bool = Field(
        default=False, description="False when chapters could not be safely derived - never fabricated"
    )
    chapters_omitted_reason: Optional[str] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Final video duration used for timing")
    output_path: Optional[str] = Field(default=None, description="Path to the written metadata JSON artifact, if any")
    llm_provider: Optional[str] = Field(default=None, description="LLM provider class/name used")
    llm_model: Optional[str] = Field(default=None, description="Underlying model actually used, if reported")
    used_fallback_model: Optional[bool] = Field(
        default=None, description="Whether the provider's fallback model was used, if reported"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal notes (e.g. dropped chapters)")
    error: Optional[str] = Field(default=None, description="Error message if metadata generation failed")
