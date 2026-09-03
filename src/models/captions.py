# Caption/subtitle data models (Caption Service)
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class CaptionSegment(BaseModel):
    """One readable, screen-ready subtitle entry.

    Not necessarily one raw transcription segment: a long transcribed
    segment is split into multiple readable CaptionSegments (see
    src/services/caption_segmentation.py); this model represents the
    final, already-formatted result.
    """

    index: int = Field(
        ge=1,
        description="1-based display order at creation time. SRT numbering is always "
        "re-derived from list order when writing the file, not trusted from this field.",
    )
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str = Field(
        min_length=1,
        description="Renderable caption text, with embedded newlines where wrapped to multiple lines",
    )

    @model_validator(mode="after")
    def _validate_timing(self) -> "CaptionSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class CaptionResult(BaseModel):
    """Structured output of the Caption Service."""

    success: bool = Field(description="Whether a captioned video was produced successfully")
    segments: List[CaptionSegment] = Field(
        default_factory=list, description="Final readable caption segments, in chronological order"
    )
    srt_path: Optional[str] = Field(
        default=None, description="Local filesystem path to the generated .srt file, if successful"
    )
    captioned_video_path: Optional[str] = Field(
        default=None, description="Local filesystem path to the captioned MP4 copy, if successful"
    )
    source_audio_path: Optional[str] = Field(
        default=None, description="Narration audio file transcription was based on"
    )
    source_video_path: Optional[str] = Field(
        default=None, description="Original assembled MP4 the captions were burned onto (never modified)"
    )
    transcription_provider: Optional[str] = Field(
        default=None, description="Transcription provider used, e.g. 'mock' or 'whisper'"
    )
    transcription_model: Optional[str] = Field(
        default=None, description="Underlying transcription model identifier, if applicable"
    )
    narration_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    video_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    captioned_duration_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Duration of the final captioned MP4, probed after rendering"
    )
    error: Optional[str] = Field(default=None, description="Error message if caption generation failed")
