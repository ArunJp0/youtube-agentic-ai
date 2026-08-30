# Video assembly data model
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class VideoAssemblyResult(BaseModel):
    """Structured output of the Video Assembly Service."""

    success: bool = Field(description="Whether a final MP4 was produced successfully")
    output_path: Optional[str] = Field(
        default=None, description="Local filesystem path to the assembled MP4, if successful"
    )
    duration_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Final video duration in seconds"
    )
    width: Optional[int] = Field(default=None, ge=0, description="Output video width in pixels")
    height: Optional[int] = Field(default=None, ge=0, description="Output video height in pixels")
    fps: Optional[float] = Field(default=None, ge=0.0, description="Output video frame rate")
    format: Optional[str] = Field(default=None, description="Output container format, e.g. 'mp4'")
    video_codec: Optional[str] = Field(default=None, description="Output video codec, e.g. 'h264'")
    audio_codec: Optional[str] = Field(default=None, description="Output audio codec, e.g. 'aac'")
    section_count: Optional[int] = Field(
        default=None, ge=0, description="Number of script sections assembled into the video"
    )
    section_durations_seconds: List[float] = Field(
        default_factory=list,
        description="Deterministically-calculated duration allocated to each section, in script order",
    )
    error: Optional[str] = Field(default=None, description="Error message if assembly failed")
