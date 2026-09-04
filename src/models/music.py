# Background music / audio mixing data models (standalone BGM milestone)
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

EnergyLevel = Literal["low", "medium", "high"]


class BGMTrack(BaseModel):
    """One track in the approved, curated local BGM catalog.

    Every field describing rights (``source``/``license_type``/
    ``attribution_required``/``attribution_text``) must reflect the track's
    actual known origin - a track is never treated as copyright-safe on the
    strength of this model alone, only on the strength of accurate catalog
    metadata behind it.
    """

    track_id: str = Field(min_length=1, description="Stable unique identifier within the catalog")
    file_path: str = Field(min_length=1, description="Local filesystem path to the audio file")
    title: str = Field(min_length=1, description="Track title, as credited by its source")
    source: str = Field(
        min_length=1, description="Where this track came from, e.g. 'YouTube Audio Library'"
    )
    license_type: str = Field(
        min_length=1,
        description="Approved license identifier, e.g. 'youtube_audio_library_no_attribution'",
    )
    attribution_required: bool = Field(
        default=False, description="Whether crediting the creator is required when this track is used"
    )
    attribution_text: Optional[str] = Field(
        default=None, description="Exact credit text to use, if attribution_required is True"
    )
    genre: Optional[str] = Field(default=None, description="e.g. 'ambient', 'cinematic', 'electronic'")
    mood_tags: List[str] = Field(default_factory=list, description="e.g. ['calm', 'thoughtful', 'subtle']")
    energy_level: EnergyLevel = Field(default="medium")
    instrumental: bool = Field(default=True, description="False if the track has vocals/lyrics")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0)


class MusicPlan(BaseModel):
    """The desired music CHARACTERISTICS for a video - never a specific
    song - produced either by MusicContextPlanner (one LLM call for the
    whole script) or, when that is unavailable/fails, an equivalent
    deterministic fallback (see ``used_semantic_planning``)."""

    topic: str = Field(min_length=1)
    primary_mood: str = Field(min_length=1, description="e.g. 'thoughtful', 'energetic', 'warm'")
    secondary_mood: Optional[str] = Field(default=None)
    energy_level: EnergyLevel = Field(default="medium")
    preferred_genres: List[str] = Field(default_factory=list)
    preferred_instrumentation: List[str] = Field(default_factory=list)
    avoid_styles: List[str] = Field(
        default_factory=list, description="Styles/moods/energy that would NOT fit this video"
    )
    requires_neutral_subtle: bool = Field(
        default=False, description="True when the subject calls for especially restrained, non-distracting music"
    )
    reasoning_summary: str = Field(default="", description="Short explanation suitable for logs")
    used_semantic_planning: bool = Field(
        description="True if this plan came from a real LLM planning call; False if the "
        "deterministic fallback was used instead"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Why the deterministic fallback was used, if it was"
    )


class AudioMixResult(BaseModel):
    """Structured output of the Audio Mixing Service."""

    success: bool = Field(description="Whether a BGM-mixed video was produced successfully")
    output_path: Optional[str] = Field(
        default=None, description="Local filesystem path to the mixed MP4 copy, if successful"
    )
    source_video_path: Optional[str] = Field(
        default=None, description="Original video the music was mixed onto (never modified)"
    )
    music_plan: Optional[MusicPlan] = Field(default=None, description="The mood/context plan used for selection")
    selected_track: Optional[BGMTrack] = Field(default=None, description="The BGM track selected, if any")
    bgm_gain_db: Optional[float] = Field(default=None, description="Gain applied to the music track, in dB")
    ducking_used: bool = Field(default=False, description="Whether sidechain ducking under narration was applied")
    looped: bool = Field(default=False, description="Whether the track was looped to reach the video's duration")
    source_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    output_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    warnings: List[str] = Field(default_factory=list, description="Non-fatal notes about the mix")
    error: Optional[str] = Field(default=None, description="Error message if mixing failed")
