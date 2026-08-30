# Visual media data models (Visual Media Service)
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MediaAsset(BaseModel):
    """A single visual asset (image or video) selected for one visual slot
    within a script section. A section may have several of these, in order."""

    provider: str = Field(description="Media provider used, e.g. 'mock' or 'pexels'", min_length=1)
    asset_type: Optional[str] = Field(
        default=None, description="'image' or 'video', set when successful"
    )
    local_file_path: Optional[str] = Field(
        default=None, description="Local filesystem path to the downloaded asset, if successful"
    )
    source_url: Optional[str] = Field(
        default=None, description="Original hosted page/URL of the asset, for attribution"
    )
    provider_asset_id: Optional[str] = Field(
        default=None,
        description="Provider-specific unique asset ID (e.g. Pexels numeric ID), used for "
        "reliable duplicate detection independent of URL formatting",
    )
    attribution: Optional[str] = Field(
        default=None, description="Author/photographer credit, if available"
    )
    search_query: str = Field(description="Search query used to find this asset", min_length=1)
    section_index: int = Field(ge=0, description="Index of the ScriptSection this asset is for")
    duration_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Clip duration in seconds, if this is a video asset"
    )
    width: Optional[int] = Field(default=None, ge=0, description="Asset width in pixels, if known")
    height: Optional[int] = Field(default=None, ge=0, description="Asset height in pixels, if known")
    reused: bool = Field(
        default=False,
        description="True if this asset reuses an already-downloaded file rather than a fresh download",
    )
    relevance_tier: Optional[str] = Field(
        default=None,
        description="Diagnostic selection tier: 'high' (matched a specific visual-plan query), "
        "'neutral' (matched a neutral/fallback query), or 'reused' (reused an already-"
        "downloaded asset). Not a guarantee of factual visual correctness.",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Diagnostic 0-1 keyword-overlap score between the selected candidate's "
        "metadata and the section's visual plan, for QC/diagnostics only",
    )
    success: bool = Field(description="Whether an asset was found and downloaded successfully")
    error: Optional[str] = Field(default=None, description="Error message if retrieval failed")


class SectionMediaMapping(BaseModel):
    """The ordered visual asset(s) selected for one specific script section.

    A section is split into one or more time slots (see
    ``VisualMediaService.calculate_slot_count``), each assigned its own
    ``MediaAsset``, so longer sections get more visual variety instead of a
    single clip looping for their entire duration.
    """

    section_index: int = Field(ge=0)
    section_heading: str = Field(min_length=1)
    search_queries: List[str] = Field(
        default_factory=list,
        description="Search query used for each visual slot in this section, in order",
    )
    planned_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="This section's total allocated timeline duration"
    )
    semantic_summary: Optional[str] = Field(
        default=None,
        description="One-sentence description of what this section's visuals are meant to "
        "convey, carried over from the visual plan",
    )
    avoid_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts the visual plan flagged to avoid selecting for this section",
    )
    assets: List[MediaAsset] = Field(
        default_factory=list,
        description="Assets selected for this section's visual slots, in playback order",
    )


class VisualResult(BaseModel):
    """Structured media plan produced by the Visual Media Service: one or
    more ordered assets per script section, ready to be handed to the
    Video Assembly Service."""

    topic: str = Field(description="The original script topic", min_length=1)
    provider: str = Field(description="Media provider used, e.g. 'mock' or 'pexels'", min_length=1)
    sections: List[SectionMediaMapping] = Field(
        default_factory=list, description="Section-to-asset mapping, in section order"
    )
    success: bool = Field(description="Whether every section got at least one usable asset")
    error: Optional[str] = Field(
        default=None, description="Summary error if one or more sections failed"
    )
    semantic_planning_used: bool = Field(
        default=False,
        description="True if a real LLM semantic visual plan (VisualContextPlanner) was used "
        "to select media for this result; False if the deterministic fallback was used",
    )
    semantic_planning_fallback_reason: Optional[str] = Field(
        default=None, description="Why the deterministic fallback was used, if it was"
    )
