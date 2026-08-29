# Visual media data models (Visual Media Service)
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MediaAsset(BaseModel):
    """A single visual asset (image or video) selected for one script section."""

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
    success: bool = Field(description="Whether an asset was found and downloaded successfully")
    error: Optional[str] = Field(default=None, description="Error message if retrieval failed")


class SectionMediaMapping(BaseModel):
    """The asset(s) selected for one specific script section."""

    section_index: int = Field(ge=0)
    section_heading: str = Field(min_length=1)
    search_query: str = Field(description="Search query derived for this section", min_length=1)
    assets: List[MediaAsset] = Field(
        default_factory=list, description="Assets selected for this section (usually one)"
    )


class VisualResult(BaseModel):
    """Structured media plan produced by the Visual Media Service: one or
    more assets per script section, ready to be handed to a future Video
    Assembly Service."""

    topic: str = Field(description="The original script topic", min_length=1)
    provider: str = Field(description="Media provider used, e.g. 'mock' or 'pexels'", min_length=1)
    sections: List[SectionMediaMapping] = Field(
        default_factory=list, description="Section-to-asset mapping, in section order"
    )
    success: bool = Field(description="Whether every section got a usable asset")
    error: Optional[str] = Field(
        default=None, description="Summary error if one or more sections failed"
    )
