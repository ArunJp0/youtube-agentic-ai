# Thumbnail Agent data models
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# A small, deterministic, renderer-owned set - the LLM only ever picks one
# of these, never arbitrary pixel coordinates (see src/services/thumbnail_renderer.py).
Composition = Literal["subject_left", "subject_right", "centered"]
TextPosition = Literal["left", "right", "center"]


class ThumbnailPlan(BaseModel):
    """The desired thumbnail CONCEPT - mood/subject/composition/hook text -
    produced either by ThumbnailPlanner (one LLM call) or, when that is
    unavailable/fails, an equivalent deterministic fallback (see
    ``used_semantic_planning``). Never a set of pixel coordinates."""

    hook_text: str = Field(min_length=1, description="Short visual hook, distinct from the full video title")
    visual_concept: str = Field(default="", description="What the thumbnail should visually convey")
    search_query: str = Field(min_length=1, description="Stock-photo search query")
    mood: str = Field(default="neutral")
    subject: str = Field(default="", description="The video's main visual subject")
    composition: Composition = Field(default="centered")
    text_position: TextPosition = Field(
        default="center", description="Informational - the renderer's layout is driven by `composition`"
    )
    avoid_concepts: List[str] = Field(
        default_factory=list, description="Literal-but-wrong interpretations to steer image selection away from"
    )
    used_semantic_planning: bool = Field(
        description="True if this plan came from a real LLM planning call; False if the "
        "deterministic fallback was used instead"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Why the deterministic fallback was used, if it was"
    )


class ThumbnailSourceAsset(BaseModel):
    """Metadata about the stock image actually used, preserved for attribution."""

    provider: str = Field(min_length=1, description="e.g. 'pexels'")
    provider_asset_id: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    attribution: Optional[str] = Field(default=None)
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)


class ThumbnailResult(BaseModel):
    """Structured output of the Thumbnail Agent."""

    success: bool = Field(description="Whether a valid 1280x720 thumbnail was produced")
    topic: str = Field(default="")
    output_path: Optional[str] = Field(default=None, description="Path to the rendered thumbnail file, if successful")
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)
    plan: Optional[ThumbnailPlan] = Field(default=None)
    selected_asset: Optional[ThumbnailSourceAsset] = Field(default=None)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    used_fallback_model: Optional[bool] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, description="Error message if thumbnail generation failed")
