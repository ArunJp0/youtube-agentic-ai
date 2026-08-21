# Script data models for structured YouTube video script output
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class ScriptSection(BaseModel):
    """A single narrated section of the video script.

    Structured so a future Voice Agent can synthesize ``narration`` directly,
    and a future Visual Agent can use ``visual_notes``/``estimated_duration_seconds``
    to plan matching footage or graphics for this section.
    """

    heading: str = Field(description="Short internal label for this section", min_length=1)
    narration: str = Field(description="Spoken narration text for this section", min_length=1)
    visual_notes: Optional[str] = Field(
        default=None,
        description="Suggested visuals/b-roll for this section (input for a future Visual Agent)",
    )
    estimated_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated narration duration for this section, in seconds",
    )
    source_refs: List[str] = Field(
        default_factory=list,
        description="Source URLs/references supporting this section's content",
    )


class ScriptResult(BaseModel):
    """Structured YouTube video script produced by the Script Agent."""

    topic: str = Field(description="The original research topic", min_length=1)
    video_title: str = Field(description="Suggested YouTube video title", min_length=1)
    hook: str = Field(
        description="Opening narration line(s) meant to grab attention in the first seconds",
        min_length=1,
    )
    introduction: str = Field(
        description="Narration introducing the video's subject and what it will cover",
        min_length=1,
    )
    sections: List[ScriptSection] = Field(
        default_factory=list,
        description="Main narrated sections of the script",
    )
    conclusion: str = Field(description="Closing narration summarizing the video", min_length=1)
    call_to_action: str = Field(
        description="Closing narration asking viewers to like/subscribe/comment",
        min_length=1,
    )
    estimated_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated narration duration for the whole script, in seconds",
    )
    sources: List[HttpUrl] = Field(
        default_factory=list,
        description="Source URLs carried over from the originating ResearchResult",
    )
    script_notes: Optional[str] = Field(
        default=None,
        description="Additional notes/caveats, typically inherited from the research",
    )
