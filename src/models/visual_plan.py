# Structured visual plan consumed by VisualMediaService for query selection
# and semantic filtering - produced either by VisualContextPlanner (one LLM
# call for the whole script) or, when that is unavailable/fails, by an
# equivalent deterministic fallback (src.services.query_generation).
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SectionVisualPlan(BaseModel):
    """The visual plan for one script section: what it's actually about, in
    context, and what to search for/avoid when selecting stock media.

    ``avoid_concepts`` exists specifically to catch lexical ambiguity - a
    literal keyword match that would be visually wrong for what the section
    actually means (e.g. "the mind constructs a narrative" should not surface
    construction/building footage just because "construct" appears).
    """

    section_index: int = Field(ge=0)
    semantic_summary: str = Field(
        default="", description="One-sentence description of what this section is actually about"
    )
    visual_intents: List[str] = Field(
        default_factory=list, description="Concrete visual concepts this section calls for"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Ordered stock-search queries for this section, most specific first",
    )
    avoid_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts a literal/lexical reading of this section's wording could "
        "wrongly surface - rejected during candidate filtering",
    )
    neutral_fallback_queries: List[str] = Field(
        default_factory=list,
        description="Safe, generic queries related to this section or the overall topic, "
        "used when no specific/relevant candidate is available",
    )
    preferred_visual_types: Optional[List[str]] = Field(
        default=None, description="Optional preferred asset types/styles, e.g. ['video']"
    )
    confidence: Optional[str] = Field(
        default=None, description="Optional planner confidence for this section, e.g. 'high'/'medium'/'low'"
    )


class VisualPlan(BaseModel):
    """The complete per-section visual plan for a script, produced by one
    VisualContextPlanner call - or, if that call fails or isn't configured,
    an equivalent deterministic fallback plan (see ``used_semantic_planning``)."""

    topic: str = Field(description="The original script topic", min_length=1)
    sections: List[SectionVisualPlan] = Field(default_factory=list)
    used_semantic_planning: bool = Field(
        description="True if this plan came from a real LLM semantic-planning call; False "
        "if the deterministic fallback was used instead"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Why the deterministic fallback was used, if it was"
    )
