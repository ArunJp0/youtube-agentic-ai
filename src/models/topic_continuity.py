# TIER 3 (Qualified Evergreen Reserve) data model - see
# src.orchestration.topic_continuity_orchestrator for the full multi-tier
# content-continuity strategy this supports.
#
# Deliberately minimal: an entry is identity + provenance only, never a
# cached ResearchResult/research content. A reserve topic is always
# re-researched FRESH (via ResearchAgent.research()) at the moment it is
# actually consumed - this is what "revalidate stale research rather than
# assuming old time-sensitive facts remain current" means in practice here,
# without needing a separate staleness-window algorithm. The reserve is
# evergreen-only by construction (see QualifiedTopicReserveStore/
# TopicContinuityOrchestrator - only evergreen-fallback candidates are ever
# added), which is also why "assume old facts are stale" is a non-issue for
# what's actually stored: a genuinely evergreen topic's IDENTITY doesn't go
# stale even though its cached research still must never be trusted.
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ReserveEntryState = Literal["available", "consumed"]


class QualifiedTopicReserveEntry(BaseModel):
    """One persisted, previously-research-qualified evergreen topic held in
    reserve for a future autonomous run's TIER 3 content-continuity
    fallback.

    A caller must only ever construct/store one of these for a topic that
    has genuinely just passed ``ResearchAgent.research()`` - this model (and
    ``QualifiedTopicReserveStore``) never itself performs or re-validates
    that judgment.
    """

    entry_id: str = Field(min_length=1, description="Stable identifier - also this entry's persisted filename")
    topic: str = Field(min_length=1, description="The exact topic string to re-research - mirrors TopicCandidate.raw_title")
    topic_source: Optional[str] = Field(
        default=None,
        description="Mirrors TopicCandidate.source at the time this entry was qualified (e.g. 'youtube', 'mock') - "
        "never 'current_news': the reserve is evergreen-only by construction.",
    )
    category: Optional[str] = Field(default=None, description="Mirrors TopicCandidate.category, when known")
    qualified_at: str = Field(min_length=1, description="ISO 8601 UTC timestamp this topic last passed the Research quality contract")
    qualification_provider: Optional[str] = Field(
        default=None,
        description="ResearchResult.source_provider from the qualifying research() call - provenance only, never cached content",
    )
    state: ReserveEntryState = Field(default="available")
    consumed_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp this entry was consumed, if any")
    consumed_reason: Optional[str] = Field(
        default=None, description="e.g. 'published' or 'revalidation_failed' - diagnostic only, never branches behavior"
    )
