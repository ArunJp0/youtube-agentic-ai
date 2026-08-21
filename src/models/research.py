# Research data models for structured research output
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class ResearchFact(BaseModel):
    """A single fact with source attribution and confidence."""
    claim: str = Field(description="The factual claim")
    source: Optional[str] = Field(default=None, description="Source of the fact")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0-1.0",
    )


class ResearchResult(BaseModel):
    """Structured research output from the Research Agent."""
    topic: str = Field(
        description="The research topic",
        min_length=1,
    )
    summary: str = Field(
        description="High-level summary of findings",
        min_length=1,
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Bullet-worthy key points",
    )
    facts: List[ResearchFact] = Field(
        default_factory=list,
        description="Verified facts with source attribution",
    )
    sources: List[HttpUrl] = Field(
        default_factory=list,
        description="URLs of source material",
    )
    research_notes: Optional[str] = Field(
        default=None,
        description="Additional notes from the research process",
    )