# Topic Planner Agent data models.
#
# The planner is a pre-Research SELECTION step, not a research/content
# agent: it only chooses ONE topic string for the existing ResearchAgent to
# investigate - it never researches, scripts, or judges factual content
# itself. See src.agents.topic_planner_agent.
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PlannerStatus = Literal[
    "selected", "no_candidates", "source_unavailable", "all_candidates_duplicate", "all_candidates_stale"
]


class TopicCandidate(BaseModel):
    """One raw candidate topic before scoring, already normalized for
    duplicate comparison.

    ``source`` identifies which TopicSourceProvider discovered this
    candidate (e.g. "youtube", "current_news") - ``source_name`` is a
    separate, optional field for the actual publisher/outlet a news
    candidate came from (e.g. "BBC News"), when the two differ.
    """

    raw_title: str = Field(min_length=1, description="The candidate's original, human-readable title")
    normalized_title: str = Field(min_length=1, description="Deterministically normalized form, for dedup/comparison")
    source: str = Field(min_length=1, description="Name of the TopicSourceProvider that produced this candidate")
    source_id: Optional[str] = Field(default=None, description="Source-specific id (e.g. a YouTube video id)")
    category: Optional[str] = Field(default=None, description="Source-reported category/niche, if any")
    popularity_signal: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Normalized 0-1 popularity/interest signal from the source, if any"
    )

    # ---- current/trending-news-specific fields (all optional/additive -
    # a YouTube-sourced evergreen candidate leaves every one of these at
    # its default, so existing behavior for evergreen mode is unaffected) --
    source_name: Optional[str] = Field(default=None, description="Actual publisher/outlet name, e.g. 'BBC News'")
    source_url: Optional[str] = Field(default=None, description="Stable URL/identifier for the source item")
    published_at: Optional[str] = Field(default=None, description="ISO 8601 UTC publish timestamp, if known")
    discovered_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp this candidate was fetched")
    market: Optional[str] = Field(default=None, description="Region/market this candidate was discovered for")
    distinct_source_count: int = Field(
        default=1, ge=1, description="How many independent sources reported substantially the same story"
    )
    is_sensitive: bool = Field(
        default=False, description="True if a deterministic keyword heuristic flagged this as a sensitive category"
    )
    sensitivity_reasons: List[str] = Field(
        default_factory=list, description="Which sensitive category keyword(s) matched, for downstream Compliance/Research"
    )


class SemanticTopicJudgment(BaseModel):
    """One candidate's semantic sub-scores from the bounded batch LLM
    ranking call - correlated back to its candidate by ``normalized_title``."""

    normalized_title: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    evergreen_score: float = Field(ge=0.0, le=1.0)
    suitability_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


class TopicRankingOutcome(BaseModel):
    """Outcome of the single bounded batch semantic-ranking LLM call.

    ``performed=False`` means the call failed/was unavailable - this is
    never treated as "no opinion", it still carries one
    ``SemanticTopicJudgment`` per input candidate, filled with a neutral
    deterministic fallback score - see TopicRankingPlanner.
    """

    performed: bool
    judgments: List[SemanticTopicJudgment] = Field(default_factory=list)
    fallback_reason: Optional[str] = Field(default=None)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    used_fallback_model: Optional[bool] = Field(default=None)


class TopicScore(BaseModel):
    """Full scoring breakdown for one candidate - combines the source's own
    popularity signal, novelty against history, and (semantic or
    deterministic-fallback) relevance/evergreen/suitability judgments into
    one bounded ``total_score``."""

    candidate: TopicCandidate
    relevance_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    evergreen_score: float = Field(ge=0.0, le=1.0)
    suitability_score: float = Field(ge=0.0, le=1.0)
    popularity_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")

    # Optional trending-mode signals - None when not applicable (e.g. an
    # evergreen YouTube candidate has no publish timestamp to score
    # freshness from). See TopicPlannerAgent._combine_scores: when every
    # one of these is None for a candidate, total_score is computed with
    # EXACTLY the original 5-signal formula, unchanged.
    freshness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    market_relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    category_relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TopicSelectionResult(BaseModel):
    """Structured output of one TopicPlannerAgent.plan_topic() call - the
    typed value later passed into the existing ResearchAgent/run_pipeline.

    ``success=False`` (see ``status``) means no topic could be safely
    selected - the planner never invents/fabricates a topic in that case.
    """

    success: bool
    status: PlannerStatus
    selected_topic: Optional[str] = Field(default=None)
    rationale: str = Field(default="")
    category: Optional[str] = Field(default=None)
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sources_used: List[str] = Field(default_factory=list)
    candidate_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    fallback_used: bool = Field(default=False, description="True if deterministic (non-semantic) scoring was used")
    used_semantic_ranking: bool = Field(default=False)
    selected_at: str = Field(default="", description="ISO 8601 UTC timestamp this selection was made")
    error: Optional[str] = Field(default=None)
    top_candidates: List[TopicScore] = Field(
        default_factory=list, description="Top-scored candidates considered, for audit/traceability"
    )
    is_sensitive: bool = Field(
        default=False,
        description="Mirrors the selected candidate's sensitivity flag - never used to suppress selection, only "
        "to signal downstream Research/Compliance to apply stricter treatment",
    )
    sensitivity_reasons: List[str] = Field(default_factory=list)
