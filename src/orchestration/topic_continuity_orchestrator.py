# Bounded, multi-tier content-continuity strategy sitting ABOVE
# TopicPlannerAgent/ResearchAgent - AutonomousRunController's own topic-
# selection step delegates here (when configured) instead of trusting a
# single TopicPlannerAgent.plan_topic() call outright.
#
# Exhausting a handful of ranked current-news candidates must not be treated
# as a normal end-of-run condition on its own. Three bounded tiers, tried in
# order, each candidate validated against the exact same, unweakened
# ResearchAgent quality contract:
#
#   TIER 1 (primary):  the ranked candidate pool from ONE
#                       TopicPlannerAgent.plan_topic() call - try
#                       TopicSelectionResult.top_candidates in ranking
#                       order, never retrying the same candidate twice.
#   TIER 2 (evergreen): a SEPARATE, evergreen-only TopicPlannerAgent's own
#                        ranked candidates - only reached when every bounded
#                        TIER 1 candidate failed Research.
#   TIER 3 (reserve):   a small persistent reserve of evergreen topics that
#                        PREVIOUSLY passed Research in an earlier run (see
#                        QualifiedTopicReserveStore) - only reached when
#                        TIER 1 and TIER 2 are both exhausted. A reserve
#                        topic is always re-researched FRESH here (never
#                        trusting cached content), so stale time-sensitive
#                        facts are never assumed current.
#
# Genuine content insufficiency (ResearchAgentError) simply moves on to the
# next candidate/tier. An infrastructure-shaped failure
# (ResearchInfrastructureError - every search provider raised outright, or
# LLM synthesis itself failed) stops the search immediately instead of
# burning through unrelated candidates for what is actually a systemic
# problem that would affect any of them equally - see
# ResearchInfrastructureError's own docstring.
#
# Only when all configured tiers are genuinely exhausted does this return
# "topic_candidates_exhausted" - the sole condition under which an
# autonomous run may terminate for topic/research insufficiency.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set

from src.agents.research import ResearchAgent, ResearchAgentError, ResearchInfrastructureError
from src.agents.topic_planner_agent import TopicPlannerAgent, TopicPlannerAgentError
from src.models.topic_planner import TopicCandidate, TopicSelectionResult
from src.services.topic_reserve_store import QualifiedTopicReserveStore

# Centralized, configurable bounds (see src.config.settings for the
# environment-driven counterparts wired in production) - every tier is
# bounded; none of these ever means "retry forever".
DEFAULT_MAX_PRIMARY_CANDIDATES = 5
DEFAULT_MAX_EVERGREEN_CANDIDATES = 5
DEFAULT_MAX_RESERVE_ATTEMPTS = 5
# How many of TIER 2's own remaining (untried) ranked candidates get an
# opportunistic extra Research pass purely to seed TIER 3 for FUTURE runs -
# bounded and small since this is pure future-benefit cost on top of the
# current run's own work, not required for the current run to succeed.
DEFAULT_RESERVE_SEED_ATTEMPTS = 2

ContinuityStatus = Literal["selected", "topic_candidates_exhausted", "infrastructure_unavailable"]
ContinuityTier = Literal["primary", "evergreen", "reserve"]


@dataclass
class ContinuityOutcome:
    """Result of one ``TopicContinuityOrchestrator.select_researchable_topic()``
    call - either a validated topic ready for the downstream content
    pipeline, or an explicit terminal reason it could not find one."""

    status: ContinuityStatus
    topic: Optional[str] = None
    topic_source: Optional[str] = None
    tier: Optional[ContinuityTier] = None
    topic_plan_status: Optional[str] = None
    error_message: Optional[str] = None
    tried_topics: List[str] = field(default_factory=list)


class TopicContinuityOrchestrator:
    """Coordinates TIER 1 -> TIER 2 -> TIER 3 as described in this module's
    docstring. Has no Research/Topic-Planning business logic of its own -
    every actual capability (candidate discovery/ranking, research
    validation, reserve persistence) is an injected dependency this class
    only calls and inspects the typed result of.
    """

    def __init__(
        self,
        primary_topic_planner: TopicPlannerAgent,
        research_agent: ResearchAgent,
        evergreen_topic_planner: Optional[TopicPlannerAgent] = None,
        reserve_store: Optional[QualifiedTopicReserveStore] = None,
        max_primary_candidates: int = DEFAULT_MAX_PRIMARY_CANDIDATES,
        max_evergreen_candidates: int = DEFAULT_MAX_EVERGREEN_CANDIDATES,
        max_reserve_attempts: int = DEFAULT_MAX_RESERVE_ATTEMPTS,
        reserve_seed_attempts: int = DEFAULT_RESERVE_SEED_ATTEMPTS,
    ) -> None:
        self._primary_topic_planner = primary_topic_planner
        self._research_agent = research_agent
        self._evergreen_topic_planner = evergreen_topic_planner
        self._reserve_store = reserve_store
        self._max_primary_candidates = max_primary_candidates
        self._max_evergreen_candidates = max_evergreen_candidates
        self._max_reserve_attempts = max_reserve_attempts
        self._reserve_seed_attempts = reserve_seed_attempts

    async def select_researchable_topic(self) -> ContinuityOutcome:
        tried: Set[str] = set()

        # ---- TIER 1: ranked primary candidates ---------------------------
        primary_result = await self._safe_plan(self._primary_topic_planner)
        if primary_result is not None and primary_result.success and primary_result.selected_topic:
            outcome = await self._try_candidates(
                self._candidates_from(primary_result, self._max_primary_candidates), tried, "primary"
            )
            if outcome is not None:
                outcome.topic_plan_status = primary_result.status
                return outcome

        # ---- TIER 2: evergreen fallback discovery -------------------------
        if self._evergreen_topic_planner is not None:
            evergreen_result = await self._safe_plan(self._evergreen_topic_planner)
            if evergreen_result is not None and evergreen_result.success and evergreen_result.selected_topic:
                evergreen_candidates = self._candidates_from(evergreen_result, self._max_evergreen_candidates)
                outcome = await self._try_candidates(evergreen_candidates, tried, "evergreen")
                if outcome is not None:
                    outcome.topic_plan_status = evergreen_result.status
                    await self._seed_reserve(evergreen_candidates, tried)
                    return outcome
                await self._seed_reserve(evergreen_candidates, tried)

        # ---- TIER 3: pre-qualified evergreen reserve -----------------------
        if self._reserve_store is not None:
            outcome = await self._try_reserve(tried)
            if outcome is not None:
                return outcome

        return ContinuityOutcome(
            status="topic_candidates_exhausted",
            tried_topics=sorted(tried),
            error_message=(
                f"All content-continuity tiers exhausted after genuinely trying {len(tried)} "
                f"candidate(s): {', '.join(sorted(tried)) or 'none'}"
            ),
        )

    # ---- shared helpers ---------------------------------------------------

    @staticmethod
    async def _safe_plan(planner: TopicPlannerAgent) -> Optional[TopicSelectionResult]:
        """A Topic Planner failure (e.g. source unavailable) at any tier is
        never fatal to the overall continuity search - it simply means that
        tier produced no candidates, and the next tier gets its chance."""
        try:
            return await planner.plan_topic()
        except TopicPlannerAgentError:
            return None

    @staticmethod
    def _candidates_from(result: TopicSelectionResult, limit: int) -> List[TopicCandidate]:
        if not result.top_candidates:
            return []
        return [score.candidate for score in result.top_candidates[:limit]]

    async def _try_candidates(
        self, candidates: List[TopicCandidate], tried: Set[str], tier: ContinuityTier
    ) -> Optional[ContinuityOutcome]:
        """Try each candidate (in order, skipping any already tried this
        run) against Research. Returns a "selected" outcome the moment one
        passes, an "infrastructure_unavailable" outcome the moment an
        infrastructure-shaped failure is seen (stopping immediately rather
        than continuing to burn through candidates), or ``None`` to fall
        through to the next tier (every candidate here was genuinely
        content-insufficient - Research thresholds were never weakened to
        get past this)."""
        for candidate in candidates:
            if candidate.raw_title in tried:
                continue
            tried.add(candidate.raw_title)
            try:
                await self._research_agent.research(candidate.raw_title, topic_source=candidate.source)
            except ResearchInfrastructureError as e:
                return ContinuityOutcome(
                    status="infrastructure_unavailable",
                    tried_topics=sorted(tried),
                    error_message=str(e),
                )
            except ResearchAgentError:
                continue
            return ContinuityOutcome(
                status="selected",
                topic=candidate.raw_title,
                topic_source=candidate.source,
                tier=tier,
                tried_topics=sorted(tried),
            )
        return None

    async def _try_reserve(self, tried: Set[str]) -> Optional[ContinuityOutcome]:
        attempts = 0
        while attempts < self._max_reserve_attempts:
            entry = self._reserve_store.get_next_available(exclude_topics=tried)
            if entry is None:
                return None
            attempts += 1
            tried.add(entry.topic)
            try:
                # Always re-validated FRESH - a reserve entry never carries
                # cached research content, and old time-sensitive facts are
                # never assumed to still be current (see
                # QualifiedTopicReserveEntry's docstring).
                await self._research_agent.research(entry.topic, topic_source=entry.topic_source)
            except ResearchInfrastructureError as e:
                return ContinuityOutcome(
                    status="infrastructure_unavailable",
                    tried_topics=sorted(tried),
                    error_message=str(e),
                )
            except ResearchAgentError:
                # No longer genuinely researchable (e.g. stale/no-longer-
                # relevant) - consumed so it is never retried indefinitely,
                # never reused just because it was once qualified.
                self._reserve_store.mark_consumed(entry.entry_id, reason="revalidation_failed")
                continue
            self._reserve_store.mark_consumed(entry.entry_id, reason="published")
            return ContinuityOutcome(
                status="selected",
                topic=entry.topic,
                topic_source=entry.topic_source,
                tier="reserve",
                tried_topics=sorted(tried),
            )
        return None

    async def _seed_reserve(self, evergreen_candidates: List[TopicCandidate], tried: Set[str]) -> None:
        """Opportunistically qualify a small, bounded number of TIER 2's own
        remaining (untried) candidates purely to persist them into the TIER
        3 reserve for a FUTURE run - never required for the current run's
        own success, and never lets a failure here affect the outcome
        already decided above (best-effort, same spirit as
        TopicPlannerAgent._persist)."""
        if self._reserve_store is None or self._reserve_seed_attempts <= 0:
            return
        seeded = 0
        for candidate in evergreen_candidates:
            if seeded >= self._reserve_seed_attempts:
                return
            if candidate.raw_title in tried:
                continue
            tried.add(candidate.raw_title)
            try:
                research_result = await self._research_agent.research(candidate.raw_title, topic_source=candidate.source)
            except ResearchInfrastructureError:
                # Same systemic problem would affect every remaining
                # candidate too - stop seeding rather than repeating it.
                return
            except ResearchAgentError:
                continue
            seeded += 1
            try:
                self._reserve_store.add(
                    topic=candidate.raw_title,
                    topic_source=candidate.source,
                    category=candidate.category,
                    qualification_provider=research_result.source_provider,
                )
            except OSError:
                pass
