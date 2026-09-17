# Tests for the multi-tier content-continuity strategy (see
# src.orchestration.topic_continuity_orchestrator): TIER 1 (ranked primary
# candidates) -> TIER 2 (dedicated evergreen fallback discovery) -> TIER 3
# (persistent qualified evergreen reserve). Topic insufficiency by itself
# must never end a run while another eligible candidate exists at any tier;
# only genuine exhaustion of every configured tier may terminate with
# "topic_candidates_exhausted". A systemic infrastructure failure must stop
# the search immediately rather than being treated as "every topic is bad".
#
# All mocked - no real network/LLM calls. TopicPlannerAgent and
# ResearchAgent are both fully stubbed; only TopicContinuityOrchestrator's
# own coordination logic is under test.
from __future__ import annotations

from typing import List, Optional

import pytest

from src.agents.research import ResearchAgentError, ResearchInfrastructureError
from src.agents.topic_planner_agent import TopicPlannerAgentError
from src.models.research import ResearchResult
from src.models.topic_planner import TopicCandidate, TopicScore, TopicSelectionResult
from src.orchestration.topic_continuity_orchestrator import TopicContinuityOrchestrator
from src.services.topic_reserve_store import QualifiedTopicReserveStore

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def make_candidate(title: str, source: str = "mock", category: Optional[str] = None) -> TopicCandidate:
    return TopicCandidate(raw_title=title, normalized_title=title.lower(), source=source, category=category)


def make_score(title: str, source: str = "mock", total_score: float = 0.8) -> TopicScore:
    return TopicScore(
        candidate=make_candidate(title, source=source),
        relevance_score=0.8,
        novelty_score=0.8,
        evergreen_score=0.8,
        suitability_score=0.8,
        popularity_score=0.8,
        total_score=total_score,
    )


def make_plan_result(titles: List[str], source: str = "mock", status: str = "selected") -> TopicSelectionResult:
    scores = [make_score(t, source=source, total_score=1.0 - i * 0.1) for i, t in enumerate(titles)]
    return TopicSelectionResult(
        success=True,
        status=status,
        selected_topic=titles[0] if titles else None,
        top_candidates=scores,
        selected_topic_source=source,
    )


def make_empty_plan_result(status: str = "no_candidates") -> TopicSelectionResult:
    return TopicSelectionResult(success=False, status=status, selected_topic=None)


def make_research_result(topic: str, provider: str = "wikipedia") -> ResearchResult:
    return ResearchResult(topic=topic, summary="A summary.", source_provider=provider)


class StubTopicPlanner:
    """Replaces TopicPlannerAgent entirely - never touches a real
    TopicSourceProvider/LLMProvider."""

    def __init__(self, result: Optional[TopicSelectionResult] = None, error: Optional[Exception] = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def plan_topic(self) -> TopicSelectionResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class StubResearchAgent:
    """Replaces ResearchAgent entirely - a topic "passes" if it's in
    ``passing_topics``; otherwise raises the configured error (content-
    insufficiency by default, or an injected infrastructure error for
    specific topics via ``infra_failure_topics``)."""

    def __init__(
        self,
        passing_topics: Optional[List[str]] = None,
        infra_failure_topics: Optional[List[str]] = None,
    ) -> None:
        self.passing_topics = set(passing_topics or [])
        self.infra_failure_topics = set(infra_failure_topics or [])
        self.calls: List[str] = []

    async def research(self, topic: str, topic_source: Optional[str] = None) -> ResearchResult:
        self.calls.append(topic)
        if topic in self.infra_failure_topics:
            raise ResearchInfrastructureError(f"simulated infrastructure outage researching '{topic}'")
        if topic in self.passing_topics:
            return make_research_result(topic, provider=topic_source or "wikipedia")
        raise ResearchAgentError(f"Insufficient DIRECT research material for '{topic}'")


# ---------------------------------------------------------------------------
# TIER 1: ranked primary candidates
# ---------------------------------------------------------------------------


class TestTier1PrimaryCandidates:
    @pytest.mark.asyncio
    async def test_primary_candidate_one_fails_another_primary_succeeds(self, tmp_path) -> None:
        """Requirement 1: primary candidate #1 fails Research -> the next
        ranked primary candidate is tried and succeeds - never treated as
        run-ending on its own."""
        primary = StubTopicPlanner(result=make_plan_result(["Topic A", "Topic B", "Topic C"]))
        research = StubResearchAgent(passing_topics=["Topic B"])
        orchestrator = TopicContinuityOrchestrator(primary_topic_planner=primary, research_agent=research)

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Topic B"
        assert outcome.tier == "primary"
        assert research.calls == ["Topic A", "Topic B"]  # never retries "Topic A"
        assert primary.calls == 1  # ONE plan_topic() call reused for every primary candidate

    @pytest.mark.asyncio
    async def test_never_retries_the_same_primary_candidate_twice(self, tmp_path) -> None:
        primary = StubTopicPlanner(result=make_plan_result(["Topic A", "Topic A", "Topic B"]))
        research = StubResearchAgent(passing_topics=["Topic B"])
        orchestrator = TopicContinuityOrchestrator(primary_topic_planner=primary, research_agent=research)

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert research.calls == ["Topic A", "Topic B"]  # duplicate "Topic A" entry never re-tried

    @pytest.mark.asyncio
    async def test_primary_candidates_bounded_by_max_primary_candidates(self, tmp_path) -> None:
        titles = [f"Topic {i}" for i in range(10)]
        primary = StubTopicPlanner(result=make_plan_result(titles))
        research = StubResearchAgent(passing_topics=[])  # nothing ever passes
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, max_primary_candidates=3
        )

        await orchestrator.select_researchable_topic()

        assert research.calls == titles[:3]  # never tries beyond the configured bound


# ---------------------------------------------------------------------------
# TIER 2: evergreen fallback discovery
# ---------------------------------------------------------------------------


class TestTier2EvergreenFallback:
    @pytest.mark.asyncio
    async def test_all_bounded_primary_candidates_fail_evergreen_fallback_succeeds(self, tmp_path) -> None:
        """Requirement 2: every bounded TIER 1 candidate fails -> a
        dedicated evergreen-only Topic Planner is consulted and ITS ranked
        candidates are tried."""
        primary = StubTopicPlanner(result=make_plan_result(["News A", "News B"], source="current_news"))
        evergreen = StubTopicPlanner(result=make_plan_result(["Evergreen X", "Evergreen Y"], source="youtube"))
        research = StubResearchAgent(passing_topics=["Evergreen Y"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, evergreen_topic_planner=evergreen
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Evergreen Y"
        assert outcome.tier == "evergreen"
        assert research.calls == ["News A", "News B", "Evergreen X", "Evergreen Y"]
        assert evergreen.calls == 1

    @pytest.mark.asyncio
    async def test_no_evergreen_planner_configured_skips_tier_2_cleanly(self, tmp_path) -> None:
        primary = StubTopicPlanner(result=make_plan_result(["News A"], source="current_news"))
        research = StubResearchAgent(passing_topics=[])
        orchestrator = TopicContinuityOrchestrator(primary_topic_planner=primary, research_agent=research)

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"

    @pytest.mark.asyncio
    async def test_evergreen_planner_failure_falls_through_without_crashing(self, tmp_path) -> None:
        primary = StubTopicPlanner(result=make_empty_plan_result())
        evergreen = StubTopicPlanner(error=TopicPlannerAgentError("evergreen source unavailable"))
        research = StubResearchAgent(passing_topics=[])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, evergreen_topic_planner=evergreen
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"


# ---------------------------------------------------------------------------
# TIER 3: pre-qualified evergreen reserve
# ---------------------------------------------------------------------------


class TestTier3QualifiedReserve:
    @pytest.mark.asyncio
    async def test_primary_and_evergreen_fail_unused_reserve_succeeds(self, tmp_path) -> None:
        """Requirement 3: TIER 1 and TIER 2 both exhausted -> an unused,
        previously-qualified reserve topic is tried and succeeds."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        reserve_store.add(topic="Reserve Topic", topic_source="youtube", qualification_provider="wikipedia")

        primary = StubTopicPlanner(result=make_plan_result(["News A"], source="current_news"))
        evergreen = StubTopicPlanner(result=make_plan_result(["Evergreen X"], source="youtube"))
        research = StubResearchAgent(passing_topics=["Reserve Topic"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary,
            research_agent=research,
            evergreen_topic_planner=evergreen,
            reserve_store=reserve_store,
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Reserve Topic"
        assert outcome.tier == "reserve"

    @pytest.mark.asyncio
    async def test_consumed_reserve_topics_are_never_reused(self, tmp_path) -> None:
        """Requirement 4: a reserve entry already consumed by a PRIOR run
        must never be returned again by a later run."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        entry = reserve_store.add(topic="Already Used Topic", topic_source="youtube")
        reserve_store.mark_consumed(entry.entry_id, reason="published")

        primary = StubTopicPlanner(result=make_empty_plan_result())
        research = StubResearchAgent(passing_topics=["Already Used Topic"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, reserve_store=reserve_store
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"
        assert "Already Used Topic" not in research.calls

    @pytest.mark.asyncio
    async def test_a_topic_consumed_within_this_run_is_not_reused_within_the_same_run(self, tmp_path) -> None:
        """Never-reuse also holds WITHIN one run: once TIER 3 consumes a
        topic (success or failed revalidation), get_next_available must not
        hand it back again in the same search."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        reserve_store.add(topic="Only Reserve Topic", topic_source="youtube")

        primary = StubTopicPlanner(result=make_empty_plan_result())
        research = StubResearchAgent(passing_topics=[])  # even the reserve topic fails
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, reserve_store=reserve_store, max_reserve_attempts=5
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"
        assert research.calls == ["Only Reserve Topic"]  # not retried 5 times

    @pytest.mark.asyncio
    async def test_stale_reserve_research_is_revalidated_and_consumed_on_failure(self, tmp_path) -> None:
        """Requirement 5: a reserve topic is ALWAYS re-researched fresh at
        consumption time (never trusting old cached research) - when that
        fresh revalidation fails (e.g. stale/no-longer-current facts), the
        entry is consumed (never retried indefinitely) and the search moves
        on to try another candidate."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        stale_entry = reserve_store.add(topic="Stale Reserve Topic", topic_source="youtube")
        reserve_store.add(topic="Fresh Reserve Topic", topic_source="youtube")

        primary = StubTopicPlanner(result=make_empty_plan_result())
        research = StubResearchAgent(passing_topics=["Fresh Reserve Topic"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, reserve_store=reserve_store
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Fresh Reserve Topic"
        assert research.calls == ["Stale Reserve Topic", "Fresh Reserve Topic"]  # both re-researched fresh

        reread = reserve_store.read_entry(stale_entry.entry_id)
        assert reread.state == "consumed"
        assert reread.consumed_reason == "revalidation_failed"


# ---------------------------------------------------------------------------
# Cross-tier invariants
# ---------------------------------------------------------------------------


class TestCrossTierInvariants:
    @pytest.mark.asyncio
    async def test_research_thresholds_unchanged_across_every_tier(self, tmp_path) -> None:
        """Requirement 6: the SAME injected ResearchAgent (same thresholds/
        contract) is used for every tier - continuity never constructs a
        separate, weaker validator for TIER 2/TIER 3."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        reserve_store.add(topic="Reserve Candidate", topic_source="youtube")
        primary = StubTopicPlanner(result=make_plan_result(["Primary Candidate"], source="current_news"))
        evergreen = StubTopicPlanner(result=make_plan_result(["Evergreen Candidate"], source="youtube"))
        research = StubResearchAgent(passing_topics=[])  # nothing passes anywhere - proves no tier weakens the gate
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary,
            research_agent=research,
            evergreen_topic_planner=evergreen,
            reserve_store=reserve_store,
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"
        # Every single candidate across all three tiers went through the
        # identical `research` object - none was ever bypassed/weakened.
        assert set(research.calls) == {"Primary Candidate", "Evergreen Candidate", "Reserve Candidate"}

    @pytest.mark.asyncio
    async def test_infrastructure_failure_does_not_blacklist_multiple_topics(self, tmp_path) -> None:
        """Requirement 7: a systemic infrastructure failure on the FIRST
        candidate must stop the search immediately - never burn through the
        remaining bounded candidates as though each were individually bad."""
        primary = StubTopicPlanner(result=make_plan_result(["Topic A", "Topic B", "Topic C", "Topic D"]))
        research = StubResearchAgent(infra_failure_topics=["Topic A"])
        orchestrator = TopicContinuityOrchestrator(primary_topic_planner=primary, research_agent=research)

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "infrastructure_unavailable"
        assert research.calls == ["Topic A"]  # never tried B/C/D for what was actually an infra problem

    @pytest.mark.asyncio
    async def test_infrastructure_failure_mid_tier_stops_before_reaching_reserve(self, tmp_path) -> None:
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        reserve_store.add(topic="Reserve Candidate", topic_source="youtube")
        primary = StubTopicPlanner(result=make_plan_result(["Topic A"]))
        evergreen = StubTopicPlanner(result=make_plan_result(["Topic B"]))
        research = StubResearchAgent(infra_failure_topics=["Topic B"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary,
            research_agent=research,
            evergreen_topic_planner=evergreen,
            reserve_store=reserve_store,
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "infrastructure_unavailable"
        assert "Reserve Candidate" not in research.calls

    @pytest.mark.asyncio
    async def test_all_tiers_exhausted_reaches_explicit_terminal_state(self, tmp_path) -> None:
        """Requirement 8: only once every configured tier is genuinely
        exhausted does the orchestrator report topic_candidates_exhausted -
        never a generic/unexplained failure."""
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        reserve_store.add(topic="Reserve Candidate", topic_source="youtube")
        primary = StubTopicPlanner(result=make_plan_result(["Topic A"]))
        evergreen = StubTopicPlanner(result=make_plan_result(["Topic B"]))
        research = StubResearchAgent(passing_topics=[])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary,
            research_agent=research,
            evergreen_topic_planner=evergreen,
            reserve_store=reserve_store,
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "topic_candidates_exhausted"
        assert outcome.topic is None
        assert set(outcome.tried_topics) == {"Topic A", "Topic B", "Reserve Candidate"}
        assert outcome.error_message  # explicit, diagnosable reason - never silent


# ---------------------------------------------------------------------------
# Reserve store persistence (TIER 3 mechanics)
# ---------------------------------------------------------------------------


class TestQualifiedTopicReserveStore:
    def test_add_write_and_read_round_trip(self, tmp_path) -> None:
        store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        entry = store.add(topic="Round Trip Topic", topic_source="youtube", qualification_provider="wikipedia")

        reread = store.read_entry(entry.entry_id)
        assert reread == entry
        assert reread.state == "available"

    def test_mark_consumed_is_idempotent_and_persists_atomically(self, tmp_path) -> None:
        store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        entry = store.add(topic="Idempotent Topic")

        first = store.mark_consumed(entry.entry_id, reason="published")
        second = store.mark_consumed(entry.entry_id, reason="published_again")

        assert first.state == "consumed"
        assert second.state == "consumed"
        assert second.consumed_reason == "published"  # first consumption wins, never re-timestamped

    def test_get_next_available_never_returns_a_consumed_entry(self, tmp_path) -> None:
        store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        entry = store.add(topic="Solo Topic")
        store.mark_consumed(entry.entry_id)

        assert store.get_next_available() is None

    def test_get_next_available_respects_exclude_topics(self, tmp_path) -> None:
        store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        store.add(topic="Topic One")
        store.add(topic="Topic Two")

        next_entry = store.get_next_available(exclude_topics={"Topic One"})

        assert next_entry.topic == "Topic Two"

    def test_mark_consumed_missing_entry_is_a_safe_noop(self, tmp_path) -> None:
        store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        assert store.mark_consumed("does-not-exist") is None


# ---------------------------------------------------------------------------
# Reserve seeding (opportunistic TIER 2 -> TIER 3 population)
# ---------------------------------------------------------------------------


class TestReserveSeeding:
    @pytest.mark.asyncio
    async def test_tier_2_success_opportunistically_seeds_leftover_qualified_candidates(self, tmp_path) -> None:
        reserve_store = QualifiedTopicReserveStore(str(tmp_path / "reserve"))
        primary = StubTopicPlanner(result=make_empty_plan_result())
        evergreen = StubTopicPlanner(
            result=make_plan_result(["Winner", "Leftover Qualified", "Leftover Unqualified"], source="youtube")
        )
        research = StubResearchAgent(passing_topics=["Winner", "Leftover Qualified"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary,
            research_agent=research,
            evergreen_topic_planner=evergreen,
            reserve_store=reserve_store,
            reserve_seed_attempts=2,
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Winner"
        available = {e.topic for e in reserve_store.list_available()}
        assert "Leftover Qualified" in available
        assert "Leftover Unqualified" not in available
        assert "Winner" not in available  # the winning topic is consumed by the current run, not banked


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestOutcomeIdempotency:
    @pytest.mark.asyncio
    async def test_calling_select_researchable_topic_twice_does_not_reuse_first_calls_tried_set(self, tmp_path) -> None:
        """Each call starts a fresh "tried" set - a second, independent
        call (e.g. a later autonomous run) may legitimately retry a
        primary candidate that failed in an EARLIER call, since it isn't
        the same run."""
        primary = StubTopicPlanner(result=make_plan_result(["Topic A"]))
        research = StubResearchAgent(passing_topics=[])
        orchestrator = TopicContinuityOrchestrator(primary_topic_planner=primary, research_agent=research)

        first = await orchestrator.select_researchable_topic()
        second = await orchestrator.select_researchable_topic()

        assert first.status == "topic_candidates_exhausted"
        assert second.status == "topic_candidates_exhausted"
        assert research.calls == ["Topic A", "Topic A"]  # each call independently tries its own candidates
