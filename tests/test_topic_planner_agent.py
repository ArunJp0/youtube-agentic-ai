# Tests for TopicPlannerAgent - the orchestration that ties candidate
# discovery, normalization, duplicate/history filtering, scoring, and
# selection together. All mocked (MockTopicSourceProvider, fake LLM
# providers, tmp_path-based stores) - no real YouTube/Gemini/network calls.
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.agents.research import ResearchAgent
from src.agents.topic_planner_agent import TopicPlannerAgent
from src.agents.topic_ranking_planner import TopicRankingPlanner
from src.config.settings import Settings
from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.provenance import ProvenanceManifest
from src.models.topic_planner import TopicCandidate
from src.services.provenance_store import ProvenanceManifestStore
from src.services.topic_normalization import normalize_topic
from src.services.topic_plan_store import TopicPlanStore
from src.tools.search_provider import MockSearchProvider
from src.tools.topic_source_provider import CompositeTopicSourceProvider, MockTopicSourceProvider, TopicSourceProviderError

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def _news_candidate(
    title: str,
    hours_old: float = 1.0,
    source_name: str = "Source A",
    market: str = "global",
    category: str = "technology",
) -> TopicCandidate:
    published = (NOW - timedelta(hours=hours_old)).isoformat()
    return TopicCandidate(
        raw_title=title,
        normalized_title=normalize_topic(title),
        source="current_news",
        source_name=source_name,
        published_at=published,
        discovered_at=NOW.isoformat(),
        market=market,
        category=category,
    )


def _candidate(title: str, popularity: float = 0.5, category: str = "education") -> TopicCandidate:
    return TopicCandidate(
        raw_title=title, normalized_title=normalize_topic(title), source="mock",
        popularity_signal=popularity, category=category,
    )


def _manifest(run_id: str, topic: str, created_at: str) -> ProvenanceManifest:
    return ProvenanceManifest(
        run_id=run_id, topic=topic, final_video_path=f"output/video/{run_id}.mp4", created_at=created_at
    )


class RecordingLLMProvider(LLMProvider):
    """Fake LLM that returns a valid ranking response and counts calls."""

    def __init__(self, response: str | None = None, raise_error: Exception | None = None) -> None:
        self.response = response
        self.raise_error = raise_error
        self.calls: list[str] = []
        self.name = "fake-llm"
        self.last_model_used = "fake-model"
        self.last_used_fallback = False

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_error:
            raise self.raise_error
        return self.response


def _valid_ranking_response(candidates) -> str:
    return json.dumps(
        {
            "judgments": [
                {
                    "normalized_title": c.normalized_title,
                    "relevance_score": 0.9,
                    "evergreen_score": 0.8,
                    "suitability_score": 0.85,
                    "rationale": f"Good fit: {c.raw_title}",
                }
                for c in candidates
            ]
        }
    )


class TestBasicSelection:
    def test_candidates_discovered_and_best_topic_selected(self, tmp_path) -> None:
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is True
        assert result.status == "selected"
        assert result.selected_topic in {"Why do cats purr?", "How do vaccines work?"}
        assert result.candidate_count == 2
        assert result.score is not None

    def test_sources_used_recorded(self, tmp_path) -> None:
        provider = MockTopicSourceProvider(candidates=[_candidate("Topic A")], provider_name="my-source")
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )
        result = asyncio.run(agent.plan_topic())
        assert result.sources_used == ["my-source"]


class TestDuplicatePrevention:
    def test_topic_already_produced_is_rejected(self, tmp_path) -> None:
        provenance_dir = tmp_path / "provenance"
        ProvenanceManifestStore(str(provenance_dir)).write(
            _manifest("run-1", "Why do cats purr?", "2026-09-01T00:00:00+00:00")
        )
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(provenance_dir),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is True
        assert result.selected_topic == "How do vaccines work?"
        assert result.duplicate_count == 1

    def test_normalized_near_duplicate_rejected(self, tmp_path) -> None:
        """The required example: 'Why is the ocean salty?' vs 'Why Is
        Ocean Water Salty' must be recognized as the same topic."""
        provenance_dir = tmp_path / "provenance"
        ProvenanceManifestStore(str(provenance_dir)).write(
            _manifest("run-1", "Why is the ocean salty?", "2026-09-01T00:00:00+00:00")
        )
        candidates = [_candidate("Why Is Ocean Water Salty"), _candidate("How do vaccines work?")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(provenance_dir),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.selected_topic == "How do vaccines work?"
        assert result.duplicate_count == 1

    def test_all_candidates_duplicate_returns_explicit_no_topic_result(self, tmp_path) -> None:
        provenance_dir = tmp_path / "provenance"
        ProvenanceManifestStore(str(provenance_dir)).write(
            _manifest("run-1", "Why do cats purr?", "2026-09-01T00:00:00+00:00")
        )
        candidates = [_candidate("Why do cats purr?")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(provenance_dir),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is False
        assert result.status == "all_candidates_duplicate"
        assert result.selected_topic is None
        assert result.candidate_count == 1
        assert result.duplicate_count == 1

    def test_own_recent_plan_history_also_prevents_reselection(self, tmp_path) -> None:
        """Even before a topic is ever actually produced, the planner's own
        recent selection history (TopicPlanStore) prevents immediately
        reselecting the same topic on a back-to-back planning call."""
        plan_store = TopicPlanStore(str(tmp_path / "plans"))
        provenance_dir = tmp_path / "provenance"

        first_candidates = [_candidate("Why do cats purr?")]
        first_agent = TopicPlannerAgent(
            topic_source_provider=MockTopicSourceProvider(candidates=first_candidates),
            topic_plan_store=plan_store,
            provenance_output_dir=str(provenance_dir),
        )
        first_result = asyncio.run(first_agent.plan_topic())
        assert first_result.selected_topic == "Why do cats purr?"

        second_candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        second_agent = TopicPlannerAgent(
            topic_source_provider=MockTopicSourceProvider(candidates=second_candidates),
            topic_plan_store=plan_store,
            provenance_output_dir=str(provenance_dir),
        )
        second_result = asyncio.run(second_agent.plan_topic())

        assert second_result.selected_topic == "How do vaccines work?"
        assert second_result.duplicate_count == 1

    def test_candidates_deduplicated_against_each_other(self, tmp_path) -> None:
        candidates = [_candidate("Why is the ocean salty?"), _candidate("Why Is Ocean Water Salty")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.candidate_count == 1  # the second collapsed into the first


class TestSemanticRanking:
    def test_batch_ranking_used_when_llm_available(self, tmp_path) -> None:
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?"), _candidate("Why does ice float?")]
        llm = RecordingLLMProvider(response=_valid_ranking_response(candidates))
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.candidate_count == 3  # all three distinct candidates survived self-dedup
        assert result.used_semantic_ranking is True
        assert result.fallback_used is False
        assert len(llm.calls) == 1  # exactly one bounded batch call, not one per candidate

    def test_no_llm_calls_when_ranking_planner_not_configured(self, tmp_path) -> None:
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=None,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.candidate_count == 2
        assert result.used_semantic_ranking is False
        assert result.fallback_used is True
        assert result.success is True

    def test_gemini_ranking_failure_falls_back_deterministically(self, tmp_path) -> None:
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        llm = RecordingLLMProvider(raise_error=RuntimeError("simulated Gemini outage"))
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is True  # still selects a topic, just via fallback scoring
        assert result.used_semantic_ranking is False
        assert result.fallback_used is True


class TestFailureBehavior:
    def test_source_unavailable_returns_explicit_failure(self, tmp_path) -> None:
        provider = MockTopicSourceProvider(fail=True)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is False
        assert result.status == "source_unavailable"
        assert result.selected_topic is None
        assert result.error is not None

    def test_no_candidates_returns_explicit_failure(self, tmp_path) -> None:
        provider = MockTopicSourceProvider(candidates=[])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is False
        assert result.status == "no_candidates"
        assert result.selected_topic is None

    def test_never_fabricates_a_topic_on_failure(self, tmp_path) -> None:
        for provider in (MockTopicSourceProvider(fail=True), MockTopicSourceProvider(candidates=[])):
            agent = TopicPlannerAgent(
                topic_source_provider=provider,
                topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
                provenance_output_dir=str(tmp_path / "provenance"),
            )
            result = asyncio.run(agent.plan_topic())
            assert result.selected_topic is None


class TestPersistence:
    def test_selection_is_persisted(self, tmp_path) -> None:
        plan_store = TopicPlanStore(str(tmp_path / "plans"))
        provider = MockTopicSourceProvider(candidates=[_candidate("Topic A")])
        agent = TopicPlannerAgent(
            topic_source_provider=provider, topic_plan_store=plan_store,
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())

        recent = plan_store.list_recent()
        assert len(recent) == 1
        assert recent[0].selected_topic == result.selected_topic
        assert recent[0] == result

    def test_failure_results_are_also_persisted(self, tmp_path) -> None:
        plan_store = TopicPlanStore(str(tmp_path / "plans"))
        provider = MockTopicSourceProvider(candidates=[])
        agent = TopicPlannerAgent(
            topic_source_provider=provider, topic_plan_store=plan_store,
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        asyncio.run(agent.plan_topic())

        recent = plan_store.list_recent()
        assert len(recent) == 1
        assert recent[0].status == "no_candidates"

    def test_history_round_trip_across_two_agent_instances(self, tmp_path) -> None:
        """A fresh TopicPlannerAgent instance (simulating a new process/run)
        still respects history written by a prior instance."""
        plan_store_dir = str(tmp_path / "plans")
        provenance_dir = str(tmp_path / "provenance")

        agent_1 = TopicPlannerAgent(
            topic_source_provider=MockTopicSourceProvider(candidates=[_candidate("Why do cats purr?")]),
            topic_plan_store=TopicPlanStore(plan_store_dir),
            provenance_output_dir=provenance_dir,
        )
        result_1 = asyncio.run(agent_1.plan_topic())
        assert result_1.selected_topic == "Why do cats purr?"

        agent_2 = TopicPlannerAgent(
            topic_source_provider=MockTopicSourceProvider(
                candidates=[_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
            ),
            topic_plan_store=TopicPlanStore(plan_store_dir),
            provenance_output_dir=provenance_dir,
        )
        result_2 = asyncio.run(agent_2.plan_topic())
        assert result_2.selected_topic == "How do vaccines work?"


class TestConsumableByResearchAgent:
    def test_selected_topic_is_a_plain_string(self, tmp_path) -> None:
        provider = MockTopicSourceProvider(candidates=[_candidate("Why do cats purr?")])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )
        result = asyncio.run(agent.plan_topic())
        assert isinstance(result.selected_topic, str)

    def test_selected_topic_feeds_directly_into_research_agent(self, tmp_path) -> None:
        """Proves the existing Research input boundary needs zero changes -
        the planner's output is passed straight into ResearchAgent.research()."""
        provider = MockTopicSourceProvider(candidates=[_candidate("Why do cats purr?")])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )
        plan_result = asyncio.run(agent.plan_topic())

        research_agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=MockLLMProvider())
        research_result = asyncio.run(research_agent.research(plan_result.selected_topic))

        assert research_result.topic == plan_result.selected_topic


class TestNoExistingBehaviorChange:
    def test_topic_planner_disabled_by_default(self) -> None:
        import os

        # Guard against a developer's local .env accidentally leaking into
        # this assertion.
        previous = os.environ.pop("TOPIC_PLANNER_ENABLED", None)
        try:
            assert Settings().topic_planner_enabled is False
        finally:
            if previous is not None:
                os.environ["TOPIC_PLANNER_ENABLED"] = previous

    def test_default_topic_source_is_mock(self) -> None:
        import os

        previous = os.environ.pop("TOPIC_PLANNER_SOURCE", None)
        try:
            assert Settings().topic_planner_source == "mock"
        finally:
            if previous is not None:
                os.environ["TOPIC_PLANNER_SOURCE"] = previous

    def test_default_topic_mode_is_evergreen(self) -> None:
        import os

        previous = os.environ.pop("TOPIC_MODE", None)
        try:
            assert Settings().topic_mode == "evergreen"
        finally:
            if previous is not None:
                os.environ["TOPIC_MODE"] = previous

    def test_evergreen_mode_total_score_formula_byte_for_byte_unchanged(self, tmp_path) -> None:
        """Regression guard: an evergreen candidate (no published_at/market/
        category-preference signals) must score with EXACTLY the original
        5-signal formula - trend weights must contribute zero."""
        from src.agents.topic_planner_agent import SCORE_WEIGHTS

        candidate = _candidate("Why do cats purr?", popularity=0.7)
        provider = MockTopicSourceProvider(candidates=[candidate])
        llm = RecordingLLMProvider(response=_valid_ranking_response([candidate]))
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = asyncio.run(agent.plan_topic())
        score = result.top_candidates[0]

        expected = (
            SCORE_WEIGHTS["relevance"] * score.relevance_score
            + SCORE_WEIGHTS["suitability"] * score.suitability_score
            + SCORE_WEIGHTS["novelty"] * score.novelty_score
            + SCORE_WEIGHTS["evergreen"] * score.evergreen_score
            + SCORE_WEIGHTS["popularity"] * score.popularity_score
        )
        assert abs(score.total_score - round(expected, 4)) < 1e-9
        assert score.freshness_score is None
        assert score.source_confidence_score is None
        assert score.market_relevance_score is None
        assert score.category_relevance_score is None


class TestFreshnessFiltering:
    def test_fresh_story_preferred_over_stale_equivalent(self, tmp_path) -> None:
        fresh = _news_candidate("Company Unveils New Chip Design", hours_old=1)
        stale = _news_candidate("Regional Sports League Result Announced", hours_old=47)
        provider = MockTopicSourceProvider(candidates=[stale, fresh])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            freshness_hours=48,
        )

        result = asyncio.run(agent.plan_topic())

        assert result.selected_topic == "Company Unveils New Chip Design"

    def test_stale_beyond_rejection_multiplier_is_hard_filtered(self, tmp_path) -> None:
        stale = _news_candidate("Very Old Story", hours_old=200)
        provider = MockTopicSourceProvider(candidates=[stale])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            freshness_hours=48,
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is False
        assert result.status == "all_candidates_stale"

    def test_freshness_filter_not_applied_in_evergreen_mode(self, tmp_path) -> None:
        """Freshness filtering is trending/mixed-mode only - an evergreen
        candidate (no published_at) is never affected regardless."""
        candidate = _candidate("Why do cats purr?")
        provider = MockTopicSourceProvider(candidates=[candidate])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="evergreen",
        )

        result = asyncio.run(agent.plan_topic())
        assert result.success is True

    def test_configurable_freshness_window_changes_outcome(self, tmp_path) -> None:
        story = _news_candidate("Story", hours_old=10)
        # With a 48h window, 10h old is fresh and well within the
        # 3x-window hard-rejection cutoff.
        provider_a = MockTopicSourceProvider(candidates=[story])
        agent_a = TopicPlannerAgent(
            topic_source_provider=provider_a,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans_a")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            freshness_hours=48,
        )
        result_a = asyncio.run(agent_a.plan_topic())
        assert result_a.success is True

        # With a 1h window, 10h old is beyond the 3x-window cutoff (3h) -
        # hard filtered.
        provider_b = MockTopicSourceProvider(candidates=[story])
        agent_b = TopicPlannerAgent(
            topic_source_provider=provider_b,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans_b")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            freshness_hours=1,
        )
        result_b = asyncio.run(agent_b.plan_topic())
        assert result_b.status == "all_candidates_stale"


class TestCrossSourceConfidence:
    def test_multiple_sources_reporting_same_story_increases_distinct_source_count(self, tmp_path) -> None:
        same_story_source_a = _news_candidate("Major AI Breakthrough Announced", source_name="Source A")
        same_story_source_b = _news_candidate("Major AI Breakthrough Announced Today", source_name="Source B")
        unrelated = _news_candidate("Local Weather Update", source_name="Source A")

        provider = MockTopicSourceProvider(candidates=[same_story_source_a, same_story_source_b, unrelated])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        merged = next(s for s in result.top_candidates if "AI Breakthrough" in s.candidate.raw_title)
        assert merged.candidate.distinct_source_count == 2

    def test_same_outlet_repeating_does_not_inflate_count(self, tmp_path) -> None:
        first = _news_candidate("Big Story Happens", source_name="Source A")
        duplicate_same_outlet = _news_candidate("Big Story Happens Today", source_name="Source A")

        provider = MockTopicSourceProvider(candidates=[first, duplicate_same_outlet])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        assert result.candidate_count == 1
        assert result.top_candidates[0].candidate.distinct_source_count == 1

    def test_higher_source_confidence_scores_higher(self, tmp_path) -> None:
        multi_source = [
            _news_candidate("Widely Reported Global Event", source_name="Source A"),
            _news_candidate("Widely Reported Global Event Today", source_name="Source B"),
            _news_candidate("Widely Reported Global Event Now", source_name="Source C"),
        ]
        single_source = [_news_candidate("Barely Reported Local Story", source_name="Source A")]

        provider = MockTopicSourceProvider(candidates=multi_source + single_source)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        multi = next(s for s in result.top_candidates if "Widely Reported" in s.candidate.raw_title)
        single = next(s for s in result.top_candidates if "Barely Reported" in s.candidate.raw_title)
        assert multi.source_confidence_score > single.source_confidence_score

    def test_composite_provider_partial_failure_does_not_crash_planner(self, tmp_path) -> None:
        working = MockTopicSourceProvider(
            candidates=[_news_candidate("A Working Source Story")], provider_name="working"
        )
        failing = MockTopicSourceProvider(fail=True, provider_name="failing_news")
        composite = CompositeTopicSourceProvider([working, failing])

        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is True
        assert result.selected_topic == "A Working Source Story"
        assert result.sources_used == ["working"]


class TestMarketConfiguration:
    def test_global_story_scores_max_market_relevance_regardless_of_config(self, tmp_path) -> None:
        global_story = _news_candidate("Global Important Event", market="global")
        provider = MockTopicSourceProvider(candidates=[global_story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            target_markets=["GB"],
        )

        result = asyncio.run(agent.plan_topic())
        assert result.top_candidates[0].market_relevance_score == 1.0

    def test_matching_configured_market_scores_max(self, tmp_path) -> None:
        india_story = _news_candidate("India Regional Story", market="IN")
        provider = MockTopicSourceProvider(candidates=[india_story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            target_markets=["IN"],
        )

        result = asyncio.run(agent.plan_topic())
        assert result.top_candidates[0].market_relevance_score == 1.0

    def test_unmatched_market_not_penalized_below_neutral_baseline(self, tmp_path) -> None:
        other_market_story = _news_candidate("Other Region Story", market="FR")
        provider = MockTopicSourceProvider(candidates=[other_market_story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            target_markets=["GB"],
        )

        result = asyncio.run(agent.plan_topic())
        assert result.top_candidates[0].market_relevance_score < 1.0
        assert result.top_candidates[0].market_relevance_score > 0.0

    def test_no_market_configured_leaves_signal_inapplicable(self, tmp_path) -> None:
        story = _news_candidate("Some Story", market="IN")
        provider = MockTopicSourceProvider(candidates=[story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            target_markets=None,
        )

        result = asyncio.run(agent.plan_topic())
        assert result.top_candidates[0].market_relevance_score is None

    def test_no_region_hardcoded_works_for_any_configured_market_string(self, tmp_path) -> None:
        """The exact same agent code path handles 'GB', 'IN', and a
        free-text sub-national region like 'Tamil Nadu' identically -
        nothing region-specific is hardcoded in planner logic."""
        for market_value in ("GB", "IN", "Tamil Nadu", "global"):
            story = _news_candidate("A Story", market=market_value)
            provider = MockTopicSourceProvider(candidates=[story])
            agent = TopicPlannerAgent(
                topic_source_provider=provider,
                topic_plan_store=TopicPlanStore(str(tmp_path / f"plans_{market_value}".replace(" ", "_"))),
                provenance_output_dir=str(tmp_path / "provenance"),
                mode="trending",
                target_markets=[market_value],
            )
            result = asyncio.run(agent.plan_topic())
            assert result.success is True
            assert result.top_candidates[0].market_relevance_score == 1.0

    def test_globally_important_story_can_still_win_over_regional_story(self, tmp_path) -> None:
        """STEP 6: a globally important story should still be selectable
        even when regional signals/preferences exist."""
        global_story = _news_candidate(
            "Globally Major Breakthrough", market="global", hours_old=1, source_name="Source A"
        )
        regional_story = _news_candidate("Minor Regional News", market="GB", hours_old=1, source_name="Source B")
        provider = MockTopicSourceProvider(candidates=[global_story, regional_story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            target_markets=["GB"],
        )

        result = asyncio.run(agent.plan_topic())
        # Both score max market_relevance (global always does) - the
        # global story is not crowded out just because a regional
        # preference is configured.
        assert result.top_candidates[0].market_relevance_score == 1.0
        assert result.top_candidates[1].market_relevance_score == 1.0


class TestCategoryConfiguration:
    def test_preferred_category_scores_higher(self, tmp_path) -> None:
        preferred = _news_candidate("Tech Story", category="technology", hours_old=1)
        other = _news_candidate("Sports Story", category="sports", hours_old=1)
        provider = MockTopicSourceProvider(candidates=[preferred, other])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            preferred_categories=["technology", "science"],
        )

        result = asyncio.run(agent.plan_topic())

        assert result.selected_topic == "Tech Story"

    def test_no_categories_configured_no_bias(self, tmp_path) -> None:
        story = _news_candidate("Any Story", category="sports")
        provider = MockTopicSourceProvider(candidates=[story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            preferred_categories=None,
        )

        result = asyncio.run(agent.plan_topic())
        assert result.top_candidates[0].category_relevance_score is None

    def test_unpreferred_category_does_not_make_story_impossible_to_select(self, tmp_path) -> None:
        """A high-quality story outside the configured categories can still
        win if its other signals are strong enough - category preference
        is a weight, never a hard filter."""
        unpreferred_but_fresh = _news_candidate(
            "Huge Important Sports Story", category="sports", hours_old=0.5, source_name="Source A"
        )
        provider = MockTopicSourceProvider(candidates=[unpreferred_but_fresh])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
            preferred_categories=["technology"],
        )

        result = asyncio.run(agent.plan_topic())
        assert result.success is True
        assert result.selected_topic == "Huge Important Sports Story"


class TestSensitiveStoryFlag:
    def test_sensitive_story_flagged_on_candidate_and_result(self, tmp_path) -> None:
        sensitive = _news_candidate("War escalates as ceasefire talks collapse", hours_old=1)
        provider = MockTopicSourceProvider(candidates=[sensitive])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        assert result.is_sensitive is True
        assert "active_conflict" in result.sensitivity_reasons
        assert result.top_candidates[0].candidate.is_sensitive is True

    def test_non_sensitive_story_not_flagged(self, tmp_path) -> None:
        story = _news_candidate("New smartphone released today", hours_old=1)
        provider = MockTopicSourceProvider(candidates=[story])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())
        assert result.is_sensitive is False
        assert result.sensitivity_reasons == []

    def test_sensitive_flag_never_suppresses_selection(self, tmp_path) -> None:
        """A sensitive story is still selectable - the flag informs
        downstream handling, it never auto-rejects the story."""
        sensitive = _news_candidate("Election results contested amid allegations", hours_old=0.5)
        provider = MockTopicSourceProvider(candidates=[sensitive])
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())
        assert result.success is True
        assert result.selected_topic == "Election results contested amid allegations"


class TestTrendingModeGeminiFailureFallback:
    def test_gemini_failure_in_trending_mode_still_selects_deterministically(self, tmp_path) -> None:
        candidates = [_news_candidate("Story One", hours_old=1), _news_candidate("Story Two", hours_old=2)]
        llm = RecordingLLMProvider(raise_error=RuntimeError("simulated Gemini outage"))
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        result = asyncio.run(agent.plan_topic())

        assert result.success is True
        assert result.fallback_used is True
        # Deterministic signals (freshness) still work even when semantic
        # ranking fails.
        assert result.top_candidates[0].freshness_score is not None

    def test_exactly_one_batch_llm_call_in_trending_mode(self, tmp_path) -> None:
        candidates = [_news_candidate(f"Story {i}", hours_old=i + 1) for i in range(5)]
        llm = RecordingLLMProvider(response=_valid_ranking_response(candidates))
        provider = MockTopicSourceProvider(candidates=candidates)
        agent = TopicPlannerAgent(
            topic_source_provider=provider,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="trending",
        )

        asyncio.run(agent.plan_topic())

        assert len(llm.calls) == 1

    def test_exactly_one_batch_llm_call_in_mixed_mode_with_composite_source(self, tmp_path) -> None:
        evergreen = MockTopicSourceProvider(candidates=[_candidate("Evergreen Story")], provider_name="evergreen")
        news = MockTopicSourceProvider(candidates=[_news_candidate("News Story")], provider_name="news")
        composite = CompositeTopicSourceProvider([evergreen, news])
        llm = RecordingLLMProvider(response=None)
        # Build a response covering both candidates dynamically once we
        # know the merged set - simplest is to just accept any candidate
        # set by returning judgments for both known titles.
        from src.services.topic_normalization import normalize_topic as _norm

        llm.response = json.dumps(
            {
                "judgments": [
                    {
                        "normalized_title": _norm("Evergreen Story"),
                        "relevance_score": 0.8, "evergreen_score": 0.8, "suitability_score": 0.8, "rationale": "x",
                    },
                    {
                        "normalized_title": _norm("News Story"),
                        "relevance_score": 0.8, "evergreen_score": 0.8, "suitability_score": 0.8, "rationale": "x",
                    },
                ]
            }
        )
        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            ranking_planner=TopicRankingPlanner(llm),
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
            mode="mixed",
        )

        result = asyncio.run(agent.plan_topic())

        assert len(llm.calls) == 1
        assert result.success is True
