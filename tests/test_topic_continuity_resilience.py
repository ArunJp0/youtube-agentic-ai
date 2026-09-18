# Integration tests proving the exact production gap a real controlled
# autonomous run exposed is fixed: TIER 2 evergreen discovery must remain
# genuinely usable when YouTube is absent/unavailable, by composing it with
# the new free WikipediaTopicSourceProvider via the existing
# CompositeTopicSourceProvider - not a new resilience mechanism of its own.
#
# These combine REAL TopicPlannerAgent + REAL CompositeTopicSourceProvider
# + REAL WikipediaTopicSourceProvider with a fake YouTube-shaped source and
# a fake Wikipedia SearchProvider - proving the actual composition works
# end to end, not just each piece in isolation. No real network calls.
from __future__ import annotations

import pytest

from src.agents.topic_planner_agent import TopicPlannerAgent
from src.orchestration.topic_continuity_orchestrator import TopicContinuityOrchestrator
from src.services.topic_plan_store import TopicPlanStore
from src.tools.search_provider import SearchProvider
from src.tools.topic_source_provider import CompositeTopicSourceProvider, TopicSourceProvider, TopicSourceProviderError
from src.tools.wikipedia_topic_source_provider import WikipediaTopicSourceProvider
from tests.test_topic_continuity_orchestrator import StubResearchAgent, StubTopicPlanner, make_plan_result


class FailingYouTubeLikeSource(TopicSourceProvider):
    """Stands in for the real YouTubeTopicSourceProvider when
    YOUTUBE_API_KEY is absent/unavailable - raises the exact same typed
    error a real unconfigured/unreachable YouTube source would."""

    def __init__(self, message: str = "YOUTUBE_API_KEY is not configured - required for real topic discovery") -> None:
        self.message = message
        self.calls = 0

    @property
    def name(self) -> str:
        return "youtube"

    async def discover_candidates(self, category=None, region=None, limit=15):
        self.calls += 1
        raise TopicSourceProviderError(self.message)


class FakeWikipediaSearchProvider(SearchProvider):
    def __init__(self, results_by_query) -> None:
        self.results_by_query = results_by_query
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "wikipedia"

    async def search(self, query: str, num_results: int = 5):
        self.calls.append(query)
        return self.results_by_query.get(query, [])[:num_results]


def _wiki_result(title: str) -> dict:
    return {"title": title, "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "snippet": f"About {title}."}


class TestResilientEvergreenDiscoveryEndToEnd:
    @pytest.mark.asyncio
    async def test_youtube_api_key_absent_composite_still_discovers_real_candidates(self, tmp_path) -> None:
        """C. A YouTube-shaped source that fails exactly like a real
        missing-YOUTUBE_API_KEY source must not prevent TopicPlannerAgent
        from discovering and selecting a real evergreen candidate via the
        composed Wikipedia source."""
        youtube = FailingYouTubeLikeSource()
        wikipedia_search = FakeWikipediaSearchProvider(
            {"science explained": [_wiki_result("Photosynthesis"), _wiki_result("Gravity")]}
        )
        wikipedia = WikipediaTopicSourceProvider(
            search_provider=wikipedia_search, category_seeds=["science explained"]
        )
        composite = CompositeTopicSourceProvider([youtube, wikipedia])

        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = await agent.plan_topic()

        assert result.success is True
        assert result.status == "selected"
        assert youtube.calls == 1  # tried, and genuinely failed
        assert result.selected_topic in {"Photosynthesis", "Gravity"}
        assert result.selected_topic_source == "wikipedia_evergreen"

    @pytest.mark.asyncio
    async def test_youtube_provider_generic_unavailability_still_discovers_real_candidates(self, tmp_path) -> None:
        """D. The SAME resilience must hold for any recoverable YouTube
        provider failure, not only the specific missing-API-key message -
        CompositeTopicSourceProvider (and WikipediaTopicSourceProvider,
        for the composed source itself) never inspect the failure reason."""
        youtube = FailingYouTubeLikeSource(message="YouTube Data API quota exceeded")
        wikipedia_search = FakeWikipediaSearchProvider({"how technology works": [_wiki_result("Battery")]})
        wikipedia = WikipediaTopicSourceProvider(
            search_provider=wikipedia_search, category_seeds=["how technology works"]
        )
        composite = CompositeTopicSourceProvider([youtube, wikipedia])

        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = await agent.plan_topic()

        assert result.success is True
        assert result.selected_topic == "Battery"

    @pytest.mark.asyncio
    async def test_youtube_and_wikipedia_both_unavailable_reports_source_unavailable(self, tmp_path) -> None:
        """The genuine "nothing at all is reachable" case must still
        surface as source_unavailable, unchanged - resilience recovers a
        SINGLE point of failure, it never masks a total outage."""
        youtube = FailingYouTubeLikeSource()
        wikipedia_search = FakeWikipediaSearchProvider({})  # every seed returns nothing... but that's not an error

        class _FailingWikipediaSearch(SearchProvider):
            @property
            def name(self) -> str:
                return "wikipedia"

            async def search(self, query, num_results=5):
                raise RuntimeError("simulated Wikipedia outage")

        wikipedia = WikipediaTopicSourceProvider(
            search_provider=_FailingWikipediaSearch(), category_seeds=["science explained"]
        )
        composite = CompositeTopicSourceProvider([youtube, wikipedia])

        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        result = await agent.plan_topic()

        assert result.success is False
        assert result.status == "source_unavailable"

    @pytest.mark.asyncio
    async def test_previously_selected_wikipedia_topic_is_not_reselected(self, tmp_path) -> None:
        """K. Wikipedia-sourced candidates go through the EXACT SAME
        duplicate/history filtering as any other source - no regression."""
        plan_store = TopicPlanStore(str(tmp_path / "plans"))
        provenance_dir = tmp_path / "provenance"

        def _agent(results_by_query):
            wikipedia = WikipediaTopicSourceProvider(
                search_provider=FakeWikipediaSearchProvider(results_by_query), category_seeds=["science explained"]
            )
            return TopicPlannerAgent(
                topic_source_provider=wikipedia,
                topic_plan_store=plan_store,
                provenance_output_dir=str(provenance_dir),
            )

        first_result = await _agent({"science explained": [_wiki_result("Photosynthesis")]}).plan_topic()
        assert first_result.selected_topic == "Photosynthesis"

        second_result = await _agent(
            {"science explained": [_wiki_result("Photosynthesis"), _wiki_result("Gravity")]}
        ).plan_topic()

        assert second_result.selected_topic == "Gravity"
        assert second_result.duplicate_count == 1

    @pytest.mark.asyncio
    async def test_no_unnecessary_wikipedia_search_calls_once_youtube_succeeds(self, tmp_path) -> None:
        """L. When YouTube itself succeeds, the composed Wikipedia source
        is still queried (CompositeTopicSourceProvider merges all
        component results - this is existing, unchanged behavior), but
        WikipediaTopicSourceProvider itself never performs more search
        calls than its own bounded per-seed limit requires."""
        class _SucceedingYouTubeLikeSource(TopicSourceProvider):
            @property
            def name(self) -> str:
                return "youtube"

            async def discover_candidates(self, category=None, region=None, limit=15):
                from src.models.topic_planner import TopicCandidate
                from src.services.topic_normalization import normalize_topic

                return [
                    TopicCandidate(raw_title="Why Do Cats Purr", normalized_title=normalize_topic("Why Do Cats Purr"), source="youtube")
                ]

        wikipedia_search = FakeWikipediaSearchProvider(
            {"science explained": [_wiki_result("Photosynthesis")], "how technology works": [_wiki_result("Battery")]}
        )
        wikipedia = WikipediaTopicSourceProvider(
            search_provider=wikipedia_search, category_seeds=["science explained", "how technology works"]
        )
        composite = CompositeTopicSourceProvider([_SucceedingYouTubeLikeSource(), wikipedia])

        agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )
        await agent.plan_topic()

        # Exactly one search call per configured seed - never more.
        assert wikipedia_search.calls == ["science explained", "how technology works"]


class TestTier2ContinuityUsesResilientEvergreenSource:
    @pytest.mark.asyncio
    async def test_tier1_exhausted_tier2_recovers_via_wikipedia_when_youtube_unavailable(self, tmp_path) -> None:
        """B + C combined at the TopicContinuityOrchestrator level: TIER 1
        exhausted -> TIER 2 (backed by the resilient composite source)
        still produces a real, research-qualified topic even though its
        YouTube component is unavailable."""
        primary = StubTopicPlanner(result=make_plan_result(["News A"], source="current_news"))

        youtube = FailingYouTubeLikeSource()
        wikipedia_search = FakeWikipediaSearchProvider({"science explained": [_wiki_result("Photosynthesis")]})
        wikipedia = WikipediaTopicSourceProvider(search_provider=wikipedia_search, category_seeds=["science explained"])
        composite = CompositeTopicSourceProvider([youtube, wikipedia])
        evergreen_agent = TopicPlannerAgent(
            topic_source_provider=composite,
            topic_plan_store=TopicPlanStore(str(tmp_path / "plans")),
            provenance_output_dir=str(tmp_path / "provenance"),
        )

        research = StubResearchAgent(passing_topics=["Photosynthesis"])
        orchestrator = TopicContinuityOrchestrator(
            primary_topic_planner=primary, research_agent=research, evergreen_topic_planner=evergreen_agent
        )

        outcome = await orchestrator.select_researchable_topic()

        assert outcome.status == "selected"
        assert outcome.topic == "Photosynthesis"
        assert outcome.tier == "evergreen"
        assert youtube.calls == 1
