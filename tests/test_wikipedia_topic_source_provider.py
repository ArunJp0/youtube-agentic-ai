# Tests for WikipediaTopicSourceProvider: the free, no-API-key evergreen
# discovery source that closes the production gap where YouTube was TIER
# 2's only real discovery path (see src.orchestration.topic_continuity_orchestrator).
# All network calls mocked via an injected fake SearchProvider - no real
# Wikipedia HTTP calls.
from __future__ import annotations

import asyncio

import pytest

from src.models.topic_planner import TopicCandidate
from src.tools.search_provider import SearchProvider
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError
from src.tools.wikipedia_topic_source_provider import (
    DEFAULT_EVERGREEN_CATEGORY_SEEDS,
    WikipediaTopicSourceProvider,
)


class FakeSearchProvider(SearchProvider):
    """Test double: returns a caller-scripted result list per query, or
    raises a caller-scripted exception for specific queries - records every
    call for assertions."""

    def __init__(self, results_by_query=None, fail_for=None) -> None:
        self.results_by_query = results_by_query or {}
        self.fail_for = fail_for or set()
        self.calls: list[tuple] = []

    @property
    def name(self) -> str:
        return "wikipedia"

    async def search(self, query: str, num_results: int = 5):
        self.calls.append((query, num_results))
        if query in self.fail_for:
            raise RuntimeError(f"simulated Wikipedia outage for '{query}'")
        return self.results_by_query.get(query, [])[:num_results]


def _result(title: str) -> dict:
    return {"title": title, "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "snippet": f"About {title}."}


class TestWikipediaTopicSourceProvider:
    def test_is_a_topic_source_provider(self) -> None:
        assert isinstance(WikipediaTopicSourceProvider(search_provider=FakeSearchProvider()), TopicSourceProvider)

    def test_name(self) -> None:
        assert WikipediaTopicSourceProvider(search_provider=FakeSearchProvider()).name == "wikipedia_evergreen"

    def test_discovers_real_distinct_candidates_across_default_seeds(self) -> None:
        results_by_query = {
            seed: [_result(f"{seed.title()} Article {i}") for i in range(2)]
            for seed in DEFAULT_EVERGREEN_CATEGORY_SEEDS
        }
        provider = WikipediaTopicSourceProvider(search_provider=FakeSearchProvider(results_by_query))

        candidates = asyncio.run(provider.discover_candidates(limit=10))

        assert len(candidates) == 10  # bounded by limit, even though more distinct candidates exist
        assert all(isinstance(c, TopicCandidate) for c in candidates)
        assert all(c.source == "wikipedia_evergreen" for c in candidates)
        assert all(c.normalized_title for c in candidates)
        # Never a hardcoded final-topic list - every returned title came
        # from the injected search results, not a fixed catalog in this module.
        all_possible_titles = {r["title"] for results in results_by_query.values() for r in results}
        assert {c.raw_title for c in candidates} <= all_possible_titles

    def test_respects_limit(self) -> None:
        results_by_query = {seed: [_result(f"{seed} {i}") for i in range(5)] for seed in DEFAULT_EVERGREEN_CATEGORY_SEEDS}
        provider = WikipediaTopicSourceProvider(search_provider=FakeSearchProvider(results_by_query))

        candidates = asyncio.run(provider.discover_candidates(limit=4))

        assert len(candidates) == 4

    def test_deduplicates_identical_titles_across_seeds(self) -> None:
        results_by_query = {
            "science explained": [_result("Photosynthesis")],
            "how technology works": [_result("Photosynthesis")],  # same real title from a different seed
        }
        provider = WikipediaTopicSourceProvider(
            search_provider=FakeSearchProvider(results_by_query),
            category_seeds=["science explained", "how technology works"],
        )

        candidates = asyncio.run(provider.discover_candidates(limit=10))

        assert len(candidates) == 1

    def test_one_seed_failure_does_not_prevent_other_seeds_from_succeeding(self) -> None:
        """One category seed's transient failure must not kill discovery
        entirely - the same resilience principle CompositeTopicSourceProvider
        applies at the whole-provider level, applied here at the seed level."""
        provider = WikipediaTopicSourceProvider(
            search_provider=FakeSearchProvider(
                results_by_query={"how technology works": [_result("How Batteries Work")]},
                fail_for={"science explained"},
            ),
            category_seeds=["science explained", "how technology works"],
        )

        candidates = asyncio.run(provider.discover_candidates(limit=10))

        assert len(candidates) == 1
        assert candidates[0].raw_title == "How Batteries Work"

    def test_all_seeds_failing_raises_topic_source_provider_error(self) -> None:
        provider = WikipediaTopicSourceProvider(
            search_provider=FakeSearchProvider(fail_for={"science explained", "how technology works"}),
            category_seeds=["science explained", "how technology works"],
        )

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates(limit=10))

    def test_no_results_anywhere_is_an_empty_list_not_an_error(self) -> None:
        """Genuinely finding nothing is never an error (see
        TopicSourceProvider's own documented contract)."""
        provider = WikipediaTopicSourceProvider(
            search_provider=FakeSearchProvider(results_by_query={}), category_seeds=["obscure seed"],
        )

        candidates = asyncio.run(provider.discover_candidates(limit=10))

        assert candidates == []

    def test_category_hint_overrides_default_seeds_with_a_single_seed(self) -> None:
        fake = FakeSearchProvider(results_by_query={"astronomy": [_result("Black Hole")]})
        provider = WikipediaTopicSourceProvider(search_provider=fake)

        candidates = asyncio.run(provider.discover_candidates(category="astronomy", limit=10))

        assert [c.raw_title for c in candidates] == ["Black Hole"]
        assert fake.calls == [("astronomy", 10)]  # only the hinted seed was queried, never the defaults

    def test_default_seeds_are_generic_categories_not_final_topics(self) -> None:
        """The built-in seed list must read as broad category search terms,
        never as specific hardcoded final video topics."""
        assert len(DEFAULT_EVERGREEN_CATEGORY_SEEDS) >= 3
        for seed in DEFAULT_EVERGREEN_CATEGORY_SEEDS:
            assert len(seed.split()) <= 5  # short category-shaped phrases, not full sentences/titles

    def test_custom_category_seeds_override_defaults(self) -> None:
        fake = FakeSearchProvider(results_by_query={"ocean life": [_result("Coral Reef")]})
        provider = WikipediaTopicSourceProvider(search_provider=fake, category_seeds=["ocean life"])

        candidates = asyncio.run(provider.discover_candidates(limit=10))

        assert [c.raw_title for c in candidates] == ["Coral Reef"]

    def test_real_wikipedia_search_provider_used_by_default_no_network_call_made(self) -> None:
        """Constructing without an injected search_provider must default to
        a real WikipediaSearchProvider - proven by type only, since this
        test never calls discover_candidates() and therefore never performs
        real network I/O."""
        from src.tools.wikipedia_provider import WikipediaSearchProvider

        provider = WikipediaTopicSourceProvider()
        assert isinstance(provider.search_provider, WikipediaSearchProvider)
