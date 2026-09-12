# Topic source provider abstraction. Mirrors every other provider interface
# in this project (SearchProvider, MediaProvider, VoiceProvider, ...): the
# Topic Planner Agent only ever depends on this interface, never on a
# concrete vendor API directly, so additional topic sources (e.g. a future
# Google Trends provider) can be added later without changing
# TopicPlannerAgent at all.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.topic_planner import TopicCandidate


class TopicSourceProviderError(Exception):
    """Raised when a topic source provider fails to retrieve candidates -
    network failure, HTTP error, or a malformed response. Never raised for
    "the source legitimately had nothing to offer" (that's an empty list,
    not an error)."""


class TopicSourceProvider(ABC):
    """Abstract base class for topic-candidate discovery sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider, e.g. 'youtube', 'mock' -
        recorded on every TopicCandidate/TopicSelectionResult it produces."""
        raise NotImplementedError

    @abstractmethod
    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = 15
    ) -> List[TopicCandidate]:
        """Discover up to ``limit`` raw candidate topics.

        Args:
            category: Optional niche/category hint (provider-specific
                interpretation - e.g. a YouTube category name/id)
            region: Optional region/locale hint (e.g. an ISO 3166-1 alpha-2
                country code)
            limit: Maximum number of candidates to return

        Returns:
            Candidates with ``normalized_title`` already populated (see
            ``src.services.topic_normalization.normalize_topic``) - an
            empty list is a valid "nothing available" outcome, never an
            error.

        Raises:
            TopicSourceProviderError: On a genuine source failure (network,
                HTTP, malformed response) - never for a merely empty result.
        """
        raise NotImplementedError


class MockTopicSourceProvider(TopicSourceProvider):
    """In-memory test double - no network. Returns a fixed/injectable list
    of candidates, or raises TopicSourceProviderError if configured to
    simulate a source failure."""

    def __init__(
        self,
        candidates: Optional[List[TopicCandidate]] = None,
        fail: bool = False,
        provider_name: str = "mock",
    ) -> None:
        self._candidates = candidates if candidates is not None else _default_mock_candidates()
        self.fail = fail
        self._name = provider_name
        self.calls: List[dict] = []

    @property
    def name(self) -> str:
        return self._name

    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = 15
    ) -> List[TopicCandidate]:
        self.calls.append({"category": category, "region": region, "limit": limit})
        if self.fail:
            raise TopicSourceProviderError("simulated topic source outage")
        return list(self._candidates[:limit])


class CompositeTopicSourceProvider(TopicSourceProvider):
    """Merges candidates from multiple TopicSourceProviders into one -
    e.g. YouTube (evergreen-popular) + CurrentNewsTopicSourceProvider
    (trending) together. TopicPlannerAgent's own constructor never needs
    to know about this - it still just takes ONE TopicSourceProvider,
    which may be a composite wrapping several real sources.

    Each component provider is called independently: one component
    failing (TopicSourceProviderError) is recorded and skipped, never
    propagated, as long as at least one component succeeds - "a news
    provider failure does not crash planning when another source is
    available" (STEP 11). Only if EVERY component fails does this raise,
    so the agent's existing "source_unavailable" path still applies when
    genuinely nothing is available.
    """

    def __init__(self, providers: List[TopicSourceProvider]) -> None:
        if not providers:
            raise ValueError("CompositeTopicSourceProvider requires at least one component provider")
        self.providers = providers
        # Populated by the most recent discover_candidates() call - which
        # component providers actually returned data (vs. failed) that
        # time. TopicPlannerAgent reads this (if present) to record an
        # accurate ``sources_used`` on the TopicSelectionResult.
        self.last_successful_sources: List[str] = []
        self.last_failures: List[str] = []

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = 15
    ) -> List[TopicCandidate]:
        merged: List[TopicCandidate] = []
        successful: List[str] = []
        failures: List[str] = []

        for provider in self.providers:
            try:
                candidates = await provider.discover_candidates(category=category, region=region, limit=limit)
            except TopicSourceProviderError as e:
                failures.append(f"{provider.name}: {e}")
                continue
            merged.extend(candidates)
            successful.append(provider.name)

        self.last_successful_sources = successful
        self.last_failures = failures

        if not successful:
            raise TopicSourceProviderError(
                "All configured topic sources failed: " + "; ".join(failures) if failures else "no sources configured"
            )
        return merged


def _default_mock_candidates() -> List[TopicCandidate]:
    from src.services.topic_normalization import normalize_topic

    titles = [
        "Why do cats purr?",
        "How do vaccines work?",
        "Why does ice float on water?",
        "What causes the northern lights?",
        "Why do we dream?",
    ]
    return [
        TopicCandidate(
            raw_title=title,
            normalized_title=normalize_topic(title),
            source="mock",
            source_id=f"mock-{index}",
            category="education",
            popularity_signal=0.5,
        )
        for index, title in enumerate(titles)
    ]
