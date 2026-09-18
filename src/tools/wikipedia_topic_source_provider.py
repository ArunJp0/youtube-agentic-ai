# Wikipedia-backed evergreen topic discovery: a free, no-API-key-required
# TopicSourceProvider used to close the exact production gap a real
# autonomous run exposed - TIER 2 (evergreen fallback) had only ONE real
# discovery path (YouTube), which is a single point of failure whenever
# YOUTUBE_API_KEY is absent/unavailable.
#
# Deliberately reuses the SAME public Wikipedia MediaWiki search API
# WikipediaSearchProvider already talks to for Research - via a
# SearchProvider dependency, not a second HTTP implementation - so this
# module owns only topic-discovery-specific concerns: turning a bounded,
# centralized list of GENERIC evergreen category seed queries (never final
# video topics) into real, distinct TopicCandidates.
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from src.models.topic_planner import TopicCandidate
from src.services.topic_normalization import normalize_topic
from src.tools.search_provider import SearchProvider
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError

# Generic evergreen CATEGORY seed queries (never a hardcoded list of final
# video topics) - each is used as a Wikipedia search term whose real
# results become candidates. Broad enough to cover the channel's existing
# evergreen educational strategy (science, technology, history, geography,
# everyday "why/how" explainers) without ever fixing what the actual
# discovered titles will be. Centralized and overridable via
# Settings.topic_evergreen_category_seeds.
DEFAULT_EVERGREEN_CATEGORY_SEEDS: List[str] = [
    "science explained",
    "how technology works",
    "history explained",
    "why does this happen",
    "how does this work",
    "geography and nature",
]

DEFAULT_DISCOVERY_LIMIT = 15


class WikipediaTopicSourceProvider(TopicSourceProvider):
    """Discovers real, distinct evergreen topic candidates from Wikipedia's
    public search API - no API key, no paid dependency, reuses the exact
    same MediaWiki search endpoint WikipediaSearchProvider already uses for
    Research (injected as a SearchProvider, never a duplicated HTTP client).

    Each configured category seed is searched independently and leniently:
    one seed's failure (network/HTTP error) is recorded and skipped, the
    same way CompositeTopicSourceProvider treats one whole PROVIDER's
    failure - only if EVERY seed fails does this raise
    TopicSourceProviderError, never for a seed that simply found nothing.
    """

    def __init__(
        self,
        search_provider: Optional[SearchProvider] = None,
        category_seeds: Optional[List[str]] = None,
    ) -> None:
        """Initialize the provider.

        Args:
            search_provider: SearchProvider used to query Wikipedia - a
                real ``WikipediaSearchProvider()`` by default (free, no API
                key), or an injected mock/fake for tests.
            category_seeds: Generic evergreen category search seeds (see
                ``DEFAULT_EVERGREEN_CATEGORY_SEEDS``) - never final topic
                titles, only search terms real Wikipedia results are found
                from.
        """
        if search_provider is None:
            from src.tools.wikipedia_provider import WikipediaSearchProvider

            search_provider = WikipediaSearchProvider()
        self.search_provider = search_provider
        self.category_seeds = list(category_seeds) if category_seeds else list(DEFAULT_EVERGREEN_CATEGORY_SEEDS)

    @property
    def name(self) -> str:
        return "wikipedia_evergreen"

    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = DEFAULT_DISCOVERY_LIMIT
    ) -> List[TopicCandidate]:
        seeds = [category] if category and category.strip() else self.category_seeds
        if not seeds:
            return []

        per_seed_limit = max(1, -(-limit // len(seeds)))  # ceiling division, bounded per seed
        candidates: List[TopicCandidate] = []
        seen_titles = set()
        seed_failures: List[str] = []
        discovered_at = datetime.now(timezone.utc).isoformat()

        for seed in seeds:
            if len(candidates) >= limit:
                break
            try:
                results = await self.search_provider.search(seed, per_seed_limit)
            except Exception as e:
                # One seed's failure is not a whole-provider failure - try
                # the remaining seeds, exactly like CompositeTopicSourceProvider
                # treats one component provider's failure.
                seed_failures.append(f"'{seed}': {e}")
                continue

            for item in results:
                title = (item.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                candidates.append(
                    TopicCandidate(
                        raw_title=title,
                        normalized_title=normalize_topic(title),
                        source=self.name,
                        source_id=item.get("url"),
                        category=seed,
                        source_name="Wikipedia",
                        source_url=item.get("url"),
                        discovered_at=discovered_at,
                    )
                )
                if len(candidates) >= limit:
                    break

        if not candidates and seed_failures:
            raise TopicSourceProviderError(
                "All Wikipedia evergreen category seeds failed: " + "; ".join(seed_failures)
            )

        return candidates[:limit]
