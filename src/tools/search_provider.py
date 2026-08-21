# Search provider abstraction layer
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import asyncio


class SearchProvider(ABC):
    """Abstract base class for search providers.

    Concrete implementations must provide the ``search`` method that
    returns raw search results for a given query.
    """

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for information about a query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of search result dictionaries with at least 'title', 'url', 'snippet' keys
        """
        raise NotImplementedError


class MockSearchProvider(SearchProvider):
    """Mock search provider for development/testing without external API keys.

    Returns predetermined realistic results for sample queries to enable
    local testing of the research workflow.
    """

    # Predefined mock data for common queries
    _MOCK_RESULTS: Dict[str, List[Dict[str, str]]] = {
        "why do humans dream": [
            {
                "title": "The Science of Dreams - Harvard Medical School",
                "url": "https://www.health.harvard.edu/vital-signs/the-science-of-dreams",
                "snippet": "Scientists believe our sleeping brain cleans out toxins and puts memories in long-term storage.",
            },
            {
                "title": "Why Do Humans Dream? - National Geographic",
                "url": "https://www.nationalgeographic.com/science/article/why-do-humans-dream",
                "snippet": "Dreams may serve important functions like learning, memory consolidation, and emotional regulation.",
            },
            {
                "title": "REM Sleep and Dreaming - Sleep Foundation",
                "url": "https://www.sleepfoundation.org/rem-sleep/what-are-rem-sleep-dreams",
                "snippet": "REM sleep is when most vivid dreaming occurs, with brain activity matching wakefulness.",
            },
            {
                "title": "Dream Functions in Psychology",
                "url": "https://www.apa.org/monitor/2017/01/dream-functions",
                "snippet": "Psychologists propose multiple theories including memory consolidation and threat simulation.",
            },
            {
                "title": "Neuroscience Research on Dreams",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876157/",
                "snippet": "Recent neuroimaging studies show the brain's default mode network is highly active during REM sleep.",
            },
        ],
        "how does photosynthesis work": [
            {
                "title": "Photosynthesis: Process, Stages, & Types",
                "url": "https://www.nature.com/articles/s41586-022-04589-9",
                "snippet": "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
            },
            {
                "title": "The Process of Photosynthesis - Biology",
                "url": "https://www.khanacademy.org/science/biology/photosynthesis-in-plants",
                "snippet": "Light-dependent reactions capture energy, while Calvin cycle uses it to make sugar.",
            },
        ],
        "what causes earthquakes": [
            {
                "title": "Earthquake Causes - USGS",
                "url": "https://www.usgs.gov/faqs/what-causes-earthquakes",
                "snippet": "Most earthquakes result from tectonic plate movement and stress accumulation.",
            },
        ],
    }

    async def search(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """Return mock search results for the query.

        If no mock data exists for the query, returns a generic placeholder.
        """
        query_lower = query.lower().strip()

        # Look for exact matches or partial matches
        for key in self._MOCK_RESULTS:
            if key in query_lower or query_lower in key:
                results = self._MOCK_RESULTS[key][:num_results]
                return [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in results]

        # Generic mock response for unknown queries
        return [
            {
                "title": f"Mock result 1 for '{query}'",
                "url": "https://example.com/result1",
                "snippet": f"Mock snippet for research on {query}",
            },
            {
                "title": f"Mock result 2 for '{query}'",
                "url": "https://example.com/result2",
                "snippet": f"Additional information about {query}",
            },
            {
                "title": f"Mock result 3 for '{query}'",
                "url": "https://example.com/result3",
                "snippet": f"Further details about {query}",
            },
        ][:num_results]


_async_search_instance: MockSearchProvider | None = None


def get_mock_search_provider() -> MockSearchProvider:
    """Singleton factory for MockSearchProvider."""
    global _async_search_instance
    if _async_search_instance is None:
        _async_search_instance = MockSearchProvider()
    return _async_search_instance


async def search_raw(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Convenience function for mock search."""
    provider = get_mock_search_provider()
    return await provider.search(query, num_results)