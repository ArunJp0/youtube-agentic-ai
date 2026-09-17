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
                "snippet": (
                    "Photosynthesis converts light energy into chemical energy stored in "
                    "glucose molecules, using carbon dioxide and water absorbed by the "
                    "plant's leaves and roots as the essential starting raw materials."
                ),
            },
            {
                "title": "The Process of Photosynthesis - Biology",
                "url": "https://www.khanacademy.org/science/biology/photosynthesis-in-plants",
                "snippet": (
                    "Light-dependent reactions capture energy in the thylakoid membrane, "
                    "while the Calvin cycle uses that captured energy in the stroma to "
                    "fix carbon and ultimately make sugar the plant can use or store."
                ),
            },
        ],
        "what causes earthquakes": [
            {
                "title": "Earthquake Causes - USGS",
                "url": "https://www.usgs.gov/faqs/what-causes-earthquakes",
                "snippet": (
                    "Most earthquakes result from tectonic plate movement and the "
                    "gradual buildup of stress accumulation along a fault line, which "
                    "is released suddenly as seismic waves once it exceeds the rock's "
                    "frictional strength."
                ),
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

        # Generic mock response for unknown queries. Each snippet is
        # deliberately substantial and distinctly worded (not just a short
        # templated phrase) so callers exercising pipeline/orchestration
        # behavior (rather than research-content-quality itself, which has
        # its own dedicated real-content fixtures/tests) don't incidentally
        # trip ResearchAgent's real-world-calibrated minimum-substance gate
        # - still obviously a mock, never real content.
        return [
            {
                "title": f"Mock background report on {query}",
                "url": "https://example.com/result1",
                "snippet": (
                    f"Mock research snippet exploring {query} in some detail, covering its "
                    "background, key mechanisms, and why it matters, with illustrative "
                    "examples drawn from representative studies and expert commentary."
                ),
            },
            {
                "title": f"Mock practical implications for {query}",
                "url": "https://example.com/result2",
                "snippet": (
                    f"A second mock perspective on {query}, focusing on practical "
                    "implications, common misconceptions, and how current understanding "
                    "has evolved based on recent findings and ongoing expert discussion."
                ),
            },
            {
                "title": f"Mock open questions about {query}",
                "url": "https://example.com/result3",
                "snippet": (
                    f"A third mock viewpoint on {query}, summarizing supporting evidence, "
                    "notable examples, and open questions left for future investigation, "
                    "plus broader context useful for understanding the bigger picture."
                ),
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