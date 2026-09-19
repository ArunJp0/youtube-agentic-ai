# Wikipedia search provider (free, no API key required)
from __future__ import annotations

import asyncio
import html
import re
from typing import Any, Dict, List, Optional

import httpx

from src.tools.search_provider import SearchProvider

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Hard OUTER ceiling wrapped around the actual network call
# (asyncio.wait_for), on top of - never instead of - the httpx client
# timeout below. See src.tools.pexels_media_provider's identical pattern
# (added after a real controlled autonomous run proved a client-level
# timeout alone did not reliably bound a network call in this pipeline) -
# the same defense in depth applied here.
DEFAULT_OUTER_TIMEOUT_SECONDS = 20.0

# Wikimedia's API gateway rejects requests without a descriptive User-Agent
# (returns 403), per https://meta.wikimedia.org/wiki/User-Agent_policy.
USER_AGENT = "YoutubeAgenticAI/0.1 (contact: local-development)"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}


class WikipediaSearchError(Exception):
    """Raised when the Wikipedia search provider fails to retrieve results."""


class WikipediaSearchProvider(SearchProvider):
    """Search provider backed by the public Wikipedia MediaWiki API.

    Wikipedia-specific request/response handling (HTML snippet cleanup, URL
    construction) is fully contained here; callers only see the normalized
    ``{title, url, snippet}`` dicts shared with every other SearchProvider.
    """

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
        outer_timeout_seconds: float = DEFAULT_OUTER_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the provider.

        Args:
            timeout_seconds: Request timeout for the Wikipedia API.
            client: Optional pre-configured httpx.AsyncClient (mainly for
                tests). When omitted, a short-lived client is created per
                request.
            outer_timeout_seconds: Hard outer ceiling on top of
                ``timeout_seconds`` - see ``DEFAULT_OUTER_TIMEOUT_SECONDS``.
        """
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.outer_timeout_seconds = outer_timeout_seconds

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia and return normalized results.

        Raises:
            WikipediaSearchError: On network failure, an HTTP error, or a
                timeout (either the inner per-request client timeout, or
                the outer hard ceiling - this never waits indefinitely
                either way).
        """
        if not query or not query.strip():
            return []

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }

        try:
            data = await asyncio.wait_for(self._do_search(params), timeout=self.outer_timeout_seconds)
        except asyncio.TimeoutError as e:
            raise WikipediaSearchError(
                f"Wikipedia search timed out after {self.outer_timeout_seconds:.0f}s "
                "(provider=wikipedia, operation=search, category=timeout)"
            ) from e
        except httpx.TimeoutException as e:
            raise WikipediaSearchError(f"Wikipedia search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise WikipediaSearchError(
                f"Wikipedia search returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise WikipediaSearchError(f"Wikipedia search request failed: {e}") from e

        raw_results = data.get("query", {}).get("search", [])

        normalized: List[Dict[str, Any]] = []
        for item in raw_results:
            title = item.get("title", "")
            if not title:
                continue
            snippet = _clean_snippet(item.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            normalized.append({"title": title, "url": url, "snippet": snippet})

        return normalized[:num_results]

    async def _do_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # http2=True: Wikimedia's edge returns 403 for plain HTTP/1.1 requests
        # (confirmed against the live API) even with a compliant User-Agent;
        # negotiating HTTP/2 (as curl/browsers do by default) resolves it.
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds, http2=True)
        try:
            response = await client.get(WIKIPEDIA_API_URL, params=params, headers=REQUEST_HEADERS)
            response.raise_for_status()
            return response.json()
        finally:
            if owns_client:
                await client.aclose()


def _clean_snippet(raw_snippet: str) -> str:
    """Strip MediaWiki search-highlight HTML tags and unescape entities."""
    without_tags = re.sub(r"<[^>]+>", "", raw_snippet)
    return html.unescape(without_tags)
