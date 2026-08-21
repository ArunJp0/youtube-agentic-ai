# Tests for WikipediaSearchProvider (all network calls are mocked)

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.tools.search_provider import SearchProvider
from src.tools.wikipedia_provider import (
    REQUEST_HEADERS,
    USER_AGENT,
    WikipediaSearchProvider,
    WikipediaSearchError,
)


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://en.wikipedia.org/w/api.php")
            raise httpx.HTTPStatusError(
                f"HTTP error {self.status_code}", request=request, response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._json_data


class FakeClient:
    """Minimal async stand-in for httpx.AsyncClient, injected for testing."""

    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.last_params: dict | None = None
        self.last_headers: dict | None = None
        self.last_url: str | None = None

    async def get(self, url, params=None, headers=None):
        self.last_url = url
        self.last_params = params
        self.last_headers = headers
        if self.exc is not None:
            raise self.exc
        return self.response


def _search_payload(results: list[dict]) -> dict:
    return {"query": {"search": results}}


class TestWikipediaSearchProvider:
    """Tests for WikipediaSearchProvider.search with the client mocked out."""

    def test_is_search_provider(self) -> None:
        assert isinstance(WikipediaSearchProvider(), SearchProvider)

    def test_empty_query_returns_empty_list(self) -> None:
        provider = WikipediaSearchProvider(client=FakeClient())
        results = asyncio.run(provider.search("   "))
        assert results == []

    def test_successful_search_returns_normalized_results(self) -> None:
        payload = _search_payload(
            [
                {
                    "title": "Dream",
                    "snippet": "A <span class=\"searchmatch\">dream</span> is a succession of images.",
                }
            ]
        )
        client = FakeClient(response=FakeResponse(payload))
        provider = WikipediaSearchProvider(client=client)

        results = asyncio.run(provider.search("dream", num_results=5))

        assert len(results) == 1
        assert results[0]["title"] == "Dream"
        assert results[0]["url"] == "https://en.wikipedia.org/wiki/Dream"
        assert "<span" not in results[0]["snippet"]
        assert "dream" in results[0]["snippet"].lower()
        assert client.last_params["srlimit"] == 5

    def test_request_includes_user_agent_and_accept_headers(self) -> None:
        """Wikimedia rejects requests without a descriptive User-Agent (403)."""
        client = FakeClient(response=FakeResponse(_search_payload([])))
        provider = WikipediaSearchProvider(client=client)

        asyncio.run(provider.search("dream"))

        assert client.last_headers == REQUEST_HEADERS
        assert client.last_headers["User-Agent"] == USER_AGENT
        assert "YoutubeAgenticAI" in client.last_headers["User-Agent"]
        assert client.last_headers["Accept"] == "application/json"

    def test_title_with_spaces_becomes_underscored_url(self) -> None:
        payload = _search_payload([{"title": "REM sleep", "snippet": "Rapid eye movement"}])
        client = FakeClient(response=FakeResponse(payload))
        provider = WikipediaSearchProvider(client=client)

        results = asyncio.run(provider.search("REM sleep"))
        assert results[0]["url"] == "https://en.wikipedia.org/wiki/REM_sleep"

    def test_no_results_found_returns_empty_list(self) -> None:
        client = FakeClient(response=FakeResponse(_search_payload([])))
        provider = WikipediaSearchProvider(client=client)

        results = asyncio.run(provider.search("asdkjhaskjdhaskjdh_nonexistent"))
        assert results == []

    def test_respects_num_results_limit(self) -> None:
        payload = _search_payload(
            [{"title": f"Topic {i}", "snippet": f"Snippet {i}"} for i in range(10)]
        )
        client = FakeClient(response=FakeResponse(payload))
        provider = WikipediaSearchProvider(client=client)

        results = asyncio.run(provider.search("topic", num_results=3))
        assert len(results) == 3

    def test_timeout_raises_search_error(self) -> None:
        client = FakeClient(exc=httpx.TimeoutException("timed out"))
        provider = WikipediaSearchProvider(client=client)

        with pytest.raises(WikipediaSearchError, match="timed out"):
            asyncio.run(provider.search("dream"))

    def test_http_status_error_raises_search_error(self) -> None:
        client = FakeClient(response=FakeResponse({}, status_code=503))
        provider = WikipediaSearchProvider(client=client)

        with pytest.raises(WikipediaSearchError, match="503"):
            asyncio.run(provider.search("dream"))

    def test_network_error_raises_search_error(self) -> None:
        client = FakeClient(exc=httpx.ConnectError("connection refused"))
        provider = WikipediaSearchProvider(client=client)

        with pytest.raises(WikipediaSearchError):
            asyncio.run(provider.search("dream"))
