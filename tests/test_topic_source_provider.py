# Tests for TopicSourceProvider (ABC + Mock) and YouTubeTopicSourceProvider
# (all network calls mocked - no real YouTube Data API calls).
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from src.models.topic_planner import TopicCandidate
from src.services.topic_normalization import normalize_topic
from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider
from src.tools.topic_source_provider import (
    CompositeTopicSourceProvider,
    MockTopicSourceProvider,
    TopicSourceProvider,
    TopicSourceProviderError,
)
from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider


class TestMockTopicSourceProvider:
    def test_is_topic_source_provider(self) -> None:
        assert isinstance(MockTopicSourceProvider(), TopicSourceProvider)

    def test_default_candidates_returned(self) -> None:
        provider = MockTopicSourceProvider()
        candidates = asyncio.run(provider.discover_candidates())
        assert len(candidates) > 0
        assert all(isinstance(c, TopicCandidate) for c in candidates)
        assert all(c.normalized_title for c in candidates)

    def test_respects_limit(self) -> None:
        provider = MockTopicSourceProvider()
        candidates = asyncio.run(provider.discover_candidates(limit=2))
        assert len(candidates) == 2

    def test_records_calls(self) -> None:
        provider = MockTopicSourceProvider()
        asyncio.run(provider.discover_candidates(category="science", region="US", limit=5))
        assert provider.calls == [{"category": "science", "region": "US", "limit": 5}]

    def test_fail_true_raises(self) -> None:
        provider = MockTopicSourceProvider(fail=True)
        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates())

    def test_custom_candidates_injectable(self) -> None:
        custom = [TopicCandidate(raw_title="Custom Topic", normalized_title="custom topic", source="mock")]
        provider = MockTopicSourceProvider(candidates=custom)
        candidates = asyncio.run(provider.discover_candidates())
        assert candidates == custom


# ---- YouTubeTopicSourceProvider (real implementation, network mocked) ------


class FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://www.googleapis.com/youtube/v3/videos")
            raise httpx.HTTPStatusError(f"HTTP error {self.status_code}", request=request, response=self)

    def json(self) -> dict:
        return self._json_data


class FakeClient:
    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.last_params: dict | None = None

    async def get(self, url, params=None):
        self.last_params = params
        if self.exc is not None:
            raise self.exc
        return self.response


def _videos_payload(items: list[dict]) -> dict:
    return {"items": items}


def _video_item(video_id: str, title: str, view_count: str = "1000000", category_id: str = "27") -> dict:
    return {
        "id": video_id,
        "snippet": {"title": title, "categoryId": category_id},
        "statistics": {"viewCount": view_count},
    }


class TestYouTubeTopicSourceProvider:
    def test_is_topic_source_provider(self) -> None:
        assert isinstance(YouTubeTopicSourceProvider(api_key="key"), TopicSourceProvider)

    def test_no_api_key_raises(self) -> None:
        provider = YouTubeTopicSourceProvider(api_key=None, client=FakeClient())
        with pytest.raises(TopicSourceProviderError, match="YOUTUBE_API_KEY"):
            asyncio.run(provider.discover_candidates())

    def test_successful_discovery_returns_normalized_candidates(self) -> None:
        payload = _videos_payload([_video_item("v1", "Why Do Cats Purr?")])
        client = FakeClient(response=FakeResponse(payload))
        provider = YouTubeTopicSourceProvider(api_key="test-key", client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert len(candidates) == 1
        assert candidates[0].raw_title == "Why Do Cats Purr?"
        assert candidates[0].normalized_title == normalize_topic("Why Do Cats Purr?")
        assert candidates[0].source == "youtube"
        assert candidates[0].source_id == "v1"

    def test_view_count_maps_to_bounded_popularity_signal(self) -> None:
        payload = _videos_payload([_video_item("v1", "Topic", view_count="5000000")])
        client = FakeClient(response=FakeResponse(payload))
        provider = YouTubeTopicSourceProvider(api_key="test-key", client=client)

        candidates = asyncio.run(provider.discover_candidates())
        assert candidates[0].popularity_signal == 1.0

    def test_missing_view_count_leaves_popularity_signal_none(self) -> None:
        item = {"id": "v1", "snippet": {"title": "Topic"}, "statistics": {}}
        client = FakeClient(response=FakeResponse(_videos_payload([item])))
        provider = YouTubeTopicSourceProvider(api_key="test-key", client=client)

        candidates = asyncio.run(provider.discover_candidates())
        assert candidates[0].popularity_signal is None

    def test_api_key_sent_as_param(self) -> None:
        client = FakeClient(response=FakeResponse(_videos_payload([])))
        provider = YouTubeTopicSourceProvider(api_key="secret-key", client=client)

        asyncio.run(provider.discover_candidates())

        assert client.last_params["key"] == "secret-key"

    def test_category_and_region_forwarded(self) -> None:
        client = FakeClient(response=FakeResponse(_videos_payload([])))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        asyncio.run(provider.discover_candidates(category="27", region="GB"))

        assert client.last_params["videoCategoryId"] == "27"
        assert client.last_params["regionCode"] == "GB"

    def test_no_category_omits_param(self) -> None:
        client = FakeClient(response=FakeResponse(_videos_payload([])))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        asyncio.run(provider.discover_candidates())

        assert "videoCategoryId" not in client.last_params

    def test_respects_limit(self) -> None:
        items = [_video_item(f"v{i}", f"Topic {i}") for i in range(10)]
        client = FakeClient(response=FakeResponse(_videos_payload(items)))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        candidates = asyncio.run(provider.discover_candidates(limit=3))
        assert len(candidates) == 3

    def test_items_with_no_title_skipped(self) -> None:
        items = [{"id": "v1", "snippet": {"title": ""}, "statistics": {}}, _video_item("v2", "Real Topic")]
        client = FakeClient(response=FakeResponse(_videos_payload(items)))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        candidates = asyncio.run(provider.discover_candidates())
        assert len(candidates) == 1
        assert candidates[0].raw_title == "Real Topic"

    def test_malformed_response_raises(self) -> None:
        client = FakeClient(response=FakeResponse({"unexpected": "shape"}))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates())

    def test_timeout_raises_source_error(self) -> None:
        client = FakeClient(exc=httpx.TimeoutException("timed out"))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        with pytest.raises(TopicSourceProviderError, match="timed out"):
            asyncio.run(provider.discover_candidates())

    def test_http_status_error_raises_source_error(self) -> None:
        client = FakeClient(response=FakeResponse({}, status_code=403))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        with pytest.raises(TopicSourceProviderError, match="403"):
            asyncio.run(provider.discover_candidates())

    def test_network_error_raises_source_error(self) -> None:
        client = FakeClient(exc=httpx.ConnectError("connection refused"))
        provider = YouTubeTopicSourceProvider(api_key="key", client=client)

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates())


# ---- CompositeTopicSourceProvider -------------------------------------------


class TestCompositeTopicSourceProvider:
    def test_is_topic_source_provider(self) -> None:
        composite = CompositeTopicSourceProvider([MockTopicSourceProvider()])
        assert isinstance(composite, TopicSourceProvider)

    def test_requires_at_least_one_provider(self) -> None:
        with pytest.raises(ValueError):
            CompositeTopicSourceProvider([])

    def test_merges_candidates_from_all_components(self) -> None:
        a = MockTopicSourceProvider(
            candidates=[TopicCandidate(raw_title="Topic A", normalized_title="topic a", source="a")],
            provider_name="a",
        )
        b = MockTopicSourceProvider(
            candidates=[TopicCandidate(raw_title="Topic B", normalized_title="topic b", source="b")],
            provider_name="b",
        )
        composite = CompositeTopicSourceProvider([a, b])

        candidates = asyncio.run(composite.discover_candidates())

        assert {c.raw_title for c in candidates} == {"Topic A", "Topic B"}

    def test_name_joins_component_names(self) -> None:
        a = MockTopicSourceProvider(provider_name="a")
        b = MockTopicSourceProvider(provider_name="b")
        composite = CompositeTopicSourceProvider([a, b])
        assert composite.name == "a+b"

    def test_one_component_failing_does_not_crash_when_another_succeeds(self) -> None:
        working = MockTopicSourceProvider(
            candidates=[TopicCandidate(raw_title="Topic A", normalized_title="topic a", source="working")],
            provider_name="working",
        )
        failing = MockTopicSourceProvider(fail=True, provider_name="failing")
        composite = CompositeTopicSourceProvider([working, failing])

        candidates = asyncio.run(composite.discover_candidates())

        assert len(candidates) == 1
        assert composite.last_successful_sources == ["working"]
        assert len(composite.last_failures) == 1

    def test_all_components_failing_raises(self) -> None:
        a = MockTopicSourceProvider(fail=True, provider_name="a")
        b = MockTopicSourceProvider(fail=True, provider_name="b")
        composite = CompositeTopicSourceProvider([a, b])

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(composite.discover_candidates())

    def test_category_and_region_forwarded_to_every_component(self) -> None:
        a = MockTopicSourceProvider(provider_name="a")
        b = MockTopicSourceProvider(provider_name="b")
        composite = CompositeTopicSourceProvider([a, b])

        asyncio.run(composite.discover_candidates(category="science", region="GB", limit=7))

        assert a.calls[0] == {"category": "science", "region": "GB", "limit": 7}
        assert b.calls[0] == {"category": "science", "region": "GB", "limit": 7}


# ---- CurrentNewsTopicSourceProvider (real implementation, network mocked) --


class FakeTextResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://news.google.com/rss")
            raise httpx.HTTPStatusError(f"HTTP error {self.status_code}", request=request, response=self)


class FakeTextClient:
    def __init__(self, response: FakeTextResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.last_url: str | None = None
        self.last_headers: dict | None = None

    async def get(self, url, headers=None):
        self.last_url = url
        self.last_headers = headers
        if self.exc is not None:
            raise self.exc
        return self.response


def _rss_feed(items_xml: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>Top stories</title>{items_xml}</channel></rss>"
    )


def _rss_item(title: str, link: str = "https://example.com/a", pub_date: str = "Thu, 10 Sep 2026 10:00:00 GMT", source: str | None = "Example News") -> str:
    source_xml = f'<source url="https://example.com">{source}</source>' if source else ""
    return f"<item><title>{title}</title><link>{link}</link><pubDate>{pub_date}</pubDate>{source_xml}</item>"


class TestCurrentNewsTopicSourceProvider:
    def test_is_topic_source_provider(self) -> None:
        assert isinstance(CurrentNewsTopicSourceProvider(client=FakeTextClient()), TopicSourceProvider)

    def test_name_is_current_news(self) -> None:
        assert CurrentNewsTopicSourceProvider(client=FakeTextClient()).name == "current_news"

    def test_successful_discovery_returns_candidates(self) -> None:
        feed = _rss_feed(_rss_item("Big Story Today - Example News"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert len(candidates) == 1
        assert candidates[0].source == "current_news"

    def test_title_source_suffix_stripped(self) -> None:
        feed = _rss_feed(_rss_item("Big Story Today - Example News", source="Example News"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert candidates[0].raw_title == "Big Story Today"
        assert candidates[0].source_name == "Example News"

    def test_published_at_parsed_to_iso(self) -> None:
        feed = _rss_feed(_rss_item("Story", pub_date="Thu, 10 Sep 2026 10:00:00 GMT"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert candidates[0].published_at == "2026-09-10T10:00:00+00:00"

    def test_discovered_at_is_populated(self) -> None:
        feed = _rss_feed(_rss_item("Story"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert candidates[0].discovered_at is not None

    def test_source_url_and_id_populated(self) -> None:
        feed = _rss_feed(_rss_item("Story", link="https://example.com/story-123"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert candidates[0].source_url == "https://example.com/story-123"
        assert candidates[0].source_id == "https://example.com/story-123"

    def test_items_with_no_title_skipped(self) -> None:
        feed = _rss_feed(_rss_item("") + _rss_item("Real Story"))
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        assert len(candidates) == 1
        assert candidates[0].raw_title == "Real Story"

    def test_respects_limit(self) -> None:
        items = "".join(_rss_item(f"Story {i}", link=f"https://example.com/{i}") for i in range(10))
        client = FakeTextClient(response=FakeTextResponse(_rss_feed(items)))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates(limit=3))
        assert len(candidates) == 3

    def test_country_code_region_uses_geo_targeting_params(self) -> None:
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(language="en", client=client)

        asyncio.run(provider.discover_candidates(region="GB"))

        assert "gl=GB" in client.last_url
        assert "ceid=GB" in client.last_url

    def test_global_region_uses_world_feed_no_country_code(self) -> None:
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(client=client)

        asyncio.run(provider.discover_candidates(region="global"))

        assert "topic/WORLD" in client.last_url

    def test_free_text_region_becomes_search_query_not_hardcoded(self) -> None:
        """A sub-national region like 'Tamil Nadu' has no native Google
        News geo-targeting param - it must become a search term instead,
        proving no region name is hardcoded in the provider itself."""
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(client=client)

        asyncio.run(provider.discover_candidates(region="Tamil Nadu"))

        assert "rss/search" in client.last_url
        assert "Tamil" in client.last_url

    def test_category_becomes_search_query_term(self) -> None:
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(client=client)

        asyncio.run(provider.discover_candidates(category="technology"))

        assert "rss/search" in client.last_url
        assert "technology" in client.last_url

    def test_no_category_no_region_uses_world_feed(self) -> None:
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(client=client)

        asyncio.run(provider.discover_candidates())

        assert "topic/WORLD" in client.last_url

    def test_configured_language_forwarded(self) -> None:
        client = FakeTextClient(response=FakeTextResponse(_rss_feed("")))
        provider = CurrentNewsTopicSourceProvider(language="ta", client=client)

        asyncio.run(provider.discover_candidates(region="IN"))

        assert "hl=ta-IN" in client.last_url

    def test_malformed_xml_raises_source_error(self) -> None:
        client = FakeTextClient(response=FakeTextResponse("not valid xml <<<"))
        provider = CurrentNewsTopicSourceProvider(client=client)

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates())

    def test_timeout_raises_source_error(self) -> None:
        client = FakeTextClient(exc=httpx.TimeoutException("timed out"))
        provider = CurrentNewsTopicSourceProvider(client=client)

        with pytest.raises(TopicSourceProviderError, match="timed out"):
            asyncio.run(provider.discover_candidates())

    def test_http_status_error_raises_source_error(self) -> None:
        client = FakeTextClient(response=FakeTextResponse("", status_code=503))
        provider = CurrentNewsTopicSourceProvider(client=client)

        with pytest.raises(TopicSourceProviderError, match="503"):
            asyncio.run(provider.discover_candidates())

    def test_network_error_raises_source_error(self) -> None:
        client = FakeTextClient(exc=httpx.ConnectError("connection refused"))
        provider = CurrentNewsTopicSourceProvider(client=client)

        with pytest.raises(TopicSourceProviderError):
            asyncio.run(provider.discover_candidates())

    def test_default_client_follows_redirects(self) -> None:
        """Regression test: real Google News RSS URLs (confirmed against
        the live endpoint) respond with a 302 redirect to the actual feed
        content. httpx defaults to NOT following redirects, and its own
        raise_for_status() treats an un-followed redirect as an
        HTTPStatusError - a provider that doesn't explicitly request
        follow_redirects=True always fails with 'error: 302' in real
        usage, even though the feed genuinely exists and is reachable."""
        captured_kwargs = {}

        class _FakeAsyncClient:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, headers=None):
                return FakeTextResponse(_rss_feed(""), status_code=200)

            async def aclose(self):
                pass

        with patch("httpx.AsyncClient", _FakeAsyncClient):
            provider = CurrentNewsTopicSourceProvider()
            asyncio.run(provider.discover_candidates())

        assert captured_kwargs.get("follow_redirects") is True

    def test_does_not_store_full_article_body(self) -> None:
        """Only title/link/source/pubDate metadata is kept - never a full
        <description> article body."""
        feed = _rss_feed(
            "<item><title>Story</title><link>https://example.com/a</link>"
            "<pubDate>Thu, 10 Sep 2026 10:00:00 GMT</pubDate>"
            "<description>Full copyrighted article text goes here in detail...</description></item>"
        )
        client = FakeTextClient(response=FakeTextResponse(feed))
        provider = CurrentNewsTopicSourceProvider(client=client)

        candidates = asyncio.run(provider.discover_candidates())

        dumped = candidates[0].model_dump()
        assert "Full copyrighted article text" not in str(dumped)
