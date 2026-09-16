# Real current/trending-story topic discovery, backed by the free, public
# Google News RSS feed - no API key, no paid news API. Region/market and
# language are pass-through configuration only (see Settings.topic_target_markets/
# topic_language) - this module has no awareness of any specific country or
# region, so nothing is hardcoded to India/Tamil Nadu/UK/etc.
#
# Only lightweight metadata (title, publisher name, link, publish time, and
# - for CurrentNewsSearchProvider below - the RSS feed's own short
# description/snippet) is kept - never full article bodies (STEP 3: "Do not
# store entire copyrighted news articles"). A snippet is the same kind of
# short excerpt any search engine already displays, not the article itself.
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.models.topic_planner import TopicCandidate
from src.services.topic_normalization import normalize_topic
from src.tools.search_provider import SearchProvider
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError

GOOGLE_NEWS_SEARCH_URL = "https://news.google.com/rss/search"
GOOGLE_NEWS_WORLD_URL = "https://news.google.com/rss/headlines/section/topic/WORLD"

USER_AGENT = "YoutubeAgenticAI/0.1 (contact: local-development)"


class CurrentNewsSearchError(Exception):
    """Raised when CurrentNewsSearchProvider fails to retrieve results."""


class CurrentNewsTopicSourceProvider(TopicSourceProvider):
    """Topic source backed by Google News RSS - current/trending stories,
    not a fixed evergreen catalog. Region and category are plain pass-
    through query parameters; this provider assigns no special meaning to
    any specific market/category name."""

    def __init__(
        self,
        language: str = "en",
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.language = language or "en"
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def name(self) -> str:
        return "current_news"

    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = 15
    ) -> List[TopicCandidate]:
        url = self._build_feed_url(category=category, region=region)

        try:
            raw_xml = await _fetch_rss_xml(url, self._client, self.timeout_seconds)
        except _RssFetchError as e:
            raise TopicSourceProviderError(str(e)) from e

        try:
            items = _parse_rss_items(raw_xml)
        except ET.ParseError as e:
            raise TopicSourceProviderError(f"Current news discovery returned a malformed feed: {e}") from e

        discovered_at = datetime.now(timezone.utc).isoformat()
        candidates: List[TopicCandidate] = []
        for item in items[:limit]:
            candidate = _candidate_from_item(item, category=category, region=region, discovered_at=discovered_at)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _build_feed_url(self, category: Optional[str], region: Optional[str]) -> str:
        """Build a Google News RSS URL from plain configuration - never
        hardcodes a specific region/category; both are pass-through
        strings. A recognizable 2-letter country code gets Google News'
        native gl/ceid geo-targeting; anything else (including "global",
        None, or a free-text region like a state/city name) becomes part
        of the search query instead, since Google News RSS itself has no
        native sub-country geo-targeting."""
        query_terms = [t for t in (category, ) if t]

        is_country_code = bool(region) and region.lower() != "global" and len(region) == 2 and region.isalpha()
        if is_country_code:
            gl = region.upper()
            hl = f"{self.language}-{gl}"
            ceid = f"{gl}:{self.language}"
        else:
            gl = "US"
            hl = self.language if "-" in self.language else f"{self.language}-{gl}"
            ceid = f"{gl}:{self.language}"
            if region and region.lower() != "global":
                query_terms.append(region)

        if query_terms:
            query = quote(" ".join(query_terms))
            return f"{GOOGLE_NEWS_SEARCH_URL}?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
        return f"{GOOGLE_NEWS_WORLD_URL}?hl={hl}&gl={gl}&ceid={ceid}"


class _RssItem:
    __slots__ = ("title", "link", "pub_date", "source_name", "description")

    def __init__(
        self,
        title: str,
        link: Optional[str],
        pub_date: Optional[str],
        source_name: Optional[str],
        description: Optional[str] = None,
    ) -> None:
        self.title = title
        self.link = link
        self.pub_date = pub_date
        self.source_name = source_name
        self.description = description


def _parse_rss_items(raw_xml: str) -> List[_RssItem]:
    root = ET.fromstring(raw_xml)
    items: List[_RssItem] = []
    for item_el in root.iter("item"):
        title_el = item_el.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        link_el = item_el.find("link")
        link = (link_el.text or "").strip() if link_el is not None else None
        pubdate_el = item_el.find("pubDate")
        pub_date = (pubdate_el.text or "").strip() if pubdate_el is not None else None
        source_el = item_el.find("source")
        source_name = (source_el.text or "").strip() if source_el is not None else None
        # Google News RSS <description> is a short HTML snippet (typically
        # just the headline again, wrapped in an <a> tag, sometimes with a
        # trailing related-coverage list) - a search-result-style excerpt,
        # never the article body itself. Only used by CurrentNewsSearchProvider
        # below (STEP 2's "content/snippets where appropriate"); discovery
        # candidates still never carry it (STEP 3's own "no article bodies").
        description_el = item_el.find("description")
        description = _clean_html_snippet(description_el.text) if description_el is not None else None
        items.append(
            _RssItem(title=title, link=link, pub_date=pub_date, source_name=source_name, description=description)
        )
    return items


def _clean_html_snippet(raw: Optional[str]) -> Optional[str]:
    """Strip HTML tags/entities from an RSS description, mirroring
    WikipediaSearchProvider's own _clean_snippet - the same "search result
    snippet" treatment, not a new parsing idiom."""
    if not raw:
        return None
    without_tags = re.sub(r"<[^>]+>", " ", raw)
    cleaned = html.unescape(without_tags)
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _parse_pub_date(pub_date: Optional[str]) -> Optional[str]:
    """Parse an RFC 2822 RSS pubDate into an ISO 8601 UTC string - shared
    by candidate discovery and search so both report the exact same
    timestamp format."""
    if not pub_date:
        return None
    try:
        parsed = parsedate_to_datetime(pub_date)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


class _RssFetchError(Exception):
    """Internal: raised by _fetch_rss_xml, translated by each caller into
    its own public exception type (TopicSourceProviderError vs
    CurrentNewsSearchError) - the HTTP fetch itself is identical either way."""


async def _fetch_rss_xml(url: str, client: Optional[httpx.AsyncClient], timeout_seconds: float) -> str:
    """Fetch raw RSS XML from a Google News URL - the one place the actual
    HTTP request/redirect/error handling lives, shared by
    CurrentNewsTopicSourceProvider.discover_candidates and
    CurrentNewsSearchProvider.search so neither duplicates it."""
    owns_client = client is None
    # Google News RSS URLs (both the WORLD section feed and search
    # results) respond with a 302 redirect to the actual feed content -
    # confirmed against the real endpoint. httpx defaults to NOT
    # following redirects, and its own raise_for_status() treats an
    # un-followed redirect as an error, so this must be explicit.
    active_client = client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
    try:
        response = await active_client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.text
    except httpx.TimeoutException as e:
        raise _RssFetchError(f"Google News request timed out: {e}") from e
    except httpx.HTTPStatusError as e:
        raise _RssFetchError(f"Google News request returned an error: {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise _RssFetchError(f"Google News request failed: {e}") from e
    finally:
        if owns_client:
            await active_client.aclose()


def _candidate_from_item(
    item: _RssItem, category: Optional[str], region: Optional[str], discovered_at: str
) -> Optional[TopicCandidate]:
    title = item.title
    # Google News titles are commonly "Headline - Source Name" - strip the
    # redundant trailing source suffix when it matches the <source> tag,
    # so the candidate's own title doesn't duplicate source_name.
    if item.source_name and title.endswith(f" - {item.source_name}"):
        title = title[: -(len(item.source_name) + 3)].strip()
    if not title:
        return None

    published_at = _parse_pub_date(item.pub_date)

    return TopicCandidate(
        raw_title=title,
        normalized_title=normalize_topic(title),
        source="current_news",
        source_id=item.link,
        category=category,
        source_name=item.source_name,
        source_url=item.link,
        published_at=published_at,
        discovered_at=discovered_at,
        market=region,
    )


class CurrentNewsSearchProvider(SearchProvider):
    """SearchProvider-interface adapter over the SAME real Google News RSS
    infrastructure as CurrentNewsTopicSourceProvider (shared HTTP fetch,
    XML parsing, and redirect handling via ``_fetch_rss_xml``/
    ``_parse_rss_items`` - never a second, duplicated implementation).

    Used by ResearchAgent for a topic whose Topic Planner classification
    is "current_news" - Wikipedia's encyclopedia index has no dedicated
    article for a specific recent event, so a topic string derived from
    breaking news is instead searched directly against the same live feed
    the Topic Planner already trusts for freshness. Returns the standard
    ``{title, url, snippet}`` dict shape every SearchProvider already
    returns, extended with two additional, purely optional keys
    (``published_at``, ``source_name``) that ResearchAgent uses for
    provenance when present - never required by the base contract.
    """

    def __init__(
        self,
        language: str = "en",
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.language = language or "en"
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def name(self) -> str:
        return "current_news"

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search Google News for ``query`` and return normalized results.

        Raises:
            CurrentNewsSearchError: On network failure, a malformed feed,
                or an empty/whitespace-only query.
        """
        if not query or not query.strip():
            return []

        hl = self.language if "-" in self.language else f"{self.language}-US"
        url = f"{GOOGLE_NEWS_SEARCH_URL}?q={quote(query)}&hl={hl}&gl=US&ceid=US:{self.language}"

        try:
            raw_xml = await _fetch_rss_xml(url, self._client, self.timeout_seconds)
        except _RssFetchError as e:
            raise CurrentNewsSearchError(str(e)) from e

        try:
            items = _parse_rss_items(raw_xml)
        except ET.ParseError as e:
            raise CurrentNewsSearchError(f"Current news search returned a malformed feed: {e}") from e

        results: List[Dict[str, Any]] = []
        for item in items[:num_results]:
            if not item.link:
                continue
            results.append(
                {
                    "title": item.title,
                    "url": item.link,
                    # The snippet is what ResearchAgent actually synthesizes
                    # from - fall back to the title itself if the feed's
                    # <description> was empty/unparseable, rather than an
                    # empty string that would silently contribute nothing.
                    "snippet": item.description or item.title,
                    "published_at": _parse_pub_date(item.pub_date),
                    "source_name": item.source_name,
                }
            )
        return results
