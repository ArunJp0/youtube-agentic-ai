# Real current/trending-story topic discovery, backed by the free, public
# Google News RSS feed - no API key, no paid news API. Region/market and
# language are pass-through configuration only (see Settings.topic_target_markets/
# topic_language) - this module has no awareness of any specific country or
# region, so nothing is hardcoded to India/Tamil Nadu/UK/etc.
#
# Only lightweight metadata (title, publisher name, link, publish time) is
# kept - never full article bodies (STEP 3: "Do not store entire
# copyrighted news articles").
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional
from urllib.parse import quote

import httpx

from src.models.topic_planner import TopicCandidate
from src.services.topic_normalization import normalize_topic
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError

GOOGLE_NEWS_SEARCH_URL = "https://news.google.com/rss/search"
GOOGLE_NEWS_WORLD_URL = "https://news.google.com/rss/headlines/section/topic/WORLD"

USER_AGENT = "YoutubeAgenticAI/0.1 (contact: local-development)"


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

        owns_client = self._client is None
        # Google News RSS URLs (both the WORLD section feed and search
        # results) respond with a 302 redirect to the actual feed content -
        # confirmed against the real endpoint. httpx defaults to NOT
        # following redirects, and its own raise_for_status() treats an
        # un-followed redirect as an error, so this must be explicit.
        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True)
        try:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            raw_xml = response.text
        except httpx.TimeoutException as e:
            raise TopicSourceProviderError(f"Current news discovery timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise TopicSourceProviderError(
                f"Current news discovery returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise TopicSourceProviderError(f"Current news discovery request failed: {e}") from e
        finally:
            if owns_client:
                await client.aclose()

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
    __slots__ = ("title", "link", "pub_date", "source_name")

    def __init__(self, title: str, link: Optional[str], pub_date: Optional[str], source_name: Optional[str]) -> None:
        self.title = title
        self.link = link
        self.pub_date = pub_date
        self.source_name = source_name


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
        items.append(_RssItem(title=title, link=link, pub_date=pub_date, source_name=source_name))
    return items


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

    published_at = None
    if item.pub_date:
        try:
            parsed = parsedate_to_datetime(item.pub_date)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            published_at = parsed.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            published_at = None

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
