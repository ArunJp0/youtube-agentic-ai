# Real topic-candidate discovery backed by the public YouTube Data API v3
# `videos.list(chart=mostPopular)` endpoint. Deliberately a plain API-key
# request (the pre-existing, previously-unused Settings.youtube_api_key
# placeholder) - NOT the OAuth-scoped YouTubeClient used for uploads, which
# has no search/discovery surface at all. This is read-only public data,
# never touches the authenticated channel or any upload/scheduling action.
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from src.models.topic_planner import TopicCandidate
from src.services.topic_normalization import normalize_topic
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError

YOUTUBE_VIDEOS_API_URL = "https://www.googleapis.com/youtube/v3/videos"

DEFAULT_REGION_CODE = "US"

# view counts at/above this are treated as maximally "popular" (signal=1.0)
# for the bounded 0-1 popularity_signal - a fixed, documented reference
# point rather than a per-call dynamically-rescaled range (which would make
# the same video score differently run to run depending on what else was
# fetched alongside it).
_POPULARITY_VIEW_COUNT_CEILING = 5_000_000.0


class YouTubeTopicSourceProvider(TopicSourceProvider):
    """Topic source backed by YouTube's public `mostPopular` chart.

    Titles are used as candidate topics as-is (further normalized by
    ``normalize_topic``) - this provider does no semantic judgment of its
    own; that stays entirely in TopicRankingPlanner.
    """

    def __init__(
        self,
        api_key: Optional[str],
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: YouTube Data API v3 key (Settings.youtube_api_key) -
                a plain API key, never the OAuth upload credentials.
            timeout_seconds: Request timeout.
            client: Optional pre-configured httpx.AsyncClient (mainly for
                tests). When omitted, a short-lived client is created per
                request.

        Raises:
            TopicSourceProviderError: If no api_key is configured - caught
                lazily at discover_candidates() time, never at
                construction, matching every other provider's "fail on
                first real use, not on wiring" convention.
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def name(self) -> str:
        return "youtube"

    async def discover_candidates(
        self, category: Optional[str] = None, region: Optional[str] = None, limit: int = 15
    ) -> List[TopicCandidate]:
        if not self.api_key:
            raise TopicSourceProviderError(
                "YOUTUBE_API_KEY is not configured - required for real topic discovery"
            )

        params: Dict[str, Any] = {
            "part": "snippet,statistics",
            "chart": "mostPopular",
            "regionCode": region or DEFAULT_REGION_CODE,
            "maxResults": max(1, min(limit, 50)),
            "key": self.api_key,
        }
        if category:
            params["videoCategoryId"] = category

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await client.get(YOUTUBE_VIDEOS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as e:
            raise TopicSourceProviderError(f"YouTube topic discovery timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise TopicSourceProviderError(
                f"YouTube topic discovery returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise TopicSourceProviderError(f"YouTube topic discovery request failed: {e}") from e
        finally:
            if owns_client:
                await client.aclose()

        items = data.get("items")
        if not isinstance(items, list):
            raise TopicSourceProviderError("YouTube topic discovery returned a malformed response")

        candidates: List[TopicCandidate] = []
        for item in items:
            candidate = _candidate_from_item(item, category)
            if candidate is not None:
                candidates.append(candidate)
        return candidates[:limit]


def _candidate_from_item(item: Dict[str, Any], requested_category: Optional[str]) -> Optional[TopicCandidate]:
    snippet = item.get("snippet") or {}
    title = str(snippet.get("title") or "").strip()
    if not title:
        return None

    statistics = item.get("statistics") or {}
    popularity_signal = None
    view_count_raw = statistics.get("viewCount")
    if view_count_raw is not None:
        try:
            view_count = float(view_count_raw)
            popularity_signal = max(0.0, min(1.0, view_count / _POPULARITY_VIEW_COUNT_CEILING))
        except (TypeError, ValueError):
            popularity_signal = None

    return TopicCandidate(
        raw_title=title,
        normalized_title=normalize_topic(title),
        source="youtube",
        source_id=item.get("id"),
        category=requested_category or snippet.get("categoryId"),
        popularity_signal=popularity_signal,
    )
