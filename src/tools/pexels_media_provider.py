# Pexels stock media provider (free tier, API key required)
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"

# Per-request httpx client timeout (connect/read/write/pool) - the normal,
# expected bound for a Pexels call.
DEFAULT_TIMEOUT_SECONDS = 15.0

# Hard OUTER ceiling wrapped around every actual network call
# (asyncio.wait_for), on top of - never instead of - the httpx client
# timeout above. A real controlled autonomous run proved a blocking
# network call elsewhere in this pipeline did not reliably terminate on
# its own client-level timeout; this is the same defense-in-depth applied
# here so a Pexels search/download can never again wait indefinitely
# regardless of why the inner timeout might fail to fire. Deliberately
# just a bit larger than the inner timeout so the inner one gets the first
# real chance to fire (the normal path); this is a last-resort net, not
# the primary bound.
DEFAULT_OUTER_TIMEOUT_SECONDS = 25.0


class PexelsMediaProvider(MediaProvider):
    """Media provider backed by the free Pexels API.

    Pexels was chosen for the MVP: its free tier (200 requests/hour, 20,000/
    month) needs only a simple API key (no OAuth flow), all content is
    royalty-free and watermark-free by license, and the API supports
    filtering by orientation server-side (used here to request landscape
    16:9-friendly results directly). All Pexels-specific request/response
    handling is contained here; VisualMediaService only sees the plain
    MediaProvider interface.
    """

    def __init__(
        self,
        api_key: Optional[str],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        outer_timeout_seconds: float = DEFAULT_OUTER_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key:
            raise MediaProviderError(
                "PEXELS_API_KEY is not set. Get a free key at https://www.pexels.com/api/ "
                "and set it in the environment or .env file to use the Pexels media provider."
            )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.outer_timeout_seconds = outer_timeout_seconds

    @property
    def name(self) -> str:
        return "pexels"

    async def search(
        self, query: str, prefer_video: bool = True, max_results: int = 5
    ) -> List[MediaCandidate]:
        """Search Pexels for candidates, videos first if ``prefer_video``.

        Raises:
            MediaProviderError: On network failure or an HTTP error.
        """
        candidates: List[MediaCandidate] = []
        if prefer_video:
            candidates = await self._search_videos(query, max_results)
        if not candidates:
            candidates = await self._search_photos(query, max_results)
        return candidates

    async def download(self, candidate: MediaCandidate, output_path: str) -> None:
        """Download ``candidate`` to ``output_path``.

        Raises:
            MediaProviderError: On network failure, an HTTP error, or a
                timeout (either the inner per-request client timeout, or
                the outer hard ceiling - see ``outer_timeout_seconds``;
                either way this never waits indefinitely).
        """
        try:
            response = await asyncio.wait_for(
                self._do_download(candidate.download_url), timeout=self.outer_timeout_seconds
            )
        except asyncio.TimeoutError as e:
            raise MediaProviderError(
                f"Pexels download timed out after {self.outer_timeout_seconds:.0f}s (provider=pexels, "
                "operation=download, category=timeout)"
            ) from e
        except httpx.HTTPStatusError as e:
            raise MediaProviderError(
                f"Pexels download returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise MediaProviderError(f"Pexels download failed: {e}") from e

        with open(output_path, "wb") as f:
            f.write(response.content)

    async def _do_download(self, download_url: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            return response

    # ---- internal --------------------------------------------------------

    async def _search_videos(self, query: str, max_results: int) -> List[MediaCandidate]:
        data = await self._get(
            PEXELS_VIDEO_SEARCH_URL,
            {"query": query, "orientation": "landscape", "per_page": max_results},
        )
        candidates = []
        for video in data.get("videos", []):
            video_file = _pick_best_video_file(video.get("video_files", []))
            if not video_file or not video_file.get("link"):
                continue
            candidates.append(
                MediaCandidate(
                    asset_type="video",
                    download_url=video_file["link"],
                    source_url=video.get("url", ""),
                    provider_asset_id=str(video["id"]) if video.get("id") is not None else None,
                    attribution=(video.get("user") or {}).get("name"),
                    width=video_file.get("width"),
                    height=video_file.get("height"),
                    duration_seconds=video.get("duration"),
                    content_hint=_slug_from_url(video.get("url", "")),
                )
            )
        return candidates

    async def _search_photos(self, query: str, max_results: int) -> List[MediaCandidate]:
        data = await self._get(
            PEXELS_PHOTO_SEARCH_URL,
            {"query": query, "orientation": "landscape", "per_page": max_results},
        )
        candidates = []
        for photo in data.get("photos", []):
            src = photo.get("src") or {}
            download_url = src.get("large2x") or src.get("original") or src.get("large")
            if not download_url:
                continue
            candidates.append(
                MediaCandidate(
                    asset_type="image",
                    download_url=download_url,
                    source_url=photo.get("url", ""),
                    provider_asset_id=str(photo["id"]) if photo.get("id") is not None else None,
                    attribution=photo.get("photographer"),
                    width=photo.get("width"),
                    height=photo.get("height"),
                    content_hint=photo.get("alt") or _slug_from_url(photo.get("url", "")),
                )
            )
        return candidates

    async def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(self._do_get(url, params), timeout=self.outer_timeout_seconds)
        except asyncio.TimeoutError as e:
            raise MediaProviderError(
                f"Pexels search timed out after {self.outer_timeout_seconds:.0f}s (provider=pexels, "
                "operation=search, category=timeout)"
            ) from e
        except httpx.HTTPStatusError as e:
            raise MediaProviderError(
                f"Pexels API returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise MediaProviderError(f"Pexels request failed: {e}") from e

    async def _do_get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": self.api_key}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()


def _slug_from_url(url: str) -> Optional[str]:
    """Pexels page URLs embed a human-readable content slug, e.g.
    ".../video/hikers-trekking-through-autumn-meadow-35759719/" - parse it
    into a plain phrase usable as lightweight content metadata (no extra
    API calls needed). Returns None if the URL has no usable slug."""
    if not url:
        return None
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    slug = path.rsplit("/", 1)[-1]
    slug = re.sub(r"-\d+$", "", slug)  # drop the trailing Pexels numeric ID
    words = [w for w in slug.split("-") if w and not w.isdigit()]
    return " ".join(words) if words else None


def _pick_best_video_file(video_files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer an mp4 file at 'hd' quality; fall back to the widest available."""
    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4"]
    pool = mp4_files or video_files
    if not pool:
        return None
    hd_files = [f for f in pool if f.get("quality") == "hd"]
    if hd_files:
        return hd_files[0]
    return max(pool, key=lambda f: f.get("width") or 0)
