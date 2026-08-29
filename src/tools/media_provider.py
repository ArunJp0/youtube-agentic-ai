# Visual media (stock image/video) provider abstraction layer
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


class MediaProviderError(Exception):
    """Raised when a MediaProvider fails to search for or download an asset."""


@dataclass
class MediaCandidate:
    """A single raw search result from a MediaProvider, before download.

    Kept separate from the ``MediaAsset`` pydantic model: a candidate is an
    internal, provider-facing detail (what to download and from where);
    ``MediaAsset`` is the public, post-download result record.
    """

    asset_type: str  # "image" or "video"
    download_url: str
    source_url: str
    attribution: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None


class MediaProvider(ABC):
    """Abstract base class for stock image/video providers.

    Concrete implementations search for candidate assets matching a query,
    then download a chosen candidate to a local path. The application
    (VisualMediaService, and everything above it) only ever depends on this
    interface - never on a concrete stock-media API (Pexels, Pixabay,
    Unsplash, etc.) directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'mock', 'pexels'."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, query: str, prefer_video: bool = True, max_results: int = 5
    ) -> List[MediaCandidate]:
        """Search for candidate assets matching ``query``.

        Args:
            query: Search query text
            prefer_video: Search video clips first if the provider supports both
            max_results: Maximum number of candidates to return

        Returns:
            List of MediaCandidate, best match first. Empty if nothing found.
        """
        raise NotImplementedError

    @abstractmethod
    async def download(self, candidate: MediaCandidate, output_path: str) -> None:
        """Download ``candidate`` to ``output_path``.

        Args:
            candidate: A MediaCandidate previously returned by ``search``
            output_path: Local filesystem path to write the asset to
        """
        raise NotImplementedError


class MockMediaProvider(MediaProvider):
    """Mock media provider for development/testing without a real stock API.

    Returns deterministic fake candidates and writes a small placeholder
    file (not real media) on "download", so VisualMediaService tests can
    exercise search/selection/download-recording behavior without any
    network dependency.
    """

    def __init__(
        self,
        results_per_query: int = 3,
        empty_for: Optional[set] = None,
    ) -> None:
        """Initialize the mock provider.

        Args:
            results_per_query: Number of fake candidates to return per search
            empty_for: Set of query strings that should return no results,
                for testing empty/no-result fallback behavior
        """
        self.results_per_query = results_per_query
        self.empty_for = empty_for or set()
        self.calls: List[Tuple[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def search(
        self, query: str, prefer_video: bool = True, max_results: int = 5
    ) -> List[MediaCandidate]:
        self.calls.append(("search", query))
        if query in self.empty_for:
            return []

        asset_type = "video" if prefer_video else "image"
        count = min(self.results_per_query, max_results)
        slug = "-".join(query.lower().split())
        candidates = []
        for i in range(count):
            candidates.append(
                MediaCandidate(
                    asset_type=asset_type,
                    download_url=f"https://mock.media/files/{slug}-{i}.{'mp4' if asset_type == 'video' else 'jpg'}",
                    source_url=f"https://mock.media/page/{slug}-{i}",
                    attribution="Mock Contributor",
                    width=1920,
                    height=1080,
                    duration_seconds=8.0 if asset_type == "video" else None,
                )
            )
        return candidates

    async def download(self, candidate: MediaCandidate, output_path: str) -> None:
        self.calls.append(("download", candidate.download_url))
        with open(output_path, "wb") as f:
            f.write(f"MOCK MEDIA for {candidate.download_url}".encode("utf-8"))
