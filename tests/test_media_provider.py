# Tests for MediaProvider abstraction and MockMediaProvider
from __future__ import annotations

import os

import pytest

from src.tools.media_provider import MediaCandidate, MediaProvider, MockMediaProvider


class TestMockMediaProvider:
    """Tests for MockMediaProvider functionality."""

    def test_is_media_provider(self) -> None:
        assert isinstance(MockMediaProvider(), MediaProvider)

    def test_name(self) -> None:
        assert MockMediaProvider().name == "mock"

    @pytest.mark.asyncio
    async def test_search_returns_candidates(self) -> None:
        provider = MockMediaProvider(results_per_query=3)
        candidates = await provider.search("ocean waves", prefer_video=True, max_results=5)

        assert len(candidates) == 3
        assert all(isinstance(c, MediaCandidate) for c in candidates)
        assert all(c.asset_type == "video" for c in candidates)

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self) -> None:
        provider = MockMediaProvider(results_per_query=5)
        candidates = await provider.search("ocean waves", max_results=2)
        assert len(candidates) == 2

    @pytest.mark.asyncio
    async def test_search_prefer_video_false_returns_images(self) -> None:
        provider = MockMediaProvider(results_per_query=1)
        candidates = await provider.search("mountains", prefer_video=False)
        assert candidates[0].asset_type == "image"
        assert candidates[0].duration_seconds is None

    @pytest.mark.asyncio
    async def test_search_empty_for_configured_query(self) -> None:
        provider = MockMediaProvider(empty_for={"no results here"})
        candidates = await provider.search("no results here")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_search_candidates_have_unique_ids_by_default(self) -> None:
        provider = MockMediaProvider(results_per_query=3)
        candidates = await provider.search("forest")
        ids = [c.provider_asset_id for c in candidates]
        assert len(set(ids)) == len(ids)
        assert all(ids)  # every candidate has a non-empty id

    @pytest.mark.asyncio
    async def test_search_candidates_have_unique_urls(self) -> None:
        provider = MockMediaProvider(results_per_query=3)
        candidates = await provider.search("forest")
        urls = [c.download_url for c in candidates]
        assert len(set(urls)) == len(urls)

    @pytest.mark.asyncio
    async def test_different_queries_produce_different_ids_by_default(self) -> None:
        provider = MockMediaProvider(results_per_query=1)
        a = await provider.search("forest")
        b = await provider.search("ocean")
        assert a[0].provider_asset_id != b[0].provider_asset_id

    @pytest.mark.asyncio
    async def test_pool_size_limits_distinct_ids_across_queries(self) -> None:
        """pool_size simulates a limited stock library: different queries
        can still surface the same small set of asset IDs, for testing
        reuse/fallback behavior deterministically."""
        provider = MockMediaProvider(results_per_query=1, pool_size=2)
        a = await provider.search("forest")
        b = await provider.search("ocean")
        c = await provider.search("mountains")
        ids = {a[0].provider_asset_id, b[0].provider_asset_id, c[0].provider_asset_id}
        assert ids <= {"0", "1"}

    @pytest.mark.asyncio
    async def test_content_hint_unset_by_default(self) -> None:
        provider = MockMediaProvider(results_per_query=3)
        candidates = await provider.search("forest")
        assert all(c.content_hint is None for c in candidates)

    @pytest.mark.asyncio
    async def test_content_hints_applied_by_index(self) -> None:
        provider = MockMediaProvider(results_per_query=3, content_hints=["a hint", None, "another hint"])
        candidates = await provider.search("forest")
        assert [c.content_hint for c in candidates] == ["a hint", None, "another hint"]

    @pytest.mark.asyncio
    async def test_content_hints_cycle_when_shorter_than_results(self) -> None:
        provider = MockMediaProvider(results_per_query=4, content_hints=["only hint"])
        candidates = await provider.search("forest")
        assert all(c.content_hint == "only hint" for c in candidates)

    @pytest.mark.asyncio
    async def test_download_writes_file(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=1)
        candidates = await provider.search("sunrise")
        output_path = str(tmp_path / "asset.mp4")

        await provider.download(candidates[0], output_path)

        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            content = f.read()
        assert candidates[0].download_url.encode() in content

    @pytest.mark.asyncio
    async def test_calls_are_recorded(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=1)
        candidates = await provider.search("rain")
        await provider.download(candidates[0], str(tmp_path / "a.mp4"))

        assert ("search", "rain") in provider.calls
        assert ("download", candidates[0].download_url) in provider.calls
