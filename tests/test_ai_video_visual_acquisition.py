# Tests for src.services.ai_video_visual_acquisition - the AI-video ->
# bounded-retry -> Pexels-fallback routing seam. No real AI/Pexels network
# calls - the "pexels_fallback" is always a local async stub.
from __future__ import annotations

import asyncio

from src.models.ai_video import AIVideoGenerationRequest
from src.models.media import MediaAsset
from src.services.ai_video_visual_acquisition import (
    acquire_visual_asset_with_ai_fallback,
    build_ai_video_request,
)
from src.tools.ai_video_provider import MockAIVideoProvider


def _request() -> AIVideoGenerationRequest:
    return build_ai_video_request(
        prompt="A close-up of ice crystals forming on a window",
        section_index=0,
        slot_index=0,
        duration_seconds=6.0,
    )


def _pexels_stub_asset() -> MediaAsset:
    return MediaAsset(
        provider="mock",
        asset_type="video",
        local_file_path="/mock/pexels/clip.mp4",
        search_query="ice crystals",
        section_index=0,
        success=True,
    )


async def _pexels_fallback():
    return _pexels_stub_asset(), "ice crystals"


class TestAcquireVisualAssetWithAiFallback:
    def test_no_provider_configured_goes_straight_to_pexels(self) -> None:
        asset, query = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(),
                ai_video_provider=None,
                pexels_fallback=_pexels_fallback,
                max_retries=2,
                stock_fallback_enabled=True,
            )
        )
        assert asset.provider == "mock"
        assert asset.local_file_path == "/mock/pexels/clip.mp4"
        assert query == "ice crystals"

    def test_successful_ai_generation_is_used_directly(self) -> None:
        provider = MockAIVideoProvider()
        asset, prompt = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(),
                ai_video_provider=provider,
                pexels_fallback=_pexels_fallback,
                max_retries=2,
                stock_fallback_enabled=True,
            )
        )
        assert asset.success is True
        assert asset.provider == "mock"
        assert asset.relevance_tier == "ai_generated"
        assert asset.asset_type == "video"
        assert prompt == _request().prompt
        assert len(provider.calls) == 1  # succeeded on the first attempt - no retry needed

    def test_ai_result_reuses_the_same_media_asset_model(self) -> None:
        """Downstream stages (Visual QC, Video Assembly) never need to know
        an asset came from AI generation vs Pexels - same typed model."""
        provider = MockAIVideoProvider()
        asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=0, stock_fallback_enabled=True,
            )
        )
        assert isinstance(asset, MediaAsset)

    def test_ai_failure_retries_up_to_the_bound_then_falls_back_to_pexels(self) -> None:
        provider = MockAIVideoProvider(fail_times=2)  # fails attempts 1-2, succeeds on 3
        asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=2, stock_fallback_enabled=True,
            )
        )
        assert len(provider.calls) == 3  # 1 initial + 2 retries
        assert asset.provider == "mock"
        assert asset.relevance_tier == "ai_generated"  # the 3rd attempt succeeded - AI clip used, not Pexels

    def test_ai_exhausted_retries_falls_back_to_pexels_when_enabled(self) -> None:
        provider = MockAIVideoProvider(fail=True)
        asset, query = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=1, stock_fallback_enabled=True,
            )
        )
        assert len(provider.calls) == 2  # 1 initial + 1 retry, bounded
        assert asset.local_file_path == "/mock/pexels/clip.mp4"
        assert query == "ice crystals"

    def test_ai_exhausted_retries_fails_cleanly_when_fallback_disabled(self) -> None:
        provider = MockAIVideoProvider(fail=True)
        asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=1, stock_fallback_enabled=False,
            )
        )
        assert asset.success is False
        assert "2 attempt" in asset.error
        assert asset.provider == "mock"

    def test_zero_retries_means_exactly_one_attempt(self) -> None:
        provider = MockAIVideoProvider(fail=True)
        asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=0, stock_fallback_enabled=True,
            )
        )
        assert len(provider.calls) == 1

    def test_negative_max_retries_still_makes_exactly_one_attempt(self) -> None:
        provider = MockAIVideoProvider(fail=True)
        asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=-5, stock_fallback_enabled=True,
            )
        )
        assert len(provider.calls) == 1


class TestExistingVisualMediaAndVisualQcCompatibility:
    """An AI-generated MediaAsset must slot into the EXACT same
    SectionMediaMapping/VisualResult contract VisualMediaService/
    VisualQCService already consume - no new/parallel result type, no
    schema change, no VisualMediaService/VisualQCService modification
    required to accept it structurally."""

    def test_ai_generated_asset_fits_into_section_media_mapping(self) -> None:
        from src.models.media import SectionMediaMapping

        provider = MockAIVideoProvider()
        asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=0, stock_fallback_enabled=True,
            )
        )

        mapping = SectionMediaMapping(
            section_index=0,
            section_heading="Ice crystal formation",
            search_queries=[asset.search_query],
            assets=[asset],
        )
        assert mapping.assets[0] is asset

    def test_ai_generated_asset_fits_into_visual_result(self) -> None:
        from src.models.media import SectionMediaMapping, VisualResult

        provider = MockAIVideoProvider()
        asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=0, stock_fallback_enabled=True,
            )
        )
        mapping = SectionMediaMapping(
            section_index=0, section_heading="Ice crystal formation", assets=[asset]
        )

        result = VisualResult(topic="Why does ice float?", provider=asset.provider, sections=[mapping], success=True)

        assert result.sections[0].assets[0].relevance_tier == "ai_generated"
        assert result.success is True

    def test_mixed_ai_and_pexels_assets_coexist_in_the_same_result(self) -> None:
        """A future integration could reasonably mix AI-generated and
        Pexels assets across slots within the same video - both are the
        same MediaAsset type, so nothing downstream needs to special-case
        either source."""
        from src.models.media import SectionMediaMapping, VisualResult

        provider = MockAIVideoProvider()
        ai_asset, _ = asyncio.run(
            acquire_visual_asset_with_ai_fallback(
                _request(), ai_video_provider=provider, pexels_fallback=_pexels_fallback,
                max_retries=0, stock_fallback_enabled=True,
            )
        )
        pexels_asset = _pexels_stub_asset()

        mapping = SectionMediaMapping(section_index=0, section_heading="Ice", assets=[ai_asset, pexels_asset])
        result = VisualResult(topic="Why does ice float?", provider="mixed", sections=[mapping], success=True)

        assert len(result.sections[0].assets) == 2
        assert {a.provider for a in result.sections[0].assets} == {"mock"}
        assert {a.relevance_tier for a in result.sections[0].assets if a.relevance_tier} == {"ai_generated"}


class TestBuildAiVideoRequest:
    def test_builds_a_valid_request(self) -> None:
        request = build_ai_video_request(
            prompt="A time-lapse of clouds over a mountain",
            section_index=1,
            slot_index=2,
            duration_seconds=8.0,
        )
        assert request.prompt == "A time-lapse of clouds over a mountain"
        assert request.section_index == 1
        assert request.slot_index == 2
        assert request.duration_seconds == 8.0
        assert request.aspect_ratio == "16:9"
