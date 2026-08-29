# Tests for the media models (MediaAsset, SectionMediaMapping, VisualResult)
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult


class TestMediaAsset:
    """Tests for the MediaAsset model."""

    def test_success_asset_with_all_fields(self) -> None:
        asset = MediaAsset(
            provider="pexels",
            asset_type="video",
            local_file_path="output/media/section-01-abcd1234.mp4",
            source_url="https://www.pexels.com/video/12345",
            attribution="Jane Doe",
            search_query="ocean waves",
            section_index=0,
            duration_seconds=12.5,
            width=1920,
            height=1080,
            success=True,
        )
        assert asset.provider == "pexels"
        assert asset.asset_type == "video"
        assert asset.success is True
        assert asset.error is None

    def test_failure_asset_with_minimal_fields(self) -> None:
        asset = MediaAsset(
            provider="pexels",
            search_query="ocean waves",
            section_index=2,
            success=False,
            error="No suitable (non-duplicate) media asset found",
        )
        assert asset.asset_type is None
        assert asset.local_file_path is None
        assert asset.success is False
        assert "No suitable" in asset.error

    def test_provider_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            MediaAsset(provider="", search_query="q", section_index=0, success=True)

    def test_search_query_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            MediaAsset(provider="mock", search_query="", section_index=0, success=True)

    def test_section_index_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            MediaAsset(provider="mock", search_query="q", section_index=-1, success=True)

    def test_duration_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            MediaAsset(
                provider="mock",
                search_query="q",
                section_index=0,
                success=True,
                duration_seconds=-1.0,
            )


class TestSectionMediaMapping:
    """Tests for the SectionMediaMapping model."""

    def test_mapping_with_asset(self) -> None:
        mapping = SectionMediaMapping(
            section_index=0,
            section_heading="Intro",
            search_query="city skyline",
            assets=[
                MediaAsset(provider="mock", search_query="city skyline", section_index=0, success=True)
            ],
        )
        assert len(mapping.assets) == 1

    def test_mapping_defaults_to_no_assets(self) -> None:
        mapping = SectionMediaMapping(section_index=0, section_heading="Intro", search_query="q")
        assert mapping.assets == []

    def test_mapping_assets_default_factory_is_isolated(self) -> None:
        m1 = SectionMediaMapping(section_index=0, section_heading="A", search_query="a")
        m2 = SectionMediaMapping(section_index=1, section_heading="B", search_query="b")
        m1.assets.append(
            MediaAsset(provider="mock", search_query="a", section_index=0, success=True)
        )
        assert m2.assets == []


class TestVisualResult:
    """Tests for the VisualResult model."""

    def test_result_with_sections(self) -> None:
        result = VisualResult(
            topic="Dreams",
            provider="mock",
            sections=[
                SectionMediaMapping(section_index=0, section_heading="Intro", search_query="q")
            ],
            success=True,
        )
        assert len(result.sections) == 1
        assert result.error is None

    def test_result_defaults(self) -> None:
        result = VisualResult(topic="Dreams", provider="mock", success=False)
        assert result.sections == []

    def test_topic_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VisualResult(topic="", provider="mock", success=True)

    def test_provider_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VisualResult(topic="Dreams", provider="", success=True)

    def test_serialization_roundtrip(self) -> None:
        result = VisualResult(
            topic="Dreams",
            provider="mock",
            sections=[
                SectionMediaMapping(
                    section_index=0,
                    section_heading="Intro",
                    search_query="q",
                    assets=[
                        MediaAsset(
                            provider="mock", search_query="q", section_index=0, success=True
                        )
                    ],
                )
            ],
            success=True,
        )
        restored = VisualResult.model_validate_json(result.model_dump_json())
        assert restored == result
