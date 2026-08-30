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
            provider_asset_id="12345",
            attribution="Jane Doe",
            search_query="ocean waves",
            section_index=0,
            duration_seconds=12.5,
            width=1920,
            height=1080,
            reused=False,
            success=True,
        )
        assert asset.provider == "pexels"
        assert asset.asset_type == "video"
        assert asset.provider_asset_id == "12345"
        assert asset.reused is False
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
        assert asset.provider_asset_id is None
        assert asset.reused is False
        assert asset.success is False
        assert "No suitable" in asset.error

    def test_reused_asset_can_be_flagged(self) -> None:
        asset = MediaAsset(
            provider="pexels",
            local_file_path="output/media/section-01-abcd1234.mp4",
            provider_asset_id="12345",
            search_query="ocean waves",
            section_index=3,
            reused=True,
            success=True,
        )
        assert asset.reused is True

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

    def test_mapping_with_multiple_ordered_assets(self) -> None:
        mapping = SectionMediaMapping(
            section_index=0,
            section_heading="Intro",
            search_queries=["city skyline", "urban night"],
            planned_duration_seconds=16.0,
            assets=[
                MediaAsset(provider="mock", search_query="city skyline", section_index=0, success=True),
                MediaAsset(provider="mock", search_query="urban night", section_index=0, success=True),
            ],
        )
        assert len(mapping.assets) == 2
        assert mapping.search_queries == ["city skyline", "urban night"]
        assert mapping.planned_duration_seconds == 16.0

    def test_mapping_defaults(self) -> None:
        mapping = SectionMediaMapping(section_index=0, section_heading="Intro")
        assert mapping.assets == []
        assert mapping.search_queries == []
        assert mapping.planned_duration_seconds == 0.0

    def test_mapping_assets_default_factory_is_isolated(self) -> None:
        m1 = SectionMediaMapping(section_index=0, section_heading="A")
        m2 = SectionMediaMapping(section_index=1, section_heading="B")
        m1.assets.append(
            MediaAsset(provider="mock", search_query="a", section_index=0, success=True)
        )
        assert m2.assets == []

    def test_planned_duration_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            SectionMediaMapping(section_index=0, section_heading="A", planned_duration_seconds=-1.0)


class TestVisualResult:
    """Tests for the VisualResult model."""

    def test_result_with_sections(self) -> None:
        result = VisualResult(
            topic="Dreams",
            provider="mock",
            sections=[
                SectionMediaMapping(section_index=0, section_heading="Intro", search_queries=["q"])
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
                    search_queries=["q1", "q2"],
                    planned_duration_seconds=20.0,
                    assets=[
                        MediaAsset(
                            provider="mock",
                            provider_asset_id="1",
                            search_query="q1",
                            section_index=0,
                            success=True,
                        ),
                        MediaAsset(
                            provider="mock",
                            provider_asset_id="2",
                            search_query="q2",
                            section_index=0,
                            reused=True,
                            success=True,
                        ),
                    ],
                )
            ],
            success=True,
        )
        restored = VisualResult.model_validate_json(result.model_dump_json())
        assert restored == result
