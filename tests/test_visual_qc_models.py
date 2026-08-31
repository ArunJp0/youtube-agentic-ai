# Tests for the Visual QC models (RawAssetVerdict, AssetQCResult,
# SectionQCResult, VisualQCResult).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.visual_qc import AssetQCResult, RawAssetVerdict, SectionQCResult, VisualQCResult


class TestRawAssetVerdict:
    def test_minimal_construction(self) -> None:
        verdict = RawAssetVerdict(asset_id="abc123", relevance_score=0.9)
        assert verdict.misleading_or_conflicting is False
        assert verdict.detected_visual_summary is None
        assert verdict.reason == ""

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawAssetVerdict(asset_id="abc", relevance_score=1.5)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawAssetVerdict(asset_id="abc", relevance_score=-0.1)

    def test_empty_asset_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawAssetVerdict(asset_id="", relevance_score=0.5)


class TestAssetQCResult:
    def test_construction(self) -> None:
        result = AssetQCResult(
            asset_id="abc",
            section_index=0,
            slot_index=1,
            approved=True,
            decision="approved",
            relevance_score=0.9,
            evaluation_source="vision",
        )
        assert result.replaced is False
        assert result.replacement_attempts == 0

    def test_section_index_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            AssetQCResult(
                asset_id="abc", section_index=-1, slot_index=0, approved=True,
                decision="approved", evaluation_source="vision",
            )

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AssetQCResult(
                asset_id="abc", section_index=0, slot_index=0, approved=True,
                decision="approved", relevance_score=2.0, evaluation_source="vision",
            )

    def test_score_can_be_none(self) -> None:
        result = AssetQCResult(
            asset_id="abc", section_index=0, slot_index=0, approved=True,
            decision="metadata_fallback", relevance_score=None, evaluation_source="metadata_fallback",
        )
        assert result.relevance_score is None


class TestSectionQCResult:
    def test_defaults(self) -> None:
        result = SectionQCResult(section_index=0)
        assert result.assets == []

    def test_assets_default_factory_is_isolated(self) -> None:
        a = SectionQCResult(section_index=0)
        b = SectionQCResult(section_index=1)
        a.assets.append(
            AssetQCResult(
                asset_id="x", section_index=0, slot_index=0, approved=True,
                decision="approved", evaluation_source="vision",
            )
        )
        assert b.assets == []


class TestVisualQCResult:
    def test_defaults(self) -> None:
        result = VisualQCResult(topic="Dreams", provider="mock", success=True)
        assert result.sections == []
        assert result.total_assets_checked == 0
        assert result.fallback_used is False
        assert result.repetition_warnings == []

    def test_topic_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VisualQCResult(topic="", provider="mock", success=True)

    def test_serialization_roundtrip(self) -> None:
        result = VisualQCResult(
            topic="Dreams",
            provider="gemini",
            model="gemini-3.6-flash",
            success=True,
            sections=[
                SectionQCResult(
                    section_index=0,
                    assets=[
                        AssetQCResult(
                            asset_id="abc",
                            section_index=0,
                            slot_index=0,
                            approved=True,
                            decision="approved",
                            relevance_score=0.9,
                            evaluation_source="vision",
                        )
                    ],
                )
            ],
            total_assets_checked=1,
            approved_count=1,
            vision_calls_made=1,
        )
        restored = VisualQCResult.model_validate_json(result.model_dump_json())
        assert restored == result
