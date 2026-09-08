# Tests for Thumbnail Agent data models (ThumbnailPlan, ThumbnailSourceAsset,
# ThumbnailResult).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.thumbnail import ThumbnailPlan, ThumbnailResult, ThumbnailSourceAsset


class TestThumbnailPlanValidation:
    def test_minimal_valid_plan(self) -> None:
        plan = ThumbnailPlan(hook_text="WHY DO WE DREAM", search_query="person sleeping", used_semantic_planning=True)
        assert plan.composition == "centered"
        assert plan.text_position == "center"
        assert plan.avoid_concepts == []

    def test_empty_hook_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailPlan(hook_text="", search_query="q", used_semantic_planning=True)

    def test_empty_search_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailPlan(hook_text="HOOK", search_query="", used_semantic_planning=True)

    def test_invalid_composition_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailPlan(hook_text="HOOK", search_query="q", composition="bottom_left", used_semantic_planning=True)

    def test_invalid_text_position_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailPlan(hook_text="HOOK", search_query="q", text_position="top", used_semantic_planning=True)

    def test_fallback_plan_shape(self) -> None:
        plan = ThumbnailPlan(
            hook_text="HOOK", search_query="q", used_semantic_planning=False, fallback_reason="LLM outage"
        )
        assert plan.used_semantic_planning is False
        assert plan.fallback_reason == "LLM outage"


class TestThumbnailSourceAssetValidation:
    def test_minimal_valid_asset(self) -> None:
        asset = ThumbnailSourceAsset(provider="pexels")
        assert asset.provider_asset_id is None

    def test_empty_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailSourceAsset(provider="")


class TestThumbnailResultValidation:
    def test_minimal_failure_result(self) -> None:
        result = ThumbnailResult(success=False, error="no image found")
        assert result.output_path is None
        assert result.plan is None
        assert result.warnings == []

    def test_success_result_carries_plan_and_asset(self) -> None:
        plan = ThumbnailPlan(hook_text="HOOK", search_query="q", used_semantic_planning=True)
        asset = ThumbnailSourceAsset(provider="pexels", provider_asset_id="123")
        result = ThumbnailResult(
            success=True,
            topic="dreams",
            output_path="output/thumbnails/dreams.jpg",
            width=1280,
            height=720,
            plan=plan,
            selected_asset=asset,
            llm_provider="gemini",
            llm_model="gemini-3.5-flash-lite",
            used_fallback_model=False,
        )
        assert result.plan.hook_text == "HOOK"
        assert result.selected_asset.provider_asset_id == "123"

    def test_negative_dimensions_raise(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailResult(success=True, width=-1)
