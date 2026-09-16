# Tests for the provider-agnostic AI Video Generation models
# (src.models.ai_video). Pure model validation - no provider/network.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.ai_video import AIVideoGenerationRequest, AIVideoGenerationResult


class TestAIVideoGenerationRequest:
    def test_minimal_valid_request(self) -> None:
        request = AIVideoGenerationRequest(
            prompt="A slow zoom over a frozen lake at dawn",
            duration_seconds=6.0,
            section_index=0,
            slot_index=0,
        )
        assert request.aspect_ratio == "16:9"
        assert request.negative_prompt is None
        assert request.seed is None

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AIVideoGenerationRequest(prompt="", duration_seconds=6.0, section_index=0, slot_index=0)

    def test_non_positive_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AIVideoGenerationRequest(prompt="x", duration_seconds=0, section_index=0, slot_index=0)
        with pytest.raises(ValidationError):
            AIVideoGenerationRequest(prompt="x", duration_seconds=-1.0, section_index=0, slot_index=0)

    def test_negative_section_or_slot_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AIVideoGenerationRequest(prompt="x", duration_seconds=6.0, section_index=-1, slot_index=0)
        with pytest.raises(ValidationError):
            AIVideoGenerationRequest(prompt="x", duration_seconds=6.0, section_index=0, slot_index=-1)

    def test_optional_fields_accepted(self) -> None:
        request = AIVideoGenerationRequest(
            prompt="A slow zoom over a frozen lake at dawn",
            negative_prompt="no text overlays, no watermarks",
            aspect_ratio="9:16",
            duration_seconds=8.0,
            resolution="1080p",
            seed=42,
            section_index=1,
            slot_index=2,
        )
        assert request.negative_prompt == "no text overlays, no watermarks"
        assert request.aspect_ratio == "9:16"
        assert request.resolution == "1080p"
        assert request.seed == 42


class TestAIVideoGenerationResult:
    def test_successful_result(self) -> None:
        result = AIVideoGenerationResult(
            success=True,
            status="succeeded",
            provider="mock",
            model="mock-model",
            local_file_path="/tmp/clip.mp4",
            duration_seconds=6.0,
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result_requires_no_file_path(self) -> None:
        result = AIVideoGenerationResult(success=False, status="failed", provider="mock", error="quota exceeded")
        assert result.success is False
        assert result.local_file_path is None
        assert result.error == "quota exceeded"

    def test_empty_provider_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AIVideoGenerationResult(success=True, status="succeeded", provider="")

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AIVideoGenerationResult(success=True, status="succeeded", provider="mock", cost_usd=-1.0)

    def test_warnings_default_to_empty_list(self) -> None:
        result = AIVideoGenerationResult(success=True, status="succeeded", provider="mock")
        assert result.warnings == []
