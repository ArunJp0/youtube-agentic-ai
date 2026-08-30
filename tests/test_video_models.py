# Tests for the VideoAssemblyResult model
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.video import VideoAssemblyResult


class TestVideoAssemblyResult:
    """Tests for the VideoAssemblyResult model."""

    def test_success_result_with_all_fields(self) -> None:
        result = VideoAssemblyResult(
            success=True,
            output_path="output/video/why-do-humans-dream-abcd1234.mp4",
            duration_seconds=120.5,
            width=1920,
            height=1080,
            fps=30,
            format="mp4",
            video_codec="h264",
            audio_codec="aac",
            section_count=5,
            section_durations_seconds=[20.0, 25.5, 18.0, 30.0, 27.0],
        )
        assert result.success is True
        assert result.duration_seconds == 120.5
        assert result.width == 1920
        assert result.error is None

    def test_failure_result_with_minimal_fields(self) -> None:
        result = VideoAssemblyResult(success=False, error="Narration audio file not found")
        assert result.output_path is None
        assert result.section_durations_seconds == []
        assert "not found" in result.error

    def test_duration_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoAssemblyResult(success=True, duration_seconds=-1.0)

    def test_width_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoAssemblyResult(success=True, width=-1)

    def test_height_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoAssemblyResult(success=True, height=-1)

    def test_fps_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoAssemblyResult(success=True, fps=-1.0)

    def test_section_count_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoAssemblyResult(success=True, section_count=-1)

    def test_serialization_roundtrip(self) -> None:
        result = VideoAssemblyResult(
            success=True,
            output_path="output/video/x.mp4",
            duration_seconds=42.0,
            width=1920,
            height=1080,
            fps=30,
            format="mp4",
            section_durations_seconds=[21.0, 21.0],
        )
        restored = VideoAssemblyResult.model_validate_json(result.model_dump_json())
        assert restored == result
