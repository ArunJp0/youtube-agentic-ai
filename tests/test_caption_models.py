# Tests for the caption models (CaptionSegment, CaptionResult).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.captions import CaptionResult, CaptionSegment


class TestCaptionSegment:
    def test_valid_construction(self) -> None:
        segment = CaptionSegment(index=1, start_seconds=0.0, end_seconds=2.5, text="Hello world.")
        assert segment.index == 1
        assert segment.start_seconds == 0.0
        assert segment.end_seconds == 2.5
        assert segment.text == "Hello world."

    def test_index_cannot_be_zero_or_negative(self) -> None:
        with pytest.raises(ValidationError):
            CaptionSegment(index=0, start_seconds=0.0, end_seconds=1.0, text="x")

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptionSegment(index=1, start_seconds=-1.0, end_seconds=1.0, text="x")

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptionSegment(index=1, start_seconds=0.0, end_seconds=1.0, text="")

    def test_end_equal_to_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptionSegment(index=1, start_seconds=1.0, end_seconds=1.0, text="x")

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptionSegment(index=1, start_seconds=2.0, end_seconds=1.0, text="x")

    def test_multiline_text_allowed(self) -> None:
        segment = CaptionSegment(index=1, start_seconds=0.0, end_seconds=2.0, text="Line one\nLine two")
        assert "\n" in segment.text


class TestCaptionResult:
    def test_defaults(self) -> None:
        result = CaptionResult(success=False)
        assert result.segments == []
        assert result.srt_path is None
        assert result.captioned_video_path is None
        assert result.error is None

    def test_success_construction(self) -> None:
        result = CaptionResult(
            success=True,
            segments=[CaptionSegment(index=1, start_seconds=0.0, end_seconds=1.0, text="Hi")],
            srt_path="output/subtitles/video.srt",
            captioned_video_path="output/video/video-captioned.mp4",
            transcription_provider="whisper",
            transcription_model="base",
        )
        assert len(result.segments) == 1
        assert result.transcription_provider == "whisper"

    def test_failure_construction(self) -> None:
        result = CaptionResult(success=False, error="Narration audio file not found: x.mp3")
        assert result.success is False
        assert "not found" in result.error

    def test_serialization_roundtrip(self) -> None:
        result = CaptionResult(
            success=True,
            segments=[
                CaptionSegment(index=1, start_seconds=0.0, end_seconds=1.5, text="First"),
                CaptionSegment(index=2, start_seconds=1.5, end_seconds=3.0, text="Second"),
            ],
            srt_path="a.srt",
            captioned_video_path="b.mp4",
            narration_duration_seconds=3.0,
        )
        restored = CaptionResult.model_validate_json(result.model_dump_json())
        assert restored == result
