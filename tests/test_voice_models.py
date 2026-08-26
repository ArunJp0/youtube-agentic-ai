# Tests for the VoiceResult model
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.voice import VoiceResult


class TestVoiceResult:
    """Tests for the VoiceResult model."""

    def test_success_result_with_all_fields(self) -> None:
        result = VoiceResult(
            audio_file_path="output/audio/narration-abcd1234.mp3",
            provider="edge",
            voice_name="en-US-AriaNeural",
            duration_seconds=42.5,
            format="mp3",
            success=True,
            error=None,
        )
        assert result.audio_file_path == "output/audio/narration-abcd1234.mp3"
        assert result.provider == "edge"
        assert result.duration_seconds == 42.5
        assert result.success is True
        assert result.error is None

    def test_failure_result_with_minimal_fields(self) -> None:
        result = VoiceResult(
            provider="edge",
            voice_name="en-US-AriaNeural",
            format="mp3",
            success=False,
            error="Voice synthesis failed: network error",
        )
        assert result.audio_file_path is None
        assert result.duration_seconds is None
        assert result.success is False
        assert "network error" in result.error

    def test_provider_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VoiceResult(provider="", voice_name="v", format="mp3", success=True)

    def test_voice_name_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VoiceResult(provider="mock", voice_name="", format="mp3", success=True)

    def test_format_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VoiceResult(provider="mock", voice_name="v", format="", success=True)

    def test_duration_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            VoiceResult(
                provider="mock", voice_name="v", format="mp3", success=True, duration_seconds=-1.0
            )

    def test_serialization_roundtrip(self) -> None:
        result = VoiceResult(
            audio_file_path="output/audio/x.mp3",
            provider="mock",
            voice_name="v",
            duration_seconds=3.0,
            format="mp3",
            success=True,
        )
        restored = VoiceResult.model_validate_json(result.model_dump_json())
        assert restored == result
