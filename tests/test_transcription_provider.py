# Tests for the TranscriptionProvider abstraction and MockTranscriptionProvider.
from __future__ import annotations

import pytest

from src.tools.transcription_provider import (
    MockTranscriptionProvider,
    TranscribedSegment,
    TranscribedWord,
    TranscriptionProvider,
)


class TestMockTranscriptionProvider:
    def test_is_transcription_provider(self) -> None:
        assert isinstance(MockTranscriptionProvider(), TranscriptionProvider)

    def test_name(self) -> None:
        assert MockTranscriptionProvider().name == "mock"

    def test_default_segments_returned(self) -> None:
        provider = MockTranscriptionProvider()
        segments = provider.transcribe("narration.mp3")
        assert len(segments) >= 1
        assert all(isinstance(s, TranscribedSegment) for s in segments)

    def test_custom_segments_returned(self) -> None:
        custom = [TranscribedSegment(text="Custom line.", start_seconds=0.0, end_seconds=1.0)]
        provider = MockTranscriptionProvider(segments=custom)
        assert provider.transcribe("x.mp3") == custom

    def test_empty_segments_list_returned_as_is(self) -> None:
        provider = MockTranscriptionProvider(segments=[])
        assert provider.transcribe("x.mp3") == []

    def test_raises_configured_error(self) -> None:
        provider = MockTranscriptionProvider(raise_error=RuntimeError("transcription outage"))
        with pytest.raises(RuntimeError, match="transcription outage"):
            provider.transcribe("x.mp3")

    def test_calls_are_recorded(self) -> None:
        provider = MockTranscriptionProvider()
        provider.transcribe("narration.mp3")
        assert provider.calls == ["narration.mp3"]

    def test_word_timings_preserved(self) -> None:
        words = [TranscribedWord(word="Hi", start_seconds=0.0, end_seconds=0.4)]
        custom = [TranscribedSegment(text="Hi", start_seconds=0.0, end_seconds=0.4, words=words)]
        provider = MockTranscriptionProvider(segments=custom)
        result = provider.transcribe("x.mp3")
        assert result[0].words == words
