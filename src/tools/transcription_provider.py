# Speech transcription provider abstraction layer
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


class TranscriptionProviderError(Exception):
    """Raised when a TranscriptionProvider fails to transcribe audio."""


@dataclass
class TranscribedWord:
    """One word-level timestamp, if the provider supports them.

    Preserved for future use (e.g. word-by-word/karaoke captions) but not
    used for rendering in this milestone - CaptionService only consumes
    segment-level text/timing plus these words when splitting a long
    segment into readable chunks.
    """

    word: str
    start_seconds: float
    end_seconds: float


@dataclass
class TranscribedSegment:
    """One raw transcribed segment, before caption readability
    segmentation/formatting (see src/services/caption_segmentation.py)."""

    text: str
    start_seconds: float
    end_seconds: float
    words: List[TranscribedWord] = field(default_factory=list)


class TranscriptionProvider(ABC):
    """Abstract base class for speech-to-text transcription providers.

    Concrete implementations transcribe an audio file into timestamped
    segments. The application (CaptionService, and everything above it)
    only ever depends on this interface - never on a concrete transcription
    engine (Whisper, etc.) directly.

    Synchronous (like LLMProvider), not async: transcription is local
    CPU-bound work, not network I/O, so there is nothing to await here -
    CaptionService (an async orchestration method, matching every other
    service in this project) calls it as a plain blocking step.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'mock', 'whisper'."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscribedSegment]:
        """Transcribe the audio file at ``audio_path`` into timestamped segments.

        Args:
            audio_path: Local filesystem path to the narration audio file

        Returns:
            Transcribed segments in chronological order. May be empty if
            the audio contains no detectable speech.

        Raises:
            TranscriptionProviderError: If transcription fails outright
                (missing file, engine error, unreadable audio).
        """
        raise NotImplementedError


# A couple of short, distinct sentences - enough for CaptionService's
# segmentation/SRT/rendering logic to be exercised meaningfully in tests
# and demos without a real transcription engine.
_DEFAULT_MOCK_SEGMENTS: List[TranscribedSegment] = [
    TranscribedSegment(text="This is a mock narration segment.", start_seconds=0.0, end_seconds=2.5),
    TranscribedSegment(
        text="It stands in for real transcription during tests.", start_seconds=2.5, end_seconds=5.0
    ),
]


class MockTranscriptionProvider(TranscriptionProvider):
    """Mock transcription provider for development/testing without a real
    speech-to-text engine.

    Returns a configurable, deterministic list of segments (or raises a
    configured error), and records every audio path it was asked to
    transcribe.
    """

    def __init__(
        self,
        segments: Optional[List[TranscribedSegment]] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the mock provider.

        Args:
            segments: Fixed segments to return from every ``transcribe``
                call. Defaults to a small built-in fixture.
            raise_error: If set, ``transcribe`` always raises this instead
                of returning segments, to simulate a transcription failure.
        """
        self.segments = segments if segments is not None else list(_DEFAULT_MOCK_SEGMENTS)
        self.raise_error = raise_error
        self.calls: List[str] = []

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, audio_path: str) -> List[TranscribedSegment]:
        self.calls.append(audio_path)
        if self.raise_error is not None:
            raise self.raise_error
        return list(self.segments)
