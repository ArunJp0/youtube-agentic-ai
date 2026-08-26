# Voice (text-to-speech) provider abstraction layer
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VoiceProvider(ABC):
    """Abstract base class for text-to-speech providers.

    Concrete implementations must synthesize ``text`` as speech using
    ``voice_name`` and write the resulting audio to ``output_path``. The
    application (VoiceService, and everything above it) only ever depends
    on this interface - never on a concrete TTS engine (Edge TTS,
    ElevenLabs, etc.) directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'mock', 'edge'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_format(self) -> str:
        """Audio file format/extension this provider writes, e.g. 'mp3'."""
        raise NotImplementedError

    @abstractmethod
    async def synthesize(self, text: str, voice_name: str, output_path: str) -> Optional[float]:
        """Synthesize ``text`` to speech and write it to ``output_path``.

        Args:
            text: Narration text to synthesize
            voice_name: Provider-specific voice identifier
            output_path: Local filesystem path to write the audio file to

        Returns:
            Duration of the generated audio in seconds, if determinable by
            this provider; otherwise None.
        """
        raise NotImplementedError


class MockVoiceProvider(VoiceProvider):
    """Mock voice provider for development/testing without a real TTS engine.

    Writes a small placeholder text file (not real audio) so VoiceService
    tests can exercise file-writing behavior without any network or TTS
    dependency.
    """

    def __init__(self, fixed_duration_seconds: Optional[float] = 1.0) -> None:
        """Initialize with an optional fixed duration to report.

        Args:
            fixed_duration_seconds: Duration value returned by every
                ``synthesize`` call; set to None to simulate a provider that
                cannot determine duration.
        """
        self.fixed_duration_seconds = fixed_duration_seconds
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def output_format(self) -> str:
        return "mp3"

    async def synthesize(self, text: str, voice_name: str, output_path: str) -> Optional[float]:
        self.calls.append({"text": text, "voice_name": voice_name, "output_path": output_path})
        with open(output_path, "wb") as f:
            f.write(f"MOCK AUDIO (voice={voice_name})\n{text}".encode("utf-8"))
        return self.fixed_duration_seconds
