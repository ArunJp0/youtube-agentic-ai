# Edge TTS voice provider (free, no API key required)
from __future__ import annotations

import os
from typing import Optional

from src.tools.voice_provider import VoiceProvider

DEFAULT_EDGE_VOICE = "en-US-AriaNeural"

# edge-tts reports offset/duration in 100-nanosecond "ticks" (the standard
# SSML/Azure Speech convention); divide by this to get seconds.
_TICKS_PER_SECOND = 10_000_000


class EdgeVoiceProviderError(Exception):
    """Raised when the Edge TTS voice provider fails to synthesize audio."""


class EdgeVoiceProvider(VoiceProvider):
    """Voice provider backed by Microsoft Edge's free neural TTS (via edge-tts).

    All edge-tts-specific usage (streaming, boundary-metadata parsing for
    duration) is contained here; VoiceService and the rest of the app only
    see the plain VoiceProvider interface.
    """

    @property
    def name(self) -> str:
        return "edge"

    @property
    def output_format(self) -> str:
        return "mp3"

    async def synthesize(self, text: str, voice_name: str, output_path: str) -> Optional[float]:
        """Synthesize ``text`` via Edge TTS and write mp3 audio to ``output_path``.

        Raises:
            EdgeVoiceProviderError: If edge-tts is not installed, or
                synthesis fails (e.g. no network access).
        """
        try:
            import edge_tts
        except ImportError as e:
            raise EdgeVoiceProviderError(
                "edge-tts is not installed. Install it with `pip install edge-tts`."
            ) from e

        last_offset_ticks = 0
        last_duration_ticks = 0

        try:
            communicate = edge_tts.Communicate(text, voice_name)
            with open(output_path, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                        last_offset_ticks = chunk.get("offset", last_offset_ticks)
                        last_duration_ticks = chunk.get("duration", last_duration_ticks)
        except Exception as e:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise EdgeVoiceProviderError(f"Edge TTS synthesis failed: {e}") from e

        total_ticks = last_offset_ticks + last_duration_ticks
        return (total_ticks / _TICKS_PER_SECOND) if total_ticks else None
