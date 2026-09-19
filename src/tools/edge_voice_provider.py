# Edge TTS voice provider (free, no API key required)
from __future__ import annotations

import asyncio
import os
from typing import Optional

from src.tools.voice_provider import VoiceProvider

DEFAULT_EDGE_VOICE = "en-US-AriaNeural"

# edge-tts reports offset/duration in 100-nanosecond "ticks" (the standard
# SSML/Azure Speech convention); divide by this to get seconds.
_TICKS_PER_SECOND = 10_000_000

# Hard OUTER ceiling wrapped around the entire streamed synthesis call
# (asyncio.wait_for) - edge-tts talks to Microsoft's TTS service over a
# websocket internally, with no configurable/documented timeout of its own
# exposed to callers. See src.tools.pexels_media_provider's identical
# defense-in-depth pattern (added after a real controlled autonomous run
# proved a client-level timeout alone did not reliably bound a network
# call in this pipeline). Narration synthesis is typically much faster
# than real-time even for a long (~8-10 minute) script, so this is sized
# generously above any realistic normal duration.
DEFAULT_OUTER_TIMEOUT_SECONDS = 120.0


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

    def __init__(self, outer_timeout_seconds: float = DEFAULT_OUTER_TIMEOUT_SECONDS) -> None:
        self.outer_timeout_seconds = outer_timeout_seconds

    async def synthesize(self, text: str, voice_name: str, output_path: str) -> Optional[float]:
        """Synthesize ``text`` via Edge TTS and write mp3 audio to ``output_path``.

        Raises:
            EdgeVoiceProviderError: If edge-tts is not installed, synthesis
                fails (e.g. no network access), or the outer ceiling is
                reached (category=timeout) - this never waits indefinitely.
        """
        try:
            import edge_tts
        except ImportError as e:
            raise EdgeVoiceProviderError(
                "edge-tts is not installed. Install it with `pip install edge-tts`."
            ) from e

        try:
            total_ticks = await asyncio.wait_for(
                self._stream_to_file(edge_tts, text, voice_name, output_path),
                timeout=self.outer_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            self._cleanup_partial_file(output_path)
            raise EdgeVoiceProviderError(
                f"Edge TTS synthesis timed out after {self.outer_timeout_seconds:.0f}s "
                "(provider=edge, operation=synthesize, category=timeout)"
            ) from e
        except Exception as e:
            self._cleanup_partial_file(output_path)
            raise EdgeVoiceProviderError(f"Edge TTS synthesis failed: {e}") from e

        return (total_ticks / _TICKS_PER_SECOND) if total_ticks else None

    @staticmethod
    async def _stream_to_file(edge_tts_module, text: str, voice_name: str, output_path: str) -> int:
        last_offset_ticks = 0
        last_duration_ticks = 0

        communicate = edge_tts_module.Communicate(text, voice_name)
        with open(output_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    last_offset_ticks = chunk.get("offset", last_offset_ticks)
                    last_duration_ticks = chunk.get("duration", last_duration_ticks)

        return last_offset_ticks + last_duration_ticks

    @staticmethod
    def _cleanup_partial_file(output_path: str) -> None:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
