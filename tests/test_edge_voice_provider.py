# Tests for EdgeVoiceProvider. edge_tts.Communicate is monkeypatched in every
# test in this module, so no real network/TTS call is ever made.
from __future__ import annotations

import os

import edge_tts
import pytest

from src.tools.edge_voice_provider import EdgeVoiceProvider, EdgeVoiceProviderError
from src.tools.voice_provider import VoiceProvider


class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate that streams canned chunks."""

    def __init__(self, chunks, raise_error: Exception | None = None) -> None:
        self._chunks = chunks
        self._raise_error = raise_error

    def __call__(self, text, voice, **kwargs):
        # edge_tts.Communicate(text, voice) is instantiated fresh per call;
        # returning self lets the test control what stream() yields.
        self.text = text
        self.voice = voice
        return self

    async def stream(self):
        if self._raise_error is not None:
            raise self._raise_error
        for chunk in self._chunks:
            yield chunk


class TestEdgeVoiceProviderBasics:
    def test_is_voice_provider(self) -> None:
        assert isinstance(EdgeVoiceProvider(), VoiceProvider)

    def test_name_and_format(self) -> None:
        provider = EdgeVoiceProvider()
        assert provider.name == "edge"
        assert provider.output_format == "mp3"


class TestEdgeVoiceProviderSynthesize:
    @pytest.mark.asyncio
    async def test_writes_audio_bytes_and_computes_duration(self, monkeypatch, tmp_path) -> None:
        fake = _FakeCommunicate(
            chunks=[
                {"type": "audio", "data": b"FAKEAUDIOBYTES1"},
                {"type": "audio", "data": b"FAKEAUDIOBYTES2"},
                {
                    "type": "SentenceBoundary",
                    "offset": 5_000_000,
                    "duration": 25_000_000,
                    "text": "Hello world.",
                },
            ]
        )
        monkeypatch.setattr(edge_tts, "Communicate", fake)

        provider = EdgeVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        duration = await provider.synthesize("Hello world.", "en-US-AriaNeural", output_path)

        assert duration == pytest.approx(3.0)  # (5_000_000 + 25_000_000) / 10_000_000
        with open(output_path, "rb") as f:
            content = f.read()
        assert content == b"FAKEAUDIOBYTES1FAKEAUDIOBYTES2"

    @pytest.mark.asyncio
    async def test_no_boundary_chunks_returns_none_duration(self, monkeypatch, tmp_path) -> None:
        fake = _FakeCommunicate(chunks=[{"type": "audio", "data": b"AUDIO"}])
        monkeypatch.setattr(edge_tts, "Communicate", fake)

        provider = EdgeVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        duration = await provider.synthesize("Text", "en-US-AriaNeural", output_path)

        assert duration is None
        assert os.path.exists(output_path)

    @pytest.mark.asyncio
    async def test_stream_failure_raises_provider_error_and_cleans_up(
        self, monkeypatch, tmp_path
    ) -> None:
        fake = _FakeCommunicate(chunks=[], raise_error=RuntimeError("network unreachable"))
        monkeypatch.setattr(edge_tts, "Communicate", fake)

        provider = EdgeVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        with pytest.raises(EdgeVoiceProviderError, match="network unreachable"):
            await provider.synthesize("Text", "en-US-AriaNeural", output_path)

        assert not os.path.exists(output_path)

    @pytest.mark.asyncio
    async def test_partial_write_then_failure_removes_partial_file(
        self, monkeypatch, tmp_path
    ) -> None:
        class _PartialThenFail(_FakeCommunicate):
            async def stream(self):
                yield {"type": "audio", "data": b"PARTIAL"}
                raise RuntimeError("connection dropped")

        fake = _PartialThenFail(chunks=[])
        monkeypatch.setattr(edge_tts, "Communicate", fake)

        provider = EdgeVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        with pytest.raises(EdgeVoiceProviderError):
            await provider.synthesize("Text", "en-US-AriaNeural", output_path)

        assert not os.path.exists(output_path)
