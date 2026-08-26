# Tests for VoiceProvider abstraction and MockVoiceProvider
from __future__ import annotations

import os

import pytest

from src.tools.voice_provider import MockVoiceProvider, VoiceProvider


class TestMockVoiceProvider:
    """Tests for MockVoiceProvider functionality."""

    def test_is_voice_provider(self) -> None:
        assert isinstance(MockVoiceProvider(), VoiceProvider)

    def test_name_and_format(self) -> None:
        provider = MockVoiceProvider()
        assert provider.name == "mock"
        assert provider.output_format == "mp3"

    @pytest.mark.asyncio
    async def test_synthesize_writes_file(self, tmp_path) -> None:
        provider = MockVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        duration = await provider.synthesize("Hello world", "en-US-AriaNeural", output_path)

        assert os.path.exists(output_path)
        assert duration == 1.0
        with open(output_path, "rb") as f:
            content = f.read()
        assert b"Hello world" in content
        assert b"en-US-AriaNeural" in content

    @pytest.mark.asyncio
    async def test_synthesize_records_calls(self, tmp_path) -> None:
        provider = MockVoiceProvider()
        output_path = str(tmp_path / "narration.mp3")

        await provider.synthesize("Some text", "voice-x", output_path)

        assert len(provider.calls) == 1
        assert provider.calls[0]["text"] == "Some text"
        assert provider.calls[0]["voice_name"] == "voice-x"
        assert provider.calls[0]["output_path"] == output_path

    @pytest.mark.asyncio
    async def test_custom_fixed_duration(self, tmp_path) -> None:
        provider = MockVoiceProvider(fixed_duration_seconds=None)
        output_path = str(tmp_path / "narration.mp3")

        duration = await provider.synthesize("Text", "voice", output_path)

        assert duration is None
