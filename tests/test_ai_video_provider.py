# Tests for src.tools.ai_video_provider (MockAIVideoProvider,
# LocalAIVideoProvider). No real AI video API, no network - LocalAIVideoProvider
# only ever copies local files that the test itself creates.
from __future__ import annotations

import asyncio
import os

import pytest

from src.models.ai_video import AIVideoGenerationRequest
from src.tools.ai_video_provider import (
    AIVideoProviderError,
    LocalAIVideoProvider,
    MockAIVideoProvider,
)


def _request(section_index: int = 0, slot_index: int = 0) -> AIVideoGenerationRequest:
    return AIVideoGenerationRequest(
        prompt="A slow pan across a snowy mountain range at sunrise",
        duration_seconds=6.0,
        section_index=section_index,
        slot_index=slot_index,
    )


class TestMockAIVideoProvider:
    def test_name(self) -> None:
        assert MockAIVideoProvider().name == "mock"
        assert MockAIVideoProvider(provider_name="mock-2").name == "mock-2"

    def test_successful_generation(self) -> None:
        provider = MockAIVideoProvider()
        result = asyncio.run(provider.generate(_request()))
        assert result.success is True
        assert result.status == "succeeded"
        assert result.local_file_path is not None
        assert result.duration_seconds == 6.0

    def test_records_calls(self) -> None:
        provider = MockAIVideoProvider()
        request = _request(section_index=2, slot_index=1)
        asyncio.run(provider.generate(request))
        assert provider.calls == [request]

    def test_always_fails_when_configured(self) -> None:
        provider = MockAIVideoProvider(fail=True)
        result = asyncio.run(provider.generate(_request()))
        assert result.success is False
        assert result.status == "failed"
        assert result.error is not None

    def test_fails_a_bounded_number_of_times_then_succeeds(self) -> None:
        provider = MockAIVideoProvider(fail_times=2)
        first = asyncio.run(provider.generate(_request()))
        second = asyncio.run(provider.generate(_request()))
        third = asyncio.run(provider.generate(_request()))
        assert first.success is False
        assert second.success is False
        assert third.success is True

    def test_cost_reported_as_zero_for_the_free_mock(self) -> None:
        provider = MockAIVideoProvider()
        result = asyncio.run(provider.generate(_request()))
        assert result.cost_usd == 0.0


class TestLocalAIVideoProvider:
    def test_requires_clips_dir(self) -> None:
        with pytest.raises(AIVideoProviderError):
            LocalAIVideoProvider(clips_dir="")

    def test_name(self, tmp_path) -> None:
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))
        assert provider.name == "local_ai_video"

    def test_finds_and_serves_a_matching_clip(self, tmp_path) -> None:
        clip_path = tmp_path / "0-0.mp4"
        clip_path.write_bytes(b"fake mp4 bytes")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        result = asyncio.run(provider.generate(_request(section_index=0, slot_index=0)))

        assert result.success is True
        assert result.local_file_path == str(clip_path)

    def test_never_claims_to_be_a_real_model(self, tmp_path) -> None:
        clip_path = tmp_path / "0-0.mp4"
        clip_path.write_bytes(b"fake mp4 bytes")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        result = asyncio.run(provider.generate(_request(section_index=0, slot_index=0)))

        assert result.model == LocalAIVideoProvider.DEMO_MODEL_LABEL
        assert any("manually generated" in w.lower() for w in result.warnings)

    def test_missing_clip_when_directory_has_no_clips_at_all_fails_cleanly(self, tmp_path) -> None:
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))
        result = asyncio.run(provider.generate(_request(section_index=5, slot_index=9)))
        assert result.success is False
        assert result.status == "failed"

    def test_missing_exact_slot_reuses_a_clip_from_the_same_section(self, tmp_path) -> None:
        (tmp_path / "0-0.mp4").write_bytes(b"clip a")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        result = asyncio.run(provider.generate(_request(section_index=0, slot_index=3)))

        assert result.success is True
        assert result.local_file_path == str(tmp_path / "0-0.mp4")
        assert any("reused another clip from the same section" in w for w in result.warnings)

    def test_missing_section_entirely_reuses_any_available_clip(self, tmp_path) -> None:
        (tmp_path / "1-0.mp4").write_bytes(b"clip b")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        result = asyncio.run(provider.generate(_request(section_index=4, slot_index=0)))

        assert result.success is True
        assert result.local_file_path == str(tmp_path / "1-0.mp4")
        assert any("reused a clip generated for a different section" in w for w in result.warnings)

    def test_limited_clip_set_covers_more_sections_than_generated(self, tmp_path) -> None:
        """The exact scenario this exists for: 2 real demo clips covering
        6 script sections - every request still succeeds, never Pexels,
        never a hard failure."""
        (tmp_path / "0-0.mp4").write_bytes(b"clip a")
        (tmp_path / "2-0.mp4").write_bytes(b"clip b")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        for section_index in range(6):
            result = asyncio.run(provider.generate(_request(section_index=section_index, slot_index=0)))
            assert result.success is True

    def test_nonexistent_clips_dir_fails_cleanly_not_a_crash(self) -> None:
        provider = LocalAIVideoProvider(clips_dir="/definitely/does/not/exist")
        result = asyncio.run(provider.generate(_request()))
        assert result.success is False

    def test_copy_into_output_copies_the_file(self, tmp_path) -> None:
        source = tmp_path / "source.mp4"
        source.write_bytes(b"clip bytes")
        destination = tmp_path / "nested" / "dest.mp4"

        LocalAIVideoProvider.copy_into_output(str(source), str(destination))

        assert destination.read_bytes() == b"clip bytes"

    def test_different_slots_map_to_different_clips(self, tmp_path) -> None:
        (tmp_path / "0-0.mp4").write_bytes(b"clip a")
        (tmp_path / "0-1.mp4").write_bytes(b"clip b")
        provider = LocalAIVideoProvider(clips_dir=str(tmp_path))

        result_a = asyncio.run(provider.generate(_request(section_index=0, slot_index=0)))
        result_b = asyncio.run(provider.generate(_request(section_index=0, slot_index=1)))

        assert result_a.local_file_path != result_b.local_file_path
