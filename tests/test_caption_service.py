# Tests for CaptionService (transcription -> segmentation -> SRT -> burned
# captions orchestration). Uses fake assembler/transcription provider only
# - no real network, Whisper model, or FFmpeg process.
from __future__ import annotations

import os

import pytest

from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.services.caption_service import CaptionService, CaptionServiceError
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.transcription_provider import (
    MockTranscriptionProvider,
    TranscribedSegment,
    TranscriptionProviderError,
)


class FakeAssembler(VideoAssembler):
    """Test double: records calls and writes tiny placeholder files instead
    of running real FFmpeg."""

    def __init__(self, duration: float = 10.0, fail_burn: bool = False, fail_captioned_probe: bool = False) -> None:
        self.duration = duration
        self.fail_burn = fail_burn
        self.fail_captioned_probe = fail_captioned_probe
        self.burn_calls: list[dict] = []

    def probe_duration_seconds(self, media_path: str) -> float:
        if self.fail_captioned_probe and "captioned" in media_path:
            raise VideoAssemblerError("simulated probe failure")
        return self.duration

    def build_section_clip(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Caption Service tests")

    def concatenate_and_mux_audio(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Caption Service tests")

    def extract_frames(self, *args, **kwargs):
        raise NotImplementedError("not exercised by Caption Service tests")

    def burn_subtitles(self, input_video_path, srt_path, output_path, force_style=None) -> None:
        self.burn_calls.append(
            {"input_video_path": input_video_path, "srt_path": srt_path, "output_path": output_path, "force_style": force_style}
        )
        if self.fail_burn:
            raise VideoAssemblerError("simulated ffmpeg subtitle burn failure")
        with open(output_path, "wb") as f:
            f.write(b"FAKE CAPTIONED VIDEO")


def _voice_result(tmp_path, success=True, missing=False) -> VoiceResult:
    audio_path = str(tmp_path / "narration.mp3")
    if not missing:
        with open(audio_path, "wb") as f:
            f.write(b"FAKE AUDIO")
    return VoiceResult(
        audio_file_path=audio_path if not missing else str(tmp_path / "gone.mp3"),
        provider="mock", voice_name="v", format="mp3", success=success, duration_seconds=5.0,
    )


def _video_result(tmp_path, success=True, missing=False) -> VideoAssemblyResult:
    video_path = str(tmp_path / "video.mp4")
    if not missing:
        with open(video_path, "wb") as f:
            f.write(b"ORIGINAL VIDEO BYTES")
    return VideoAssemblyResult(
        success=success, output_path=video_path if not missing else str(tmp_path / "gone.mp4"),
        duration_seconds=5.0, format="mp4",
    )


class TestCaptionServiceValidation:
    @pytest.mark.asyncio
    async def test_missing_required_inputs_raises(self, tmp_path) -> None:
        service = CaptionService(transcription_provider=MockTranscriptionProvider(), assembler=FakeAssembler())
        with pytest.raises(CaptionServiceError):
            await service.generate_captions(None, _video_result(tmp_path))

    @pytest.mark.asyncio
    async def test_unsuccessful_voice_result_returns_clean_failure(self, tmp_path) -> None:
        service = CaptionService(transcription_provider=MockTranscriptionProvider(), assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path, success=False), _video_result(tmp_path))
        assert result.success is False
        assert "VoiceResult" in result.error

    @pytest.mark.asyncio
    async def test_missing_audio_file_on_disk_returns_clean_failure(self, tmp_path) -> None:
        service = CaptionService(transcription_provider=MockTranscriptionProvider(), assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path, missing=True), _video_result(tmp_path))
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unsuccessful_video_result_returns_clean_failure(self, tmp_path) -> None:
        service = CaptionService(transcription_provider=MockTranscriptionProvider(), assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path, success=False))
        assert result.success is False
        assert "VideoAssemblyResult" in result.error

    @pytest.mark.asyncio
    async def test_missing_video_file_on_disk_returns_clean_failure(self, tmp_path) -> None:
        service = CaptionService(transcription_provider=MockTranscriptionProvider(), assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path, missing=True))
        assert result.success is False
        assert "not found" in result.error.lower()


class TestCaptionServiceFailures:
    @pytest.mark.asyncio
    async def test_transcription_provider_failure_returns_clean_result(self, tmp_path) -> None:
        provider = MockTranscriptionProvider(raise_error=TranscriptionProviderError("engine crashed"))
        service = CaptionService(transcription_provider=provider, assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))
        assert result.success is False
        assert "Transcription failed" in result.error

    @pytest.mark.asyncio
    async def test_unexpected_transcription_error_also_returns_clean_result(self, tmp_path) -> None:
        provider = MockTranscriptionProvider(raise_error=RuntimeError("engine crashed"))
        service = CaptionService(transcription_provider=provider, assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))
        assert result.success is False
        assert "engine crashed" in result.error

    @pytest.mark.asyncio
    async def test_empty_transcription_returns_clean_failure(self, tmp_path) -> None:
        provider = MockTranscriptionProvider(segments=[])
        service = CaptionService(transcription_provider=provider, assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))
        assert result.success is False
        assert "no valid caption segments" in result.error.lower()

    @pytest.mark.asyncio
    async def test_all_whitespace_transcription_returns_clean_failure(self, tmp_path) -> None:
        provider = MockTranscriptionProvider(
            segments=[TranscribedSegment(text="   ", start_seconds=0.0, end_seconds=1.0)]
        )
        service = CaptionService(transcription_provider=provider, assembler=FakeAssembler())
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ffmpeg_burn_failure_returns_clean_result(self, tmp_path) -> None:
        assembler = FakeAssembler(fail_burn=True)
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))
        assert result.success is False
        assert "Subtitle rendering failed" in result.error

    @pytest.mark.asyncio
    async def test_original_video_untouched_on_burn_failure(self, tmp_path) -> None:
        video_result = _video_result(tmp_path)
        with open(video_result.output_path, "rb") as f:
            original_bytes = f.read()

        assembler = FakeAssembler(fail_burn=True)
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        await service.generate_captions(_voice_result(tmp_path), video_result)

        with open(video_result.output_path, "rb") as f:
            assert f.read() == original_bytes


class TestCaptionServiceSuccess:
    @pytest.mark.asyncio
    async def test_successful_captioned_output(self, tmp_path) -> None:
        assembler = FakeAssembler(duration=5.0)
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert result.success is True
        assert result.error is None
        assert result.srt_path is not None and os.path.exists(result.srt_path)
        assert result.captioned_video_path is not None and os.path.exists(result.captioned_video_path)
        assert len(result.segments) > 0
        assert result.transcription_provider == "mock"

    @pytest.mark.asyncio
    async def test_output_path_behavior(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        video_result = _video_result(tmp_path)
        result = await service.generate_captions(_voice_result(tmp_path), video_result)

        assert os.path.basename(result.srt_path) == "video.srt"
        assert result.captioned_video_path == str(tmp_path / "video-captioned.mp4")
        # Original file is a sibling, not overwritten.
        assert result.captioned_video_path != video_result.output_path

    @pytest.mark.asyncio
    async def test_original_video_untouched_on_success(self, tmp_path) -> None:
        video_result = _video_result(tmp_path)
        with open(video_result.output_path, "rb") as f:
            original_bytes = f.read()

        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        await service.generate_captions(_voice_result(tmp_path), video_result)

        with open(video_result.output_path, "rb") as f:
            assert f.read() == original_bytes

    @pytest.mark.asyncio
    async def test_srt_file_is_valid_and_non_empty(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        with open(result.srt_path, encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("1\n")
        assert "-->" in content

    @pytest.mark.asyncio
    async def test_output_dir_created_if_missing(self, tmp_path) -> None:
        nested_dir = str(tmp_path / "nested" / "subs")
        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=nested_dir
        )
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert os.path.isdir(nested_dir)
        assert result.srt_path.startswith(nested_dir)

    @pytest.mark.asyncio
    async def test_custom_subtitle_style_passed_to_assembler(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler,
            output_dir=str(tmp_path / "subs"), subtitle_style="FontName=Custom",
        )
        await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert assembler.burn_calls[0]["force_style"] == "FontName=Custom"

    @pytest.mark.asyncio
    async def test_default_subtitle_style_used_when_not_overridden(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert assembler.burn_calls[0]["force_style"] is not None

    @pytest.mark.asyncio
    async def test_captioned_duration_probe_failure_does_not_fail_result(self, tmp_path) -> None:
        assembler = FakeAssembler(fail_captioned_probe=True)
        service = CaptionService(
            transcription_provider=MockTranscriptionProvider(), assembler=assembler, output_dir=str(tmp_path / "subs")
        )
        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert result.success is True
        assert result.captioned_duration_seconds is None

    @pytest.mark.asyncio
    async def test_segments_are_synchronized_to_real_narration_timing_not_estimated(self, tmp_path) -> None:
        """Segment timing must come from the transcription provider's own
        timestamps, not any estimate derived from narration/video duration."""
        custom_segments = [TranscribedSegment(text="Custom timed line.", start_seconds=1.23, end_seconds=3.45)]
        provider = MockTranscriptionProvider(segments=custom_segments)
        assembler = FakeAssembler()
        service = CaptionService(transcription_provider=provider, assembler=assembler, output_dir=str(tmp_path / "subs"))

        result = await service.generate_captions(_voice_result(tmp_path), _video_result(tmp_path))

        assert result.segments[0].start_seconds == 1.23
        assert result.segments[0].end_seconds == 3.45
