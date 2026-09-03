# Caption Service: transcribes the real narration audio and burns
# synchronized, readable subtitles into a copy of the already-assembled MP4.
#
# This is a deterministic orchestration service, not an LLM-driven
# reasoning agent - the only "intelligence" involved is speech-to-text
# (delegated to a swappable TranscriptionProvider); segmentation into
# readable captions is fixed rules (see caption_segmentation.py), SRT
# formatting is fixed rules (see srt_writer.py), and subtitle rendering is
# delegated to the existing VideoAssembler/FFmpeg infrastructure - no
# second video-processing stack.
#
# Standalone for this milestone: not wired into the main LangGraph
# pipeline yet (see src/caption_demo.py).
from __future__ import annotations

import os
from typing import Optional

from src.models.captions import CaptionResult
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.services.caption_segmentation import build_caption_segments
from src.services.srt_writer import write_srt_file
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.transcription_provider import TranscriptionProvider, TranscriptionProviderError

DEFAULT_SUBTITLE_OUTPUT_DIR = os.path.join("output", "subtitles")

# Professional, YouTube-style basic caption styling (ASS force_style
# fields for FFmpeg's subtitles filter). Centralized here rather than
# scattered - the tool layer (FFmpegVideoAssembler) stays styling-agnostic
# and just runs whatever style string it's given.
DEFAULT_SUBTITLE_FONT_NAME = "Arial"
DEFAULT_SUBTITLE_FONT_SIZE = 22
DEFAULT_SUBTITLE_PRIMARY_COLOUR = "&H00FFFFFF"  # white (ASS &HAABBGGRR)
DEFAULT_SUBTITLE_OUTLINE_COLOUR = "&H00000000"  # black
DEFAULT_SUBTITLE_OUTLINE = 2
DEFAULT_SUBTITLE_SHADOW = 1
DEFAULT_SUBTITLE_ALIGNMENT = 2  # ASS numpad alignment: bottom-center
DEFAULT_SUBTITLE_MARGIN_V = 50  # px safe margin from the bottom edge


def build_default_subtitle_style() -> str:
    """The default ASS force_style string: readable sans-serif, white text
    with a black outline/shadow (no solid box) for contrast against any
    background, bottom-center with a safe margin - no flashy styling."""
    return (
        f"FontName={DEFAULT_SUBTITLE_FONT_NAME},FontSize={DEFAULT_SUBTITLE_FONT_SIZE},"
        f"PrimaryColour={DEFAULT_SUBTITLE_PRIMARY_COLOUR},OutlineColour={DEFAULT_SUBTITLE_OUTLINE_COLOUR},"
        f"BorderStyle=1,Outline={DEFAULT_SUBTITLE_OUTLINE},Shadow={DEFAULT_SUBTITLE_SHADOW},"
        f"Alignment={DEFAULT_SUBTITLE_ALIGNMENT},MarginV={DEFAULT_SUBTITLE_MARGIN_V}"
    )


class CaptionServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Processing failures (missing files, transcription/rendering errors)
    are NOT raised - they are captured in the returned CaptionResult
    (success=False, error=...) so callers always get a structured result
    back, and the original video is never touched on failure.
    """


class CaptionService:
    """Deterministic service that produces synchronized subtitles for an
    already-assembled video, from the real narration audio's actual
    transcribed timing - never estimated from script section durations.
    """

    def __init__(
        self,
        transcription_provider: TranscriptionProvider,
        assembler: VideoAssembler,
        output_dir: str = DEFAULT_SUBTITLE_OUTPUT_DIR,
        subtitle_style: Optional[str] = None,
    ) -> None:
        """Initialize the Caption Service.

        Args:
            transcription_provider: Implementation of TranscriptionProvider
                for speech-to-text
            assembler: VideoAssembler implementation, used for duration
                probing and burning subtitles - the same FFmpeg
                infrastructure Video Assembly already uses
            output_dir: Local directory to write generated .srt files into
                (expected to be excluded from version control)
            subtitle_style: Optional ASS force_style override; defaults to
                build_default_subtitle_style()
        """
        self.transcription_provider = transcription_provider
        self.assembler = assembler
        self.output_dir = output_dir
        self.subtitle_style = subtitle_style or build_default_subtitle_style()

    async def generate_captions(
        self, voice_result: VoiceResult, video_result: VideoAssemblyResult
    ) -> CaptionResult:
        """Transcribe the real narration audio and burn synchronized
        captions into a copy of the assembled video.

        Never raises for processing failures (missing files, transcription
        errors, rendering errors) - those are captured in the returned
        CaptionResult. Only raises CaptionServiceError for configuration/
        programmer errors (e.g. missing required arguments). The original
        video at ``video_result.output_path`` is never modified, on
        success or failure.

        Args:
            voice_result: Narration audio produced by the Voice Service
            video_result: Assembled MP4 produced by the Video Assembly Service

        Returns:
            Structured CaptionResult describing the outcome
        """
        if voice_result is None or video_result is None:
            raise CaptionServiceError("VoiceResult and VideoAssemblyResult are both required")

        if not voice_result.success or not voice_result.audio_file_path:
            return self._failure("VoiceResult was not successful; cannot caption without narration audio")
        if not os.path.exists(voice_result.audio_file_path):
            return self._failure(f"Narration audio file not found: {voice_result.audio_file_path}")

        if not video_result.success or not video_result.output_path:
            return self._failure("VideoAssemblyResult was not successful; cannot caption without an assembled video")
        if not os.path.exists(video_result.output_path):
            return self._failure(f"Assembled video file not found: {video_result.output_path}")

        try:
            raw_segments = self.transcription_provider.transcribe(voice_result.audio_file_path)
        except TranscriptionProviderError as e:
            return self._failure(f"Transcription failed: {e}")
        except Exception as e:
            return self._failure(f"Unexpected transcription error: {e}")

        max_duration = voice_result.duration_seconds or video_result.duration_seconds
        caption_segments = build_caption_segments(raw_segments, max_duration_seconds=max_duration)
        if not caption_segments:
            return self._failure("Transcription produced no valid caption segments")

        os.makedirs(self.output_dir, exist_ok=True)
        srt_path = os.path.join(self.output_dir, self._build_srt_filename(video_result.output_path))
        try:
            write_srt_file(caption_segments, srt_path)
        except OSError as e:
            return self._failure(f"Failed to write SRT file: {e}")

        captioned_path = self._build_captioned_video_path(video_result.output_path)
        try:
            self.assembler.burn_subtitles(
                video_result.output_path, srt_path, captioned_path, force_style=self.subtitle_style
            )
        except VideoAssemblerError as e:
            return self._failure(f"Subtitle rendering failed: {e}")
        except Exception as e:
            return self._failure(f"Unexpected subtitle rendering error: {e}")

        if not os.path.exists(captioned_path):
            return self._failure("Subtitle rendering reported success but the output file is missing")

        try:
            captioned_duration = self.assembler.probe_duration_seconds(captioned_path)
        except Exception:
            captioned_duration = None

        return CaptionResult(
            success=True,
            segments=caption_segments,
            srt_path=srt_path,
            captioned_video_path=captioned_path,
            source_audio_path=voice_result.audio_file_path,
            source_video_path=video_result.output_path,
            transcription_provider=self.transcription_provider.name,
            transcription_model=getattr(self.transcription_provider, "model_size", None),
            narration_duration_seconds=voice_result.duration_seconds,
            video_duration_seconds=video_result.duration_seconds,
            captioned_duration_seconds=captioned_duration,
        )

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _build_srt_filename(video_output_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_output_path))[0]
        return f"{base}.srt"

    @staticmethod
    def _build_captioned_video_path(video_output_path: str) -> str:
        root, ext = os.path.splitext(video_output_path)
        return f"{root}-captioned{ext}"

    @staticmethod
    def _failure(error: str) -> CaptionResult:
        return CaptionResult(success=False, error=error)
