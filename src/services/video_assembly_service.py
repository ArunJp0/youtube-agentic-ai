# Video Assembly Service: combines a VoiceResult's narration audio and a
# VisualResult's section media into one final MP4.
#
# This is a deterministic service, not an LLM-driven reasoning agent:
# section timing is a fixed proportional calculation from narration word
# counts, and section-to-media mapping is taken directly from the given
# VisualResult. Only the actual video encoding/muxing is delegated to a
# swappable VideoAssembler (FFmpeg by default).
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Dict, List, Optional

from src.models.media import MediaAsset, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError

DEFAULT_VIDEO_OUTPUT_DIR = os.path.join("output", "video")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


class VideoAssemblyServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Processing failures (missing files, FFmpeg errors) are NOT raised -
    they are captured in the returned VideoAssemblyResult (success=False,
    error=...) so callers always get a structured result back.
    """


class VideoAssemblyService:
    """Deterministic service that assembles a final MP4 from an already-
    generated ScriptResult, VoiceResult, and VisualResult.

    Never regenerates the script, narration, or section media - it only
    consumes what those three earlier stages already produced. Each
    section's on-screen duration is calculated deterministically from that
    section's narration word count as a proportion of the total narration
    audio duration; the audio (not any per-section estimate) is the
    authoritative timeline.
    """

    def __init__(
        self,
        assembler: VideoAssembler,
        output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
    ) -> None:
        """Initialize the Video Assembly Service.

        Args:
            assembler: Implementation of VideoAssembler for encoding/muxing
            output_dir: Local directory to write the final MP4 into
                (expected to be excluded from version control)
            width: Output video width in pixels
            height: Output video height in pixels
            fps: Output video frame rate
        """
        self.assembler = assembler
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps

    async def assemble_video(
        self,
        script: ScriptResult,
        voice_result: VoiceResult,
        visual_result: VisualResult,
    ) -> VideoAssemblyResult:
        """Assemble the final MP4 for a script's narration + section media.

        Never raises for processing failures (missing files, FFmpeg
        errors) - those are captured in the returned VideoAssemblyResult.
        Only raises VideoAssemblyServiceError for configuration/programmer
        errors (e.g. a missing required argument).

        Args:
            script: Structured script produced by the Script Agent
            voice_result: Narration audio produced by the Voice Service
            visual_result: Section media produced by the Visual Media Service

        Returns:
            Structured VideoAssemblyResult describing the outcome
        """
        if script is None or voice_result is None or visual_result is None:
            raise VideoAssemblyServiceError(
                "ScriptResult, VoiceResult, and VisualResult are all required"
            )

        if not voice_result.success or not voice_result.audio_file_path:
            return self._failure("VoiceResult was not successful; cannot assemble video without narration audio")
        if not os.path.exists(voice_result.audio_file_path):
            return self._failure(f"Narration audio file not found: {voice_result.audio_file_path}")

        if not visual_result.success:
            return self._failure("VisualResult was not successful; cannot assemble video without section media")

        if not script.sections:
            return self._failure("ScriptResult has no sections to assemble")

        assets_by_section, error = self._map_assets_by_section(script.sections, visual_result)
        if error:
            return self._failure(error)

        try:
            audio_duration = self.assembler.probe_duration_seconds(voice_result.audio_file_path)
        except VideoAssemblerError as e:
            return self._failure(f"Failed to probe narration audio duration: {e}")
        except Exception as e:
            return self._failure(f"Unexpected error probing narration audio: {e}")

        section_durations = self.calculate_section_durations(script.sections, audio_duration)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self._build_filename(script))

        try:
            with tempfile.TemporaryDirectory(prefix="video_assembly_") as tmp_dir:
                section_clip_paths = []
                for index, duration in enumerate(section_durations):
                    asset = assets_by_section[index]
                    clip_path = os.path.join(tmp_dir, f"section-{index + 1:02d}.mp4")
                    self.assembler.build_section_clip(
                        input_path=asset.local_file_path,
                        output_path=clip_path,
                        target_duration_seconds=duration,
                        width=self.width,
                        height=self.height,
                        fps=self.fps,
                        is_video=(asset.asset_type == "video"),
                    )
                    section_clip_paths.append(clip_path)

                self.assembler.concatenate_and_mux_audio(
                    section_clip_paths=section_clip_paths,
                    audio_path=voice_result.audio_file_path,
                    output_path=output_path,
                    tmp_dir=tmp_dir,
                )
        except VideoAssemblerError as e:
            return self._failure(f"FFmpeg processing failed: {e}")
        except Exception as e:
            return self._failure(f"Unexpected video assembly error: {e}")

        if not os.path.exists(output_path):
            return self._failure("Video assembly reported success but the output file is missing")

        try:
            final_duration = round(self.assembler.probe_duration_seconds(output_path), 2)
        except Exception:
            final_duration = round(sum(section_durations), 2)

        return VideoAssemblyResult(
            success=True,
            output_path=output_path,
            duration_seconds=final_duration,
            width=self.width,
            height=self.height,
            fps=self.fps,
            format="mp4",
            video_codec="h264",
            audio_codec="aac",
            section_count=len(script.sections),
            section_durations_seconds=[round(d, 2) for d in section_durations],
        )

    # ---- section timing -------------------------------------------------

    @staticmethod
    def calculate_section_durations(
        sections: List[ScriptSection], total_duration_seconds: float
    ) -> List[float]:
        """Deterministically split ``total_duration_seconds`` across sections.

        Each section's share is proportional to its narration word count
        relative to the combined word count of all sections (no LLM call).
        The last section absorbs any rounding drift so the durations always
        sum to exactly ``total_duration_seconds``.

        Args:
            sections: Script sections, in order
            total_duration_seconds: The narration audio's total duration -
                the authoritative timeline for the whole video

        Returns:
            One duration (seconds) per section, in the same order, summing
            to exactly ``total_duration_seconds``
        """
        word_counts = [max(len(section.narration.split()), 1) for section in sections]
        total_words = sum(word_counts)

        durations = [total_duration_seconds * (count / total_words) for count in word_counts]

        if durations:
            drift = total_duration_seconds - sum(durations)
            durations[-1] += drift

        return durations

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _map_assets_by_section(
        sections: List[ScriptSection], visual_result: VisualResult
    ) -> tuple[Dict[int, MediaAsset], Optional[str]]:
        """Build section_index -> successful MediaAsset, or an error message
        describing what's missing/broken."""
        assets_by_section: Dict[int, MediaAsset] = {}
        for mapping in visual_result.sections:
            if mapping.assets and mapping.assets[0].success:
                assets_by_section[mapping.section_index] = mapping.assets[0]

        missing_indices = [i for i in range(len(sections)) if i not in assets_by_section]
        if missing_indices:
            return {}, f"Missing media for section(s): {[i + 1 for i in missing_indices]}"

        for index, asset in assets_by_section.items():
            if not asset.local_file_path or not os.path.exists(asset.local_file_path):
                return {}, f"Media file not found for section {index + 1}: {asset.local_file_path}"

        return assets_by_section, None

    @staticmethod
    def _build_filename(script: ScriptResult) -> str:
        slug = "".join(c if c.isalnum() else "-" for c in script.topic.lower())
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug.strip("-")[:50] or "video"
        return f"{slug}-{uuid.uuid4().hex[:8]}.mp4"

    @staticmethod
    def _failure(error: str) -> VideoAssemblyResult:
        return VideoAssemblyResult(success=False, error=error)
