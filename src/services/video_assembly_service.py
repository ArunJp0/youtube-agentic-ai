# Video Assembly Service: combines a VoiceResult's narration audio and a
# VisualResult's ordered per-section media into one final MP4.
#
# This is a deterministic service, not an LLM-driven reasoning agent:
# section timing is a fixed proportional calculation from narration word
# counts (shared with VisualMediaService via section_timing), and each
# section's already-planned assets are used in the exact order
# VisualMediaService selected them. Only the actual video encoding/muxing
# is delegated to a swappable VideoAssembler (FFmpeg by default).
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Dict, List, Optional

from src.models.media import MediaAsset, VisualResult
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.services.section_timing import calculate_section_durations
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
    consumes what those three earlier stages already produced, including
    VisualMediaService's own per-section slot planning: however many
    assets a section has, they are used in that exact order, each scaled
    to an equal share of that section's planned duration. The narration
    audio (not any per-section estimate) is the authoritative timeline.
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

        section_assets, error = self._map_usable_assets_by_section(script, visual_result)
        if error:
            return self._failure(error)

        try:
            audio_duration = self.assembler.probe_duration_seconds(voice_result.audio_file_path)
        except VideoAssemblerError as e:
            return self._failure(f"Failed to probe narration audio duration: {e}")
        except Exception as e:
            return self._failure(f"Unexpected error probing narration audio: {e}")

        section_durations = calculate_section_durations(script.sections, audio_duration)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self._build_filename(script))

        try:
            with tempfile.TemporaryDirectory(prefix="video_assembly_") as tmp_dir:
                clip_paths = self._build_all_clips(section_assets, section_durations, tmp_dir)
                self.assembler.concatenate_and_mux_audio(
                    section_clip_paths=clip_paths,
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

    # ---- clip building --------------------------------------------------

    def _build_all_clips(
        self,
        section_assets: Dict[int, List[MediaAsset]],
        section_durations: List[float],
        tmp_dir: str,
    ) -> List[str]:
        """Build one normalized clip per visual slot, across all sections,
        in playback order. Each section's assets split that section's
        planned duration evenly among themselves."""
        clip_paths: List[str] = []
        for section_index, duration in enumerate(section_durations):
            assets = section_assets[section_index]
            slot_duration = duration / len(assets)
            for slot_index, asset in enumerate(assets):
                clip_path = os.path.join(
                    tmp_dir, f"section-{section_index + 1:02d}-slot-{slot_index + 1:02d}.mp4"
                )
                self.assembler.build_section_clip(
                    input_path=asset.local_file_path,
                    output_path=clip_path,
                    target_duration_seconds=slot_duration,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    is_video=(asset.asset_type == "video"),
                )
                clip_paths.append(clip_path)
        return clip_paths

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _map_usable_assets_by_section(
        script: ScriptResult, visual_result: VisualResult
    ) -> tuple[Dict[int, List[MediaAsset]], Optional[str]]:
        """Build section_index -> ordered list of usable (successful, on-disk)
        MediaAssets, or an error message describing what's missing/broken.

        A section is usable if it has at least one successful asset with a
        file that actually exists on disk - VisualMediaService's own slot
        planning may occasionally leave a section with fewer successful
        assets than planned (see its fallback-reuse behavior), which is
        fine as long as at least one usable asset remains.
        """
        assets_by_section: Dict[int, List[MediaAsset]] = {
            mapping.section_index: mapping.assets for mapping in visual_result.sections
        }

        usable_by_section: Dict[int, List[MediaAsset]] = {}
        missing_indices: List[int] = []
        broken_paths: List[str] = []

        for index in range(len(script.sections)):
            assets = assets_by_section.get(index, [])
            usable = [a for a in assets if a.success and a.local_file_path]
            existing = [a for a in usable if os.path.exists(a.local_file_path)]
            broken_paths.extend(a.local_file_path for a in usable if not os.path.exists(a.local_file_path))

            if not existing:
                missing_indices.append(index)
            else:
                usable_by_section[index] = existing

        if missing_indices:
            return {}, f"Missing media for section(s): {[i + 1 for i in missing_indices]}"
        if broken_paths:
            return {}, f"Media file(s) not found: {broken_paths}"

        return usable_by_section, None

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
