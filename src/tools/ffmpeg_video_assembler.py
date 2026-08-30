# FFmpeg-backed video assembly tool.
#
# This module contains all FFmpeg-specific command construction and process
# handling. VideoAssemblyService depends only on the VideoAssembler
# interface below, never on FFmpeg directly - the same pattern used for
# every other external tool in this project (EdgeVoiceProvider,
# PexelsMediaProvider, etc.).
from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional

_WINDOWS_INSTALL_HINT = (
    "FFmpeg is required for video assembly but was not found on PATH.\n"
    "Windows install options:\n"
    "  winget install --id=Gyan.FFmpeg -e\n"
    "  (or) choco install ffmpeg\n"
    "  (or) download a build from https://www.gyan.dev/ffmpeg/builds/ "
    "and add its bin/ folder to PATH.\n"
    "After installing, restart your terminal and verify with: ffmpeg -version"
)


class VideoAssemblerError(Exception):
    """Raised when FFmpeg is unavailable or a processing step fails."""


class VideoAssembler(ABC):
    """Abstract interface VideoAssemblyService depends on.

    Keeping this as an interface (rather than calling FFmpeg directly from
    the service) lets VideoAssemblyService's orchestration logic - timing
    calculation, section ordering, error handling - be tested with a fake
    assembler, with no real FFmpeg process involved.
    """

    @abstractmethod
    def probe_duration_seconds(self, media_path: str) -> float:
        """Return the duration of a media file in seconds."""
        raise NotImplementedError

    @abstractmethod
    def build_section_clip(
        self,
        input_path: str,
        output_path: str,
        target_duration_seconds: float,
        width: int,
        height: int,
        fps: int,
        is_video: bool,
    ) -> None:
        """Produce a silent, normalized clip at ``output_path``.

        Exactly ``target_duration_seconds`` long, ``width``x``height`` at
        ``fps``, scaled and center-cropped to fill the frame without
        stretching/distorting (preserves aspect ratio). The source is
        trimmed if longer than the target duration, or looped if shorter.
        """
        raise NotImplementedError

    @abstractmethod
    def concatenate_and_mux_audio(
        self,
        section_clip_paths: List[str],
        audio_path: str,
        output_path: str,
        tmp_dir: str,
    ) -> None:
        """Concatenate section clips in the given order, then mux in
        ``audio_path`` as the final narration track, writing an MP4
        (H.264 video / AAC audio) to ``output_path``."""
        raise NotImplementedError


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoAssemblerError(_WINDOWS_INSTALL_HINT)
    return path


class FFmpegVideoAssembler(VideoAssembler):
    """Video assembly backed by the local ffmpeg/ffprobe binaries.

    Raises VideoAssemblerError immediately on construction if either binary
    is not found on PATH, with a clear Windows install hint - assembly
    never silently proceeds without FFmpeg.
    """

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path or _require_binary("ffmpeg")
        self.ffprobe_path = ffprobe_path or _require_binary("ffprobe")
        self.timeout_seconds = timeout_seconds

    def probe_duration_seconds(self, media_path: str) -> float:
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ]
        result = self._run(command)
        try:
            return float(result.stdout.strip())
        except ValueError as e:
            raise VideoAssemblerError(
                f"Could not parse duration from ffprobe output: {result.stdout!r}"
            ) from e

    def build_section_clip(
        self,
        input_path: str,
        output_path: str,
        target_duration_seconds: float,
        width: int,
        height: int,
        fps: int,
        is_video: bool,
    ) -> None:
        # scale to cover the frame (may overshoot one dimension), then
        # center-crop the overflow - this is FFmpeg's default crop
        # behavior when x/y aren't given - so the result fills the frame
        # with no stretching and no letterboxing.
        scale_crop = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        duration_str = f"{target_duration_seconds:.3f}"

        if is_video:
            # -stream_loop -1 plus -t handles both cases with one command:
            # if the source is already longer than the target, FFmpeg stops
            # at -t before it would ever need to loop (trim); if shorter,
            # it loops seamlessly to fill the target duration.
            input_args = ["-stream_loop", "-1", "-i", input_path]
        else:
            input_args = ["-loop", "1", "-i", input_path]

        command = [
            self.ffmpeg_path, "-y",
            *input_args,
            "-t", duration_str,
            "-vf", scale_crop,
            "-r", str(fps),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        self._run(command)

    def concatenate_and_mux_audio(
        self,
        section_clip_paths: List[str],
        audio_path: str,
        output_path: str,
        tmp_dir: str,
    ) -> None:
        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for clip_path in section_clip_paths:
                # ffmpeg's concat demuxer format: forward slashes and
                # escaped single quotes keep this safe on Windows paths too.
                escaped = clip_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        concatenated_path = os.path.join(tmp_dir, "concatenated.mp4")
        self._run(
            [
                self.ffmpeg_path, "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                concatenated_path,
            ]
        )

        # All section clips share identical codec/format/fps, so the concat
        # step above is a clean stream copy with no re-encoding, no gaps,
        # and no black frames between sections. -shortest here is a safety
        # net against sub-frame rounding drift between the (deterministically
        # duration-matched) video and audio - not expected to trim anything
        # perceptible.
        self._run(
            [
                self.ffmpeg_path, "-y",
                "-i", concatenated_path,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]
        )

    def _run(self, command: List[str]) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout_seconds
            )
        except FileNotFoundError as e:
            raise VideoAssemblerError(f"FFmpeg binary not found: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise VideoAssemblerError(f"FFmpeg command timed out after {self.timeout_seconds}s: {e}") from e

        if result.returncode != 0:
            raise VideoAssemblerError(
                f"FFmpeg command failed (exit {result.returncode}): {' '.join(command)}\n"
                f"{result.stderr[-2000:]}"
            )
        return result
