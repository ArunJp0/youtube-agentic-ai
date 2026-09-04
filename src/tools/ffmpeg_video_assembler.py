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

    @abstractmethod
    def extract_frames(
        self,
        input_path: str,
        timestamps_seconds: List[float],
        output_dir: str,
        basename: str,
    ) -> List[str]:
        """Extract one representative JPEG frame per timestamp from a video.

        For Visual QC's frame sampling - never every frame, just the given
        timestamps (see src/services/frame_sampling.py for how those are
        chosen).

        Args:
            input_path: Source video file
            timestamps_seconds: Timestamps to extract, in seconds
            output_dir: Directory to write the extracted frame files into
            basename: Filename prefix for the extracted frames

        Returns:
            Extracted frame file paths, in the same order as
            ``timestamps_seconds``
        """
        raise NotImplementedError

    @abstractmethod
    def burn_subtitles(
        self,
        input_video_path: str,
        srt_path: str,
        output_path: str,
        force_style: Optional[str] = None,
    ) -> None:
        """Hardcode (burn) subtitles from an .srt file into a copy of a video.

        Re-encodes video (subtitles can't be burned via stream copy); the
        audio track is copied unchanged. ``input_video_path`` is never
        modified - the captioned result is always written to a separate
        ``output_path``.

        Args:
            input_video_path: Source MP4 to caption
            srt_path: Subtitle file to burn in
            output_path: Local filesystem path for the captioned copy
            force_style: Optional ASS ``force_style`` override string
                (font/size/color/margins/etc.) for the subtitles filter -
                styling policy belongs to the caller (CaptionService), not
                this tool layer.
        """
        raise NotImplementedError

    @abstractmethod
    def mix_background_audio(
        self,
        input_video_path: str,
        music_path: str,
        output_path: str,
        target_duration_seconds: float,
        music_gain_db: float,
        fade_in_seconds: float,
        fade_out_seconds: float,
        use_ducking: bool = True,
    ) -> None:
        """Mix a background music track underneath a video's existing
        narration audio, writing a new file at ``output_path``.
        ``input_video_path`` is never modified.

        The music track is looped seamlessly if shorter than
        ``target_duration_seconds``, or trimmed if longer - the same
        ``-stream_loop -1`` plus final-duration-capping strategy already
        used by ``build_section_clip``, so no gaps/clicks from manual
        splicing. ``music_gain_db`` attenuates the music before mixing
        (a negative value quietens it, keeping narration dominant);
        ``fade_in_seconds``/``fade_out_seconds`` apply smooth fades so the
        music never starts or stops abruptly, and never continues past the
        video's own end. When ``use_ducking`` is True, the narration track
        sidechain-compresses the music so it recedes further under speech
        and returns to its base level during silence, instead of sitting
        at one flat level throughout.

        Args:
            input_video_path: Source MP4 (already has narration audio)
            music_path: Background music audio file
            output_path: Local filesystem path for the mixed copy
            target_duration_seconds: The source video's duration - the
                output is capped to this length
            music_gain_db: Gain applied to the music track, in dB
                (negative attenuates)
            fade_in_seconds: Music fade-in duration at the start
            fade_out_seconds: Music fade-out duration, ending at
                ``target_duration_seconds``
            use_ducking: Whether to sidechain-compress the music under
                narration
        """
        raise NotImplementedError


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoAssemblerError(_WINDOWS_INSTALL_HINT)
    return path


def _escape_subtitles_filter_path(path: str) -> str:
    """Escape a filesystem path for use inside FFmpeg's ``subtitles=`` filter.

    FFmpeg's filtergraph parser treats ``:`` as an option separator, so a
    Windows drive letter (``C:\\...``) breaks the filter unless escaped.
    Forward slashes avoid a second layer of backslash-escaping.
    """
    return path.replace("\\", "/").replace(":", "\\:")


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

    def extract_frames(
        self,
        input_path: str,
        timestamps_seconds: List[float],
        output_dir: str,
        basename: str,
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        for index, timestamp in enumerate(timestamps_seconds):
            output_path = os.path.join(output_dir, f"{basename}-{index + 1:02d}.jpg")
            command = [
                self.ffmpeg_path, "-y",
                "-ss", f"{max(timestamp, 0.0):.3f}",
                "-i", input_path,
                "-frames:v", "1",
                "-q:v", "2",
                output_path,
            ]
            self._run(command)
            frame_paths.append(output_path)
        return frame_paths

    def burn_subtitles(
        self,
        input_video_path: str,
        srt_path: str,
        output_path: str,
        force_style: Optional[str] = None,
    ) -> None:
        subtitles_filter = f"subtitles='{_escape_subtitles_filter_path(srt_path)}'"
        if force_style:
            subtitles_filter += f":force_style='{force_style}'"

        command = [
            self.ffmpeg_path, "-y",
            "-i", input_video_path,
            "-vf", subtitles_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path,
        ]
        self._run(command)

    def mix_background_audio(
        self,
        input_video_path: str,
        music_path: str,
        output_path: str,
        target_duration_seconds: float,
        music_gain_db: float,
        fade_in_seconds: float,
        fade_out_seconds: float,
        use_ducking: bool = True,
    ) -> None:
        fade_out_start = max(target_duration_seconds - fade_out_seconds, 0.0)
        bgm_filter = (
            f"[1:a]volume={music_gain_db}dB,"
            f"afade=t=in:st=0:d={fade_in_seconds:.3f},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_seconds:.3f}[bgm]"
        )
        if use_ducking:
            filter_complex = (
                f"{bgm_filter};"
                "[bgm][0:a]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=300[bgm_ducked];"
                "[0:a][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
        else:
            filter_complex = (
                f"{bgm_filter};"
                "[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )

        command = [
            self.ffmpeg_path, "-y",
            "-i", input_video_path,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]
        self._run(command)

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
