# Neutral fallback visual generation: the absolute-last-resort Visual QC
# recovery strategy, used only when a script section would otherwise end up
# with zero safe usable visual coverage after every replace/drop attempt
# (see src.services.visual_qc_service).
#
# Deliberately NOT a stock-media source and NOT LLM/API-driven: a neutral
# fallback clip is a plain background with, at most, an already-known-safe
# text label (e.g. the section's own heading) rendered on it - it never
# depicts photographic content and can therefore never itself assert an
# unsupported factual claim the way a wrong/misleading stock clip could.
# This keeps the fallback safe by construction, not by another round of
# semantic judgment.
from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional

_WINDOWS_INSTALL_HINT = (
    "FFmpeg is required for neutral fallback visual generation but was not found on PATH.\n"
    "Windows install options:\n"
    "  winget install --id=Gyan.FFmpeg -e\n"
    "  (or) choco install ffmpeg\n"
    "  (or) download a build from https://www.gyan.dev/ffmpeg/builds/ "
    "and add its bin/ folder to PATH.\n"
    "After installing, restart your terminal and verify with: ffmpeg -version"
)


class NeutralVisualGeneratorError(Exception):
    """Raised when a neutral fallback visual could not be generated -
    callers must treat this as "no fallback available", never invent a
    substitute."""


class NeutralVisualGenerator(ABC):
    """Generates a safe, generic, non-claim-specific background clip.

    Kept as its own small interface (not a new VideoAssembler abstract
    method) so this optional, rarely-needed capability never has to be
    implemented by every existing VideoAssembler test double/mock - see
    docs/DECISIONS.md.
    """

    @abstractmethod
    def generate(
        self,
        output_path: str,
        duration_seconds: float,
        width: int,
        height: int,
        fps: int,
        label: Optional[str] = None,
    ) -> None:
        """Write a neutral background clip to ``output_path``.

        Args:
            output_path: Destination file path (mp4)
            duration_seconds: Exact clip duration
            width/height/fps: Target video parameters, mirroring
                VideoAssembler.build_section_clip's own parameters, so the
                output slots directly into the existing assembly pipeline
                with no further processing
            label: Optional short plain text to render on the background
                (e.g. the section's own heading) - never invented content,
                only an already-known-safe string the caller supplies.

        Raises:
            NeutralVisualGeneratorError: If generation fails for any reason.
        """
        raise NotImplementedError


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise NeutralVisualGeneratorError(_WINDOWS_INSTALL_HINT)
    return path


def _escape_drawtext(text: str) -> str:
    """Escape a label for FFmpeg's drawtext filter (backslash, colon, and
    single-quote are all filtergraph-meaningful characters)."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


class FFmpegNeutralVisualGenerator(NeutralVisualGenerator):
    """Real implementation: a static neutral-color background produced
    directly by FFmpeg's ``lavfi color`` source, with an optional centered
    plain-text label - no external API, no network call, no stock media."""

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        background_color: str = "0x1a1a2e",
        font_color: str = "white",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path or _require_binary("ffmpeg")
        self.background_color = background_color
        self.font_color = font_color
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        output_path: str,
        duration_seconds: float,
        width: int,
        height: int,
        fps: int,
        label: Optional[str] = None,
    ) -> None:
        duration_str = f"{max(duration_seconds, 0.1):.3f}"
        color_source = f"color=c={self.background_color}:s={width}x{height}:d={duration_str}:r={fps}"

        command: List[str] = [self.ffmpeg_path, "-y", "-f", "lavfi", "-i", color_source]
        clean_label = (label or "").strip()
        if clean_label:
            escaped = _escape_drawtext(clean_label)
            command += [
                "-vf",
                f"drawtext=text='{escaped}':fontcolor={self.font_color}:fontsize=48:"
                "x=(w-text_w)/2:y=(h-text_h)/2",
            ]
        command += ["-pix_fmt", "yuv420p", output_path]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_seconds)
        except FileNotFoundError as e:
            raise NeutralVisualGeneratorError(f"FFmpeg binary not found: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise NeutralVisualGeneratorError(f"Neutral visual generation timed out after {self.timeout_seconds}s: {e}") from e

        if result.returncode != 0:
            raise NeutralVisualGeneratorError(
                f"Neutral visual generation failed (exit {result.returncode}): {result.stderr[-1000:]}"
            )
