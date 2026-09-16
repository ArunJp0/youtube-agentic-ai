# Deterministic FFmpeg visual-variety helpers for the single-AI-clip demo
# scenario (see src.ai_video_demo) - demo-specific, NOT part of the normal
# production pipeline. VideoAssemblyService/FFmpegVideoAssembler are reused
# completely unchanged for actual per-section trimming/looping/scaling/
# concatenation/audio muxing; this module only produces a small number of
# visually DISTINCT local video files from ONE source clip, so a video
# built from a single short AI-generated clip doesn't look like the same
# footage looping identically behind every section.
#
# Command construction is a pure function (testable without a real FFmpeg
# binary); execution is a thin, separately-testable wrapper - mirroring
# FFmpegVideoAssembler's own internal shape without touching that class.
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List

DEFAULT_FFMPEG_PATH = "ffmpeg"


class VisualVarietyError(Exception):
    """Raised when FFmpeg is unavailable or a variant-generation command fails."""


@dataclass(frozen=True)
class CropSpeedVariant:
    """One deterministic reframe + pace transform of a source clip -
    never a random/AI effect, always the same output for the same input."""

    name: str
    crop_expr: str  # ffmpeg crop filter expression (evaluated against input dimensions)
    speed_factor: float = 1.0  # 1.0 = unchanged; < 1.0 = slower; > 1.0 = faster


# A small, fixed set of named variants - deliberately few and distinct
# (center vs left vs right framing; one slowed for a calmer pace) rather
# than an open-ended/random set, so output is fully deterministic and easy
# to reason about.
CENTER_SLOW_ZOOM = CropSpeedVariant(
    name="center_slow_zoom",
    crop_expr="crop=in_w*0.86:in_h*0.86:in_w*0.07:in_h*0.07",
    speed_factor=0.85,
)
LEFT_REFRAME = CropSpeedVariant(
    name="left_reframe",
    crop_expr="crop=in_w*0.82:in_h*0.92:0:in_h*0.04",
    speed_factor=1.0,
)
RIGHT_REFRAME = CropSpeedVariant(
    name="right_reframe",
    crop_expr="crop=in_w*0.82:in_h*0.92:in_w*0.18:in_h*0.04",
    speed_factor=1.0,
)


def build_variant_command(
    input_path: str, output_path: str, variant: CropSpeedVariant, ffmpeg_path: str = DEFAULT_FFMPEG_PATH
) -> List[str]:
    """Build the FFmpeg command for one crop/speed variant - pure
    construction, no subprocess call, so this is directly unit-testable
    without a real FFmpeg binary installed."""
    filters = [variant.crop_expr]
    if variant.speed_factor != 1.0:
        filters.append(f"setpts={1.0 / variant.speed_factor:.4f}*PTS")
    return [
        ffmpeg_path, "-y",
        "-i", input_path,
        "-vf", ",".join(filters),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]


def run_ffmpeg_command(command: List[str]) -> None:
    """Execute an already-built FFmpeg command. Separated from command
    construction purely for testability (tests exercise build_variant_command
    without ever invoking this)."""
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise VisualVarietyError(
            f"FFmpeg not found at '{command[0]}' - install FFmpeg and ensure it's on PATH"
        ) from e
    except subprocess.CalledProcessError as e:
        raise VisualVarietyError(f"FFmpeg variant generation failed: {e.stderr}") from e


def generate_variant_clip(
    input_path: str, output_path: str, variant: CropSpeedVariant, ffmpeg_path: str = DEFAULT_FFMPEG_PATH
) -> str:
    """Produce one deterministic crop/speed variant of ``input_path`` at
    ``output_path``. Raises VisualVarietyError on failure - never silently
    substitutes a different source."""
    command = build_variant_command(input_path, output_path, variant, ffmpeg_path)
    run_ffmpeg_command(command)
    return output_path


def resolve_ffmpeg_path(configured_path: str = DEFAULT_FFMPEG_PATH) -> str:
    """Resolve the FFmpeg executable path, raising a clear error if it's
    genuinely unavailable - mirrors FFmpegVideoAssembler's own real
    "fail fast, clear message" behavior rather than a cryptic subprocess
    error deep inside a variant-generation call."""
    resolved = shutil.which(configured_path)
    if resolved is None:
        raise VisualVarietyError(
            f"FFmpeg executable '{configured_path}' not found on PATH - install FFmpeg first."
        )
    return resolved
