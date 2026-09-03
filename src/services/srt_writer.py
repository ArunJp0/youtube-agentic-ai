# SRT (SubRip) subtitle file formatting/writing.
#
# Deliberately separate from caption_segmentation.py (readability rules)
# and caption_service.py (orchestration): this module only knows how to
# turn already-finalized CaptionSegments into standards-compliant SRT text,
# nothing about transcription, timing normalization, or rendering.
from __future__ import annotations

from typing import List

from src.models.captions import CaptionSegment


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: HH:MM:SS,mmm."""
    total_ms = max(round(seconds * 1000), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt(segments: List[CaptionSegment]) -> str:
    """Render ``segments`` (assumed already in chronological order) as SRT text.

    Numbering is always re-derived from list order (1, 2, 3, ...) rather
    than trusting each segment's own ``index``, so the output is correctly
    sequential even if segments were filtered/reordered upstream.
    """
    blocks = []
    for position, segment in enumerate(segments, start=1):
        blocks.append(
            f"{position}\n"
            f"{format_timestamp(segment.start_seconds)} --> {format_timestamp(segment.end_seconds)}\n"
            f"{segment.text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def write_srt_file(segments: List[CaptionSegment], output_path: str) -> None:
    """Write ``segments`` to ``output_path`` as a UTF-8 .srt file."""
    content = format_srt(segments)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
