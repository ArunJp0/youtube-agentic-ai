# Deterministic caption segmentation/readability rules.
#
# Turns raw TranscribedSegment objects (Whisper's own sentence-ish chunks)
# into professional, YouTube-style CaptionSegments: split where too long,
# timed monotonically with no overlaps, bounded to a comfortable reading
# duration, and wrapped to at most a couple of on-screen lines. Pure and
# deterministic - no LLM call, no randomness - so the same transcription
# always produces the same captions.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.models.captions import CaptionSegment
from src.tools.transcription_provider import TranscribedSegment, TranscribedWord

# ---- readability constants (centralized, not scattered magic numbers) -----

MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SEGMENT = 2
MAX_CHARS_PER_SEGMENT = MAX_CHARS_PER_LINE * MAX_LINES_PER_SEGMENT

# A caption is never shown for less than this long, even if the aligned
# audio span is shorter - avoids uncomfortably rapid flashing captions.
MIN_SEGMENT_DURATION_SECONDS = 1.0

# A single caption is never shown for longer than this, even if the
# aligned audio span (or a generous reading-speed allowance) would be
# longer - keeps very long pauses from leaving stale text on screen.
MAX_SEGMENT_DURATION_SECONDS = 6.0

# Comfortable reading speed ceiling (characters/second) - a common
# subtitling guideline is roughly 17-21 cps for adult viewers. Used as a
# floor on segment duration, not a hard cap, so dense text isn't shown too
# briefly to read.
MAX_READING_CHARS_PER_SECOND = 21.0

# How far a segment's end may be clamped beyond the narration/video's own
# duration - accounts for small rounding differences between the
# transcription engine's and FFmpeg's duration estimates, not a real
# extension of the timeline.
DURATION_TOLERANCE_SECONDS = 0.5

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_END_CHARS = (".", "!", "?")


@dataclass
class _RawChunk:
    """An intermediate caption chunk: readable text with a provisional
    (not yet normalized/deduplicated) time span."""

    text: str
    start: float
    end: float


def build_caption_segments(
    raw_segments: List[TranscribedSegment], max_duration_seconds: Optional[float] = None
) -> List[CaptionSegment]:
    """Turn raw transcribed segments into final, screen-ready CaptionSegments.

    Args:
        raw_segments: Segments as returned by a TranscriptionProvider, in
            chronological order
        max_duration_seconds: The narration/video's own duration, if known -
            no caption is allowed to end more than DURATION_TOLERANCE_SECONDS
            past this

    Returns:
        Readable CaptionSegments with monotonic, non-overlapping timestamps,
        1-based ``index`` in chronological order. Empty if no usable
        speech was found.
    """
    cleaned = [s for s in raw_segments if s.text and s.text.strip()]

    chunks: List[_RawChunk] = []
    for segment in cleaned:
        chunks.extend(_split_segment_into_chunks(segment))

    normalized = _normalize_timing(chunks, max_duration_seconds)

    return [
        CaptionSegment(
            index=i + 1,
            start_seconds=round(chunk.start, 3),
            end_seconds=round(chunk.end, 3),
            text=_wrap_caption_text(chunk.text),
        )
        for i, chunk in enumerate(normalized)
    ]


# ---- splitting a raw segment into readable chunks --------------------------


def _split_segment_into_chunks(segment: TranscribedSegment) -> List[_RawChunk]:
    text = segment.text.strip()
    if not text:
        return []
    if segment.words:
        return _chunk_by_words(segment.words, segment.start_seconds, segment.end_seconds)
    if len(text) <= MAX_CHARS_PER_SEGMENT:
        return [_RawChunk(text=text, start=segment.start_seconds, end=segment.end_seconds)]
    return _chunk_by_text_proportional(text, segment.start_seconds, segment.end_seconds)


def _chunk_by_words(
    words: List[TranscribedWord], segment_start: float, segment_end: float
) -> List[_RawChunk]:
    chunks: List[_RawChunk] = []
    current: List[TranscribedWord] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            text = " ".join(w.word.strip() for w in current if w.word.strip())
            if text:
                chunks.append(_RawChunk(text=text, start=current[0].start_seconds, end=current[-1].end_seconds))
        current = []
        current_len = 0

    for word in words:
        token = word.word.strip()
        if not token:
            continue
        added_len = len(token) + (1 if current else 0)
        if current and current_len + added_len > MAX_CHARS_PER_SEGMENT:
            flush()
            added_len = len(token)
        current.append(word)
        current_len += added_len
        if token.endswith(_SENTENCE_END_CHARS):
            flush()
    flush()

    if not chunks:
        return [_RawChunk(text=" ".join(w.word.strip() for w in words), start=segment_start, end=segment_end)]
    return chunks


def _chunk_by_text_proportional(text: str, segment_start: float, segment_end: float) -> List[_RawChunk]:
    """Fallback when no word-level timestamps are available: split the
    segment's text at readable boundaries and distribute the segment's own
    time span proportionally by character count."""
    pieces = _split_text_into_readable_pieces(text)
    if not pieces:
        return []
    if len(pieces) == 1:
        return [_RawChunk(text=pieces[0], start=segment_start, end=segment_end)]

    total_chars = sum(len(p) for p in pieces) or 1
    duration = max(segment_end - segment_start, 0.0)

    chunks = []
    cursor = segment_start
    for i, piece in enumerate(pieces):
        is_last = i == len(pieces) - 1
        end = segment_end if is_last else cursor + duration * (len(piece) / total_chars)
        chunks.append(_RawChunk(text=piece, start=cursor, end=end))
        cursor = end
    return chunks


def _split_text_into_readable_pieces(text: str) -> List[str]:
    """Pure text splitter: sentence boundaries first, then a greedy
    word-wrap for any sentence still longer than MAX_CHARS_PER_SEGMENT.
    Never splits a word."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    pieces: List[str] = []
    for sentence in sentences:
        if len(sentence) <= MAX_CHARS_PER_SEGMENT:
            pieces.append(sentence)
            continue
        words = sentence.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > MAX_CHARS_PER_SEGMENT:
                pieces.append(current)
                current = word
            else:
                current = candidate
        if current:
            pieces.append(current)
    return pieces


# ---- timing normalization ---------------------------------------------------


def _normalize_timing(chunks: List[_RawChunk], max_duration_seconds: Optional[float]) -> List[_RawChunk]:
    """Enforce monotonic, non-overlapping, comfortably-timed captions:
    every segment starts no earlier than the previous one ended, lasts at
    least MIN_SEGMENT_DURATION_SECONDS and at least enough to read at
    MAX_READING_CHARS_PER_SECOND, never longer than
    MAX_SEGMENT_DURATION_SECONDS, and never past max_duration_seconds (plus
    a small tolerance). Empty/degenerate chunks are dropped."""
    normalized: List[_RawChunk] = []
    previous_end = 0.0

    for chunk in chunks:
        text = _clean_caption_text(chunk.text)
        if not text:
            continue

        start = max(chunk.start, previous_end, 0.0)
        reading_floor = len(text) / MAX_READING_CHARS_PER_SECOND
        end = max(chunk.end, start + MIN_SEGMENT_DURATION_SECONDS, start + reading_floor)
        end = min(end, start + MAX_SEGMENT_DURATION_SECONDS)
        if max_duration_seconds is not None:
            end = min(end, max_duration_seconds + DURATION_TOLERANCE_SECONDS)

        if end <= start:
            continue

        normalized.append(_RawChunk(text=text, start=start, end=end))
        previous_end = end

    return normalized


def _clean_caption_text(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


# ---- on-screen line wrapping ------------------------------------------------


def _wrap_caption_text(text: str) -> str:
    """Wrap to lines of <= MAX_CHARS_PER_LINE characters, breaking only at
    whitespace (never mid-word).

    Targets MAX_LINES_PER_SEGMENT lines (guaranteed for any chunk within
    MAX_CHARS_PER_SEGMENT that has at least one natural break every
    ~MAX_CHARS_PER_LINE characters). In the rare case of a long run of
    words with no such break, more lines are produced rather than merging
    the remainder into one over-long line - never exceeding the per-line
    character limit is the harder requirement.
    """
    if len(text) <= MAX_CHARS_PER_LINE:
        return text

    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > MAX_CHARS_PER_LINE:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    return "\n".join(lines)
