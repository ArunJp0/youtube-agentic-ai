# Deterministic normalization/validation for Metadata Agent output. No LLM
# call here - this is fixed-rule cleanup applied AFTER the single semantic
# generation request, and the last line of defense against malformed or
# borderline LLM output before it's accepted.
from __future__ import annotations

from typing import List, Optional, Tuple

from src.models.metadata import Chapter

# YouTube's real title character limit.
MAX_TITLE_LENGTH = 100
# YouTube's real description character limit.
MAX_DESCRIPTION_LENGTH = 5000
# Practical caps - not exactly YouTube's byte-based tag budget, but a safe
# margin under it (YouTube's real limit is ~500 total characters across tags).
MAX_TAGS = 30
MAX_TAGS_TOTAL_CHARS = 460
# "Typically 3-5" per project convention - capped, not just suggested.
MAX_HASHTAGS = 5


class ChapterValidationError(Exception):
    """Raised when a chapter list cannot be safely repaired - chapters
    should be omitted (with a reason), never fabricated to fill the gap."""


def normalize_title(raw: str) -> str:
    """Trim, collapse whitespace, strip wrapping quotes, and enforce the
    YouTube title length limit with a clean (non-mid-word) truncation."""
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = " ".join(text.split())
    if len(text) <= MAX_TITLE_LENGTH:
        return text
    truncated = text[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip(" -–—:,")
    return truncated or text[:MAX_TITLE_LENGTH]


def normalize_description(raw: str) -> str:
    """Trim and enforce the YouTube description length limit with a clean
    (non-mid-word) truncation."""
    text = (raw or "").strip()
    if len(text) <= MAX_DESCRIPTION_LENGTH:
        return text
    truncated = text[:MAX_DESCRIPTION_LENGTH].rsplit(" ", 1)[0]
    return truncated or text[:MAX_DESCRIPTION_LENGTH]


def normalize_tags(raw_tags: List[str]) -> List[str]:
    """Deduplicate case-insensitively (first-seen casing kept), drop empty
    entries, and cap both count and total character budget deterministically."""
    seen = set()
    normalized: List[str] = []
    total_chars = 0
    for tag in raw_tags or []:
        cleaned = " ".join((tag or "").strip().split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if len(normalized) >= MAX_TAGS:
            break
        if total_chars + len(cleaned) > MAX_TAGS_TOTAL_CHARS:
            continue
        seen.add(key)
        normalized.append(cleaned)
        total_chars += len(cleaned)
    return normalized


def normalize_hashtags(raw_hashtags: List[str]) -> List[str]:
    """Ensure a leading '#', strip whitespace, dedupe case-insensitively,
    and cap to a small practical set (no hashtag stuffing)."""
    seen = set()
    normalized: List[str] = []
    for tag in raw_hashtags or []:
        cleaned = "".join((tag or "").split())  # hashtags never contain internal whitespace
        cleaned = "#" + cleaned.lstrip("#")
        if len(cleaned) <= 1:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if len(normalized) >= MAX_HASHTAGS:
            break
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def format_youtube_timestamp(total_seconds: float) -> str:
    """Format seconds as a YouTube chapter timestamp: 'M:SS' or 'H:MM:SS'."""
    total = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def validate_and_clean_chapters(
    chapters: List[Chapter], duration_seconds: Optional[float]
) -> Tuple[List[Chapter], List[str]]:
    """Deterministically validate/repair a chapter list - drop (never
    invent) anything unsafe: empty labels, non-increasing/duplicate
    timestamps, or timestamps at/beyond the video's own duration.

    Returns:
        (cleaned_chapters, warnings) - warnings describe anything dropped.

    Raises:
        ChapterValidationError: If nothing usable remains, or the first
            remaining chapter isn't at 0:00.
    """
    warnings: List[str] = []
    cleaned: List[Chapter] = []
    last_timestamp = -1.0

    for chapter in chapters:
        label = (chapter.title or "").strip()
        if not label:
            warnings.append(f"Dropped chapter at {chapter.timestamp_text}: empty label")
            continue
        if chapter.timestamp_seconds <= last_timestamp:
            warnings.append(f"Dropped chapter '{label}': timestamp not strictly increasing")
            continue
        if duration_seconds is not None and chapter.timestamp_seconds >= duration_seconds:
            warnings.append(f"Dropped chapter '{label}': timestamp at/beyond video duration")
            continue
        cleaned.append(chapter.model_copy(update={"title": label}) if label != chapter.title else chapter)
        last_timestamp = chapter.timestamp_seconds

    if not cleaned:
        raise ChapterValidationError("No valid chapters remained after validation")
    if cleaned[0].timestamp_seconds != 0.0:
        raise ChapterValidationError("First chapter must start at 0:00")

    return cleaned, warnings
