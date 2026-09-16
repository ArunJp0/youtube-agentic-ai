# Deterministic normalization/validation for Metadata Agent output. No LLM
# call here - this is fixed-rule cleanup applied AFTER the single semantic
# generation request, and the last line of defense against malformed or
# borderline LLM output before it's accepted.
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.models.metadata import Chapter

# YouTube's real title character limit - the platform's absolute ceiling,
# kept as a last-resort safety net. This project's own house style is much
# stricter (see TITLE_HARD_MAX_CHARS below), so this constant is rarely the
# operative limit in practice.
MAX_TITLE_LENGTH = 100
# YouTube's real description character limit - same role as above relative
# to DESCRIPTION_HARD_MAX_WORDS.
MAX_DESCRIPTION_LENGTH = 5000
# Practical caps - not exactly YouTube's byte-based tag budget, but a safe
# margin under it (YouTube's real limit is ~500 total characters across tags).
MAX_TAGS = 30
MAX_TAGS_TOTAL_CHARS = 460
# "Typically 3-5" per project convention - capped, not just suggested.
MAX_HASHTAGS = 5

# House-style title/description targets - centralized here (never scattered
# as magic numbers in the agent/prompt) so both prompt guidance and
# deterministic repair reference the exact same numbers. These are this
# project's own concise/curiosity-driven content-quality goals, not
# YouTube's platform limits above.
TITLE_TARGET_MIN_CHARS = 45
TITLE_TARGET_MAX_CHARS = 60
TITLE_HARD_MAX_CHARS = 65

DESCRIPTION_TARGET_MIN_WORDS = 60
DESCRIPTION_TARGET_MAX_WORDS = 90
DESCRIPTION_HARD_MAX_WORDS = 110

# Matches a "Title: Subtitle" or "Title - Subtitle"/"Title — Subtitle"
# separator - a colon followed by whitespace, or a hyphen/en-dash/em-dash
# surrounded by whitespace. Deliberately requires surrounding whitespace so
# a mid-word hyphen (e.g. "e-commerce", "well-known") is never matched.
_SUBTITLE_SPLIT_RE = re.compile(r":\s+|\s[-–—]\s")

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_trailing_subtitle(text: str) -> str:
    """Drop a trailing explanatory subtitle after the first ':'/'-'/'–'/'—'
    separator, when the remaining prefix alone is still substantial enough
    to stand as a title on its own - never invents replacement text, only
    removes a redundant trailing clause. Returns ``text`` unchanged if no
    such separator exists or the prefix would be too thin to be meaningful.
    """
    match = _SUBTITLE_SPLIT_RE.search(text)
    if not match:
        return text
    prefix = text[: match.start()].strip()
    if len(prefix.split()) < 3:
        return text
    return prefix


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class ChapterValidationError(Exception):
    """Raised when a chapter list cannot be safely repaired - chapters
    should be omitted (with a reason), never fabricated to fill the gap."""


def normalize_title(raw: str) -> str:
    """Trim, collapse whitespace, strip wrapping quotes, and enforce this
    project's house-style hard title limit (``TITLE_HARD_MAX_CHARS``).

    An over-length title is repaired, never blindly truncated mid-sentence
    as the first resort: a trailing explanatory subtitle after a ':'/'-'/
    '–'/'—' separator is dropped first (see ``_strip_trailing_subtitle``)
    when the remaining prefix already stands on its own - only if that
    alone doesn't bring the title within the limit does a clean,
    non-mid-word truncation apply as the final safety net.
    """
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = " ".join(text.split())
    if not text or len(text) <= TITLE_HARD_MAX_CHARS:
        return text

    shortened = _strip_trailing_subtitle(text)
    if len(shortened) <= TITLE_HARD_MAX_CHARS:
        return shortened

    truncated = shortened[:TITLE_HARD_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" -–—:,")
    return truncated or shortened[:TITLE_HARD_MAX_CHARS]


def normalize_description(raw: str) -> str:
    """Trim and enforce this project's house-style hard description word
    limit (``DESCRIPTION_HARD_MAX_WORDS``), after first applying YouTube's
    absolute character ceiling as an outer safety net.

    An over-length description is repaired by dropping whole trailing
    sentences (grouped by paragraph, so a kept prefix's paragraph structure
    survives) until the word count fits - never a mid-sentence cut. Only
    when a single sentence alone exceeds the word cap does a last-resort
    clean word-boundary truncation apply.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    if len(text) > MAX_DESCRIPTION_LENGTH:
        text = text[:MAX_DESCRIPTION_LENGTH].rsplit(" ", 1)[0] or text[:MAX_DESCRIPTION_LENGTH]

    if len(text.split()) <= DESCRIPTION_HARD_MAX_WORDS:
        return text

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    kept_paragraphs: List[str] = []
    word_count = 0
    for paragraph in paragraphs:
        kept_sentences: List[str] = []
        for sentence in _split_sentences(paragraph):
            sentence_words = len(sentence.split())
            if word_count == 0 and not kept_sentences:
                # Always attempt the very first sentence overall, so a
                # short description is never reduced to nothing - but if
                # even this one sentence alone exceeds the cap, bail
                # straight to last-resort word truncation instead of
                # force-keeping a single over-cap "sentence".
                if sentence_words > DESCRIPTION_HARD_MAX_WORDS:
                    return " ".join(text.split()[:DESCRIPTION_HARD_MAX_WORDS])
            elif word_count + sentence_words > DESCRIPTION_HARD_MAX_WORDS:
                break
            kept_sentences.append(sentence)
            word_count += sentence_words
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
        if word_count >= DESCRIPTION_HARD_MAX_WORDS:
            break

    return "\n\n".join(kept_paragraphs) if kept_paragraphs else " ".join(text.split()[:DESCRIPTION_HARD_MAX_WORDS])


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
