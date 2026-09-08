# Deterministic Thumbnail Agent validation: hook-text normalization/
# topic-alignment (applied before rendering) and output-image validation
# (applied after rendering). No LLM/network involved - the last line of
# defense before a thumbnail is reported successful.
from __future__ import annotations

import os
import re
from typing import Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from src.services.thumbnail_planning import deterministic_hook_from_title_or_topic
from src.services.thumbnail_renderer import THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH

# Generous character cap - the ~2-6 word guidance is a planning-time
# instruction to the LLM, not a hard limit; this is the deterministic,
# always-enforced backstop so a runaway response can never overflow layout.
MAX_HOOK_TEXT_LENGTH = 60

# Generic English function words, excluded when comparing a hook's words
# against the video's own topic/title vocabulary - not topic-specific.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "do",
    "does", "did", "why", "how", "what", "when", "where", "who", "this", "that", "these",
    "those", "we", "you", "your", "our", "it", "its", "with", "from", "by", "about", "into",
    "up", "down", "out", "over", "under", "again", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
}

# Generic English number words - not topic-specific, just a linguistic
# resource (like the stopword list above) used to recognize "this hook is
# mostly a bare number/count" regardless of subject matter.
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "half", "dozen",
}

_DURATION_UNIT_WORDS = {
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week",
    "weeks", "month", "months", "year", "years", "percent",
}


class ThumbnailValidationError(Exception):
    """Raised when the rendered output cannot be safely accepted."""


def normalize_hook_text(raw: str) -> str:
    """Trim, collapse whitespace, and enforce the hook-text length cap
    with a clean (non-mid-word) truncation."""
    text = " ".join((raw or "").strip().split())
    if len(text) <= MAX_HOOK_TEXT_LENGTH:
        return text
    truncated = text[:MAX_HOOK_TEXT_LENGTH].rsplit(" ", 1)[0]
    return truncated or text[:MAX_HOOK_TEXT_LENGTH]


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def looks_like_isolated_statistic(hook_text: str) -> bool:
    """True when a hook is dominated by a bare number/duration - the kind
    of supporting fact that reads as ambiguous without its surrounding
    context (e.g. "Two Hours Every Night" - two hours of *what*?)."""
    words = set(re.findall(r"[a-z']+", (hook_text or "").lower()))
    has_digit = bool(re.search(r"\d", hook_text or ""))
    has_number_word = bool(words & _NUMBER_WORDS)
    has_duration_unit = bool(words & _DURATION_UNIT_WORDS) or "%" in (hook_text or "")
    return has_digit or has_number_word or has_duration_unit


def has_topical_overlap(hook_text: str, *reference_texts: str) -> bool:
    """True when the hook shares at least one significant word with any of
    the given reference texts (typically the topic/title) - a simple,
    generic lexical-overlap check, not a classifier or an LLM call."""
    hook_words = _significant_words(hook_text)
    if not hook_words:
        return False
    reference_words: set[str] = set()
    for text in reference_texts:
        reference_words |= _significant_words(text)
    return bool(hook_words & reference_words)


def is_ambiguous_supporting_fact_hook(hook_text: str, *reference_texts: str) -> bool:
    """True when a hook looks like an isolated statistic/duration AND has
    no lexical connection to the video's own topic/title - i.e. a viewer
    could not tell what the video is about from the hook text alone."""
    return looks_like_isolated_statistic(hook_text) and not has_topical_overlap(hook_text, *reference_texts)


def resolve_hook_text(
    hook_text: str,
    topic: str,
    metadata_title: str | None = None,
    extra_reference_texts: Sequence[str] = (),
) -> Tuple[str, str | None]:
    """Normalize a candidate hook and, if it reads as an ambiguous isolated
    statistic/duration with no clear connection to the video's own topic/
    title, deterministically replace it with a topic/title-derived hook -
    the standalone-clarity guard: a viewer must be able to tell what the
    video is broadly about from the thumbnail text alone. Never a second
    LLM call - this is a fixed lexical-overlap rule, mirroring the same
    "generic keyword overlap, not a classifier" approach already used by
    src/services/semantic_visual_filter.py for the main visual pipeline.

    Returns:
        (final_hook_text, warning) - warning is None unless the hook was replaced.
    """
    normalized = normalize_hook_text(hook_text)
    reference_texts = (topic, metadata_title or "", *extra_reference_texts)

    if normalized and not is_ambiguous_supporting_fact_hook(normalized, *reference_texts):
        return normalized, None

    fallback = normalize_hook_text(deterministic_hook_from_title_or_topic(metadata_title, topic))
    if not normalized:
        return fallback, None

    warning = (
        f"Hook text '{normalized}' read as an isolated statistic/detail with no clear "
        "connection to the video's topic - replaced with a topic/title-derived hook for "
        "standalone clarity"
    )
    return fallback, warning


def validate_output_image(output_path: str) -> Tuple[int, int]:
    """Open and validate a rendered thumbnail file.

    Returns:
        (width, height)

    Raises:
        ThumbnailValidationError: If the file is missing/empty, not a
            valid image, or not exactly THUMBNAIL_WIDTH x THUMBNAIL_HEIGHT.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ThumbnailValidationError(f"Thumbnail output file is missing or empty: {output_path}")

    try:
        with Image.open(output_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ThumbnailValidationError(f"Thumbnail output file is not a valid image: {e}") from e

    # img.verify() invalidates the file handle - reopen to read attributes.
    try:
        with Image.open(output_path) as img:
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError) as e:
        raise ThumbnailValidationError(f"Thumbnail output file could not be re-read: {e}") from e

    if (width, height) != (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT):
        raise ThumbnailValidationError(
            f"Thumbnail dimensions {width}x{height} do not match required {THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT}"
        )
    if mode not in ("RGB", "RGBA"):
        raise ThumbnailValidationError(f"Unexpected thumbnail image mode: {mode}")

    return width, height
