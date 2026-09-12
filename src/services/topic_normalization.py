# Deterministic topic normalization/similarity - no LLM call. Used to
# recognize superficial variations of the same topic (different phrasing,
# word order, capitalization, punctuation) as duplicates, e.g. "Why is the
# ocean salty?" vs "Why Is Ocean Water Salty" - never relies on exact
# string equality alone.
from __future__ import annotations

import re
from typing import Iterable, Set

_WORD_RE = re.compile(r"[a-z0-9]+")

# Generic English function/question words - not topic-specific. Stripping
# these before comparison is what lets word-order/phrasing differences
# ("ocean salty" vs "ocean water salty") still overlap on their real
# content words instead of being swamped by shared boilerplate ("why",
# "is", "the").
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
        "why", "how", "what", "when", "where", "who", "which", "will",
        "can", "could", "should", "would", "to", "of", "in", "on", "for",
        "and", "or", "but", "this", "that", "these", "those", "it", "its",
        "be", "been", "being", "with", "from", "as", "at", "by",
    }
)

# Jaccard similarity (over significant words) at/above which two topics are
# treated as the same underlying subject. Calibrated so "why is the ocean
# salty" vs "why is ocean water salty" (2/3 = 0.667 shared significant
# words) is caught, while genuinely distinct topics (0 overlap) are not.
DUPLICATE_SIMILARITY_THRESHOLD = 0.6


def normalize_topic(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - a stable,
    human-readable normalized form (used for exact-match comparison and as
    TopicCandidate.normalized_title)."""
    words = _WORD_RE.findall((text or "").lower())
    return " ".join(words)


def _significant_words(normalized_text: str) -> Set[str]:
    return {w for w in normalized_text.split() if w not in _STOPWORDS and len(w) > 1}


def topic_similarity(a: str, b: str) -> float:
    """Jaccard similarity of significant (non-stopword) words between two
    ALREADY-NORMALIZED topic strings. 0.0 if either has no significant
    words at all."""
    words_a = _significant_words(a)
    words_b = _significant_words(b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def is_duplicate_topic(
    normalized_title: str, history: Iterable[str], threshold: float = DUPLICATE_SIMILARITY_THRESHOLD
) -> bool:
    """True if ``normalized_title`` is the same or a near-duplicate of any
    entry in ``history`` (also expected already-normalized) - exact match
    is always caught (similarity 1.0); near-duplicates are caught via
    ``topic_similarity`` against the configured threshold."""
    for past in history:
        if normalized_title == past:
            return True
        if topic_similarity(normalized_title, past) >= threshold:
            return True
    return False
