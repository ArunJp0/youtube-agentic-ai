# Deterministic thumbnail image selection: picks one candidate from an
# already-searched MediaCandidate list against a ThumbnailPlan's
# avoid_concepts. No LLM call, no network - a fixed keyword-overlap rule,
# consistent with src/services/semantic_visual_filter.py's approach for
# the main visual pipeline (kept as its own small function here rather
# than reused directly, since that one is typed to SectionVisualPlan).
from __future__ import annotations

from typing import List, Optional, Tuple

from src.tools.media_provider import MediaCandidate


def select_thumbnail_candidate(
    candidates: List[MediaCandidate], avoid_concepts: List[str]
) -> Tuple[Optional[MediaCandidate], List[str]]:
    """Select the first candidate whose available content metadata doesn't
    clearly match one of ``avoid_concepts``.

    Never raises - an empty candidate list is reported via the returned
    ``(None, warnings)`` rather than an exception. If every candidate
    matches an avoid concept, the first candidate is used anyway as a
    best-available last resort (matching this project's established
    "keep something rather than nothing" fallback philosophy), and that
    choice is recorded as a warning rather than silently accepted.

    Args:
        candidates: Search results, best match first
        avoid_concepts: Literal-but-wrong interpretations to steer away from

    Returns:
        (selected candidate or None, list of warnings/explanations)
    """
    if not candidates:
        return None, ["No candidate images found"]

    avoid_terms = [a.strip().lower() for a in avoid_concepts if a and a.strip()]
    if not avoid_terms:
        return candidates[0], []

    for candidate in candidates:
        hint = (candidate.content_hint or "").strip().lower()
        if hint and any(term in hint for term in avoid_terms):
            continue
        return candidate, []

    return candidates[0], ["Every candidate matched an avoid_concept; used the best-available match as a last resort"]
