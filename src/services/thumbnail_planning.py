# Deterministic fallback thumbnail plan and composition/text-position
# normalization - used directly when no LLM planner is configured, and
# internally by ThumbnailPlanner when its single LLM call fails or returns
# something unusable. Exactly one fallback implementation, mirroring
# src/services/music_planning.py's role for BGM mood planning.
from __future__ import annotations

import re

from src.models.thumbnail import Composition, TextPosition, ThumbnailPlan

VALID_COMPOSITIONS = {"subject_left", "subject_right", "centered"}
VALID_TEXT_POSITIONS = {"left", "right", "center"}
DEFAULT_COMPOSITION: Composition = "centered"
DEFAULT_TEXT_POSITION: TextPosition = "center"

# Deliberately generous - the renderer itself further caps/wraps text
# safely regardless of this value (see thumbnail_validation.py's stricter
# character cap, applied on top of this word cap).
MAX_DETERMINISTIC_HOOK_WORDS = 6


def normalize_composition(raw) -> Composition:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in VALID_COMPOSITIONS else DEFAULT_COMPOSITION


def normalize_text_position(raw) -> TextPosition:
    value = str(raw or "").strip().lower()
    return value if value in VALID_TEXT_POSITIONS else DEFAULT_TEXT_POSITION


def deterministic_hook_from_topic(topic: str) -> str:
    """A safe, never-misleading hook: the topic's own words, since a topic
    is definitionally accurate for its own video. Never invents a claim."""
    words = re.findall(r"[A-Za-z0-9']+", topic)[:MAX_DETERMINISTIC_HOOK_WORDS]
    if not words:
        return topic.strip().upper() or "WATCH NOW"
    return " ".join(w.upper() for w in words)


def deterministic_hook_from_title_or_topic(metadata_title: str | None, topic: str) -> str:
    """A safe, on-topic hook derived from whichever real content is
    available, preferring the already-generated video title (more
    specific/polished) and falling back to the raw topic. Used both as the
    top-level fallback when no LLM planner is configured, and as the
    repair target when a semantically-planned hook reads as an ambiguous
    isolated fact (see thumbnail_validation.resolve_hook_text)."""
    source = metadata_title.strip() if metadata_title and metadata_title.strip() else topic
    return deterministic_hook_from_topic(source)


def build_deterministic_thumbnail_plan(topic: str, fallback_reason: str) -> ThumbnailPlan:
    """Build the same safe, generic ThumbnailPlan regardless of script
    content - deliberately not topic-specific beyond using the topic's own
    words for the hook, since that's the only way to guarantee accuracy
    without an LLM."""
    return ThumbnailPlan(
        hook_text=deterministic_hook_from_topic(topic),
        visual_concept=topic,
        search_query=topic,
        mood="neutral",
        subject=topic,
        composition=DEFAULT_COMPOSITION,
        text_position=DEFAULT_TEXT_POSITION,
        avoid_concepts=[],
        used_semantic_planning=False,
        fallback_reason=fallback_reason,
    )
