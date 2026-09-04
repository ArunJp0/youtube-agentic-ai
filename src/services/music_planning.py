# Deterministic fallback music plan - used directly when no LLM planner is
# configured, and internally by MusicContextPlanner when its single LLM
# call fails or returns something unusable. Exactly one fallback
# implementation (mirrors src/services/query_generation.py's role for
# visual planning), so the two "no semantic plan available" call sites
# never drift apart.
from __future__ import annotations

from src.models.music import MusicPlan
from src.models.script import ScriptResult

# A safe, generic profile that suits almost any narrated educational/
# explainer video without knowing anything topic-specific: quiet, restrained,
# unlikely to distract from or clash with narration.
FALLBACK_PRIMARY_MOOD = "neutral"
FALLBACK_SECONDARY_MOOD = "calm"
FALLBACK_ENERGY_LEVEL = "low"
FALLBACK_GENRES = ["ambient", "cinematic"]
FALLBACK_INSTRUMENTATION = ["piano", "soft synth", "strings"]
FALLBACK_AVOID_STYLES = ["aggressive", "lyrics", "heavy drums", "distorted"]


def build_deterministic_music_plan(topic: str, script: ScriptResult) -> MusicPlan:
    """Build the same safe, generic MusicPlan regardless of ``script``
    content - deliberately not topic-aware, since the whole point is a
    dependable fallback when semantic planning isn't available."""
    return MusicPlan(
        topic=topic,
        primary_mood=FALLBACK_PRIMARY_MOOD,
        secondary_mood=FALLBACK_SECONDARY_MOOD,
        energy_level=FALLBACK_ENERGY_LEVEL,
        preferred_genres=list(FALLBACK_GENRES),
        preferred_instrumentation=list(FALLBACK_INSTRUMENTATION),
        avoid_styles=list(FALLBACK_AVOID_STYLES),
        requires_neutral_subtle=True,
        reasoning_summary=(
            "Deterministic fallback: neutral, subtle background music suitable for any "
            "narrated topic when semantic mood planning is unavailable."
        ),
        used_semantic_planning=False,
    )
