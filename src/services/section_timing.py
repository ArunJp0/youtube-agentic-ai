# Deterministic section-duration calculation, shared by VisualMediaService
# (to plan how many visual slots a section needs) and VideoAssemblyService
# (to know how long each section's clips should play for). Kept in one
# place so both derive timing from the exact same rule - no LLM call.
from __future__ import annotations

from typing import List

from src.models.script import ScriptSection


def calculate_section_durations(
    sections: List[ScriptSection], total_duration_seconds: float
) -> List[float]:
    """Deterministically split ``total_duration_seconds`` across sections.

    Each section's share is proportional to its narration word count
    relative to the combined word count of all sections. The last section
    absorbs any rounding drift so the durations always sum to exactly
    ``total_duration_seconds``.

    Args:
        sections: Script sections, in order
        total_duration_seconds: The narration audio's total duration - the
            authoritative timeline for the whole video

    Returns:
        One duration (seconds) per section, in the same order, summing to
        exactly ``total_duration_seconds``
    """
    word_counts = [max(len(section.narration.split()), 1) for section in sections]
    total_words = sum(word_counts)

    durations = [total_duration_seconds * (count / total_words) for count in word_counts]

    if durations:
        drift = total_duration_seconds - sum(durations)
        durations[-1] += drift

    return durations
