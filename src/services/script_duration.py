# Centralized target-video-duration -> content/word-budget derivation for
# the Script Agent. This is the ONLY place duration-profile minute values
# and the word-budget formula live - every other module (Settings, the
# pipeline builder, ScriptAgent itself) references these, never a second
# hardcoded copy of "5 minutes" or "150 words per minute" scattered
# elsewhere.
#
# Duration must come from appropriate script DEPTH (more/longer body
# sections), never from padding: slowed narration, repeated scenes, or
# repeated script content. This module only decides how many words a
# script should aim for and how that's split across sections - it never
# touches narration pacing/speed, visual selection, or video assembly.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Named profiles (minutes) - a caller picks one via Settings
# (SCRIPT_DURATION_PROFILE), or supplies an explicit
# SCRIPT_TARGET_DURATION_MINUTES override. Extend this dict to add a new
# profile; nothing else in the codebase needs to change.
DURATION_PROFILES: Dict[str, float] = {
    "short": 4.0,  # ~3-5 minutes
    "standard": 6.5,  # ~5-8 minutes - the current next-demo target
    "long": 12.5,  # ~10-15 minutes
}
DEFAULT_DURATION_PROFILE = "standard"

# Matches ScriptAgent.DEFAULT_WORDS_PER_MINUTE - kept here too (imported by
# ScriptAgent, not redefined a second time) since word-budget math and
# duration estimation must always agree on the same speaking rate.
DEFAULT_WORDS_PER_MINUTE = 150.0

# Roughly-fixed structural overhead (hook + introduction + conclusion +
# call-to-action combined, in words) - these stay short regardless of
# target length; only body-section depth and count grow to hit a longer
# target. Derived from this project's own observed typical short-form
# output for these four fixed-short-prompt parts (~20+45+45+25 words).
STRUCTURAL_OVERHEAD_WORDS = 140.0

# Bounds on how the body word budget is split into sections - never fewer
# than MIN_SECTIONS (a script that thin isn't "sections", it's a blob) or
# more than MAX_SECTIONS (avoids fragmenting into too many tiny chapters).
MIN_SECTIONS = 3
MAX_SECTIONS = 16
# The body budget is divided by this to get a REQUESTED section count
# (research availability may still yield fewer real sections - see
# WordBudget.words_per_section, which redistributes the full budget across
# however many sections actually end up available).
TARGET_WORDS_PER_SECTION = 100.0
# A floor under which a section isn't worth writing as its own point -
# used both to bound the requested count and as the redistribution floor.
MIN_WORDS_PER_SECTION = 45


class ScriptDurationError(Exception):
    """Raised for an unrecognized duration profile or a non-positive
    duration override - never silently substitutes a default."""


@dataclass(frozen=True)
class WordBudget:
    """A deterministically-derived content budget for one script.

    ``target_section_count`` is a REQUEST, not a guarantee: ScriptAgent
    still only ever creates as many sections as Research actually provided
    key points for (an existing, unchanged safety property). When fewer
    real sections are available than requested, ``words_per_section``
    redistributes the SAME total body budget across however many sections
    actually exist, so a shorter research result still reaches roughly the
    same target duration via extra depth per section, never by inventing
    extra sections or padding.
    """

    target_duration_minutes: float
    words_per_minute: float
    target_total_words: int
    body_words_budget: int
    target_section_count: int

    def words_per_section(self, actual_section_count: int) -> int:
        count = max(actual_section_count, 1)
        return max(int(round(self.body_words_budget / count)), MIN_WORDS_PER_SECTION)


def resolve_target_duration_minutes(
    profile: Optional[str] = None, override_minutes: Optional[float] = None
) -> float:
    """Resolve the effective target duration in minutes.

    An explicit positive ``override_minutes`` always wins (fine-grained
    control); otherwise the named ``profile`` (default
    ``DEFAULT_DURATION_PROFILE``) is looked up in ``DURATION_PROFILES``.

    Raises:
        ScriptDurationError: If ``profile`` isn't a recognized name and no
            valid override was given.
    """
    if override_minutes is not None and override_minutes > 0:
        return override_minutes

    key = (profile or DEFAULT_DURATION_PROFILE).strip().lower()
    if key not in DURATION_PROFILES:
        raise ScriptDurationError(
            f"Unknown script duration profile '{profile}'. Expected one of "
            f"{sorted(DURATION_PROFILES)}, or a positive override_minutes."
        )
    return DURATION_PROFILES[key]


def calculate_word_budget(
    target_duration_minutes: float, words_per_minute: float = DEFAULT_WORDS_PER_MINUTE
) -> WordBudget:
    """Deterministically derive a requested section count and body word
    budget from a target video duration - never a fixed count/length
    regardless of target, and never a mechanism that relies on padding or
    repetition to reach it.

    Raises:
        ScriptDurationError: If ``target_duration_minutes`` isn't positive.
    """
    if target_duration_minutes <= 0:
        raise ScriptDurationError("target_duration_minutes must be positive")
    if words_per_minute <= 0:
        raise ScriptDurationError("words_per_minute must be positive")

    target_total_words = max(int(round(target_duration_minutes * words_per_minute)), 1)
    body_words_budget = max(
        int(round(target_total_words - STRUCTURAL_OVERHEAD_WORDS)),
        int(MIN_WORDS_PER_SECTION * MIN_SECTIONS),
    )

    requested_section_count = round(body_words_budget / TARGET_WORDS_PER_SECTION)
    target_section_count = max(MIN_SECTIONS, min(MAX_SECTIONS, requested_section_count))

    return WordBudget(
        target_duration_minutes=target_duration_minutes,
        words_per_minute=words_per_minute,
        target_total_words=target_total_words,
        body_words_budget=body_words_budget,
        target_section_count=target_section_count,
    )
