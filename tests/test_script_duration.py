# Deterministic tests for src.services.script_duration - the centralized
# target-duration -> content/word-budget derivation used by ScriptAgent. No
# LLM/network involved anywhere in this file.
from __future__ import annotations

import pytest

from src.services.script_duration import (
    DURATION_PROFILES,
    MAX_SECTIONS,
    MIN_SECTIONS,
    MIN_WORDS_PER_SECTION,
    ScriptDurationError,
    WordBudget,
    calculate_word_budget,
    resolve_target_duration_minutes,
)


class TestResolveTargetDurationMinutes:
    def test_default_profile_is_standard(self) -> None:
        assert resolve_target_duration_minutes() == DURATION_PROFILES["standard"]

    def test_named_profile_resolves_to_its_value(self) -> None:
        assert resolve_target_duration_minutes(profile="short") == DURATION_PROFILES["short"]
        assert resolve_target_duration_minutes(profile="long") == DURATION_PROFILES["long"]

    def test_profile_name_is_case_insensitive(self) -> None:
        assert resolve_target_duration_minutes(profile="STANDARD") == DURATION_PROFILES["standard"]

    def test_explicit_override_wins_over_profile(self) -> None:
        assert resolve_target_duration_minutes(profile="short", override_minutes=9.0) == 9.0

    def test_zero_or_negative_override_falls_back_to_profile(self) -> None:
        assert resolve_target_duration_minutes(profile="short", override_minutes=0) == DURATION_PROFILES["short"]
        assert resolve_target_duration_minutes(profile="short", override_minutes=-5) == DURATION_PROFILES["short"]

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ScriptDurationError):
            resolve_target_duration_minutes(profile="ultra-mega-long")


class TestCalculateWordBudget:
    def test_returns_word_budget_instance(self) -> None:
        budget = calculate_word_budget(6.5)
        assert isinstance(budget, WordBudget)
        assert budget.target_duration_minutes == 6.5

    def test_longer_target_never_produces_a_smaller_body_budget(self) -> None:
        short_budget = calculate_word_budget(DURATION_PROFILES["short"])
        standard_budget = calculate_word_budget(DURATION_PROFILES["standard"])
        long_budget = calculate_word_budget(DURATION_PROFILES["long"])

        assert short_budget.body_words_budget <= standard_budget.body_words_budget
        assert standard_budget.body_words_budget <= long_budget.body_words_budget

    def test_longer_target_never_produces_fewer_requested_sections(self) -> None:
        short_budget = calculate_word_budget(DURATION_PROFILES["short"])
        standard_budget = calculate_word_budget(DURATION_PROFILES["standard"])

        assert short_budget.target_section_count <= standard_budget.target_section_count

    def test_section_count_bounded_within_min_and_max(self) -> None:
        tiny_budget = calculate_word_budget(0.5)
        huge_budget = calculate_word_budget(60.0)

        assert MIN_SECTIONS <= tiny_budget.target_section_count <= MAX_SECTIONS
        assert MIN_SECTIONS <= huge_budget.target_section_count <= MAX_SECTIONS
        assert huge_budget.target_section_count == MAX_SECTIONS

    def test_standard_profile_targets_the_5_to_8_minute_demo_range(self) -> None:
        """The concrete regression this milestone exists to fix: the
        DEFAULT profile actually targets the requested next-demo range."""
        budget = calculate_word_budget(DURATION_PROFILES["standard"])
        implied_minutes = budget.target_total_words / budget.words_per_minute
        assert 5.0 <= implied_minutes <= 8.0

    def test_non_positive_duration_raises(self) -> None:
        with pytest.raises(ScriptDurationError):
            calculate_word_budget(0)
        with pytest.raises(ScriptDurationError):
            calculate_word_budget(-3.0)

    def test_non_positive_words_per_minute_raises(self) -> None:
        with pytest.raises(ScriptDurationError):
            calculate_word_budget(5.0, words_per_minute=0)


class TestWordBudgetRedistribution:
    """words_per_section must redistribute the SAME total body budget
    across however many sections actually end up available - depth
    compensates for fewer real sections, never invented extra sections."""

    def test_fewer_actual_sections_increase_words_per_section(self) -> None:
        budget = calculate_word_budget(6.5)
        fewer = budget.words_per_section(actual_section_count=3)
        more_sections = budget.words_per_section(actual_section_count=budget.target_section_count)
        assert fewer > more_sections

    def test_words_per_section_times_count_approximates_body_budget(self) -> None:
        budget = calculate_word_budget(6.5)
        for count in (3, 5, budget.target_section_count):
            per_section = budget.words_per_section(count)
            total = per_section * count
            # Rounding means this is approximate, not exact.
            assert abs(total - budget.body_words_budget) <= per_section

    def test_never_below_the_minimum_words_per_section_floor(self) -> None:
        budget = calculate_word_budget(4.0)
        # An unrealistically large section count must still floor out
        # rather than produce a near-empty "section".
        assert budget.words_per_section(actual_section_count=1000) >= MIN_WORDS_PER_SECTION

    def test_single_section_gets_the_whole_budget(self) -> None:
        budget = calculate_word_budget(6.5)
        assert budget.words_per_section(actual_section_count=1) == budget.body_words_budget

    def test_zero_actual_sections_does_not_raise(self) -> None:
        budget = calculate_word_budget(6.5)
        assert budget.words_per_section(actual_section_count=0) >= MIN_WORDS_PER_SECTION
