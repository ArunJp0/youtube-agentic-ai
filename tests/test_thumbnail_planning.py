# Tests for deterministic thumbnail planning fallback and normalization
# helpers (src/services/thumbnail_planning.py). No LLM/network involved.
from __future__ import annotations

from src.services.thumbnail_planning import (
    build_deterministic_thumbnail_plan,
    deterministic_hook_from_topic,
    normalize_composition,
    normalize_text_position,
)


class TestNormalizeComposition:
    def test_valid_values_pass_through(self) -> None:
        assert normalize_composition("subject_left") == "subject_left"
        assert normalize_composition("subject_right") == "subject_right"
        assert normalize_composition("centered") == "centered"

    def test_hyphenated_and_spaced_variants_normalized(self) -> None:
        assert normalize_composition("subject-left") == "subject_left"
        assert normalize_composition("subject right") == "subject_right"

    def test_case_insensitive(self) -> None:
        assert normalize_composition("SUBJECT_LEFT") == "subject_left"

    def test_unrecognized_value_defaults_to_centered(self) -> None:
        assert normalize_composition("bottom_banner") == "centered"

    def test_none_defaults_to_centered(self) -> None:
        assert normalize_composition(None) == "centered"


class TestNormalizeTextPosition:
    def test_valid_values_pass_through(self) -> None:
        assert normalize_text_position("left") == "left"
        assert normalize_text_position("right") == "right"
        assert normalize_text_position("center") == "center"

    def test_unrecognized_value_defaults_to_center(self) -> None:
        assert normalize_text_position("top") == "center"

    def test_none_defaults_to_center(self) -> None:
        assert normalize_text_position(None) == "center"


class TestDeterministicHookFromTopic:
    def test_uses_topic_words_uppercased(self) -> None:
        assert deterministic_hook_from_topic("Why do humans dream?") == "WHY DO HUMANS DREAM"

    def test_caps_at_six_words(self) -> None:
        hook = deterministic_hook_from_topic("one two three four five six seven eight")
        assert len(hook.split()) == 6

    def test_empty_topic_returns_safe_default(self) -> None:
        assert deterministic_hook_from_topic("") == "WATCH NOW"


class TestBuildDeterministicThumbnailPlan:
    def test_plan_is_grounded_in_topic_only(self) -> None:
        plan = build_deterministic_thumbnail_plan("Why do humans dream?", "LLM outage")

        assert plan.used_semantic_planning is False
        assert plan.fallback_reason == "LLM outage"
        assert plan.hook_text == "WHY DO HUMANS DREAM"
        assert plan.search_query == "Why do humans dream?"
        assert plan.composition == "centered"
        assert plan.avoid_concepts == []

    def test_never_invents_claims_beyond_topic(self) -> None:
        plan = build_deterministic_thumbnail_plan("The history of Rome", "outage")
        assert plan.visual_concept == "The history of Rome"
        assert plan.subject == "The history of Rome"
