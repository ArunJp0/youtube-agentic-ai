# Tests for deterministic topic normalization/similarity
# (src/services/topic_normalization.py). No LLM/network calls - pure
# functions only.
from __future__ import annotations

from src.services.topic_normalization import is_duplicate_topic, normalize_topic, topic_similarity


class TestNormalizeTopic:
    def test_lowercases_and_strips_punctuation(self) -> None:
        assert normalize_topic("Why is the Ocean Salty?") == "why is the ocean salty"

    def test_collapses_whitespace(self) -> None:
        assert normalize_topic("Why   is the   ocean salty") == "why is the ocean salty"

    def test_empty_string(self) -> None:
        assert normalize_topic("") == ""

    def test_none_safe(self) -> None:
        assert normalize_topic(None) == ""


class TestTopicSimilarity:
    def test_identical_topics_score_one(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        assert topic_similarity(a, a) == 1.0

    def test_required_example_ocean_salty_variants(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        b = normalize_topic("Why Is Ocean Water Salty")
        assert topic_similarity(a, b) == 2 / 3

    def test_unrelated_topics_score_zero(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        b = normalize_topic("Why do humans dream?")
        assert topic_similarity(a, b) == 0.0

    def test_stopword_only_strings_score_zero(self) -> None:
        a = normalize_topic("Why is the")
        b = normalize_topic("What is the")
        assert topic_similarity(a, b) == 0.0


class TestIsDuplicateTopic:
    def test_exact_match_is_duplicate(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        assert is_duplicate_topic(a, [a]) is True

    def test_near_duplicate_phrasing_is_duplicate(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        b = normalize_topic("Why Is Ocean Water Salty")
        assert is_duplicate_topic(a, [b]) is True

    def test_unrelated_topic_is_not_duplicate(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        b = normalize_topic("Why do humans dream?")
        assert is_duplicate_topic(a, [b]) is False

    def test_empty_history_is_not_duplicate(self) -> None:
        a = normalize_topic("Why is the ocean salty?")
        assert is_duplicate_topic(a, []) is False

    def test_custom_threshold_respected(self) -> None:
        a = normalize_topic("blue ocean waves")
        b = normalize_topic("blue mountain trail")
        # Only "blue" overlaps - low similarity, should not match a strict threshold.
        assert is_duplicate_topic(a, [b], threshold=0.9) is False
