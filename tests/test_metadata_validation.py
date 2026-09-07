# Tests for deterministic Metadata Agent normalization/validation
# (src/services/metadata_validation.py). No LLM/network involved.
from __future__ import annotations

import pytest

from src.models.metadata import Chapter
from src.services.metadata_validation import (
    MAX_HASHTAGS,
    MAX_TAGS,
    MAX_TITLE_LENGTH,
    ChapterValidationError,
    format_youtube_timestamp,
    normalize_description,
    normalize_hashtags,
    normalize_tags,
    normalize_title,
    validate_and_clean_chapters,
)


def _chapter(seconds: float, title: str = "Chapter") -> Chapter:
    return Chapter(timestamp_seconds=seconds, timestamp_text=format_youtube_timestamp(seconds), title=title)


class TestNormalizeTitle:
    def test_trims_and_collapses_whitespace(self) -> None:
        assert normalize_title("  Why   Do Humans  Dream?  ") == "Why Do Humans Dream?"

    def test_strips_wrapping_quotes(self) -> None:
        assert normalize_title('"Why Do Humans Dream?"') == "Why Do Humans Dream?"

    def test_empty_title_stays_empty(self) -> None:
        assert normalize_title("   ") == ""

    def test_title_within_limit_unchanged(self) -> None:
        title = "A short valid title"
        assert normalize_title(title) == title

    def test_over_length_title_truncated_cleanly(self) -> None:
        long_title = "word " * 40  # far over MAX_TITLE_LENGTH
        result = normalize_title(long_title)
        assert len(result) <= MAX_TITLE_LENGTH
        assert not result.endswith(" ")
        assert result == result.rstrip()

    def test_truncation_does_not_cut_mid_word(self) -> None:
        long_title = "Supercalifragilisticexpialidocious " * 5
        result = normalize_title(long_title)
        assert len(result) <= MAX_TITLE_LENGTH
        # Every remaining "word" should be a full word from the source.
        assert all(word in long_title for word in result.split())


class TestNormalizeDescription:
    def test_trims_whitespace(self) -> None:
        assert normalize_description("  Hello world  ") == "Hello world"

    def test_empty_description_stays_empty(self) -> None:
        assert normalize_description(None) == ""

    def test_within_limit_unchanged(self) -> None:
        text = "A perfectly normal description."
        assert normalize_description(text) == text


class TestNormalizeTags:
    def test_deduplicates_case_insensitively_keeping_first_casing(self) -> None:
        tags = normalize_tags(["Dreams", "dreams", "DREAMS", "Sleep"])
        assert tags == ["Dreams", "Sleep"]

    def test_drops_empty_entries(self) -> None:
        assert normalize_tags(["dreams", "  ", "", "sleep"]) == ["dreams", "sleep"]

    def test_collapses_internal_whitespace(self) -> None:
        assert normalize_tags(["  rem   sleep  "]) == ["rem sleep"]

    def test_caps_total_count(self) -> None:
        many_tags = [f"tag{i}" for i in range(MAX_TAGS + 10)]
        assert len(normalize_tags(many_tags)) == MAX_TAGS

    def test_caps_total_character_budget(self) -> None:
        huge_tags = ["x" * 100 for _ in range(10)]  # 1000 chars, over budget
        result = normalize_tags(huge_tags)
        assert sum(len(t) for t in result) <= 460

    def test_empty_input(self) -> None:
        assert normalize_tags([]) == []
        assert normalize_tags(None) == []


class TestNormalizeHashtags:
    def test_adds_missing_hash_prefix(self) -> None:
        assert normalize_hashtags(["dreams"]) == ["#dreams"]

    def test_keeps_existing_hash_prefix(self) -> None:
        assert normalize_hashtags(["#dreams"]) == ["#dreams"]

    def test_deduplicates_case_insensitively(self) -> None:
        assert normalize_hashtags(["#Dreams", "#dreams"]) == ["#Dreams"]

    def test_caps_to_max_hashtags(self) -> None:
        many = [f"#tag{i}" for i in range(MAX_HASHTAGS + 5)]
        assert len(normalize_hashtags(many)) == MAX_HASHTAGS

    def test_bare_hash_dropped(self) -> None:
        assert normalize_hashtags(["#", "  "]) == []

    def test_strips_internal_whitespace(self) -> None:
        assert normalize_hashtags(["dream science"]) == ["#dreamscience"]


class TestFormatYoutubeTimestamp:
    def test_zero_seconds(self) -> None:
        assert format_youtube_timestamp(0.0) == "0:00"

    def test_minutes_and_seconds(self) -> None:
        assert format_youtube_timestamp(135.0) == "2:15"

    def test_hours_included_when_over_an_hour(self) -> None:
        assert format_youtube_timestamp(3725.0) == "1:02:05"

    def test_negative_clamped_to_zero(self) -> None:
        assert format_youtube_timestamp(-5.0) == "0:00"


class TestValidateAndCleanChapters:
    def test_valid_chapters_pass_through_unchanged(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(30.0, "Middle"), _chapter(60.0, "End")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=90.0)
        assert [c.title for c in cleaned] == ["Intro", "Middle", "End"]
        assert warnings == []

    def test_first_chapter_not_at_zero_raises(self) -> None:
        chapters = [_chapter(5.0, "Intro"), _chapter(30.0, "Middle")]
        with pytest.raises(ChapterValidationError):
            validate_and_clean_chapters(chapters, duration_seconds=60.0)

    def test_empty_label_dropped(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(30.0, "   "), _chapter(60.0, "End")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=90.0)
        assert [c.title for c in cleaned] == ["Intro", "End"]
        assert len(warnings) == 1

    def test_duplicate_timestamp_dropped(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(0.0, "Duplicate"), _chapter(30.0, "End")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=60.0)
        assert [c.title for c in cleaned] == ["Intro", "End"]
        assert any("increasing" in w for w in warnings)

    def test_non_increasing_timestamp_dropped(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(20.0, "A"), _chapter(10.0, "B")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=60.0)
        assert [c.title for c in cleaned] == ["Intro", "A"]

    def test_timestamp_beyond_duration_dropped(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(30.0, "Middle"), _chapter(100.0, "TooLate")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=90.0)
        assert [c.title for c in cleaned] == ["Intro", "Middle"]
        assert any("duration" in w for w in warnings)

    def test_timestamp_equal_to_duration_dropped(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(90.0, "AtEnd")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=90.0)
        assert [c.title for c in cleaned] == ["Intro"]

    def test_no_duration_skips_duration_check(self) -> None:
        chapters = [_chapter(0.0, "Intro"), _chapter(999999.0, "Later")]
        cleaned, warnings = validate_and_clean_chapters(chapters, duration_seconds=None)
        assert [c.title for c in cleaned] == ["Intro", "Later"]

    def test_all_chapters_invalid_raises(self) -> None:
        chapters = [_chapter(0.0, "   ")]
        with pytest.raises(ChapterValidationError):
            validate_and_clean_chapters(chapters, duration_seconds=60.0)

    def test_empty_chapter_list_raises(self) -> None:
        with pytest.raises(ChapterValidationError):
            validate_and_clean_chapters([], duration_seconds=60.0)
