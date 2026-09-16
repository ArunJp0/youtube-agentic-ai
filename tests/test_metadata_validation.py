# Tests for deterministic Metadata Agent normalization/validation
# (src/services/metadata_validation.py). No LLM/network involved.
from __future__ import annotations

import pytest

from src.models.metadata import Chapter
from src.services.metadata_validation import (
    DESCRIPTION_HARD_MAX_WORDS,
    DESCRIPTION_TARGET_MAX_WORDS,
    DESCRIPTION_TARGET_MIN_WORDS,
    MAX_HASHTAGS,
    MAX_TAGS,
    MAX_TITLE_LENGTH,
    TITLE_HARD_MAX_CHARS,
    TITLE_TARGET_MAX_CHARS,
    TITLE_TARGET_MIN_CHARS,
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
        long_title = "word " * 40  # far over the house-style hard max
        result = normalize_title(long_title)
        assert len(result) <= TITLE_HARD_MAX_CHARS
        assert not result.endswith(" ")
        assert result == result.rstrip()

    def test_truncation_does_not_cut_mid_word(self) -> None:
        long_title = "Supercalifragilisticexpialidocious " * 5
        result = normalize_title(long_title)
        assert len(result) <= TITLE_HARD_MAX_CHARS
        # Every remaining "word" should be a full word from the source.
        assert all(word in long_title for word in result.split())


class TestNormalizeTitleHouseStyle:
    """New house-style constraints: target 45-60 chars, hard max 65 -
    repaired by dropping a trailing subtitle before falling back to
    word-boundary truncation, never a topic-specific hack."""

    def test_title_at_or_under_hard_max_passes_through_unchanged(self) -> None:
        title = "Why Does Ice Float Instead of Sink?"
        assert len(title) <= TITLE_HARD_MAX_CHARS
        assert normalize_title(title) == title

    def test_title_in_preferred_target_range_unchanged(self) -> None:
        # A realistic hook-style title landing inside the 45-60 char target.
        title = "Why Does Ice Float on Water Instead of Sinking?"
        assert TITLE_TARGET_MIN_CHARS <= len(title) <= TITLE_TARGET_MAX_CHARS
        assert normalize_title(title) == title

    def test_hard_max_is_always_enforced(self) -> None:
        long_title = (
            "Why Does Ice Float Instead of Sinking to the Bottom of the Ocean Floor Every Single Time It Forms"
        )
        result = normalize_title(long_title)
        assert len(result) <= TITLE_HARD_MAX_CHARS

    def test_over_length_title_repaired_by_dropping_colon_subtitle(self) -> None:
        title = "Why Does Ice Float on Water: The Fascinating Science Explained in Full Detail"
        assert len(title) > TITLE_HARD_MAX_CHARS
        result = normalize_title(title)
        assert result == "Why Does Ice Float on Water"
        assert len(result) <= TITLE_HARD_MAX_CHARS

    def test_over_length_title_repaired_by_dropping_em_dash_subtitle(self) -> None:
        title = "Why Does Ice Float on Water — A Deep Dive Into the Real Science Behind It"
        assert len(title) > TITLE_HARD_MAX_CHARS
        result = normalize_title(title)
        assert result == "Why Does Ice Float on Water"

    def test_thin_subtitle_prefix_is_not_used_alone(self) -> None:
        # The prefix before ':' ("Ice") is too thin (<3 words) to stand as
        # a title on its own, so repair falls through to word-boundary
        # truncation of the whole title instead of collapsing to "Ice".
        title = "Ice: " + ("Really Amazing Frozen Water Facts Explained In Great Detail " * 2)
        result = normalize_title(title)
        assert len(result) <= TITLE_HARD_MAX_CHARS
        assert result != "Ice"
        assert result.startswith("Ice")

    def test_repair_never_cuts_mid_word(self) -> None:
        long_title = "Supercalifragilisticexpialidocious " * 5
        result = normalize_title(long_title)
        assert len(result) <= TITLE_HARD_MAX_CHARS
        assert all(word in long_title for word in result.split())

    def test_mid_word_hyphen_never_treated_as_subtitle_separator(self) -> None:
        # If the mid-word hyphen in "e-commerce" were incorrectly matched
        # as a subtitle separator, this would wrongly collapse to "The
        # Complete Guide to e" - the regex requires whitespace around the
        # separator precisely to prevent that.
        title = "The Complete Guide to e-commerce Growth Strategies for Modern Businesses Everywhere"
        assert len(title) > TITLE_HARD_MAX_CHARS
        result = normalize_title(title)
        assert result != "The Complete Guide to e"
        assert "e-commerce" in result
        assert len(result) <= TITLE_HARD_MAX_CHARS

    def test_title_remains_meaningful_and_topic_related_after_repair(self) -> None:
        title = "Why Do Volcanoes Erupt: A Complete Scientific Breakdown of Magma and Pressure"
        result = normalize_title(title)
        assert "Volcanoes" in result
        assert "Erupt" in result
        assert len(result) <= TITLE_HARD_MAX_CHARS


class TestNormalizeDescription:
    def test_trims_whitespace(self) -> None:
        assert normalize_description("  Hello world  ") == "Hello world"

    def test_empty_description_stays_empty(self) -> None:
        assert normalize_description(None) == ""

    def test_within_limit_unchanged(self) -> None:
        text = "A perfectly normal description."
        assert normalize_description(text) == text


class TestNormalizeDescriptionHouseStyle:
    """New house-style constraints: target 60-90 words, hard max 110 -
    repaired by dropping whole trailing sentences (paragraph-aware), never
    a mid-sentence cut."""

    def test_description_at_or_under_hard_max_unchanged(self) -> None:
        text = "This is a normal, reasonably short video description. " * 3
        text = text.strip()
        assert len(text.split()) <= DESCRIPTION_HARD_MAX_WORDS
        assert normalize_description(text) == text

    def test_description_in_preferred_target_range_unchanged(self) -> None:
        sentence = "This sentence has exactly eight words in it."
        text = " ".join([sentence] * 9)  # 9 * 8 = 72 words, inside 60-90
        assert DESCRIPTION_TARGET_MIN_WORDS <= len(text.split()) <= DESCRIPTION_TARGET_MAX_WORDS
        assert normalize_description(text) == text

    def test_hard_max_is_always_enforced(self) -> None:
        sentence = "This sentence has exactly eight words in it."
        text = " ".join([sentence] * 30)  # 240 words, far over the cap
        result = normalize_description(text)
        assert len(result.split()) <= DESCRIPTION_HARD_MAX_WORDS

    def test_repair_keeps_only_whole_sentences_never_cuts_mid_sentence(self) -> None:
        sentence = "This sentence has exactly eight words in it."
        text = " ".join([sentence] * 30)
        result = normalize_description(text)
        assert result.endswith(".")
        # 110 // 8 == 13 whole sentences fit (104 words); a 14th would be 112 > 110.
        assert result == " ".join([sentence] * 13)
        assert len(result.split()) == 104

    def test_repair_preserves_paragraph_breaks(self) -> None:
        sentence = "This sentence has exactly eight words in it."
        paragraph_one = " ".join([sentence] * 10)  # 80 words
        paragraph_two = " ".join([sentence] * 10)  # 80 words
        text = f"{paragraph_one}\n\n{paragraph_two}"
        result = normalize_description(text)
        assert len(result.split()) <= DESCRIPTION_HARD_MAX_WORDS
        assert "\n\n" in result

    def test_single_oversized_sentence_falls_back_to_word_truncation(self) -> None:
        huge_sentence = ("word " * 200).strip()  # one "sentence", no punctuation at all
        result = normalize_description(huge_sentence)
        assert len(result.split()) <= DESCRIPTION_HARD_MAX_WORDS

    def test_does_not_reproduce_a_large_script_passage(self) -> None:
        """A description built by naively concatenating many long narration
        sentences must be safely cut down to size, never passed through as
        a near-transcript."""
        narration_sentences = [f"The narrator explains fact number {i} about the topic in detail." for i in range(20)]
        long_description = " ".join(narration_sentences)
        result = normalize_description(long_description)
        assert len(result.split()) <= DESCRIPTION_HARD_MAX_WORDS
        assert result != long_description


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
