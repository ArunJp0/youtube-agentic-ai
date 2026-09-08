# Tests for deterministic Thumbnail Agent validation
# (src/services/thumbnail_validation.py). No LLM/network involved.
from __future__ import annotations

import pytest
from PIL import Image

from src.services.thumbnail_validation import (
    MAX_HOOK_TEXT_LENGTH,
    ThumbnailValidationError,
    has_topical_overlap,
    is_ambiguous_supporting_fact_hook,
    looks_like_isolated_statistic,
    normalize_hook_text,
    resolve_hook_text,
    validate_output_image,
)


class TestNormalizeHookText:
    def test_trims_and_collapses_whitespace(self) -> None:
        assert normalize_hook_text("  WHY   DO WE  DREAM  ") == "WHY DO WE DREAM"

    def test_empty_stays_empty(self) -> None:
        assert normalize_hook_text("") == ""
        assert normalize_hook_text(None) == ""

    def test_within_limit_unchanged(self) -> None:
        text = "SHORT HOOK"
        assert normalize_hook_text(text) == text

    def test_over_length_truncated_cleanly(self) -> None:
        long_text = "word " * 30
        result = normalize_hook_text(long_text)
        assert len(result) <= MAX_HOOK_TEXT_LENGTH
        assert result == result.rstrip()


class TestValidateOutputImage:
    def test_valid_1280x720_image_passes(self, tmp_path) -> None:
        path = str(tmp_path / "thumb.jpg")
        Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)

        width, height = validate_output_image(path)

        assert (width, height) == (1280, 720)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ThumbnailValidationError, match="missing"):
            validate_output_image(str(tmp_path / "nope.jpg"))

    def test_empty_file_raises(self, tmp_path) -> None:
        path = str(tmp_path / "empty.jpg")
        open(path, "wb").close()
        with pytest.raises(ThumbnailValidationError):
            validate_output_image(path)

    def test_corrupt_file_raises(self, tmp_path) -> None:
        path = str(tmp_path / "corrupt.jpg")
        with open(path, "wb") as f:
            f.write(b"not a real image")
        with pytest.raises(ThumbnailValidationError, match="not a valid image"):
            validate_output_image(path)

    def test_wrong_dimensions_raise(self, tmp_path) -> None:
        path = str(tmp_path / "wrong.jpg")
        Image.new("RGB", (640, 480), (10, 20, 30)).save(path)
        with pytest.raises(ThumbnailValidationError, match="dimensions"):
            validate_output_image(path)


class TestLooksLikeIsolatedStatistic:
    def test_digit_detected(self) -> None:
        assert looks_like_isolated_statistic("300 Years Later") is True

    def test_number_word_detected(self) -> None:
        assert looks_like_isolated_statistic("Two Hours Every Night") is True

    def test_duration_unit_detected(self) -> None:
        assert looks_like_isolated_statistic("A Whole Decade Passed") is False  # "decade" isn't in the unit list
        assert looks_like_isolated_statistic("Every Single Hour") is True

    def test_percent_detected(self) -> None:
        assert looks_like_isolated_statistic("Only 10% Survive") is True

    def test_plain_phrase_not_flagged(self) -> None:
        assert looks_like_isolated_statistic("The Mystery Explained") is False


class TestHasTopicalOverlap:
    def test_shared_word_detected(self) -> None:
        assert has_topical_overlap("WHY DO WE DREAM", "Why do humans dream?") is True

    def test_no_shared_word_returns_false(self) -> None:
        assert has_topical_overlap("Two Hours Every Night", "Why do humans dream?") is False

    def test_overlap_checked_against_multiple_reference_texts(self) -> None:
        assert has_topical_overlap("The Science Explained", "dreams", "The Science of Sleep") is True


class TestIsAmbiguousSupportingFactHook:
    def test_isolated_duration_with_no_topic_overlap_is_ambiguous(self) -> None:
        assert is_ambiguous_supporting_fact_hook("Two Hours Every Night", "Why do humans dream?") is True

    def test_isolated_statistic_for_unrelated_topic_is_ambiguous(self) -> None:
        """Same failure pattern, a completely different topic - proves this
        is a generic rule, not hardcoded to the dreams example."""
        assert is_ambiguous_supporting_fact_hook("300 Years Later", "The history of the Roman Empire") is True

    def test_number_word_hook_with_topic_overlap_is_not_ambiguous(self) -> None:
        """A number is fine when the hook still clearly connects to the topic."""
        assert is_ambiguous_supporting_fact_hook("Five Roman Emperors", "The history of the Roman Empire") is False

    def test_clear_shortened_title_hook_is_not_ambiguous(self) -> None:
        assert is_ambiguous_supporting_fact_hook("WHY DO WE DREAM?", "Why do humans dream?") is False

    def test_clear_contextual_hook_is_not_ambiguous(self) -> None:
        assert is_ambiguous_supporting_fact_hook("THE MYSTERY OF DREAMS", "Why do humans dream?", "dream mystery") is False


class TestResolveHookText:
    def test_clear_hook_passes_through_unchanged(self) -> None:
        final, warning = resolve_hook_text("WHY DO WE DREAM?", "Why do humans dream?")
        assert final == "WHY DO WE DREAM?"
        assert warning is None

    def test_clear_shortened_title_hook_accepted(self) -> None:
        final, warning = resolve_hook_text(
            "Why Do Humans Dream?", "Why do humans dream?", metadata_title="Why Do Humans Dream? The Science of Sleep"
        )
        assert final == "Why Do Humans Dream?"
        assert warning is None

    def test_clear_contextual_hook_accepted(self) -> None:
        final, warning = resolve_hook_text("THE MYSTERY OF SLEEP", "Why do humans dream?", metadata_title="Why Do Humans Dream? The Mystery of Sleep")
        assert final == "THE MYSTERY OF SLEEP"
        assert warning is None

    def test_ambiguous_isolated_statistic_replaced_with_fallback(self) -> None:
        final, warning = resolve_hook_text(
            "Two Hours Every Night", "Why do humans dream?", metadata_title="Why Do Humans Dream? The Science of Sleep"
        )
        assert final != "Two Hours Every Night"
        assert warning is not None
        assert "isolated statistic" in warning.lower()

    def test_ambiguous_duration_hook_replaced_with_fallback(self) -> None:
        final, warning = resolve_hook_text("Every Single Hour", "The history of the Roman Empire")
        assert final != "Every Single Hour"
        assert warning is not None

    def test_fallback_prefers_metadata_title_over_topic(self) -> None:
        final, _ = resolve_hook_text(
            "Two Hours Every Night", "dreams", metadata_title="Why Do Humans Dream? The Science of Sleep"
        )
        assert final == "WHY DO HUMANS DREAM THE SCIENCE"  # first 6 words of the title, uppercased

    def test_fallback_uses_topic_when_no_metadata_title(self) -> None:
        final, _ = resolve_hook_text("Two Hours Every Night", "Why do humans dream?", metadata_title=None)
        assert final == "WHY DO HUMANS DREAM"

    def test_no_topic_specific_hardcoding_generic_across_unrelated_topics(self) -> None:
        """The exact same rule/function applied to a completely unrelated
        topic must behave the same way - proves genericity."""
        final, warning = resolve_hook_text("50 Percent Faster", "How volcanoes erupt", metadata_title="How Volcanoes Erupt Explained")
        assert final != "50 Percent Faster"
        assert warning is not None
        assert "volcano" not in warning.lower()  # the warning message itself is generic, not topic-specific text

    def test_empty_hook_falls_back_silently_without_warning(self) -> None:
        final, warning = resolve_hook_text("   ", "Why do humans dream?")
        assert final == "WHY DO HUMANS DREAM"
        assert warning is None
