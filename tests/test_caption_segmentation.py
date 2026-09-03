# Tests for deterministic caption segmentation/readability rules
# (src.services.caption_segmentation). Pure functions only - no network,
# no transcription engine, no FFmpeg.
from __future__ import annotations

from src.services.caption_segmentation import (
    DURATION_TOLERANCE_SECONDS,
    MAX_CHARS_PER_LINE,
    MAX_CHARS_PER_SEGMENT,
    MAX_LINES_PER_SEGMENT,
    MAX_READING_CHARS_PER_SECOND,
    MAX_SEGMENT_DURATION_SECONDS,
    MIN_SEGMENT_DURATION_SECONDS,
    build_caption_segments,
)
from src.tools.transcription_provider import TranscribedSegment, TranscribedWord


def _seg(text: str, start: float, end: float, words=None) -> TranscribedSegment:
    return TranscribedSegment(text=text, start_seconds=start, end_seconds=end, words=words or [])


def _word(word: str, start: float, end: float) -> TranscribedWord:
    return TranscribedWord(word=word, start_seconds=start, end_seconds=end)


class TestEmptyAndDegenerateInput:
    def test_empty_input_returns_empty(self) -> None:
        assert build_caption_segments([]) == []

    def test_all_empty_text_segments_dropped(self) -> None:
        raw = [_seg("", 0.0, 1.0), _seg("   ", 1.0, 2.0)]
        assert build_caption_segments(raw) == []

    def test_mixed_empty_and_real_segments(self) -> None:
        raw = [_seg("", 0.0, 1.0), _seg("   ", 1.0, 2.0), _seg("Real text.", 2.0, 4.0)]
        result = build_caption_segments(raw)
        assert len(result) == 1
        assert result[0].text == "Real text."


class TestBasicPassthrough:
    def test_single_short_segment(self) -> None:
        raw = [_seg("Hello there.", 0.0, 2.0)]
        result = build_caption_segments(raw)
        assert len(result) == 1
        assert result[0].start_seconds == 0.0
        assert result[0].end_seconds == 2.0
        assert result[0].text == "Hello there."

    def test_index_is_one_based_sequential(self) -> None:
        raw = [_seg("First.", 0.0, 2.0), _seg("Second.", 2.0, 4.0), _seg("Third.", 4.0, 6.0)]
        result = build_caption_segments(raw)
        assert [s.index for s in result] == list(range(1, len(result) + 1))

    def test_deterministic(self) -> None:
        raw = [_seg("First one.", 0.0, 2.0), _seg("Second one.", 2.0, 4.0)]
        first_run = build_caption_segments(raw)
        second_run = build_caption_segments(raw)
        assert [(s.start_seconds, s.end_seconds, s.text) for s in first_run] == [
            (s.start_seconds, s.end_seconds, s.text) for s in second_run
        ]


class TestTimestampOrdering:
    def test_multiple_segments_are_chronological_and_non_overlapping(self) -> None:
        raw = [_seg("First segment.", 0.0, 2.0), _seg("Second segment.", 2.0, 4.0), _seg("Third segment.", 4.0, 6.0)]
        result = build_caption_segments(raw)
        for a, b in zip(result, result[1:]):
            assert a.end_seconds <= b.start_seconds

    def test_starts_are_monotonically_non_decreasing(self) -> None:
        raw = [_seg("First segment.", 0.0, 2.0), _seg("Second segment.", 2.0, 4.0), _seg("Third segment.", 4.0, 6.0)]
        result = build_caption_segments(raw)
        starts = [s.start_seconds for s in result]
        assert starts == sorted(starts)

    def test_overlapping_raw_timestamps_are_resolved(self) -> None:
        """A transcription glitch where segment 2 claims to start before
        segment 1 ends must be clamped, not passed through as an overlap."""
        raw = [_seg("First segment here.", 0.0, 2.0), _seg("Second segment here.", 1.5, 3.5)]
        result = build_caption_segments(raw)
        assert len(result) == 2
        assert result[1].start_seconds >= result[0].end_seconds
        assert result[1].start_seconds == result[0].end_seconds  # clamped forward exactly, no gap invented


class TestDurationBoundaries:
    def test_minimum_duration_enforced_for_very_short_segment(self) -> None:
        raw = [_seg("Hi.", 0.0, 0.2)]
        result = build_caption_segments(raw)
        assert result[0].end_seconds - result[0].start_seconds >= MIN_SEGMENT_DURATION_SECONDS

    def test_reading_speed_floor_extends_duration_for_long_text(self) -> None:
        # Short enough to stay a single chunk (< MAX_CHARS_PER_SEGMENT) so
        # the reading-speed floor applies to this exact text's length.
        text = "word " * 12 + "end."
        assert len(text) <= MAX_CHARS_PER_SEGMENT
        expected_floor = len(text) / MAX_READING_CHARS_PER_SECOND
        raw = [_seg(text, 0.0, 0.5)]  # implausibly short natural duration
        result = build_caption_segments(raw)
        assert len(result) == 1
        assert result[0].end_seconds - result[0].start_seconds >= expected_floor - 0.01

    def test_maximum_duration_is_capped(self) -> None:
        raw = [_seg("Hi.", 0.0, 100.0)]  # e.g. a long silence before the next word
        result = build_caption_segments(raw)
        assert result[0].end_seconds - result[0].start_seconds <= MAX_SEGMENT_DURATION_SECONDS

    def test_end_clamped_to_max_duration_plus_tolerance(self) -> None:
        raw = [_seg("Short line.", 9.0, 10.0)]
        result = build_caption_segments(raw, max_duration_seconds=9.3)
        assert result[0].end_seconds <= 9.3 + DURATION_TOLERANCE_SECONDS

    def test_segment_starting_after_max_duration_is_dropped(self) -> None:
        raw = [_seg("First.", 0.0, 2.0), _seg("Too late.", 9.0, 11.0)]
        result = build_caption_segments(raw, max_duration_seconds=2.0)
        assert all(s.text != "Too late." for s in result)

    def test_no_max_duration_means_no_clamp(self) -> None:
        raw = [_seg("Hi.", 0.0, 1.5)]
        result = build_caption_segments(raw, max_duration_seconds=None)
        assert result[0].end_seconds == 1.5


class TestPunctuationAndTextCleaning:
    def test_extra_whitespace_collapsed(self) -> None:
        raw = [_seg("Hello    there,   world.", 0.0, 2.0)]
        result = build_caption_segments(raw)
        assert result[0].text == "Hello there, world."

    def test_space_before_punctuation_removed(self) -> None:
        raw = [_seg("Hello , world .", 0.0, 2.0)]
        result = build_caption_segments(raw)
        assert result[0].text == "Hello, world."

    def test_special_characters_preserved(self) -> None:
        raw = [_seg("It's a test - with punctuation!", 0.0, 2.0)]
        result = build_caption_segments(raw)
        assert result[0].text == "It's a test - with punctuation!"


class TestReadableSegmentation:
    def test_long_segment_without_words_is_split_at_sentence_boundaries(self) -> None:
        text = (
            "This is the first sentence in a long narration block. "
            "Here comes a second sentence that continues the thought. "
            "And finally a third sentence wraps everything up nicely."
        )
        assert len(text) > MAX_CHARS_PER_SEGMENT
        raw = [_seg(text, 0.0, 15.0)]
        result = build_caption_segments(raw)
        assert len(result) > 1
        for segment in result:
            for line in segment.text.split("\n"):
                assert len(line) <= MAX_CHARS_PER_LINE
            assert segment.text.count("\n") + 1 <= MAX_LINES_PER_SEGMENT

    def test_long_sentence_without_punctuation_is_wrapped_without_splitting_words(self) -> None:
        words = ["reallylongword"] * 15  # no sentence punctuation anywhere
        text = " ".join(words)
        assert len(text) > MAX_CHARS_PER_SEGMENT
        raw = [_seg(text, 0.0, 10.0)]
        result = build_caption_segments(raw)
        all_tokens = " ".join(s.text.replace("\n", " ") for s in result).split()
        # Every original word token must still appear intact, none merged/split.
        assert all_tokens == words
        for segment in result:
            for line in segment.text.split("\n"):
                assert len(line) <= MAX_CHARS_PER_LINE

    def test_word_level_chunking_splits_at_sentence_boundaries_with_accurate_timing(self) -> None:
        words = [
            _word("This", 0.0, 0.3),
            _word("is", 0.3, 0.5),
            _word("a", 0.5, 0.6),
            _word("test.", 0.6, 1.0),
            _word("Another", 1.0, 1.4),
            _word("sentence", 1.4, 1.9),
            _word("follows.", 1.9, 2.4),
        ]
        raw = [_seg("This is a test. Another sentence follows.", 0.0, 2.4, words=words)]
        result = build_caption_segments(raw)

        assert len(result) == 2
        assert result[0].text == "This is a test."
        assert result[0].start_seconds == 0.0
        assert result[0].end_seconds == 1.0
        assert result[1].text == "Another sentence follows."
        assert result[1].start_seconds == 1.0
        assert result[1].end_seconds == 2.4

    def test_long_word_sequence_splits_by_length_when_no_punctuation(self) -> None:
        words = [_word(f"word{i}", float(i), float(i) + 0.9) for i in range(30)]
        raw = [_seg(" ".join(w.word for w in words), 0.0, 30.0, words=words)]
        result = build_caption_segments(raw)
        assert len(result) > 1

    def test_no_two_lines_exceed_max_chars_per_line(self) -> None:
        text = "word " * 30  # forces both segment-splitting and line-wrapping
        raw = [_seg(text.strip() + ".", 0.0, 20.0)]
        result = build_caption_segments(raw)
        for segment in result:
            for line in segment.text.split("\n"):
                assert len(line) <= MAX_CHARS_PER_LINE

    def test_realistic_narration_stays_within_two_lines(self) -> None:
        """With real sentence structure (the normal case), each caption
        stays within the target line count."""
        text = (
            "This is the first sentence in a long narration block. "
            "Here comes a second sentence that continues the thought. "
            "And finally a third sentence wraps everything up nicely."
        )
        raw = [_seg(text, 0.0, 15.0)]
        result = build_caption_segments(raw)
        for segment in result:
            assert segment.text.count("\n") + 1 <= MAX_LINES_PER_SEGMENT

    def test_line_length_limit_takes_priority_over_line_count_in_rare_edge_case(self) -> None:
        """A run of words with no punctuation/natural break anywhere may
        rarely need more than MAX_LINES_PER_SEGMENT lines - the per-line
        character limit (the harder readability requirement) is never
        sacrificed to force a 2-line cap."""
        text = "word " * 30
        raw = [_seg(text.strip() + ".", 0.0, 20.0)]
        result = build_caption_segments(raw)
        for segment in result:
            for line in segment.text.split("\n"):
                assert len(line) <= MAX_CHARS_PER_LINE

    def test_short_text_is_not_wrapped(self) -> None:
        raw = [_seg("Short line.", 0.0, 2.0)]
        result = build_caption_segments(raw)
        assert "\n" not in result[0].text
