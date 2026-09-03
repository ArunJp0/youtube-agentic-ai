# Tests for SRT formatting/writing (src.services.srt_writer).
from __future__ import annotations

from src.models.captions import CaptionSegment
from src.services.srt_writer import format_srt, format_timestamp, write_srt_file


def _segment(index: int, start: float, end: float, text: str) -> CaptionSegment:
    return CaptionSegment(index=index, start_seconds=start, end_seconds=end, text=text)


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_sub_second(self) -> None:
        assert format_timestamp(1.234) == "00:00:01,234"

    def test_minutes(self) -> None:
        assert format_timestamp(65.5) == "00:01:05,500"

    def test_hours(self) -> None:
        assert format_timestamp(3661.001) == "01:01:01,001"

    def test_negative_clamped_to_zero(self) -> None:
        assert format_timestamp(-1.0) == "00:00:00,000"

    def test_rounds_milliseconds(self) -> None:
        assert format_timestamp(1.9996) == "00:00:02,000"


class TestFormatSrt:
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_srt([]) == ""

    def test_single_segment_shape(self) -> None:
        content = format_srt([_segment(1, 0.0, 1.5, "Hello world.")])
        assert content == "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n"

    def test_sequential_numbering_derived_from_order_not_index_field(self) -> None:
        # Deliberately mismatched index values - format_srt must still
        # number 1, 2, 3... from list position.
        segments = [
            _segment(5, 0.0, 1.0, "First."),
            _segment(9, 1.0, 2.0, "Second."),
        ]
        content = format_srt(segments)
        blocks = content.strip().split("\n\n")
        assert blocks[0].startswith("1\n")
        assert blocks[1].startswith("2\n")

    def test_chronological_ordering_preserved_in_output(self) -> None:
        segments = [_segment(1, 0.0, 1.0, "A"), _segment(2, 1.0, 2.0, "B"), _segment(3, 2.0, 3.0, "C")]
        content = format_srt(segments)
        assert content.index("A") < content.index("B") < content.index("C")

    def test_multiline_text_preserved(self) -> None:
        content = format_srt([_segment(1, 0.0, 2.0, "Line one\nLine two")])
        assert "Line one\nLine two" in content

    def test_special_characters_preserved(self) -> None:
        content = format_srt([_segment(1, 0.0, 1.0, "It's a test - 100% \"quoted\"!")])
        assert "It's a test - 100% \"quoted\"!" in content

    def test_no_empty_entries(self) -> None:
        content = format_srt([_segment(1, 0.0, 1.0, "Only content.")])
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        assert len(blocks) == 1


class TestWriteSrtFile:
    def test_writes_utf8_file(self, tmp_path) -> None:
        output_path = str(tmp_path / "out.srt")
        write_srt_file([_segment(1, 0.0, 1.0, "Café résumé")], output_path)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "Café résumé" in content

    def test_valid_srt_structure_written(self, tmp_path) -> None:
        output_path = str(tmp_path / "out.srt")
        segments = [_segment(1, 0.0, 1.0, "First."), _segment(2, 1.0, 2.5, "Second.")]
        write_srt_file(segments, output_path)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("1\n")
        assert "-->" in content
        assert "2\n" in content

    def test_empty_segments_writes_empty_file(self, tmp_path) -> None:
        output_path = str(tmp_path / "out.srt")
        write_srt_file([], output_path)
        with open(output_path, encoding="utf-8") as f:
            assert f.read() == ""
