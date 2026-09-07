# Tests for the shared script-context reconstruction module (used by both
# src/bgm_demo.py and src/metadata_demo.py). No LLM/network involved.
from __future__ import annotations

from src.services.script_context_reconstruction import (
    build_context_from_srt,
    build_topic_only_context,
    derive_title_from_base_name,
    extract_narration_from_srt,
    find_matching_srt,
    original_base_name,
)

_SAMPLE_SRT = """1
00:00:00,000 --> 00:00:04,300
Did you know you spend over two hours
every single night dreaming?

2
00:00:04,300 --> 00:00:05,300
Interesting.

3
00:00:05,300 --> 00:00:07,443
Science still leaves us wondering why.
"""


class TestOriginalBaseName:
    def test_plain_video_unchanged(self) -> None:
        assert original_base_name("output/video/why-do-humans-dream-7c674e5c.mp4") == "why-do-humans-dream-7c674e5c"

    def test_captioned_suffix_stripped(self) -> None:
        assert (
            original_base_name("output/video/why-do-humans-dream-7c674e5c-captioned.mp4")
            == "why-do-humans-dream-7c674e5c"
        )

    def test_captioned_bgm_suffix_stripped(self) -> None:
        assert (
            original_base_name("output/video/why-do-humans-dream-7c674e5c-captioned-bgm.mp4")
            == "why-do-humans-dream-7c674e5c"
        )

    def test_bgm_only_suffix_stripped(self) -> None:
        assert original_base_name("output/video/why-do-humans-dream-7c674e5c-bgm.mp4") == "why-do-humans-dream-7c674e5c"


class TestFindMatchingSrt:
    def test_finds_matching_srt_for_plain_video(self, tmp_path) -> None:
        subtitle_dir = tmp_path / "subtitles"
        subtitle_dir.mkdir()
        (subtitle_dir / "my-video-abcd1234.srt").write_text(_SAMPLE_SRT, encoding="utf-8")

        result = find_matching_srt("output/video/my-video-abcd1234.mp4", str(subtitle_dir))

        assert result == str(subtitle_dir / "my-video-abcd1234.srt")

    def test_finds_matching_srt_for_captioned_bgm_video(self, tmp_path) -> None:
        subtitle_dir = tmp_path / "subtitles"
        subtitle_dir.mkdir()
        (subtitle_dir / "my-video-abcd1234.srt").write_text(_SAMPLE_SRT, encoding="utf-8")

        result = find_matching_srt("output/video/my-video-abcd1234-captioned-bgm.mp4", str(subtitle_dir))

        assert result == str(subtitle_dir / "my-video-abcd1234.srt")

    def test_returns_none_when_no_match(self, tmp_path) -> None:
        subtitle_dir = tmp_path / "subtitles"
        subtitle_dir.mkdir()

        assert find_matching_srt("output/video/nonexistent-abcd1234.mp4", str(subtitle_dir)) is None


class TestExtractNarrationFromSrt:
    def test_concatenates_all_caption_text_in_order(self, tmp_path) -> None:
        srt_path = tmp_path / "sample.srt"
        srt_path.write_text(_SAMPLE_SRT, encoding="utf-8")

        narration = extract_narration_from_srt(str(srt_path))

        assert narration.startswith("Did you know you spend over two hours")
        assert "Interesting." in narration
        assert narration.endswith("Science still leaves us wondering why.")

    def test_empty_srt_produces_empty_narration(self, tmp_path) -> None:
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("", encoding="utf-8")

        assert extract_narration_from_srt(str(srt_path)) == ""


class TestDeriveTitleFromBaseName:
    def test_strips_hash_suffix_and_title_cases(self) -> None:
        assert derive_title_from_base_name("why-do-humans-dream-7c674e5c") == "Why Do Humans Dream"

    def test_no_hash_suffix_still_works(self) -> None:
        assert derive_title_from_base_name("why-do-humans-dream") == "Why Do Humans Dream"


class TestBuildContextFromSrt:
    def test_reconstructed_script_has_real_narration(self, tmp_path) -> None:
        srt_path = tmp_path / "why-do-humans-dream-7c674e5c.srt"
        srt_path.write_text(_SAMPLE_SRT, encoding="utf-8")

        script = build_context_from_srt(
            "Why do humans dream?", str(srt_path), "output/video/why-do-humans-dream-7c674e5c-captioned.mp4"
        )

        assert script.topic == "Why do humans dream?"
        assert script.video_title == "Why Do Humans Dream"
        assert len(script.sections) == 1
        assert "dreaming" in script.sections[0].narration

    def test_empty_srt_falls_back_to_topic(self, tmp_path) -> None:
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("", encoding="utf-8")

        script = build_context_from_srt("Why do humans dream?", str(srt_path), "output/video/x-abcd1234.mp4")

        assert script.sections[0].narration == "Why do humans dream?"


class TestBuildTopicOnlyContext:
    def test_uses_topic_as_narration(self) -> None:
        script = build_topic_only_context("Why do humans dream?")

        assert script.topic == "Why do humans dream?"
        assert script.video_title == "Why do humans dream?"
        assert script.sections[0].narration == "Why do humans dream?"
