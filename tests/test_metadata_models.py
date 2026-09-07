# Tests for Metadata Agent data models (Chapter, MetadataResult).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.metadata import Chapter, MetadataResult


class TestChapterValidation:
    def test_minimal_valid_chapter(self) -> None:
        chapter = Chapter(timestamp_seconds=0.0, timestamp_text="0:00", title="Introduction")
        assert chapter.timestamp_seconds == 0.0
        assert chapter.title == "Introduction"

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chapter(timestamp_seconds=0.0, timestamp_text="0:00", title="")

    def test_empty_timestamp_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chapter(timestamp_seconds=0.0, timestamp_text="", title="Intro")

    def test_negative_timestamp_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chapter(timestamp_seconds=-1.0, timestamp_text="0:00", title="Intro")


class TestMetadataResultValidation:
    def test_minimal_failure_result(self) -> None:
        result = MetadataResult(success=False, error="LLM unavailable")
        assert result.title is None
        assert result.tags == []
        assert result.chapters == []
        assert result.chapters_available is False

    def test_success_result_carries_full_shape(self) -> None:
        chapter = Chapter(timestamp_seconds=0.0, timestamp_text="0:00", title="Intro")
        result = MetadataResult(
            success=True,
            topic="dreams",
            title="Why Do Humans Dream?",
            description="A video about dreams.",
            seo_summary="Learn why humans dream.",
            tags=["dreams", "sleep"],
            hashtags=["#dreams", "#sleep"],
            chapters=[chapter],
            chapters_available=True,
            duration_seconds=120.0,
            output_path="output/metadata/why-do-humans-dream.json",
            llm_provider="gemini",
            llm_model="gemini-3.5-flash-lite",
            used_fallback_model=False,
        )
        assert result.title == "Why Do Humans Dream?"
        assert result.chapters[0].title == "Intro"

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetadataResult(success=True, duration_seconds=-5.0)
