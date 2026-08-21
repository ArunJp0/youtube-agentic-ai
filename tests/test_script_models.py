# Tests for ScriptResult and ScriptSection models
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.script import ScriptResult, ScriptSection


class TestScriptSection:
    """Tests for the ScriptSection model."""

    def test_section_creation_with_all_fields(self) -> None:
        section = ScriptSection(
            heading="Why we dream",
            narration="Dreams occur mostly during REM sleep.",
            visual_notes="B-roll of a sleeping brain scan",
            estimated_duration_seconds=6.5,
            source_refs=["https://en.wikipedia.org/wiki/Dream"],
        )
        assert section.heading == "Why we dream"
        assert section.narration == "Dreams occur mostly during REM sleep."
        assert section.visual_notes == "B-roll of a sleeping brain scan"
        assert section.estimated_duration_seconds == 6.5
        assert section.source_refs == ["https://en.wikipedia.org/wiki/Dream"]

    def test_section_creation_with_minimal_fields(self) -> None:
        section = ScriptSection(heading="Intro", narration="Some narration.")
        assert section.visual_notes is None
        assert section.estimated_duration_seconds == 0.0
        assert section.source_refs == []

    def test_section_heading_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            ScriptSection(heading="", narration="Text")

    def test_section_narration_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            ScriptSection(heading="Heading", narration="")

    def test_section_duration_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            ScriptSection(heading="H", narration="N", estimated_duration_seconds=-1.0)

    def test_section_source_refs_default_factory_is_isolated(self) -> None:
        s1 = ScriptSection(heading="A", narration="A")
        s2 = ScriptSection(heading="B", narration="B")
        s1.source_refs.append("https://example.com")
        assert s2.source_refs == []


class TestScriptResult:
    """Tests for the ScriptResult model."""

    def test_result_creation_with_minimal_fields(self) -> None:
        result = ScriptResult(
            topic="Dreams",
            video_title="Why Do We Dream?",
            hook="Ever wonder why you dream?",
            introduction="Today we explore dreaming.",
            conclusion="That's the science of dreams.",
            call_to_action="Like and subscribe!",
        )
        assert result.topic == "Dreams"
        assert result.sections == []
        assert result.sources == []
        assert result.script_notes is None
        assert result.estimated_duration_seconds == 0.0

    def test_result_creation_with_all_fields(self) -> None:
        result = ScriptResult(
            topic="Dreams",
            video_title="Why Do We Dream?",
            hook="Hook line",
            introduction="Intro line",
            sections=[ScriptSection(heading="H1", narration="N1")],
            conclusion="Conclusion line",
            call_to_action="CTA line",
            estimated_duration_seconds=42.5,
            sources=["https://example.com"],
            script_notes="Some caveats",
        )
        assert len(result.sections) == 1
        assert result.estimated_duration_seconds == 42.5
        assert len(result.sources) == 1
        assert result.script_notes == "Some caveats"

    @pytest.mark.parametrize(
        "field_name",
        ["topic", "video_title", "hook", "introduction", "conclusion", "call_to_action"],
    )
    def test_required_text_fields_cannot_be_empty(self, field_name: str) -> None:
        kwargs = dict(
            topic="Dreams",
            video_title="Title",
            hook="Hook",
            introduction="Intro",
            conclusion="Conclusion",
            call_to_action="CTA",
        )
        kwargs[field_name] = ""
        with pytest.raises(ValidationError):
            ScriptResult(**kwargs)

    def test_result_sections_default_factory_is_isolated(self) -> None:
        r1 = ScriptResult(
            topic="A", video_title="A", hook="A", introduction="A", conclusion="A", call_to_action="A"
        )
        r2 = ScriptResult(
            topic="B", video_title="B", hook="B", introduction="B", conclusion="B", call_to_action="B"
        )
        r1.sections.append(ScriptSection(heading="H", narration="N"))
        assert r2.sections == []

    def test_result_serialization(self) -> None:
        result = ScriptResult(
            topic="Dreams",
            video_title="Title",
            hook="Hook",
            introduction="Intro",
            sections=[ScriptSection(heading="H", narration="N", estimated_duration_seconds=3.0)],
            conclusion="Conclusion",
            call_to_action="CTA",
        )
        data = result.model_dump()
        assert data["topic"] == "Dreams"
        assert data["sections"][0]["estimated_duration_seconds"] == 3.0

    def test_result_json_roundtrip(self) -> None:
        result = ScriptResult(
            topic="Dreams",
            video_title="Title",
            hook="Hook",
            introduction="Intro",
            sections=[ScriptSection(heading="H", narration="N")],
            conclusion="Conclusion",
            call_to_action="CTA",
        )
        json_str = result.model_dump_json()
        restored = ScriptResult.model_validate_json(json_str)
        assert restored.topic == result.topic
        assert restored.sections[0].heading == "H"
