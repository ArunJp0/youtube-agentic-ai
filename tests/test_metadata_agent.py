# Tests for MetadataAgent: one semantic LLM call, deterministic chapter
# timing, and deterministic validation/normalization of the result. Uses
# fake LLMProvider test doubles only - no real Gemini calls.
from __future__ import annotations

import json
import os

import pytest

from src.agents.metadata_agent import MetadataAgent, MetadataAgentError
from src.llm.provider import LLMProvider
from src.models.metadata import MetadataResult
from src.models.script import ScriptResult, ScriptSection


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(heading="REM Sleep", narration="Dreams occur mainly during REM sleep."),
            ScriptSection(heading="Memory", narration="The brain consolidates memories while dreaming."),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _valid_payload(**overrides) -> dict:
    payload = {
        "title": "Why Do Humans Dream? The Science Explained",
        "description": "A deep dive into the real science behind dreaming, grounded in current research.",
        "seo_summary": "Learn why humans dream and what the science says.",
        "tags": ["dreams", "sleep science", "REM sleep"],
        "hashtags": ["#dreams", "#sleep", "#science"],
    }
    payload.update(overrides)
    return payload


def _valid_response(chapter_labels=None, **overrides) -> str:
    payload = _valid_payload(**overrides)
    if chapter_labels is not None:
        payload["chapter_labels"] = chapter_labels
    return json.dumps(payload)


class FakeLLMProvider(LLMProvider):
    """Test double returning a fixed canned response and counting calls."""

    def __init__(self, response: str = "", raise_error: Exception | None = None) -> None:
        self.response = response
        self.raise_error = raise_error
        self.calls: list[str] = []
        self.name = "fake"
        self.last_model_used = "fake-model"
        self.last_used_fallback = False

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_error:
            raise self.raise_error
        return self.response


class TestMetadataAgentValidation:
    def test_missing_topic_raises(self) -> None:
        agent = MetadataAgent(llm_provider=FakeLLMProvider())
        with pytest.raises(MetadataAgentError, match="required"):
            agent.generate_metadata("", _script())

    def test_missing_script_raises(self) -> None:
        agent = MetadataAgent(llm_provider=FakeLLMProvider())
        with pytest.raises(MetadataAgentError, match="required"):
            agent.generate_metadata("dreams", None)


class TestMetadataAgentOneCallMaximum:
    def test_exactly_one_llm_call(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        agent.generate_metadata("Why do humans dream?", _script(), duration_seconds=120.0)

        assert len(llm.calls) == 1

    def test_prompt_includes_topic_and_sections(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))
        script = _script()

        agent.generate_metadata(script.topic, script, duration_seconds=120.0)

        prompt = llm.calls[0]
        assert script.topic in prompt
        for section in script.sections:
            assert section.heading in prompt
            assert section.narration in prompt

    def test_prompt_instructs_against_fabrication(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert "never invent facts" in llm.calls[0].lower()


class TestMetadataAgentSuccess:
    def test_valid_response_produces_success_result(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("Why do humans dream?", _script(), duration_seconds=120.0)

        assert isinstance(result, MetadataResult)
        assert result.success is True
        assert result.title == "Why Do Humans Dream? The Science Explained"
        assert result.tags == ["dreams", "sleep science", "REM sleep"]
        assert result.hashtags == ["#dreams", "#sleep", "#science"]

    def test_llm_provider_diagnostics_captured(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.llm_provider == "fake"
        assert result.llm_model == "fake-model"
        assert result.used_fallback_model is False

    def test_response_wrapped_in_markdown_fence_is_parsed(self, tmp_path) -> None:
        fenced = f"```json\n{_valid_response()}\n```"
        agent = MetadataAgent(llm_provider=FakeLLMProvider(response=fenced), output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is True


class TestMetadataAgentTitleHandling:
    def test_over_length_title_normalized(self, tmp_path) -> None:
        long_title = "Why Do Humans Dream " + ("Really " * 20) + "Explained"
        llm = FakeLLMProvider(response=_valid_response(title=long_title))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is True
        assert len(result.title) <= 100

    def test_empty_title_after_normalization_fails(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(title="   "))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is False
        assert result.error is not None


class TestMetadataAgentDescriptionHandling:
    def test_empty_description_after_normalization_fails(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(description=" "))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is False


class TestMetadataAgentTagsHashtags:
    def test_duplicate_tags_deduplicated(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(tags=["Dreams", "dreams", "Sleep"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.tags == ["Dreams", "Sleep"]

    def test_hashtags_normalized_with_prefix(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(hashtags=["dreams", "#sleep"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.hashtags == ["#dreams", "#sleep"]


class TestMetadataAgentChapters:
    def test_chapters_derived_deterministically_first_at_zero(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(chapter_labels=["REM Sleep Explained", "Memory Consolidation"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))
        script = _script()

        result = agent.generate_metadata(script.topic, script, duration_seconds=100.0)

        assert result.chapters_available is True
        assert len(result.chapters) == 2
        assert result.chapters[0].timestamp_seconds == 0.0
        assert result.chapters[0].timestamp_text == "0:00"
        assert result.chapters[0].title == "REM Sleep Explained"
        assert result.chapters[1].timestamp_seconds > 0.0

    def test_chapter_timestamps_strictly_increasing(self, tmp_path) -> None:
        script = _script(
            sections=[
                ScriptSection(heading="A", narration="word " * 10),
                ScriptSection(heading="B", narration="word " * 30),
                ScriptSection(heading="C", narration="word " * 5),
            ]
        )
        llm = FakeLLMProvider(response=_valid_response(chapter_labels=["A", "B", "C"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata(script.topic, script, duration_seconds=180.0)

        timestamps = [c.timestamp_seconds for c in result.chapters]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)

    def test_missing_chapter_label_falls_back_to_section_heading(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(chapter_labels=["Only One Label"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))
        script = _script()

        result = agent.generate_metadata(script.topic, script, duration_seconds=100.0)

        assert result.chapters[0].title == "Only One Label"
        assert result.chapters[1].title == script.sections[1].heading

    def test_no_duration_omits_chapters_without_failing(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=None)

        assert result.success is True
        assert result.chapters_available is False
        assert result.chapters == []
        assert result.chapters_omitted_reason is not None

    def test_single_section_omits_chapters_without_failing(self, tmp_path) -> None:
        script = _script(sections=[ScriptSection(heading="Only", narration="One section only.")])
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata(script.topic, script, duration_seconds=60.0)

        assert result.success is True
        assert result.chapters_available is False

    def test_chapter_prompt_never_asks_llm_for_timestamps(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(chapter_labels=["A", "B"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))
        script = _script()

        agent.generate_metadata(script.topic, script, duration_seconds=100.0)

        prompt = llm.calls[0]
        assert "ALREADY FIXED" in prompt


class TestMetadataAgentFailures:
    def test_llm_exception_produces_clean_failure(self, tmp_path) -> None:
        llm = FakeLLMProvider(raise_error=RuntimeError("Gemini outage"))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is False
        assert "Gemini outage" in result.error
        assert result.title is None

    def test_malformed_json_produces_clean_failure(self, tmp_path) -> None:
        llm = FakeLLMProvider(response="not json at all")
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is False
        assert result.error is not None

    def test_missing_title_or_description_in_json_fails(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=json.dumps({"tags": ["dreams"]}))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is False

    def test_failure_never_writes_artifact(self, tmp_path) -> None:
        llm = FakeLLMProvider(raise_error=RuntimeError("outage"))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.output_path is None
        assert list(tmp_path.iterdir()) == []


class TestMetadataAgentArtifactOutput:
    def test_json_artifact_written_and_valid(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(chapter_labels=["A", "B"]))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))
        script = _script()

        result = agent.generate_metadata(script.topic, script, duration_seconds=100.0)

        assert result.output_path is not None
        assert os.path.exists(result.output_path)
        with open(result.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["title"] == result.title
        assert data["chapters_available"] is True
        assert len(data["chapters"]) == 2

    def test_output_path_uses_video_slug_when_given(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata(
            "dreams", _script(), duration_seconds=120.0, video_slug="why-do-humans-dream-7c674e5c"
        )

        assert result.output_path == os.path.join(str(tmp_path), "why-do-humans-dream-7c674e5c.json")

    def test_output_path_falls_back_to_title_slug(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_response(title="Why Do Humans Dream?"))
        agent = MetadataAgent(llm_provider=llm, output_dir=str(tmp_path))

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.output_path == os.path.join(str(tmp_path), "why-do-humans-dream.json")

    def test_output_directory_created_if_missing(self, tmp_path) -> None:
        nested_dir = os.path.join(str(tmp_path), "nested", "metadata")
        llm = FakeLLMProvider(response=_valid_response())
        agent = MetadataAgent(llm_provider=llm, output_dir=nested_dir)

        result = agent.generate_metadata("dreams", _script(), duration_seconds=120.0)

        assert result.success is True
        assert os.path.isdir(nested_dir)
