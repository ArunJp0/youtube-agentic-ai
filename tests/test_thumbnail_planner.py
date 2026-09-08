# Tests for ThumbnailPlanner: one LLM call per video, structured JSON
# response parsing, and a safe deterministic fallback on any failure. Uses
# fake LLMProvider test doubles only - no real Gemini calls.
from __future__ import annotations

import json

import pytest

from src.agents.thumbnail_planner import ThumbnailPlanner, ThumbnailPlannerError
from src.llm.provider import LLMProvider
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


def _valid_response(**overrides) -> str:
    payload = {
        "hook_text": "WHY DO WE DREAM?",
        "visual_concept": "A person sleeping peacefully with abstract dream imagery",
        "search_query": "person sleeping peacefully bed",
        "mood": "curious",
        "subject": "a sleeping person",
        "composition": "subject_left",
        "text_position": "right",
        "avoid_concepts": ["nightmare horror imagery"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeLLMProvider(LLMProvider):
    """Test double returning a fixed canned response and counting calls."""

    def __init__(self, response: str = "", raise_error: Exception | None = None) -> None:
        self.response = response
        self.raise_error = raise_error
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_error:
            raise self.raise_error
        return self.response


class TestThumbnailPlannerValidation:
    def test_missing_topic_raises(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider())
        with pytest.raises(ThumbnailPlannerError, match="required"):
            planner.plan_thumbnail("", _script())

    def test_missing_script_raises(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider())
        with pytest.raises(ThumbnailPlannerError, match="required"):
            planner.plan_thumbnail("dreams", None)


class TestThumbnailPlannerOneCallMaximum:
    def test_exactly_one_llm_call(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        planner = ThumbnailPlanner(llm_provider=llm)

        planner.plan_thumbnail("Why do humans dream?", _script())

        assert len(llm.calls) == 1

    def test_prompt_includes_topic_and_sections(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        planner = ThumbnailPlanner(llm_provider=llm)
        script = _script()

        planner.plan_thumbnail(script.topic, script)

        prompt = llm.calls[0]
        assert script.topic in prompt
        for section in script.sections:
            assert section.heading in prompt

    def test_prompt_includes_metadata_context_when_given(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        planner = ThumbnailPlanner(llm_provider=llm)
        script = _script()

        planner.plan_thumbnail(
            script.topic, script, metadata_title="Why Do Humans Dream? Explained", seo_summary="Learn the science."
        )

        prompt = llm.calls[0]
        assert "Why Do Humans Dream? Explained" in prompt
        assert "Learn the science." in prompt

    def test_prompt_instructs_hook_not_to_repeat_title(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        planner = ThumbnailPlanner(llm_provider=llm)

        planner.plan_thumbnail("dreams", _script())

        prompt = llm.calls[0].lower()
        assert "does not need to repeat the full title verbatim" in prompt

    def test_prompt_instructs_primary_topic_clarity_and_avoids_isolated_facts(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        planner = ThumbnailPlanner(llm_provider=llm)

        planner.plan_thumbnail("dreams", _script())

        prompt = llm.calls[0].lower()
        assert "primary topic clarity" in prompt
        assert "isolated statistic" in prompt


class TestThumbnailPlannerParsing:
    def test_valid_response_is_used_directly(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=_valid_response()))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.used_semantic_planning is True
        assert plan.fallback_reason is None
        assert plan.hook_text == "WHY DO WE DREAM?"
        assert plan.composition == "subject_left"
        assert plan.avoid_concepts == ["nightmare horror imagery"]

    def test_response_wrapped_in_markdown_fence_is_parsed(self) -> None:
        fenced = f"```json\n{_valid_response()}\n```"
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=fenced))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.used_semantic_planning is True

    def test_invalid_composition_normalized_not_rejected(self) -> None:
        payload = _valid_response(composition="bottom-banner")
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.composition == "centered"

    def test_missing_avoid_concepts_defaults_to_empty_list(self) -> None:
        payload = json.loads(_valid_response())
        del payload["avoid_concepts"]
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=json.dumps(payload)))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.avoid_concepts == []


class TestThumbnailPlannerFallback:
    def test_llm_exception_falls_back_cleanly(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(raise_error=RuntimeError("Gemini outage")))

        plan = planner.plan_thumbnail("Why do humans dream?", _script())

        assert plan.used_semantic_planning is False
        assert "Gemini outage" in plan.fallback_reason
        assert plan.hook_text  # fallback still produces a usable hook

    def test_malformed_json_falls_back_cleanly(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response="not json at all"))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.used_semantic_planning is False
        assert plan.fallback_reason is not None

    def test_missing_hook_text_falls_back_cleanly(self) -> None:
        payload = json.dumps({"search_query": "person sleeping"})
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.used_semantic_planning is False

    def test_missing_search_query_falls_back_cleanly(self) -> None:
        payload = json.dumps({"hook_text": "WHY DREAM?"})
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.used_semantic_planning is False

    def test_fallback_plan_is_still_a_valid_thumbnail_plan(self) -> None:
        planner = ThumbnailPlanner(llm_provider=FakeLLMProvider(response="garbage"))

        plan = planner.plan_thumbnail("dreams", _script())

        assert plan.composition in {"subject_left", "subject_right", "centered"}
        assert isinstance(plan.avoid_concepts, list)
