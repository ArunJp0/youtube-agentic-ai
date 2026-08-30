# Tests for VisualContextPlanner: one LLM call per script, structured
# JSON-plan parsing, and a safe deterministic fallback on any failure.
# Uses fake LLMProvider test doubles only - no real Gemini calls.
from __future__ import annotations

import json

import pytest

from src.agents.visual_context_planner import VisualContextPlanner, VisualContextPlannerError
from src.llm.provider import LLMProvider
from src.models.script import ScriptResult, ScriptSection


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(
                heading="Dream Narratives",
                narration="The mind constructs a narrative from fragmented memories while dreaming.",
            ),
            ScriptSection(
                heading="Memory and Emotion",
                narration="The brain sorts through emotions and experiences during sleep.",
            ),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _valid_plan_json(sections_payload) -> str:
    return json.dumps({"sections": sections_payload})


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


class TestVisualContextPlannerValidation:
    def test_none_script_raises(self) -> None:
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider())
        with pytest.raises(VisualContextPlannerError, match="ScriptResult is required"):
            planner.plan_visuals(None)

    def test_no_sections_returns_fallback_without_calling_llm(self) -> None:
        llm = FakeLLMProvider()
        planner = VisualContextPlanner(llm_provider=llm)
        plan = planner.plan_visuals(_script(sections=[]))

        assert plan.used_semantic_planning is False
        assert plan.sections == []
        assert llm.calls == []


class TestVisualContextPlannerOneCallPerScript:
    def test_exactly_one_llm_call_for_whole_script(self) -> None:
        script = _script()
        payload = _valid_plan_json(
            [
                {
                    "section_index": i,
                    "semantic_summary": "summary",
                    "visual_intents": ["a"],
                    "search_queries": ["a query"],
                    "avoid_concepts": [],
                    "neutral_fallback_queries": ["b query"],
                }
                for i in range(len(script.sections))
            ]
        )
        llm = FakeLLMProvider(response=payload)
        planner = VisualContextPlanner(llm_provider=llm)

        planner.plan_visuals(script)

        assert len(llm.calls) == 1

    def test_prompt_includes_topic_title_and_every_section(self) -> None:
        script = _script()
        llm = FakeLLMProvider(response="not json")
        planner = VisualContextPlanner(llm_provider=llm)

        planner.plan_visuals(script)

        prompt = llm.calls[0]
        assert script.topic in prompt
        assert script.video_title in prompt
        for section in script.sections:
            assert section.heading in prompt
            assert section.narration in prompt


class TestVisualContextPlannerParsing:
    def test_valid_response_is_used_directly(self) -> None:
        script = _script()
        payload = _valid_plan_json(
            [
                {
                    "section_index": 0,
                    "semantic_summary": "The mind forms a dream narrative from memories.",
                    "visual_intents": ["person dreaming", "human brain activity"],
                    "search_queries": ["person dreaming sleep", "human brain neuroscience"],
                    "avoid_concepts": ["construction site", "building construction"],
                    "neutral_fallback_queries": ["person sleeping at night"],
                },
                {
                    "section_index": 1,
                    "semantic_summary": "The brain processes emotion during sleep.",
                    "visual_intents": ["emotions", "sleep"],
                    "search_queries": ["person sleeping emotional"],
                    "avoid_concepts": [],
                    "neutral_fallback_queries": ["sleep dream"],
                },
            ]
        )
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is True
        assert plan.fallback_reason is None
        assert len(plan.sections) == 2
        assert plan.sections[0].avoid_concepts == ["construction site", "building construction"]

    def test_response_wrapped_in_markdown_fence_is_parsed(self) -> None:
        script = _script(sections=_script().sections[:1])
        payload = _valid_plan_json(
            [{"section_index": 0, "search_queries": ["q"], "neutral_fallback_queries": ["n"]}]
        )
        fenced = f"```json\n{payload}\n```"
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=fenced))

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is True
        assert plan.sections[0].search_queries == ["q"]

    def test_response_with_leading_commentary_is_parsed(self) -> None:
        script = _script(sections=_script().sections[:1])
        payload = _valid_plan_json(
            [{"section_index": 0, "search_queries": ["q"], "neutral_fallback_queries": ["n"]}]
        )
        noisy = f"Sure, here is the plan:\n{payload}\nHope that helps!"
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=noisy))

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is True


class TestVisualContextPlannerFallback:
    """Section-11 requirement: planner failure falls back safely to
    deterministic visual generation, and the fallback is clearly marked."""

    def test_llm_exception_falls_back_cleanly(self) -> None:
        script = _script()
        planner = VisualContextPlanner(
            llm_provider=FakeLLMProvider(raise_error=RuntimeError("Gemini outage"))
        )

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is False
        assert "Gemini outage" in plan.fallback_reason
        assert len(plan.sections) == len(script.sections)
        assert all(sp.search_queries for sp in plan.sections)

    def test_malformed_json_falls_back_cleanly(self) -> None:
        script = _script()
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response="not json at all"))

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is False
        assert plan.fallback_reason is not None
        assert len(plan.sections) == len(script.sections)

    def test_wrong_section_count_falls_back_cleanly(self) -> None:
        script = _script()  # 2 sections
        payload = _valid_plan_json([{"section_index": 0, "search_queries": ["q"]}])  # only 1
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is False
        assert len(plan.sections) == len(script.sections)

    def test_missing_sections_key_falls_back_cleanly(self) -> None:
        script = _script()
        planner = VisualContextPlanner(
            llm_provider=FakeLLMProvider(response=json.dumps({"oops": []}))
        )

        plan = planner.plan_visuals(script)

        assert plan.used_semantic_planning is False

    def test_fallback_plan_is_still_usable(self) -> None:
        """The fallback plan must be structurally identical in shape to a
        real plan - VisualMediaService should never need to special-case it."""
        script = _script()
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response="garbage"))

        plan = planner.plan_visuals(script)

        for section_plan in plan.sections:
            assert isinstance(section_plan.search_queries, list)
            assert isinstance(section_plan.avoid_concepts, list)
            assert isinstance(section_plan.neutral_fallback_queries, list)


class TestVisualContextPlannerAmbiguity:
    """Contextual ambiguity resolution: a well-formed semantic plan (as a
    real LLM call would return) must steer AWAY from a literal keyword
    reading, via avoid_concepts - proving the planner's output plumbing
    correctly carries disambiguation through, rather than falling back to
    the naive keyword-only interpretation.
    """

    def test_constructs_a_narrative_does_not_imply_construction_footage(self) -> None:
        script = _script()  # section 0 narration contains "constructs a narrative"
        payload = _valid_plan_json(
            [
                {
                    "section_index": 0,
                    "semantic_summary": "The mind mentally forms a story from memory fragments.",
                    "visual_intents": ["person dreaming", "abstract thought", "memory"],
                    "search_queries": ["person dreaming sleep", "abstract thought concept"],
                    "avoid_concepts": ["construction site", "building construction", "architecture"],
                    "neutral_fallback_queries": ["dream sleep"],
                },
                {
                    "section_index": 1,
                    "search_queries": ["person sleeping emotional"],
                    "neutral_fallback_queries": ["sleep dream"],
                },
            ]
        )
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_visuals(script)

        section_plan = plan.sections[0]
        all_positive_text = " ".join(section_plan.search_queries + section_plan.visual_intents).lower()
        assert "construction" not in all_positive_text
        assert "building" not in all_positive_text
        assert "construction site" in section_plan.avoid_concepts

    def test_unrelated_ambiguity_example_bug_in_software(self) -> None:
        """A second, unrelated ambiguity: "bug" in a software-development
        section should not be interpreted as an insect."""
        script = _script(
            sections=[
                ScriptSection(
                    heading="Finding The Bug",
                    narration="The engineer finally found the bug causing the app to crash.",
                )
            ]
        )
        payload = _valid_plan_json(
            [
                {
                    "section_index": 0,
                    "semantic_summary": "A software engineer debugs an application crash.",
                    "visual_intents": ["programmer coding", "computer screen with code"],
                    "search_queries": ["software developer debugging", "computer code screen"],
                    "avoid_concepts": ["insect bug", "beetle macro", "ladybug"],
                    "neutral_fallback_queries": ["technology computer"],
                }
            ]
        )
        planner = VisualContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_visuals(script)

        section_plan = plan.sections[0]
        all_positive_text = " ".join(section_plan.search_queries + section_plan.visual_intents).lower()
        assert "insect" not in all_positive_text
        assert "beetle" not in all_positive_text
        assert "insect bug" in section_plan.avoid_concepts
