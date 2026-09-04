# Tests for MusicContextPlanner: one LLM call per video, structured JSON
# plan parsing, mood/energy normalization, and a safe deterministic
# fallback on any failure. Uses fake LLMProvider test doubles only - no
# real Gemini calls.
from __future__ import annotations

import json

import pytest

from src.agents.music_context_planner import MusicContextPlanner, MusicContextPlannerError
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


def _valid_plan_json(**overrides) -> str:
    payload = {
        "primary_mood": "thoughtful",
        "secondary_mood": "calm",
        "energy_level": "low",
        "preferred_genres": ["ambient"],
        "preferred_instrumentation": ["piano"],
        "avoid_styles": ["aggressive"],
        "requires_neutral_subtle": True,
        "reasoning_summary": "A calm, reflective topic suits subtle ambient music.",
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


class TestMusicContextPlannerValidation:
    def test_none_script_raises(self) -> None:
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider())
        with pytest.raises(MusicContextPlannerError, match="ScriptResult is required"):
            planner.plan_music("dreams", None)


class TestMusicContextPlannerOneCallPerVideo:
    def test_exactly_one_llm_call_for_whole_video(self) -> None:
        script = _script()
        llm = FakeLLMProvider(response=_valid_plan_json())
        planner = MusicContextPlanner(llm_provider=llm)

        planner.plan_music(script.topic, script)

        assert len(llm.calls) == 1

    def test_prompt_includes_topic_title_and_every_section(self) -> None:
        script = _script()
        llm = FakeLLMProvider(response="not json")
        planner = MusicContextPlanner(llm_provider=llm)

        planner.plan_music(script.topic, script)

        prompt = llm.calls[0]
        assert script.topic in prompt
        assert script.video_title in prompt
        for section in script.sections:
            assert section.heading in prompt

    def test_prompt_never_asks_for_a_specific_song(self) -> None:
        script = _script()
        llm = FakeLLMProvider(response=_valid_plan_json())
        planner = MusicContextPlanner(llm_provider=llm)

        planner.plan_music(script.topic, script)

        assert "never a specific song" in llm.calls[0].lower()


class TestMusicContextPlannerParsing:
    def test_valid_response_is_used_directly(self) -> None:
        script = _script()
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=_valid_plan_json()))

        plan = planner.plan_music(script.topic, script)

        assert plan.used_semantic_planning is True
        assert plan.fallback_reason is None
        assert plan.primary_mood == "thoughtful"
        assert plan.secondary_mood == "calm"
        assert plan.energy_level == "low"
        assert plan.preferred_genres == ["ambient"]
        assert plan.requires_neutral_subtle is True

    def test_response_wrapped_in_markdown_fence_is_parsed(self) -> None:
        script = _script()
        fenced = f"```json\n{_valid_plan_json()}\n```"
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=fenced))

        plan = planner.plan_music(script.topic, script)

        assert plan.used_semantic_planning is True

    def test_energy_level_normalized_to_lowercase(self) -> None:
        script = _script()
        payload = _valid_plan_json(energy_level="LOW")
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_music(script.topic, script)

        assert plan.energy_level == "low"

    def test_unrecognized_energy_level_defaults_to_medium(self) -> None:
        script = _script()
        payload = _valid_plan_json(energy_level="super-hyper")
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_music(script.topic, script)

        assert plan.energy_level == "medium"

    def test_primary_mood_whitespace_stripped(self) -> None:
        script = _script()
        payload = _valid_plan_json(primary_mood="  warm  ")
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_music(script.topic, script)

        assert plan.primary_mood == "warm"

    def test_null_secondary_mood_becomes_none(self) -> None:
        script = _script()
        payload = _valid_plan_json(secondary_mood=None)
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_music(script.topic, script)

        assert plan.secondary_mood is None


class TestMusicContextPlannerFallback:
    def test_llm_exception_falls_back_cleanly(self) -> None:
        script = _script()
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(raise_error=RuntimeError("Gemini outage")))

        plan = planner.plan_music(script.topic, script)

        assert plan.used_semantic_planning is False
        assert "Gemini outage" in plan.fallback_reason
        assert plan.primary_mood  # fallback still produces a usable mood

    def test_malformed_json_falls_back_cleanly(self) -> None:
        script = _script()
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response="not json at all"))

        plan = planner.plan_music(script.topic, script)

        assert plan.used_semantic_planning is False
        assert plan.fallback_reason is not None

    def test_missing_primary_mood_falls_back_cleanly(self) -> None:
        script = _script()
        payload = json.dumps({"energy_level": "low"})
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response=payload))

        plan = planner.plan_music(script.topic, script)

        assert plan.used_semantic_planning is False

    def test_fallback_plan_is_still_a_valid_music_plan(self) -> None:
        script = _script()
        planner = MusicContextPlanner(llm_provider=FakeLLMProvider(response="garbage"))

        plan = planner.plan_music(script.topic, script)

        assert plan.energy_level in {"low", "medium", "high"}
        assert isinstance(plan.avoid_styles, list)
        assert isinstance(plan.preferred_genres, list)
