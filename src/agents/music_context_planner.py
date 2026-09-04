# Music Context Planner: understands a video's overall mood/emotional tone
# from its topic and full script in ONE LLM call, producing a structured
# MusicPlan describing music CHARACTERISTICS (mood/energy/genre/
# instrumentation) - never a specific song - for MusicSelectionService to
# rank the approved local catalog against.
#
# Mirrors VisualContextPlanner's shape and constraints exactly: a small,
# focused component (not a LangGraph agent), one request/response, reuses
# the existing LLMProvider abstraction, and never raises for an LLM/parsing
# failure - it falls back to the same deterministic plan used when no
# planner is configured at all.
from __future__ import annotations

import json

from src.llm.provider import LLMProvider
from src.models.music import MusicPlan
from src.models.script import ScriptResult
from src.services.llm_json import extract_json_object
from src.services.music_planning import build_deterministic_music_plan

_VALID_ENERGY_LEVELS = {"low", "medium", "high"}


class MusicContextPlannerError(Exception):
    """Raised only for programmer/configuration errors (e.g. a missing
    ScriptResult).

    LLM failures, malformed responses, and validation failures are NEVER
    raised from ``plan_music`` - they are caught internally and result in
    the deterministic fallback plan instead.
    """


class MusicContextPlanner:
    """Produces one MusicPlan for an entire video using a single LLM call,
    with a deterministic fallback if that call fails.

    Deliberately scoped to ONE LLM request per video (never per-section),
    consistent with VisualContextPlanner's cost discipline.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the planner.

        Args:
            llm_provider: Implementation of LLMProvider used for the single
                whole-video music-planning call
        """
        self.llm_provider = llm_provider

    def plan_music(self, topic: str, script: ScriptResult) -> MusicPlan:
        """Produce a MusicPlan for ``topic``/``script``.

        Never raises for LLM/parsing failures - those fall back to
        ``build_deterministic_music_plan`` and are reported via
        ``MusicPlan.used_semantic_planning``/``fallback_reason``.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent

        Returns:
            A MusicPlan describing the desired music characteristics

        Raises:
            MusicContextPlannerError: If ``script`` is None
        """
        if script is None:
            raise MusicContextPlannerError("ScriptResult is required")

        try:
            prompt = self._build_prompt(topic, script)
            raw_response = self.llm_provider.generate_text(prompt)
            return self._parse_response(raw_response, topic)
        except Exception as e:
            fallback = build_deterministic_music_plan(topic, script)
            fallback.fallback_reason = (
                f"Semantic music planning unavailable, used deterministic fallback: {e}"
            )
            return fallback

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(topic: str, script: ScriptResult) -> str:
        section_summaries = "\n".join(
            f"- {section.heading}: {section.narration}" for section in script.sections
        )

        return (
            "You are planning BACKGROUND MUSIC CHARACTERISTICS - never a specific song or "
            "artist - for a narrated, faceless YouTube video. Infer the video's overall mood "
            "and emotional tone from its topic and script content using general reasoning; do "
            "not rely on any hardcoded rule for a specific subject. Favor subtle, professional, "
            "non-distracting choices - background music for narration, not a soundtrack meant "
            "to be noticed. When the subject is serious, sensitive, or ambiguous in tone, prefer "
            "especially neutral and restrained music.\n\n"
            f"Video topic: {topic}\n"
            f"Video title: {script.video_title}\n"
            f"Hook: {script.hook}\n\n"
            f"Sections, in order:\n{section_summaries}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly "
            "this shape:\n"
            "{\n"
            '  "primary_mood": "short mood descriptor, e.g. thoughtful, energetic, warm, restrained",\n'
            '  "secondary_mood": "optional secondary descriptor, or null",\n'
            '  "energy_level": "low, medium, or high",\n'
            '  "preferred_genres": ["e.g. ambient", "cinematic", "..."],\n'
            '  "preferred_instrumentation": ["e.g. piano", "soft synth", "..."],\n'
            '  "avoid_styles": ["styles, instruments, or energy levels that would NOT fit"],\n'
            '  "requires_neutral_subtle": true or false,\n'
            '  "reasoning_summary": "one or two sentences explaining this choice, for logs"\n'
            "}"
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str, topic: str) -> MusicPlan:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise MusicContextPlannerError("Music plan response was not a JSON object")

        primary_mood = str(data.get("primary_mood") or "").strip()
        if not primary_mood:
            raise MusicContextPlannerError("Music plan response had no primary_mood")

        energy_level = str(data.get("energy_level") or "medium").strip().lower()
        if energy_level not in _VALID_ENERGY_LEVELS:
            energy_level = "medium"

        secondary_mood = data.get("secondary_mood") or None

        return MusicPlan(
            topic=topic,
            primary_mood=primary_mood,
            secondary_mood=str(secondary_mood).strip() if secondary_mood else None,
            energy_level=energy_level,
            preferred_genres=[str(g) for g in (data.get("preferred_genres") or [])],
            preferred_instrumentation=[str(i) for i in (data.get("preferred_instrumentation") or [])],
            avoid_styles=[str(a) for a in (data.get("avoid_styles") or [])],
            requires_neutral_subtle=bool(data.get("requires_neutral_subtle") or False),
            reasoning_summary=str(data.get("reasoning_summary") or ""),
            used_semantic_planning=True,
        )
