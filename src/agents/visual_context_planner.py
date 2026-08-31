# Visual Context Planner: understands the whole script's topic and each
# section's actual meaning in ONE LLM call, producing a structured
# VisualPlan for VisualMediaService to search/filter/select stock media
# against - instead of VisualMediaService interpreting isolated keywords
# out of context (which can misread e.g. "the mind constructs a narrative"
# as a request for construction/building footage).
#
# This is intentionally a small, focused component - not a LangGraph agent
# with multi-step reasoning - because the task is a single request/response:
# read the whole script once, return one structured plan. It reuses the
# existing LLMProvider abstraction (the same one Research/Script agents
# use), so it works against mock or real (Gemini) providers unchanged.
from __future__ import annotations

import json
from typing import List

from src.llm.provider import LLMProvider
from src.models.script import ScriptResult
from src.models.visual_plan import SectionVisualPlan, VisualPlan
from src.services.llm_json import extract_json_object
from src.services.query_generation import build_deterministic_visual_plan


class VisualContextPlannerError(Exception):
    """Raised only for programmer/configuration errors (e.g. a missing
    ScriptResult).

    LLM failures, malformed responses, and schema-validation failures are
    NEVER raised from ``plan_visuals`` - they are caught internally and
    result in a deterministic fallback plan instead, so a Gemini outage or
    a bad response never breaks the pipeline when a safe fallback exists.
    """


class VisualContextPlanner:
    """Produces a per-section VisualPlan for an entire ScriptResult using a
    single LLM call, with a deterministic fallback if that call fails.

    Deliberately scoped to ONE LLM request per script (never per-section,
    never per-visual-slot) to keep this cheap in LLM quota regardless of how
    many sections or visual slots the video ends up needing.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the planner.

        Args:
            llm_provider: Implementation of LLMProvider used for the single
                whole-script visual-planning call
        """
        self.llm_provider = llm_provider

    def plan_visuals(self, script: ScriptResult) -> VisualPlan:
        """Produce a VisualPlan for every section of ``script``.

        Never raises for LLM/parsing failures - those fall back to
        ``build_deterministic_visual_plan`` and are reported via
        ``VisualPlan.used_semantic_planning``/``fallback_reason``.

        Args:
            script: Structured script produced by the Script Agent

        Returns:
            A VisualPlan covering every section, in order

        Raises:
            VisualContextPlannerError: If ``script`` is None
        """
        if script is None:
            raise VisualContextPlannerError("ScriptResult is required")
        if not script.sections:
            return VisualPlan(
                topic=script.topic,
                sections=[],
                used_semantic_planning=False,
                fallback_reason="ScriptResult has no sections",
            )

        try:
            prompt = self._build_prompt(script)
            raw_response = self.llm_provider.generate_text(prompt)
            sections = self._parse_response(raw_response, expected_count=len(script.sections))
            return VisualPlan(topic=script.topic, sections=sections, used_semantic_planning=True)
        except Exception as e:
            fallback = build_deterministic_visual_plan(script)
            fallback.fallback_reason = (
                f"Semantic visual planning unavailable, used deterministic fallback: {e}"
            )
            return fallback

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(script: ScriptResult) -> str:
        section_blocks = [
            f'Section {index} - "{section.heading}":\n{section.narration}'
            for index, section in enumerate(script.sections)
        ]
        sections_text = "\n\n".join(section_blocks)
        last_index = len(script.sections) - 1

        return (
            "You are planning stock video/photo search queries for a YouTube video, "
            "one plan per narrated section below. Understand each section's actual "
            "MEANING in context - not just isolated keywords. Words can be ambiguous: "
            'for example, in "the mind constructs a narrative from memories", the word '
            '"constructs" means mentally forming/creating something, NOT building, '
            "construction, or architecture. Resolve this kind of ambiguity using the "
            "surrounding sentence and the video's overall topic - for ANY subject "
            "matter, not just this example.\n\n"
            f"Video topic: {script.topic}\n"
            f"Video title: {script.video_title}\n\n"
            f"Sections, in order:\n{sections_text}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with "
            "exactly this shape:\n"
            "{\n"
            '  "sections": [\n'
            "    {\n"
            '      "section_index": 0,\n'
            '      "semantic_summary": "one sentence describing what this section is actually about",\n'
            '      "visual_intents": ["short concrete visual concept", "..."],\n'
            '      "search_queries": ["specific stock-footage search query", "... 3 to 5, most specific first"],\n'
            '      "avoid_concepts": ["a literal-but-wrong visual interpretation to avoid for this section", "..."],\n'
            '      "neutral_fallback_queries": ["a safe generic visual query related to this section or the overall topic", "..."]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Include exactly one object per section, for all {len(script.sections)} "
            f"sections (section_index 0 through {last_index}), in the order given above. "
            "Every entry in search_queries and neutral_fallback_queries must be a short, "
            "concrete, visually searchable phrase (something a stock video site could "
            "actually show) - never an abstract or scientific term on its own."
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str, expected_count: int) -> List[SectionVisualPlan]:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)

        sections_data = data.get("sections") if isinstance(data, dict) else None
        if not isinstance(sections_data, list) or not sections_data:
            raise VisualContextPlannerError("Visual plan response had no sections")

        plans = [SectionVisualPlan.model_validate(entry) for entry in sections_data]
        plans.sort(key=lambda p: p.section_index)

        indices = [p.section_index for p in plans]
        if indices != list(range(expected_count)):
            raise VisualContextPlannerError(
                f"Visual plan section_index values {indices} did not match the expected "
                f"range 0..{expected_count - 1}"
            )
        return plans
