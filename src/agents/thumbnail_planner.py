# Thumbnail Planner: understands a video's actual topic/script (and, when
# available, its already-generated metadata) in ONE LLM call, producing a
# structured ThumbnailPlan - semantic/creative decisions only (hook text,
# visual concept, search query, mood, composition intent). The LLM never
# decides pixel coordinates or file operations; the deterministic renderer
# (src/services/thumbnail_renderer.py) owns all of that.
#
# Mirrors MusicContextPlanner/VisualContextPlanner's shape and constraints
# exactly: a small, focused component (not a LangGraph agent), one request/
# response, reuses the existing LLMProvider abstraction, and never raises
# for an LLM/parsing failure - it falls back to the same deterministic plan
# used when no planner is configured at all.
from __future__ import annotations

import json
from typing import Optional

from src.llm.provider import LLMProvider
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailPlan
from src.services.llm_json import extract_json_object
from src.services.thumbnail_planning import (
    build_deterministic_thumbnail_plan,
    normalize_composition,
    normalize_text_position,
)


class ThumbnailPlannerError(Exception):
    """Raised only for programmer/configuration errors (e.g. a missing
    topic).

    LLM failures, malformed responses, and validation failures are NEVER
    raised from ``plan_thumbnail`` - they are caught internally and result
    in the deterministic fallback plan instead.
    """


class ThumbnailPlanner:
    """Produces one ThumbnailPlan for a video using a single LLM call, with
    a deterministic fallback if that call fails.

    Deliberately scoped to ONE LLM request per video (never per-candidate-
    image), consistent with every other planner's cost discipline in this
    project.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the planner.

        Args:
            llm_provider: LLMProvider implementation used for the single
                whole-video thumbnail-planning call
        """
        self.llm_provider = llm_provider

    def plan_thumbnail(
        self,
        topic: str,
        script: ScriptResult,
        metadata_title: Optional[str] = None,
        seo_summary: Optional[str] = None,
    ) -> ThumbnailPlan:
        """Produce a ThumbnailPlan for ``topic``/``script``.

        Never raises for LLM/parsing failures - those fall back to
        ``build_deterministic_thumbnail_plan`` and are reported via
        ``ThumbnailPlan.used_semantic_planning``/``fallback_reason``.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent
            metadata_title: Optional already-generated video title (from
                MetadataResult), used as extra context if available
            seo_summary: Optional already-generated SEO summary (from
                MetadataResult), used as extra context if available

        Returns:
            A ThumbnailPlan describing the desired thumbnail concept

        Raises:
            ThumbnailPlannerError: If ``topic`` or ``script`` is missing
        """
        if not topic or script is None:
            raise ThumbnailPlannerError("topic and ScriptResult are both required")

        try:
            prompt = self._build_prompt(topic, script, metadata_title, seo_summary)
            raw_response = self.llm_provider.generate_text(prompt)
            return self._parse_response(raw_response)
        except Exception as e:
            return build_deterministic_thumbnail_plan(
                topic, f"Semantic thumbnail planning unavailable, used deterministic fallback: {e}"
            )

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(
        topic: str, script: ScriptResult, metadata_title: Optional[str], seo_summary: Optional[str]
    ) -> str:
        section_summaries = "\n".join(f"- {s.heading}: {s.narration}" for s in script.sections)
        title_line = f"Video title: {metadata_title or script.video_title}\n"
        seo_line = f"SEO summary: {seo_summary}\n" if seo_summary else ""

        return (
            "You are planning a YouTube thumbnail CONCEPT for a narrated video - semantic/"
            "creative decisions only, never pixel coordinates or file operations. The plan must "
            "accurately reflect ONLY the actual content below - never invent facts or misleading "
            "claims.\n\n"
            "Choosing the hook text - follow this priority order:\n"
            "1. PRIMARY TOPIC CLARITY: the video's core subject must be recognizable from the hook "
            "text itself. A viewer who sees ONLY the thumbnail image and hook text - not the title, "
            "not the video - must be able to broadly tell what the video is about.\n"
            "2. A concise, shortened reformulation of the actual video title is a safe, preferred "
            "option - the hook does NOT need to repeat the full title verbatim, just stay clearly "
            "recognizable as the same subject.\n"
            "3. A creative/contextual hook is also fine, but only if it stays clearly connected to "
            "the video's primary topic without requiring the viewer to already know the title.\n"
            "4. AVOID choosing an isolated statistic, duration, quote, number, or other secondary "
            "supporting detail as the hook if doing so would leave the primary subject unclear - "
            "e.g. a bare duration or count that could belong to many different topics is a bad "
            "hook even if it's factually grounded in the script, because a viewer couldn't tell "
            "what the video is actually about from it alone.\n"
            "5. Ideally 2-6 words, immediately understandable, readable at small/mobile size, no "
            "clickbait, no unsupported claims, no keyword stuffing, no excessive punctuation, no "
            "ALL CAPS beyond normal thumbnail styling, no emojis.\n\n"
            f"Video topic: {topic}\n"
            f"{title_line}"
            f"{seo_line}"
            f"Hook: {script.hook}\n\n"
            f"Sections, in order:\n{section_summaries}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly "
            "this shape:\n"
            "{\n"
            '  "hook_text": "short 2-6 word visual hook, distinct from the full title",\n'
            '  "visual_concept": "one sentence describing what the thumbnail should visually convey",\n'
            '  "search_query": "a concrete, visually searchable stock-photo query, 2-6 words",\n'
            '  "mood": "short mood descriptor",\n'
            '  "subject": "the video\'s main visual subject, briefly",\n'
            '  "composition": "one of: subject_left, subject_right, centered",\n'
            '  "text_position": "one of: left, right, center",\n'
            '  "avoid_concepts": ["literal-but-wrong visual interpretations to avoid, if any"]\n'
            "}"
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> ThumbnailPlan:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ThumbnailPlannerError("Thumbnail plan response was not a JSON object")

        hook_text = str(data.get("hook_text") or "").strip()
        search_query = str(data.get("search_query") or "").strip()
        if not hook_text or not search_query:
            raise ThumbnailPlannerError("Thumbnail plan response is missing hook_text or search_query")

        return ThumbnailPlan(
            hook_text=hook_text,
            visual_concept=str(data.get("visual_concept") or "").strip(),
            search_query=search_query,
            mood=str(data.get("mood") or "neutral").strip() or "neutral",
            subject=str(data.get("subject") or "").strip(),
            composition=normalize_composition(data.get("composition")),
            text_position=normalize_text_position(data.get("text_position")),
            avoid_concepts=[str(a) for a in (data.get("avoid_concepts") or [])],
            used_semantic_planning=True,
        )
