# Metadata Agent: generates professional YouTube upload metadata (title,
# description, tags, hashtags, chapters, SEO summary) from a video's real
# topic and ScriptResult - never inventing facts not present in the script.
#
# Exactly ONE semantic LLM call per video, mirroring VisualContextPlanner/
# MusicContextPlanner's cost discipline. Chapter TIMESTAMPS are never asked
# of the LLM - they're deterministically derived from the same
# section_timing.calculate_section_durations() logic VideoAssemblyService/
# VisualMediaService already trust as the real per-section timeline; the
# LLM only supplies chapter LABELS for those fixed timestamps.
#
# Unlike VisualContextPlanner/MusicContextPlanner, there is no deterministic
# *content* fallback here - title/description/tags cannot be meaningfully
# synthesized without an LLM, so a total LLM failure is a real failure
# (MetadataResult(success=False, ...)), never a fake placeholder. A
# chapters-only failure, however, degrades gracefully: the rest of the
# metadata still succeeds, with chapters_available=False and a reason.
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from src.llm.provider import LLMProvider
from src.models.metadata import Chapter, MetadataResult
from src.models.script import ScriptResult
from src.services.llm_json import extract_json_object
from src.services.metadata_validation import (
    ChapterValidationError,
    format_youtube_timestamp,
    normalize_description,
    normalize_hashtags,
    normalize_tags,
    normalize_title,
    validate_and_clean_chapters,
)
from src.services.section_timing import calculate_section_durations

DEFAULT_METADATA_OUTPUT_DIR = os.path.join("output", "metadata")

# Fewer than this many script sections doesn't produce meaningfully
# distinct chapters - chapters are simply omitted rather than emitting a
# single degenerate 0:00-only "chapter".
MIN_SECTIONS_FOR_CHAPTERS = 2

_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")


class MetadataAgentError(Exception):
    """Raised only for configuration/programmer errors (e.g. missing topic
    or ScriptResult).

    LLM failures and malformed/unparseable responses are NEVER raised -
    they are captured in the returned MetadataResult (success=False,
    error=...) so callers always get a structured result back.
    """


class MetadataAgent:
    """Produces YouTube upload metadata for a video from its real topic and
    ScriptResult, using exactly one LLM call plus deterministic chapter
    timing and output validation/normalization."""

    def __init__(self, llm_provider: LLMProvider, output_dir: str = DEFAULT_METADATA_OUTPUT_DIR) -> None:
        """Initialize the Metadata Agent.

        Args:
            llm_provider: LLMProvider implementation for the single
                metadata-generation call (the same provider/model chain
                already used by Research/Script/Visual/BGM planning)
            output_dir: Local directory to write the metadata JSON
                artifact into (expected to be excluded from version control)
        """
        self.llm_provider = llm_provider
        self.output_dir = output_dir

    def generate_metadata(
        self,
        topic: str,
        script: ScriptResult,
        duration_seconds: Optional[float] = None,
        video_slug: Optional[str] = None,
    ) -> MetadataResult:
        """Generate YouTube metadata for ``topic``/``script``.

        Never raises for LLM/parsing/chapter-validation failures - those
        are captured in the returned MetadataResult. Only raises
        MetadataAgentError for configuration/programmer errors.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent (never
                regenerated here - Research/Script are the caller's job)
            duration_seconds: Final video duration, if known - required to
                derive chapter timestamps; metadata still succeeds without
                it, just without chapters
            video_slug: Optional stable slug (e.g. the video's own output
                filename base) used for the JSON artifact's filename;
                falls back to a slug of the generated title

        Returns:
            Structured MetadataResult describing the outcome
        """
        if not topic or script is None:
            raise MetadataAgentError("topic and ScriptResult are both required")

        chapter_timestamps, chapters_available, chapters_reason = self._derive_chapter_timestamps(
            script, duration_seconds
        )

        try:
            prompt = self._build_prompt(
                topic, script, duration_seconds, chapter_timestamps if chapters_available else None
            )
            raw_response = self.llm_provider.generate_text(prompt)
            parsed = self._parse_response(raw_response)
        except Exception as e:
            return self._failure(topic, f"Metadata generation failed: {e}")

        title = normalize_title(parsed["title"])
        description = normalize_description(parsed["description"])
        if not title:
            return self._failure(topic, "LLM response produced an empty title after normalization")
        if not description:
            return self._failure(topic, "LLM response produced an empty description after normalization")

        tags = normalize_tags(parsed["tags"])
        hashtags = normalize_hashtags(parsed["hashtags"])
        seo_summary = parsed["seo_summary"].strip() or None if parsed["seo_summary"] else None

        warnings: List[str] = []
        final_chapters: List[Chapter] = []
        final_chapters_available = False
        final_chapters_reason = chapters_reason

        if chapters_available:
            candidate_chapters = self._build_candidate_chapters(script, chapter_timestamps, parsed["chapter_labels"])
            try:
                final_chapters, chapter_warnings = validate_and_clean_chapters(candidate_chapters, duration_seconds)
                final_chapters_available = True
                final_chapters_reason = None
                warnings.extend(chapter_warnings)
            except ChapterValidationError as e:
                final_chapters_reason = str(e)
                warnings.append(f"Chapters omitted: {e}")

        output_path = None
        try:
            output_path = self._write_artifact(
                topic=topic,
                title=title,
                description=description,
                seo_summary=seo_summary,
                tags=tags,
                hashtags=hashtags,
                chapters=final_chapters,
                chapters_available=final_chapters_available,
                duration_seconds=duration_seconds,
                video_slug=video_slug,
            )
        except OSError as e:
            warnings.append(f"Failed to write metadata JSON artifact: {e}")

        return MetadataResult(
            success=True,
            topic=topic,
            title=title,
            description=description,
            seo_summary=seo_summary,
            tags=tags,
            hashtags=hashtags,
            chapters=final_chapters,
            chapters_available=final_chapters_available,
            chapters_omitted_reason=None if final_chapters_available else final_chapters_reason,
            duration_seconds=duration_seconds,
            output_path=output_path,
            llm_provider=getattr(self.llm_provider, "name", None) or type(self.llm_provider).__name__,
            llm_model=getattr(self.llm_provider, "last_model_used", None),
            used_fallback_model=getattr(self.llm_provider, "last_used_fallback", None),
            warnings=warnings,
        )

    # ---- deterministic chapter timing ----------------------------------------

    @staticmethod
    def _derive_chapter_timestamps(
        script: ScriptResult, duration_seconds: Optional[float]
    ) -> Tuple[List[float], bool, Optional[str]]:
        """Derive chapter start timestamps from the same per-section timing
        VideoAssemblyService/VisualMediaService already use - never a
        second, independently-invented timing scheme.

        Returns:
            (timestamps, available, reason_if_unavailable)
        """
        if duration_seconds is None or duration_seconds <= 0:
            return [], False, "No final video duration available to derive chapter timestamps from"
        if len(script.sections) < MIN_SECTIONS_FOR_CHAPTERS:
            return (
                [],
                False,
                f"Fewer than {MIN_SECTIONS_FOR_CHAPTERS} script sections - chapters would not be meaningful",
            )

        section_durations = calculate_section_durations(script.sections, duration_seconds)
        timestamps = []
        cumulative = 0.0
        for section_duration in section_durations:
            timestamps.append(cumulative)
            cumulative += section_duration
        return timestamps, True, None

    @staticmethod
    def _build_candidate_chapters(
        script: ScriptResult, timestamps: List[float], labels: List[str]
    ) -> List[Chapter]:
        """Pair deterministic timestamps with LLM-provided labels, falling
        back to the section's own heading when a label is missing/empty -
        bounded deterministic repair, not a failure."""
        chapters = []
        for index, timestamp in enumerate(timestamps):
            label = labels[index].strip() if index < len(labels) and labels[index] else ""
            if not label:
                label = script.sections[index].heading
            chapters.append(
                Chapter(timestamp_seconds=timestamp, timestamp_text=format_youtube_timestamp(timestamp), title=label)
            )
        return chapters

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(
        topic: str, script: ScriptResult, duration_seconds: Optional[float], chapter_timestamps: Optional[List[float]]
    ) -> str:
        section_blocks = "\n\n".join(
            f'Section {index} - "{section.heading}":\n{section.narration}'
            for index, section in enumerate(script.sections)
        )

        duration_line = ""
        if duration_seconds:
            minutes, seconds = divmod(int(duration_seconds), 60)
            duration_line = f"Approximate video length: {minutes}m {seconds}s\n"

        schema_fields = [
            '"title": "one final YouTube title, natural English, accurate, concise, no clickbait, '
            'under 100 characters"',
            '"description": "a professional multi-paragraph YouTube description grounded only in the '
            'actual script content"',
            '"seo_summary": "one or two sentence SEO-friendly summary"',
            '"tags": ["focused list of semantically relevant tags/keywords, no spam"]',
            '"hashtags": ["3 to 5 relevant hashtags, each starting with #"]',
        ]
        chapter_instruction = ""
        if chapter_timestamps:
            schema_fields.append(
                f'"chapter_labels": [exactly {len(chapter_timestamps)} short chapter titles, one per '
                "section above, in the same order]"
            )
            chapter_instruction = (
                "\nAlso provide chapter_labels: exactly one short, concise chapter title per section "
                "above, in the same order. The timestamps for these chapters are ALREADY FIXED and "
                "computed separately - only supply the labels, never timestamps."
            )
        schema_block = "{\n  " + ",\n  ".join(schema_fields) + "\n}"

        return (
            "You are writing professional YouTube upload metadata for a narrated educational video. "
            "The metadata must accurately reflect ONLY the actual content below - never invent facts, "
            "claims, statistics, or hooks the script doesn't support. Avoid clickbait, fake urgency, "
            "misleading claims, ALL CAPS, excessive punctuation/emojis, keyword stuffing, and spammy or "
            "unrelated tags/hashtags.\n\n"
            f"Video topic: {topic}\n"
            f"Working title: {script.video_title}\n"
            f"{duration_line}"
            f"Hook: {script.hook}\n"
            f"Introduction: {script.introduction}\n\n"
            f"Sections, in order:\n{section_blocks}\n\n"
            f"Conclusion: {script.conclusion}\n"
            f"Call to action: {script.call_to_action}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly this "
            f"shape:\n{schema_block}"
            f"{chapter_instruction}"
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> dict:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise MetadataAgentError("Metadata response was not a JSON object")

        title = str(data.get("title") or "").strip()
        description = str(data.get("description") or "").strip()
        if not title or not description:
            raise MetadataAgentError("Metadata response is missing a title or description")

        return {
            "title": title,
            "description": description,
            "seo_summary": str(data.get("seo_summary") or "").strip(),
            "tags": [str(t) for t in (data.get("tags") or [])],
            "hashtags": [str(h) for h in (data.get("hashtags") or [])],
            "chapter_labels": [str(c) for c in (data.get("chapter_labels") or [])],
        }

    # ---- artifact output --------------------------------------------------

    def _write_artifact(
        self,
        *,
        topic: str,
        title: str,
        description: str,
        seo_summary: Optional[str],
        tags: List[str],
        hashtags: List[str],
        chapters: List[Chapter],
        chapters_available: bool,
        duration_seconds: Optional[float],
        video_slug: Optional[str],
    ) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        slug = video_slug or self._slugify(title)
        output_path = os.path.join(self.output_dir, f"{slug}.json")

        payload = {
            "topic": topic,
            "title": title,
            "description": description,
            "seo_summary": seo_summary,
            "tags": tags,
            "hashtags": hashtags,
            "chapters": [chapter.model_dump() for chapter in chapters],
            "chapters_available": chapters_available,
            "duration_seconds": duration_seconds,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_path

    @staticmethod
    def _slugify(text: str) -> str:
        slug = _SLUG_INVALID_RE.sub("-", text.lower()).strip("-")
        return slug or "video"

    @staticmethod
    def _failure(topic: str, error: str) -> MetadataResult:
        return MetadataResult(success=False, topic=topic, error=error)
