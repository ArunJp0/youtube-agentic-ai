# Compliance Reviewer: the ONE bounded semantic LLM call the Copyright /
# Compliance Agent makes, focused only on observable, describable semantic
# risks a deterministic rule cannot reliably determine (title/description/
# thumbnail-vs-content consistency, misleading presentation, unsupported
# strong claims). It is advisory only and NEVER asked to make a legal or
# copyright determination - see the prompt below.
#
# Mirrors ThumbnailPlanner/MetadataAgent/MusicContextPlanner's shape and
# constraints exactly: a small, focused component (not a LangGraph agent),
# one request/response, reuses the existing LLMProvider abstraction, and
# never raises for an LLM/parsing failure - a failure is reported via
# SemanticReviewResult.performed=False/fallback_reason, never a fabricated
# clean result (see src/services/compliance_rules.py, which treats an
# unperformed review as REVIEW, never PASS).
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from src.llm.provider import LLMProvider
from src.models.metadata import MetadataResult
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailResult
from src.services.llm_json import extract_json_object
from src.models.compliance import SemanticReviewFinding, SemanticReviewResult

_VALID_SEVERITIES = {"low", "medium", "high"}


class ComplianceReviewerError(Exception):
    """Raised only for programmer/configuration errors (e.g. a missing topic).

    LLM failures, malformed responses, and validation failures are NEVER
    raised from ``review`` - they are caught internally and result in an
    unperformed (``performed=False``) SemanticReviewResult instead.
    """


class ComplianceReviewer:
    """Produces one SemanticReviewResult for a video's publishing package
    using a single LLM call, with a conservative unperformed-review result
    if that call fails.

    Deliberately scoped to ONE LLM request per review (never one call per
    artifact), consistent with every other planner's cost discipline in
    this project.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the reviewer.

        Args:
            llm_provider: LLMProvider implementation used for the single
                whole-package semantic compliance review call
        """
        self.llm_provider = llm_provider

    def review(
        self,
        topic: str,
        script: Optional[ScriptResult],
        metadata_result: Optional[MetadataResult],
        thumbnail_result: Optional[ThumbnailResult],
    ) -> SemanticReviewResult:
        """Produce a SemanticReviewResult for ``topic`` and whatever real
        artifacts are available.

        Never raises for LLM/parsing failures - those result in
        ``performed=False`` with ``fallback_reason`` set. Only raises
        ComplianceReviewerError for configuration/programmer errors.

        Args:
            topic: Overall video topic
            script: Structured script, if available (standalone callers may
                only have a reconstructed/topic-only context)
            metadata_result: Already-generated MetadataResult, if available
            thumbnail_result: Already-generated ThumbnailResult, if available

        Returns:
            A SemanticReviewResult describing the outcome
        """
        if not topic:
            raise ComplianceReviewerError("topic is required")

        provider_name = getattr(self.llm_provider, "name", None) or type(self.llm_provider).__name__
        model_used = getattr(self.llm_provider, "last_model_used", None)
        fallback_used = getattr(self.llm_provider, "last_used_fallback", None)

        try:
            prompt = self._build_prompt(topic, script, metadata_result, thumbnail_result)
            raw_response = self.llm_provider.generate_text(prompt)
            findings, summary = self._parse_response(raw_response)
        except Exception as e:
            return SemanticReviewResult(
                performed=False,
                fallback_reason=f"Semantic compliance review unavailable: {e}",
                llm_provider=provider_name,
                llm_model=getattr(self.llm_provider, "last_model_used", None),
                used_fallback_model=getattr(self.llm_provider, "last_used_fallback", None),
            )

        return SemanticReviewResult(
            performed=True,
            findings=findings,
            summary=summary,
            llm_provider=provider_name,
            llm_model=getattr(self.llm_provider, "last_model_used", None),
            used_fallback_model=getattr(self.llm_provider, "last_used_fallback", None),
        )

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(
        topic: str,
        script: Optional[ScriptResult],
        metadata_result: Optional[MetadataResult],
        thumbnail_result: Optional[ThumbnailResult],
    ) -> str:
        context_blocks: List[str] = [f"Video topic: {topic}"]

        if script is not None:
            section_summaries = "\n".join(f"- {s.heading}: {s.narration}" for s in script.sections)
            context_blocks.append(
                f"Script hook: {script.hook}\nScript introduction: {script.introduction}\n"
                f"Script sections:\n{section_summaries}\nScript conclusion: {script.conclusion}"
            )
        else:
            context_blocks.append("Script content: not available for this review - judge only on the fields below.")

        if metadata_result is not None and metadata_result.success:
            context_blocks.append(
                f"Published title: {metadata_result.title}\n"
                f"Published description: {metadata_result.description}\n"
                f"Published tags: {metadata_result.tags}\n"
                f"Published hashtags: {metadata_result.hashtags}"
            )
        else:
            context_blocks.append("Published metadata: not available for this review.")

        if thumbnail_result is not None and thumbnail_result.plan is not None:
            plan = thumbnail_result.plan
            context_blocks.append(
                f"Thumbnail hook text: {plan.hook_text}\nThumbnail visual concept: {plan.visual_concept}"
            )
        else:
            context_blocks.append("Thumbnail concept: not available for this review.")

        context = "\n\n".join(context_blocks)

        return (
            "You are performing a SEMANTIC COMPLIANCE REVIEW of a YouTube video's publishing package "
            "(title, description, thumbnail, and script content) before it is published. This is an "
            "ADVISORY content-consistency review only - you are NOT a legal authority and must NEVER "
            "make a legal, copyright, or licensing determination or claim something is 'safe' to "
            "publish. Only identify SPECIFIC, OBSERVABLE risks that a human/automated deterministic "
            "rule could not reliably catch, such as:\n"
            "- the published title or description not matching what the script/content actually covers\n"
            "- the thumbnail's hook text or visual concept not matching the actual topic/content\n"
            "- materially misleading or clickbait presentation not supported by the actual content\n"
            "- strong claims in the title/description/thumbnail that the script content does not support\n"
            "- other obvious factual/consistency inconsistencies between the script, metadata, and thumbnail\n\n"
            "If you find no such issues, return an empty findings list - do not invent problems.\n\n"
            f"{context}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly this shape:\n"
            "{\n"
            '  "findings": [\n'
            '    {"category": "short category slug, e.g. title_content_mismatch", '
            '"description": "specific, concrete description of the observed risk", '
            '"severity": "low, medium, or high", '
            '"related_section_heading": "the EXACT script section heading above this finding concerns, '
            'copied verbatim, or null if it concerns the title/thumbnail/overall content rather than one '
            'specific section"}\n'
            "  ],\n"
            '  "summary": "one or two sentence overall summary"\n'
            "}"
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> Tuple[List[SemanticReviewFinding], str]:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ComplianceReviewerError("Semantic compliance review response was not a JSON object")

        findings: List[SemanticReviewFinding] = []
        for item in data.get("findings") or []:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "").strip()
            description = str(item.get("description") or "").strip()
            if not category or not description:
                continue
            severity = str(item.get("severity") or "medium").strip().lower()
            if severity not in _VALID_SEVERITIES:
                severity = "medium"
            related_section_heading = item.get("related_section_heading")
            related_section_heading = (
                str(related_section_heading).strip() if related_section_heading not in (None, "") else None
            )
            findings.append(
                SemanticReviewFinding(
                    category=category,
                    description=description,
                    severity=severity,
                    related_section_heading=related_section_heading,
                )
            )

        summary = str(data.get("summary") or "").strip()
        return findings, summary
