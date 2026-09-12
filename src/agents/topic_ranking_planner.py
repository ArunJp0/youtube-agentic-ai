# Topic Ranking Planner: the ONE bounded semantic LLM call the Topic
# Planner Agent makes - judges relevance/evergreen-ness/pipeline
# suitability for a whole BATCH of surviving candidates in a single
# request, never one call per candidate. Mirrors
# VisualContextPlanner/MusicContextPlanner/ComplianceReviewer's shape and
# constraints exactly: a small, focused component (not a LangGraph agent),
# reuses the existing LLMProvider abstraction, and never raises for an
# LLM/parsing failure - it always returns a COMPLETE outcome (one
# SemanticTopicJudgment per input candidate), falling back to a neutral
# deterministic judgment internally rather than a fabricated opinion or a
# caller-visible exception.
from __future__ import annotations

import json
from typing import List, Optional

from src.llm.provider import LLMProvider
from src.models.topic_planner import SemanticTopicJudgment, TopicCandidate, TopicRankingOutcome
from src.services.llm_json import JsonExtractionError, extract_json_object

# Neutral deterministic sub-scores used when semantic ranking is
# unavailable (no LLM provider configured, or the call/parse failed) -
# genuinely "no opinion", never a guessed high/low score.
_FALLBACK_SCORE = 0.5
_FALLBACK_RATIONALE = "Semantic ranking unavailable - neutral deterministic score"


class TopicRankingPlannerError(Exception):
    """Raised only for programmer/configuration errors (e.g. an empty
    candidate list). LLM failures, malformed responses, and validation
    failures are NEVER raised from ``rank`` - they are caught internally
    and result in a fallback-filled (``performed=False``) outcome instead.
    """


class TopicRankingPlanner:
    """Produces one TopicRankingOutcome for a batch of TopicCandidates
    using a single LLM call, with a conservative neutral-fallback outcome
    if that call fails."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    def rank(self, candidates: List[TopicCandidate], niche: Optional[str] = None) -> TopicRankingOutcome:
        """Produce a TopicRankingOutcome for ``candidates``.

        Never raises for LLM/parsing failures - those result in
        ``performed=False`` with every candidate still receiving a neutral
        fallback judgment. Only raises TopicRankingPlannerError for
        configuration/programmer errors.

        Args:
            candidates: Surviving (already deduplicated) candidates to judge
            niche: Optional configured niche/category for relevance judging

        Returns:
            A TopicRankingOutcome with exactly one judgment per candidate
        """
        if not candidates:
            raise TopicRankingPlannerError("candidates is required and must be non-empty")

        provider_name = getattr(self.llm_provider, "name", None) or type(self.llm_provider).__name__

        try:
            prompt = self._build_prompt(candidates, niche)
            raw_response = self.llm_provider.generate_text(prompt)
            judgments = self._parse_response(raw_response, candidates)
        except Exception as e:
            return TopicRankingOutcome(
                performed=False,
                judgments=[self._fallback_judgment(c) for c in candidates],
                fallback_reason=f"Topic semantic ranking unavailable: {e}",
                llm_provider=provider_name,
                llm_model=getattr(self.llm_provider, "last_model_used", None),
                used_fallback_model=getattr(self.llm_provider, "last_used_fallback", None),
            )

        return TopicRankingOutcome(
            performed=True,
            judgments=judgments,
            llm_provider=provider_name,
            llm_model=getattr(self.llm_provider, "last_model_used", None),
            used_fallback_model=getattr(self.llm_provider, "last_used_fallback", None),
        )

    # ---- prompt building -----------------------------------------------

    @staticmethod
    def _build_prompt(candidates: List[TopicCandidate], niche: Optional[str]) -> str:
        niche_line = f"Configured content niche: {niche}\n" if niche else "No specific niche is configured - judge general educational/evergreen suitability.\n"
        candidate_lines = "\n".join(
            f'- "{c.normalized_title}" (original: "{c.raw_title}"{f", category: {c.category}" if c.category else ""})'
            for c in candidates
        )
        return (
            "You are ranking candidate video topics for a faceless educational YouTube channel's "
            "automated production pipeline (narrated stock-footage videos, 2-4 minutes long). For EACH "
            f"candidate below, judge three things on a 0.0-1.0 scale:\n{niche_line}\n"
            "- relevance_score: how well the topic fits the configured niche (or general educational "
            "content if no niche is configured)\n"
            "- evergreen_score: how timeless/lasting the topic's interest is, as opposed to tied to a "
            "specific current event or moment\n"
            "- suitability_score: how well the topic suits this pipeline specifically - explainable with "
            "narration and illustrative stock footage, not requiring live footage, breaking news, or "
            "a specific real person's likeness\n\n"
            f"Candidates:\n{candidate_lines}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly this shape:\n"
            "{\n"
            '  "judgments": [\n'
            '    {"normalized_title": "the exact normalized title copied from above", '
            '"relevance_score": 0.0-1.0, "evergreen_score": 0.0-1.0, "suitability_score": 0.0-1.0, '
            '"rationale": "one short sentence"}\n'
            "  ]\n"
            "}\n"
            "Include exactly one judgment per candidate listed above."
        )

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str, candidates: List[TopicCandidate]) -> List[SemanticTopicJudgment]:
        try:
            payload = extract_json_object(raw_text)
            data = json.loads(payload)
        except (JsonExtractionError, json.JSONDecodeError) as e:
            raise TopicRankingPlannerError(f"Malformed ranking response: {e}") from e
        if not isinstance(data, dict):
            raise TopicRankingPlannerError("Ranking response was not a JSON object")

        by_title = {}
        for item in data.get("judgments") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("normalized_title") or "").strip()
            if not title:
                continue
            by_title[title] = item

        judgments: List[SemanticTopicJudgment] = []
        for candidate in candidates:
            item = by_title.get(candidate.normalized_title)
            if item is None:
                # This specific candidate wasn't judged (a partial/
                # malformed response) - neutral fallback for it alone,
                # never block the whole batch over one missing entry.
                judgments.append(TopicRankingPlanner._fallback_judgment(candidate))
                continue
            judgments.append(
                SemanticTopicJudgment(
                    normalized_title=candidate.normalized_title,
                    relevance_score=_clamp(item.get("relevance_score")),
                    evergreen_score=_clamp(item.get("evergreen_score")),
                    suitability_score=_clamp(item.get("suitability_score")),
                    rationale=str(item.get("rationale") or ""),
                )
            )
        return judgments

    @staticmethod
    def _fallback_judgment(candidate: TopicCandidate) -> SemanticTopicJudgment:
        return SemanticTopicJudgment(
            normalized_title=candidate.normalized_title,
            relevance_score=_FALLBACK_SCORE,
            evergreen_score=_FALLBACK_SCORE,
            suitability_score=_FALLBACK_SCORE,
            rationale=_FALLBACK_RATIONALE,
        )


def _clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _FALLBACK_SCORE
    return max(0.0, min(1.0, number))
