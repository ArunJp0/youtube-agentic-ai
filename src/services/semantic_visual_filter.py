# Lightweight deterministic semantic filtering, applied on top of a
# VisualContextPlanner (or deterministic-fallback) SectionVisualPlan.
#
# The LLM/fallback plan provides the semantic understanding (what a section
# is actually about, and which literal-but-wrong interpretations to avoid);
# this module only *enforces* that plan against a candidate's available
# metadata. It is deliberately simple - a keyword-overlap check, not a
# classifier - so it stays explainable and testable, consistent with "no
# giant brittle keyword engine".
from __future__ import annotations

from src.models.visual_plan import SectionVisualPlan
from src.tools.media_provider import MediaCandidate


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def passes_avoid_filter(candidate: MediaCandidate, plan: SectionVisualPlan) -> bool:
    """Reject a candidate whose available content metadata clearly matches
    one of the section's ``avoid_concepts``.

    Candidate metadata (``content_hint``, e.g. a Pexels page-URL slug or
    photo "alt" text) is the only signal available without downloading the
    asset. When no metadata is available at all, the candidate is not
    blocked - this filter only rejects on positive evidence of a mismatch,
    never on absence of information.
    """
    hint = _normalize(candidate.content_hint or "")
    if not hint:
        return True
    return not any(
        _normalize(avoid_phrase) in hint for avoid_phrase in plan.avoid_concepts if avoid_phrase.strip()
    )


def relevance_score(candidate: MediaCandidate, plan: SectionVisualPlan) -> float:
    """A simple 0-1 keyword-overlap score between a candidate's content
    metadata and the section's positive visual intents/search queries.

    Diagnostic only - it measures lexical overlap with the plan, not actual
    visual/factual correctness, and is intended for logging and future QC
    integration, not as a hard selection gate.
    """
    hint_words = set(_normalize(candidate.content_hint or "").split())
    if not hint_words:
        return 0.0

    positive_words = set()
    for phrase in list(plan.visual_intents) + list(plan.search_queries):
        positive_words.update(_normalize(phrase).split())
    if not positive_words:
        return 0.0

    overlap = hint_words & positive_words
    return round(len(overlap) / len(positive_words), 3)
