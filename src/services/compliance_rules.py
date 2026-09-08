# Centralized, testable publish-decision rules for the Copyright /
# Compliance Agent - deliberately the ONLY place that decides PASS / REVIEW
# / BLOCK, so the policy can be audited/changed in one place rather than
# scattered across check functions or the agent's orchestration.
from __future__ import annotations

from typing import List, Optional, Tuple

from src.models.compliance import PublishDecision, RiskLevel, SemanticReviewResult


def decide_publish_status(
    blockers: List[str], warnings: List[str], semantic_review: Optional[SemanticReviewResult]
) -> Tuple[PublishDecision, RiskLevel]:
    """Decide the final PASS / REVIEW / BLOCK decision and risk level.

    Rules (in priority order):
      1. Any deterministic blocker -> BLOCK, high risk. A blocker always
         wins - even a clean semantic review never overrides it.
      2. The semantic review didn't run/complete (LLM failure, timeout,
         malformed response, or no LLM provider configured at all) -> REVIEW,
         never a fabricated PASS.
      3. The semantic review found any observable risk, or any deterministic
         warning exists (e.g. provenance that couldn't be verified in this
         context) -> REVIEW, medium risk. Semantic findings are advisory
         only and never escalate to BLOCK on their own.
      4. Otherwise -> PASS, low risk: no known blocking issue was found from
         the evidence available to this system (see
         ``src.models.compliance.DISCLAIMER`` - never a legal/copyright
         guarantee).
    """
    if blockers:
        return "BLOCK", "high"

    if semantic_review is None or not semantic_review.performed:
        return "REVIEW", "medium"

    if semantic_review.findings or warnings:
        return "REVIEW", "medium"

    return "PASS", "low"
