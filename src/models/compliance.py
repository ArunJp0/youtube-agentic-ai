# Copyright / Compliance Agent data models.
#
# This agent is a pre-publishing SAFETY GATE, not a legal authority: PASS
# means no known blocking issue was found from the evidence available to
# this system - it is never a legal or copyright guarantee. See
# DISCLAIMER below, which every ComplianceResult carries verbatim.
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PublishDecision = Literal["PASS", "REVIEW", "BLOCK"]
RiskLevel = Literal["low", "medium", "high"]
CheckStatus = Literal["ok", "warning", "blocker", "unavailable"]

DISCLAIMER = (
    "PASS means no known blocking issue was found from the evidence available to this "
    "system; it is not a legal or copyright guarantee."
)


class ProvenanceCheckResult(BaseModel):
    """One deterministic check's outcome - never fabricated, always traceable
    to a specific rule."""

    check_name: str = Field(min_length=1)
    status: CheckStatus
    detail: str = Field(min_length=1)


class RequiredAttribution(BaseModel):
    """Machine-readable required-attribution record, intended for a future
    Upload Agent / metadata stage to consume directly rather than parsing
    human-readable logs."""

    asset_type: str = Field(min_length=1, description="e.g. 'bgm', 'visual'")
    asset_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    attribution_text: str = Field(min_length=1)


class SemanticReviewFinding(BaseModel):
    """One observable semantic risk flagged by the advisory LLM review -
    never a legal/copyright conclusion, only a specific, describable
    consistency/presentation concern."""

    category: str = Field(min_length=1, description="e.g. 'title_content_mismatch', 'unsupported_claim'")
    description: str = Field(min_length=1)
    severity: str = Field(default="medium", description="'low', 'medium', or 'high'")
    related_section_heading: Optional[str] = Field(
        default=None,
        description=(
            "The exact ScriptSection.heading this finding concerns, if any (verbatim, as supplied in the "
            "review prompt) - None if the finding concerns the title/thumbnail/overall content rather than "
            "one specific section. Used only for remediation's finding-to-section localization; never "
            "affects the PASS/REVIEW/BLOCK decision itself."
        ),
    )


class SemanticReviewResult(BaseModel):
    """Outcome of the single bounded semantic compliance review LLM call.

    ``performed=False`` means the call failed/was unavailable (429, 503,
    timeout, malformed response, or no LLM provider configured at all) -
    this is never treated as a clean pass; see ``compliance_rules.py``.
    """

    performed: bool
    findings: List[SemanticReviewFinding] = Field(default_factory=list)
    summary: str = Field(default="")
    fallback_reason: Optional[str] = Field(default=None)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    used_fallback_model: Optional[bool] = Field(default=None)


class ComplianceResult(BaseModel):
    """Structured output of the Copyright / Compliance Agent."""

    success: bool = Field(description="Whether the compliance review itself completed (not the publish decision)")
    topic: str = Field(default="")
    publish_decision: Optional[PublishDecision] = Field(default=None)
    risk_level: Optional[RiskLevel] = Field(default=None)
    checks: List[ProvenanceCheckResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    attribution_required: bool = Field(
        default=False, description="True only when a confirmed catalog/provenance record requires attribution"
    )
    required_attributions: List[RequiredAttribution] = Field(default_factory=list)
    provenance_summary: str = Field(default="")
    semantic_review: Optional[SemanticReviewResult] = Field(default=None)
    llm_used: bool = Field(default=False)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    used_fallback_model: Optional[bool] = Field(default=None)
    artifacts_inspected: List[str] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)
    error: Optional[str] = Field(default=None, description="Error message if the review itself failed to run")
