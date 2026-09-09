# Compliance Remediation data models - typed records for the bounded
# REVIEW -> targeted script correction -> fresh Compliance re-evaluation
# loop (see src/workflows/pipeline_graph.py's script_revision_node/
# route_after_compliance and src/services/finding_localizer.py).
#
# Deliberately small and additive: nothing here ever decides PASS/REVIEW/
# BLOCK itself (that stays exclusively in src.services.compliance_rules) -
# these models only record what a remediation attempt looked at, what it
# changed, and what happened, for typed state and durable audit purposes.
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.models.compliance import ComplianceResult

# Bounded retry: original evaluation + at most this many corrected
# candidates. Centralized here (not scattered across pipeline_graph.py) so
# the policy can be tuned/overridden in one place - mirrors the existing
# bounded-retry precedent in this codebase (VisualQCService's
# DEFAULT_MAX_REPLACEMENT_ATTEMPTS, ScriptAgent's
# MAX_SECTION_GENERATION_ATTEMPTS).
DEFAULT_MAX_REMEDIATION_ATTEMPTS = 2


class LocalizedFinding(BaseModel):
    """One semantic finding, mapped (or not) to a specific ScriptSection.

    ``actionable`` is the single source of truth for "can remediation do
    anything about this finding" - False whenever no confident section
    match exists, never a guess.
    """

    finding_category: str = Field(min_length=1)
    finding_description: str = Field(min_length=1)
    section_index: Optional[int] = Field(default=None, ge=0)
    section_heading: Optional[str] = Field(default=None)
    localization_source: Optional[str] = Field(
        default=None, description="'llm' (from SemanticReviewFinding.related_section_heading) or 'fallback_match'"
    )
    actionable: bool = Field(default=False)


class ScriptCorrection(BaseModel):
    """One targeted section rewrite applied during a remediation attempt."""

    section_index: int = Field(ge=0)
    section_heading: str = Field(min_length=1)
    original_narration: str = Field(min_length=1)
    revised_narration: str = Field(min_length=1)
    finding_description: str = Field(min_length=1)
    used_research_refresh: bool = Field(default=False)
    research_refresh_reason: Optional[str] = Field(default=None)


class RemediationAttemptRecord(BaseModel):
    """One bounded remediation attempt's audit trail: what was found, what
    was corrected, and (once the next Compliance evaluation runs) what it
    resulted in. ``resulting_decision``/``resulting_run_id`` start ``None``
    and are filled in by the following compliance_node call - never
    guessed ahead of the real evaluation."""

    attempt_number: int = Field(ge=1)
    parent_run_id: Optional[str] = Field(default=None)
    triggering_decision: str = Field(default="REVIEW")
    findings_considered: int = Field(ge=0)
    findings_localized: int = Field(ge=0)
    corrections: List[ScriptCorrection] = Field(default_factory=list)
    started_at: str = Field(min_length=1)
    completed_at: Optional[str] = Field(default=None)
    resulting_run_id: Optional[str] = Field(default=None)
    resulting_decision: Optional[str] = Field(default=None)


class PersistedComplianceRecord(BaseModel):
    """The durable envelope ComplianceRecordStore persists - full run/
    revision identity plus the genuine ComplianceResult it wraps."""

    run_id: str = Field(min_length=1)
    parent_run_id: Optional[str] = Field(default=None)
    attempt_number: int = Field(default=0, ge=0)
    # Empty string means "unknown" - only ever seen when transparently
    # upgrading a legacy pre-envelope record (see ComplianceRecordStore.
    # read_envelope) that predates this field existing at all; every
    # record newly WRITTEN by this store always supplies a real timestamp.
    evaluated_at: str = Field(default="")
    result: ComplianceResult
