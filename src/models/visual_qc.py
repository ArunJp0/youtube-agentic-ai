# Visual QC data models: a vision evaluator's raw per-asset judgment
# (RawAssetVerdict), and the structured results VisualQCService produces
# after applying centralized thresholds/policy on top of that judgment.
#
# Severity/disposition model (see docs/DECISIONS.md for the full rationale):
#   PASS        - an individual asset is fine as-is (approved/neutral/
#                 metadata_fallback/error).
#   REPLACEABLE - an individual asset is weak/misleading; bounded
#                 replacement is attempted, then the asset is DROPPED
#                 (never kept as a known-bad "last resort") if replacement
#                 doesn't resolve it. A single replaceable asset is never,
#                 by itself, a whole-video failure.
#   CRITICAL    - a SECTION-level (never a single-asset-level) outcome:
#                 after every recovery strategy (replace, drop, neutral
#                 fallback) is exhausted, that section still has zero safe
#                 usable visual coverage. Only CRITICAL halts the pipeline.
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AssetDisposition = Literal["pass", "replaceable"]
SectionDisposition = Literal["pass", "recovered", "critical"]
QCDisposition = Literal["pass", "recovered", "critical"]


class RawAssetVerdict(BaseModel):
    """One vision evaluator's raw judgment for a single asset, from its
    representative frame(s) - before VisualQCService applies centralized
    thresholds to turn it into an approved/rejected decision. Evaluators
    only report what they observed; they never decide policy."""

    asset_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0, description="0 (irrelevant) to 1 (highly relevant)")
    misleading_or_conflicting: bool = Field(
        default=False,
        description="True if the visual content materially misrepresents the section's "
        "meaning or matches one of its avoid_concepts, regardless of lexical score",
    )
    detected_visual_summary: Optional[str] = Field(
        default=None, description="One short phrase describing what the frame(s) actually show"
    )
    reason: str = Field(default="", description="One short sentence explaining the score/verdict")


class AssetQCResult(BaseModel):
    """The final QC outcome for one visual slot, after any bounded
    replacement attempts - what VideoAssemblyService's asset should
    ultimately be judged as, and why."""

    asset_id: str = Field(min_length=1)
    section_index: int = Field(ge=0)
    slot_index: int = Field(ge=0)
    approved: bool = Field(description="Whether this asset should be kept in the final video")
    decision: str = Field(
        description="'approved' (highly relevant), 'neutral' (acceptable/neutral B-roll), "
        "'weak' (below threshold, replacement recommended but none available/exhausted), "
        "'rejected' (misleading, kept only as an unavoidable last resort), "
        "'metadata_fallback' (vision QC unavailable, kept on the upstream metadata filter's "
        "say-so), or 'error' (no usable verdict, kept unvalidated)"
    )
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(default="")
    detected_visual_summary: Optional[str] = Field(default=None)
    misleading_or_conflicting: bool = Field(default=False)
    retry_recommended: bool = Field(
        default=False, description="Whether this specific asset's own verdict called for replacement"
    )
    evaluation_source: str = Field(
        default="vision",
        description="'vision' (a real vision-model verdict was used), 'metadata_fallback' "
        "(the vision evaluator failed for this section - fell back to the upstream metadata "
        "filter's prior approval), or 'error' (a verdict was expected but not usable)",
    )
    replaced: bool = Field(default=False, description="True if this asset replaced an earlier rejection")
    replacement_attempts: int = Field(default=0, ge=0)
    disposition: AssetDisposition = Field(
        default="pass", description="'pass' (kept as-is) or 'replaceable' (was weak/misleading for this slot)"
    )
    dropped: bool = Field(
        default=False,
        description="True if this slot's asset was dropped (excluded from the final video) after bounded "
        "replacement failed to resolve it - a dropped slot never appears in the final asset list",
    )
    used_as_neutral_fallback: bool = Field(
        default=False, description="True if this slot's final asset is a locally-generated neutral fallback visual"
    )


class SectionQCResult(BaseModel):
    """QC outcomes for every visual slot in one script section, in slot
    order, plus the section-level coverage outcome computed AFTER any
    assets were dropped/replaced (see module docstring)."""

    section_index: int = Field(ge=0)
    assets: List[AssetQCResult] = Field(default_factory=list)
    dropped_count: int = Field(default=0, ge=0, description="How many of this section's slots were dropped")
    usable_asset_count: int = Field(
        default=0, ge=0, description="How many assets remain in the final video for this section (kept + fallback)"
    )
    neutral_fallback_used: bool = Field(default=False)
    disposition: SectionDisposition = Field(
        default="pass",
        description="'pass' (no recovery needed), 'recovered' (a drop/replacement/fallback resolved this "
        "section safely), or 'critical' (this section has zero safe usable visual coverage even after every "
        "recovery strategy was exhausted)",
    )


class VisualQCResult(BaseModel):
    """Structured Visual QC outcome for an entire VisualResult."""

    topic: str = Field(min_length=1)
    provider: str = Field(min_length=1, description="Vision evaluator used, e.g. 'mock' or 'gemini'")
    model: Optional[str] = Field(default=None, description="Underlying vision model name, if applicable")
    success: bool = Field(description="Whether the QC process itself completed without a hard failure")
    sections: List[SectionQCResult] = Field(default_factory=list)
    total_assets_checked: int = Field(default=0, ge=0)
    approved_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    replaced_count: int = Field(default=0, ge=0)
    vision_calls_made: int = Field(default=0, ge=0)
    fallback_used: bool = Field(
        default=False, description="True if any section fell back to metadata-only approval"
    )
    fallback_reason: Optional[str] = Field(default=None)
    repetition_warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, description="Set only if QC could not run at all")

    # ---- disposition/recovery observability (see module docstring) --------
    disposition: QCDisposition = Field(
        default="pass",
        description="Overall outcome: 'pass' (nothing needed recovery), 'recovered' (at least one asset was "
        "dropped/replaced/neutral-fallback-covered but every section still has safe coverage), or 'critical' "
        "(at least one section has zero safe usable visual coverage - the ONLY condition that should halt "
        "the pipeline before Video Assembly)",
    )
    dropped_count: int = Field(default=0, ge=0, description="Total slots dropped across all sections")
    neutral_fallback_count: int = Field(default=0, ge=0, description="Total slots covered by a neutral fallback visual")
    critical_section_indices: List[int] = Field(
        default_factory=list, description="Section indices with zero safe usable visual coverage, if any"
    )
