# Autonomous Orchestration data models.
#
# AutonomousRunController is coordination-only - it has no Research/Script/
# Video/Compliance/YouTube business logic of its own. These models exist to
# make ONE autonomous run's outcome typed, durable, and diagnosable, never
# to duplicate the typed results the underlying stages already produce
# (TopicSelectionResult, PipelineState, UploadResult) - this module only
# records REFERENCES to those (topic, pipeline run_id, compliance decision,
# publishing video id), never a copy of their full content.
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# "skipped_locked" and "planned_only" are additive beyond the milestone
# spec's core list - required by STEP 6 (concurrency) and STEP 10 (dry-run)
# respectively, and named consistently with the rest (lowercase snake-ish
# strings, matching PlannerStatus/UploadStatus/PublishDecision throughout
# this project).
AutonomousRunStatus = Literal[
    "started",
    "no_topic",
    "generating",
    "compliance_blocked",
    "publishing_disabled",
    "published_private",
    "scheduled",
    "publishing_failed",
    "failed",
    "completed",
    "skipped_locked",
    "planned_only",
]

TriggerSource = Literal["scheduled", "manual", "dry_run"]


class AutonomousRunRecord(BaseModel):
    """Durable, typed result of one AutonomousRunController.run_once() call.

    Persisted for every outcome (including NO_TOPIC/FAILED/skipped-locked)
    so an operator can diagnose any run without needing to have been
    watching it live. Never carries secrets - only the same public
    identifiers/decisions the underlying real stores (TopicPlanStore,
    ProvenanceManifestStore, ComplianceRecordStore, PublishingRecordStore)
    already persist.
    """

    run_id: str = Field(min_length=1, description="This autonomous run's own stable identifier")
    trigger_source: TriggerSource
    scheduled_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of the scheduled occurrence this run corresponds to - the "
        "idempotency key for 'did this exact occurrence already run'. None for manual/dry-run invocations.",
    )
    started_at: str = Field(min_length=1, description="ISO 8601 UTC timestamp this run started")
    finished_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp this run reached a terminal state")
    status: AutonomousRunStatus = Field(default="started")

    # Topic Planner reference - never a duplicate of the full TopicSelectionResult.
    topic_plan_status: Optional[str] = Field(default=None, description="The underlying TopicSelectionResult.status")
    selected_topic: Optional[str] = Field(default=None)

    # Content pipeline reference.
    pipeline_run_id: Optional[str] = Field(default=None, description="The content run's own run_id, e.g. from provenance")
    pipeline_status: Optional[str] = Field(default=None, description="The underlying PipelineState.status, verbatim")
    compliance_decision: Optional[str] = Field(default=None, description="PASS/REVIEW/BLOCK, when a compliance review ran")

    # Publishing reference.
    publishing_status: Optional[str] = Field(default=None, description="The underlying UploadResult.status, when publishing was attempted")
    publishing_video_id: Optional[str] = Field(default=None)
    publishing_video_url: Optional[str] = Field(default=None)

    error_type: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    warnings: List[str] = Field(default_factory=list)
