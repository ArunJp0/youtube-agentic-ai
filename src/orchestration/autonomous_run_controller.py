# Autonomous Orchestration - coordination layer only.
#
# AutonomousRunController has NO Research/Script/Video/Compliance/YouTube
# business logic of its own. Every actual capability (topic selection, the
# full content pipeline, Compliance/Remediation, Publishing) is an injected
# dependency this class only calls and inspects the typed result of:
#
#   AutonomousScheduler / manual --run-once
#           |
#   AutonomousRunController.run_once()   <- this file
#           |
#   TopicPlannerAgent.plan_topic()        (existing, unchanged)
#           |
#   run_pipeline(...)                     (existing, unchanged - includes
#           |                              Compliance/Remediation/Publishing)
#   AutonomousRunRecord (persisted via AutonomousRunStore)
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from src.agents.topic_planner_agent import TopicPlannerAgent
from src.models.autonomous import AutonomousRunRecord, TriggerSource
from src.models.youtube_upload import PublishingIntent
from src.services.autonomous_run_store import AutonomousRunStore
from src.services.run_lock import RunLock, RunLockError
from src.services.script_context_reconstruction import original_base_name
from src.workflows.pipeline_graph import PipelineState

# (topic, topic_source, publishing_intent) -> final PipelineState.
# topic_source mirrors TopicSelectionResult.selected_topic_source (e.g.
# "current_news") - threaded through so run_pipeline can route Research to
# an appropriate search provider; None for a topic with no classification.
# In production this closes over real providers built from Settings (see
# build_default_pipeline_runner below); tests inject a stub directly, so
# the controller itself never imports/constructs a single provider.
PipelineRunner = Callable[[str, Optional[str], Optional[PublishingIntent]], Awaitable[PipelineState]]

# PipelineState.status values that mean "Compliance did not clear this run
# for publishing" - review_exhausted is the bounded-remediation-loop's own
# terminal-without-PASS outcome (see pipeline_graph.py), grouped here with
# review_required/blocked since all three represent the same thing to an
# autonomous run: no publishing happened, and none should be attempted.
_COMPLIANCE_BLOCKED_STATUSES = {"review_required", "review_exhausted", "blocked"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AutonomousRunController:
    """Coordinates exactly one autonomous run end-to-end and returns the
    durable ``AutonomousRunRecord`` describing its outcome.

    The SAME instance/entry point is used for a scheduled trigger, a manual
    ``--run-once`` invocation, and a dry-run - only ``trigger_source``/
    ``scheduled_time``/``dry_run`` differ (see STEP 10: no separate code
    path for manual vs scheduled production execution).
    """

    def __init__(
        self,
        topic_planner_agent: TopicPlannerAgent,
        pipeline_runner: PipelineRunner,
        run_store: AutonomousRunStore,
        run_lock: RunLock,
        publishing_intent: Optional[PublishingIntent] = None,
    ) -> None:
        self._topic_planner_agent = topic_planner_agent
        self._pipeline_runner = pipeline_runner
        self._run_store = run_store
        self._run_lock = run_lock
        self._publishing_intent = publishing_intent or PublishingIntent()

    async def run_once(
        self,
        trigger_source: TriggerSource,
        scheduled_time: Optional[str] = None,
        dry_run: bool = False,
    ) -> AutonomousRunRecord:
        """Run one autonomous occurrence to completion (or a safe early
        stop) and persist its outcome.

        Args:
            trigger_source: "scheduled", "manual", or "dry_run" - recorded
                for diagnosis only, never branches business behavior.
            scheduled_time: ISO 8601 UTC timestamp identifying the specific
                scheduled occurrence this call corresponds to (the STEP 9
                idempotency key). ``None`` for manual/dry-run invocations,
                which have no notion of a repeatable "occurrence".
            dry_run: When True, Topic Planner runs but the content pipeline
                is never invoked (planning-only / STEP 10).

        Returns:
            The final ``AutonomousRunRecord`` (also persisted via
            ``AutonomousRunStore``) - never raises for an expected failure
            mode (no topic, pipeline failure, compliance block, publishing
            failure, lock contention); those are all represented as typed
            statuses instead.
        """
        if scheduled_time is not None:
            existing = self._run_store.find_by_scheduled_time(scheduled_time)
            if existing is not None and existing.status != "started":
                # STEP 9: a retried/replayed trigger for an occurrence that
                # already reached a terminal outcome must not start another
                # topic/video/upload - return the prior outcome as-is.
                return existing

        record = AutonomousRunRecord(
            run_id=f"autonomous-{uuid.uuid4().hex[:12]}",
            trigger_source=trigger_source,
            scheduled_time=scheduled_time,
            started_at=_now_iso(),
            status="started",
        )
        self._run_store.write(record)

        try:
            self._run_lock.acquire()
        except RunLockError:
            record.status = "skipped_locked"
            record.finished_at = _now_iso()
            record.warnings.append("Skipped: another autonomous run already holds the run lock")
            self._run_store.write(record)
            return record
        except Exception as e:
            # A lock-infrastructure failure (e.g. an unwritable lock
            # directory) is still just this one run's failure - it must not
            # propagate and take down a scheduler process handling later
            # occurrences too.
            record.status = "failed"
            record.error_type = type(e).__name__
            record.error_message = f"Failed to acquire run lock: {e}"
            record.finished_at = _now_iso()
            self._run_store.write(record)
            return record

        try:
            record = await self._execute(record, dry_run=dry_run)
        finally:
            # Released even when _execute raises something it didn't
            # already catch itself - STEP 6's "finally-style cleanup".
            self._run_lock.release()

        return record

    async def _execute(self, record: AutonomousRunRecord, dry_run: bool) -> AutonomousRunRecord:
        try:
            topic_result = await self._topic_planner_agent.plan_topic()
        except Exception as e:
            # STEP 8: Topic Planner failure -> record failure -> never
            # start the expensive content pipeline.
            return self._fail(record, e)

        record.topic_plan_status = topic_result.status
        if not topic_result.success or not topic_result.selected_topic:
            record.status = "no_topic"
            record.error_message = topic_result.error
            return self._finish(record)

        record.selected_topic = topic_result.selected_topic
        record.status = "generating"
        self._run_store.write(record)

        if dry_run:
            record.status = "planned_only"
            return self._finish(record)

        try:
            pipeline_state = await self._pipeline_runner(
                topic_result.selected_topic, topic_result.selected_topic_source, self._publishing_intent
            )
        except Exception as e:
            # STEP 8: pipeline generation failure -> record FAILED -> never publish.
            return self._fail(record, e)

        self._apply_pipeline_outcome(record, pipeline_state)
        return self._finish(record)

    def _apply_pipeline_outcome(self, record: AutonomousRunRecord, pipeline_state: PipelineState) -> None:
        record.pipeline_status = pipeline_state.status
        if pipeline_state.audio_mix_result and pipeline_state.audio_mix_result.output_path:
            record.pipeline_run_id = original_base_name(pipeline_state.audio_mix_result.output_path)
        if pipeline_state.compliance_result is not None:
            record.compliance_decision = pipeline_state.compliance_result.publish_decision

        status = pipeline_state.status
        if status in _COMPLIANCE_BLOCKED_STATUSES:
            # STEP 8: existing Compliance/Remediation behavior preserved
            # unchanged (it already ran, inside run_pipeline) - no
            # publishing happens for a REVIEW/BLOCK outcome.
            record.status = "compliance_blocked"
            record.error_message = pipeline_state.error
        elif status == "completed":
            # Compliance PASSed but PublishingIntent.mode == "disabled", so
            # run_pipeline never reached publishing_node at all.
            record.status = "publishing_disabled"
        elif status == "published":
            result = pipeline_state.publishing_result
            record.publishing_status = result.status if result else None
            record.publishing_video_id = result.video_id if result else None
            record.publishing_video_url = result.video_url if result else None
            # A "published" pipeline status can still carry a partial-failure
            # warning (e.g. thumbnail_failed) - result.warnings is the ONLY
            # place that diagnostic text exists; dropping it here would
            # leave the durable record permanently unable to explain a
            # thumbnail_failed/etc. outcome after this process exits.
            if result and result.warnings:
                record.warnings.extend(result.warnings)
            record.status = "scheduled" if self._publishing_intent.mode == "scheduled" else "published_private"
        elif status == "publishing_failed":
            # STEP 8: publishing failure -> do NOT regenerate content ->
            # already-produced content run_id above is retained as-is.
            result = pipeline_state.publishing_result
            record.publishing_status = result.status if result else None
            record.error_message = (result.error if result else None) or pipeline_state.error
            if result and result.warnings:
                record.warnings.extend(result.warnings)
            record.status = "publishing_failed"
        else:
            record.status = "failed"
            record.error_message = pipeline_state.error or f"Unexpected pipeline status: {status}"

    def _fail(self, record: AutonomousRunRecord, error: Exception) -> AutonomousRunRecord:
        record.status = "failed"
        record.error_type = type(error).__name__
        record.error_message = str(error)
        return self._finish(record)

    def _finish(self, record: AutonomousRunRecord) -> AutonomousRunRecord:
        record.finished_at = _now_iso()
        self._run_store.write(record)
        return record
