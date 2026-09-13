# Mocked/local tests for Autonomous Orchestration - no real Gemini/Google
# News/YouTube Data API/TTS/Pexels/Whisper/FFmpeg calls anywhere in this
# file. TopicPlannerAgent and the content pipeline are both fully stubbed;
# only the coordination logic in AutonomousRunController/AutonomousScheduler/
# RunLock/AutonomousRunStore is under test.
from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import pytest

from src.models.autonomous import AutonomousRunRecord
from src.models.compliance import ComplianceResult
from src.models.music import AudioMixResult
from src.models.topic_planner import TopicSelectionResult
from src.models.youtube_upload import PublishingIntent, UploadResult
from src.orchestration.autonomous_run_controller import AutonomousRunController
from src.orchestration.autonomous_scheduler import AutonomousScheduler
from src.orchestration.cron_schedule import CronSchedule, CronScheduleError
from src.services.autonomous_run_store import AutonomousRunStore, AutonomousRunStoreError
from src.services.run_lock import RunLock, RunLockError
from src.workflows.pipeline_graph import PipelineState


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubTopicPlannerAgent:
    """Replaces TopicPlannerAgent entirely - never touches a real
    TopicSourceProvider/LLMProvider."""

    def __init__(self, result: Optional[TopicSelectionResult] = None, error: Optional[Exception] = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def plan_topic(self) -> TopicSelectionResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def make_topic_result(topic: str = "Why is the sky blue?", success: bool = True, status: str = "selected") -> TopicSelectionResult:
    return TopicSelectionResult(success=success, status=status, selected_topic=topic if success else None)


class StubPipelineRunner:
    """Replaces run_pipeline() entirely - never touches any real provider."""

    def __init__(self, state: Optional[PipelineState] = None, error: Optional[Exception] = None) -> None:
        self.state = state
        self.error = error
        self.calls: List[tuple] = []

    async def __call__(self, topic: str, publishing_intent: Optional[PublishingIntent]) -> PipelineState:
        self.calls.append((topic, publishing_intent))
        if self.error is not None:
            raise self.error
        assert self.state is not None
        return self.state


def make_completed_state(topic: str = "Why is the sky blue?", run_id: str = "run-abc123") -> PipelineState:
    return PipelineState(
        topic=topic,
        status="completed",
        audio_mix_result=AudioMixResult(success=True, output_path=os.path.join("output", "video", f"{run_id}-bgm.mp4")),
        compliance_result=ComplianceResult(success=True, publish_decision="PASS", topic=topic),
    )


def make_blocked_state(topic: str = "Why is the sky blue?", run_id: str = "run-blocked") -> PipelineState:
    return PipelineState(
        topic=topic,
        status="blocked",
        error="Compliance blocked this run",
        audio_mix_result=AudioMixResult(success=True, output_path=os.path.join("output", "video", f"{run_id}-bgm.mp4")),
        compliance_result=ComplianceResult(success=True, publish_decision="BLOCK", topic=topic),
    )


def make_published_state(topic: str = "Why is the sky blue?", run_id: str = "run-published", video_id: str = "abc123") -> PipelineState:
    return PipelineState(
        topic=topic,
        status="published",
        audio_mix_result=AudioMixResult(success=True, output_path=os.path.join("output", "video", f"{run_id}-bgm.mp4")),
        compliance_result=ComplianceResult(success=True, publish_decision="PASS", topic=topic),
        publishing_result=UploadResult(success=True, status="private_uploaded", video_id=video_id, video_url=f"https://youtu.be/{video_id}"),
    )


def make_thumbnail_failed_state(topic: str = "Why is the sky blue?", run_id: str = "run-thumbfail", video_id: str = "vidthumb") -> PipelineState:
    """A 'published' pipeline outcome where the video itself uploaded fine
    but setting the thumbnail failed (e.g. a phone-unverified channel) -
    UploadResult.success stays True (the video exists), status is
    'thumbnail_failed', and the diagnostic text lives only in .warnings."""
    return PipelineState(
        topic=topic,
        status="published",
        audio_mix_result=AudioMixResult(success=True, output_path=os.path.join("output", "video", f"{run_id}-bgm.mp4")),
        compliance_result=ComplianceResult(success=True, publish_decision="PASS", topic=topic),
        publishing_result=UploadResult(
            success=True,
            status="thumbnail_failed",
            video_id=video_id,
            video_url=f"https://youtu.be/{video_id}",
            warnings=[f"Video uploaded successfully (video_id={video_id}), but setting the thumbnail failed: 403 forbidden"],
        ),
    )


def make_publishing_failed_state(topic: str = "Why is the sky blue?", run_id: str = "run-pubfail") -> PipelineState:
    return PipelineState(
        topic=topic,
        status="publishing_failed",
        audio_mix_result=AudioMixResult(success=True, output_path=os.path.join("output", "video", f"{run_id}-bgm.mp4")),
        compliance_result=ComplianceResult(success=True, publish_decision="PASS", topic=topic),
        publishing_result=UploadResult(success=False, status="upload_failed", error="quota exceeded"),
    )


def make_failed_state(topic: str = "Why is the sky blue?") -> PipelineState:
    return PipelineState(topic=topic, status="failed", error="Research failed")


def build_controller(
    topic_planner_agent,
    pipeline_runner,
    tmp_path,
    publishing_intent: Optional[PublishingIntent] = None,
    lock_timeout: int = 10800,
) -> AutonomousRunController:
    run_store = AutonomousRunStore(str(tmp_path / "runs"))
    run_lock = RunLock(str(tmp_path / "run.lock"), timeout_seconds=lock_timeout)
    return AutonomousRunController(
        topic_planner_agent=topic_planner_agent,
        pipeline_runner=pipeline_runner,
        run_store=run_store,
        run_lock=run_lock,
        publishing_intent=publishing_intent,
    )


# ---------------------------------------------------------------------------
# Controller: topic selection -> pipeline call
# ---------------------------------------------------------------------------


async def test_controller_selects_topic_then_calls_existing_pipeline(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result("Why is the ocean salty?"))
    state = make_completed_state(topic="Why is the ocean salty?")
    pipeline = StubPipelineRunner(state=state)
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert planner.calls == 1
    assert pipeline.calls == [("Why is the ocean salty?", controller._publishing_intent)]
    assert record.selected_topic == "Why is the ocean salty?"
    assert record.status == "publishing_disabled"


async def test_topic_planner_no_topic_means_no_pipeline_call(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result(success=False, status="no_candidates"))
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert pipeline.calls == []
    assert record.status == "no_topic"
    assert record.topic_plan_status == "no_candidates"


async def test_topic_planner_failure_produces_safe_run_record(tmp_path):
    planner = StubTopicPlannerAgent(error=RuntimeError("news source unreachable"))
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert pipeline.calls == []
    assert record.status == "failed"
    assert record.error_type == "RuntimeError"
    assert "news source unreachable" in record.error_message


async def test_pipeline_failure_means_no_publishing_continuation(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(error=RuntimeError("gemini quota exhausted"))
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert record.status == "failed"
    assert record.publishing_status is None
    assert record.publishing_video_id is None


# ---------------------------------------------------------------------------
# Compliance / publishing outcome representation
# ---------------------------------------------------------------------------


async def test_compliance_blocked_result_represented_correctly(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_blocked_state())
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert record.status == "compliance_blocked"
    assert record.compliance_decision == "BLOCK"
    assert record.pipeline_run_id == "run-blocked"


async def test_publishing_disabled_represented_correctly(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path, publishing_intent=PublishingIntent(mode="disabled"))

    record = await controller.run_once("manual")

    assert record.status == "publishing_disabled"
    assert record.publishing_video_id is None


async def test_private_publishing_result_represented_correctly(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_published_state(video_id="vid123"))
    controller = build_controller(planner, pipeline, tmp_path, publishing_intent=PublishingIntent(mode="private"))

    record = await controller.run_once("manual")

    assert record.status == "published_private"
    assert record.publishing_video_id == "vid123"
    assert record.publishing_video_url == "https://youtu.be/vid123"


async def test_thumbnail_failure_warning_is_preserved_in_run_record(tmp_path):
    """A partial-failure warning (video uploaded, thumbnail rejected by
    YouTube) must survive into the durable AutonomousRunRecord - it is the
    only place that diagnostic text exists once the process exits."""
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_thumbnail_failed_state(video_id="vidthumb"))
    controller = build_controller(planner, pipeline, tmp_path, publishing_intent=PublishingIntent(mode="private"))

    record = await controller.run_once("manual")

    assert record.status == "published_private"
    assert record.publishing_status == "thumbnail_failed"
    assert record.publishing_video_id == "vidthumb"
    assert any("thumbnail" in warning.lower() for warning in record.warnings)


async def test_scheduled_publishing_result_represented_correctly(tmp_path):
    import datetime

    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_published_state(video_id="vid456"))
    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    controller = build_controller(
        planner, pipeline, tmp_path, publishing_intent=PublishingIntent(mode="scheduled", scheduled_publish_at=future)
    )

    record = await controller.run_once("manual")

    assert record.status == "scheduled"
    assert record.publishing_video_id == "vid456"


async def test_publishing_failure_does_not_regenerate_content(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_publishing_failed_state(run_id="run-pubfail"))
    controller = build_controller(planner, pipeline, tmp_path, publishing_intent=PublishingIntent(mode="private"))

    record = await controller.run_once("manual")

    assert record.status == "publishing_failed"
    assert record.pipeline_run_id == "run-pubfail"
    assert "quota exceeded" in record.error_message
    assert pipeline.calls  # the pipeline was called exactly once - never re-invoked to "regenerate"
    assert len(pipeline.calls) == 1


# ---------------------------------------------------------------------------
# Run lock / concurrency
# ---------------------------------------------------------------------------


async def test_overlapping_trigger_prevented_by_durable_lock(tmp_path):
    lock_path = str(tmp_path / "run.lock")
    holder_lock = RunLock(lock_path, timeout_seconds=10800)
    holder_lock.acquire()

    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_completed_state())
    run_store = AutonomousRunStore(str(tmp_path / "runs"))
    controller = AutonomousRunController(
        topic_planner_agent=planner,
        pipeline_runner=pipeline,
        run_store=run_store,
        run_lock=RunLock(lock_path, timeout_seconds=10800),
    )

    record = await controller.run_once("manual")

    assert record.status == "skipped_locked"
    assert pipeline.calls == []
    holder_lock.release()


def test_stale_lock_recovery(tmp_path):
    lock_path = str(tmp_path / "run.lock")
    stale_lock = RunLock(lock_path, timeout_seconds=0)
    stale_lock.acquire()
    stale_lock._held = False  # simulate the holder process having crashed without releasing

    import time

    time.sleep(0.05)

    recovering_lock = RunLock(lock_path, timeout_seconds=0)
    recovering_lock.acquire()  # must not raise - the stale lock is recovered
    recovering_lock.release()


def test_lock_released_after_exception(tmp_path):
    lock_path = str(tmp_path / "run.lock")
    lock = RunLock(lock_path, timeout_seconds=10800)
    lock.acquire()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        pass
    finally:
        lock.release()

    assert not os.path.exists(lock_path)
    # A fresh acquire succeeds immediately - proof the lock file is truly gone.
    lock2 = RunLock(lock_path, timeout_seconds=10800)
    lock2.acquire()
    lock2.release()


async def test_lock_released_after_controller_exception(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())

    class ExplodingPipelineRunner:
        async def __call__(self, topic, publishing_intent):
            raise RuntimeError("unexpected crash")

    lock_path = str(tmp_path / "run.lock")
    run_store = AutonomousRunStore(str(tmp_path / "runs"))
    controller = AutonomousRunController(
        topic_planner_agent=planner,
        pipeline_runner=ExplodingPipelineRunner(),
        run_store=run_store,
        run_lock=RunLock(lock_path, timeout_seconds=10800),
    )

    record = await controller.run_once("manual")

    assert record.status == "failed"
    assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_same_scheduled_occurrence_does_not_duplicate_completed_run(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path)

    scheduled_time = "2026-01-01T00:00:00+00:00"
    first = await controller.run_once("scheduled", scheduled_time=scheduled_time)
    assert first.status == "publishing_disabled"
    assert planner.calls == 1

    second = await controller.run_once("scheduled", scheduled_time=scheduled_time)

    assert second.run_id == first.run_id
    assert planner.calls == 1  # not called again
    assert pipeline.calls == [(first.selected_topic, controller._publishing_intent)]  # still just one call


async def test_in_progress_scheduled_occurrence_is_retried_not_skipped(tmp_path):
    """A record still in 'started' (e.g. process crashed mid-run) is not
    treated as already-completed - the occurrence gets a fresh attempt."""
    run_store = AutonomousRunStore(str(tmp_path / "runs"))
    stuck_record = AutonomousRunRecord(
        run_id="autonomous-stuck",
        trigger_source="scheduled",
        scheduled_time="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:00+00:00",
        status="started",
    )
    run_store.write(stuck_record)

    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = AutonomousRunController(
        topic_planner_agent=planner,
        pipeline_runner=pipeline,
        run_store=run_store,
        run_lock=RunLock(str(tmp_path / "run.lock"), timeout_seconds=10800),
    )

    record = await controller.run_once("scheduled", scheduled_time="2026-01-01T00:00:00+00:00")

    assert record.run_id != "autonomous-stuck"
    assert planner.calls == 1


# ---------------------------------------------------------------------------
# Manual run-once / dry-run
# ---------------------------------------------------------------------------


async def test_manual_run_once_uses_the_same_controller(tmp_path):
    from src.autonomous_runner import _run_manual

    planner = StubTopicPlannerAgent(result=make_topic_result())
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("manual")

    assert record.trigger_source == "manual"
    assert pipeline.calls  # manual invocation reached the same pipeline call path as scheduled would


async def test_planning_dry_run_does_not_invoke_content_pipeline(tmp_path):
    planner = StubTopicPlannerAgent(result=make_topic_result("Why do cats purr?"))
    pipeline = StubPipelineRunner(state=make_completed_state())
    controller = build_controller(planner, pipeline, tmp_path)

    record = await controller.run_once("dry_run", dry_run=True)

    assert planner.calls == 1
    assert pipeline.calls == []
    assert record.status == "planned_only"
    assert record.selected_topic == "Why do cats purr?"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


async def test_scheduled_trigger_invokes_controller_once():
    calls = []

    class RecordingController:
        async def run_once(self, trigger_source, scheduled_time=None, dry_run=False):
            calls.append((trigger_source, scheduled_time))

    async def instant_sleep(_seconds):
        return None

    scheduler = AutonomousScheduler(RecordingController(), "* * * * *", "UTC", sleep_fn=instant_sleep)
    await scheduler.run_forever(max_iterations=1)

    assert len(calls) == 1
    assert calls[0][0] == "scheduled"
    assert calls[0][1] is not None


async def test_scheduler_failure_on_one_run_does_not_stop_later_runs():
    calls = []

    class FlakyController:
        async def run_once(self, trigger_source, scheduled_time=None, dry_run=False):
            calls.append(scheduled_time)
            if len(calls) == 1:
                raise RuntimeError("transient failure")

    async def instant_sleep(_seconds):
        return None

    scheduler = AutonomousScheduler(FlakyController(), "* * * * *", "UTC", sleep_fn=instant_sleep)
    await scheduler.run_forever(max_iterations=2)

    assert len(calls) == 2


def test_cron_schedule_rejects_invalid_expression():
    with pytest.raises(CronScheduleError):
        CronSchedule("not a cron expression")


def test_cron_schedule_matches_and_finds_next_occurrence():
    import datetime

    schedule = CronSchedule("0 */6 * * *")
    after = datetime.datetime(2026, 1, 1, 1, 30)
    occurrence = schedule.next_occurrence_after(after)
    assert occurrence == datetime.datetime(2026, 1, 1, 6, 0)
    assert schedule.matches(occurrence)


# ---------------------------------------------------------------------------
# Autonomous run store
# ---------------------------------------------------------------------------


def test_autonomous_run_store_round_trip(tmp_path):
    store = AutonomousRunStore(str(tmp_path))
    record = AutonomousRunRecord(
        run_id="autonomous-1",
        trigger_source="manual",
        started_at="2026-01-01T00:00:00+00:00",
        status="completed",
    )
    store.write(record)

    loaded = store.read("autonomous-1")
    assert loaded == record
    assert store.read("does-not-exist") is None


def test_autonomous_run_store_raises_on_corrupt_file(tmp_path):
    store = AutonomousRunStore(str(tmp_path))
    os.makedirs(tmp_path, exist_ok=True)
    with open(tmp_path / "autonomous-bad.json", "w", encoding="utf-8") as f:
        f.write("{not valid json")

    with pytest.raises(AutonomousRunStoreError):
        store.read("autonomous-bad")


# ---------------------------------------------------------------------------
# Existing behavior unchanged
# ---------------------------------------------------------------------------


async def test_existing_topic_planner_and_pipeline_signatures_unchanged():
    """Guards against an accidental signature change to the exact seams
    AutonomousRunController depends on."""
    import inspect

    from src.agents.topic_planner_agent import TopicPlannerAgent
    from src.workflows.pipeline_graph import run_pipeline

    plan_topic_params = inspect.signature(TopicPlannerAgent.plan_topic).parameters
    assert list(plan_topic_params) == ["self"]

    run_pipeline_params = inspect.signature(run_pipeline).parameters
    assert "topic" in run_pipeline_params
    assert "publishing_intent" in run_pipeline_params
    assert "youtube_client" in run_pipeline_params
