# Manual operations entry point for Autonomous Orchestration.
#
# Uses the exact same AutonomousRunController as scheduled execution (see
# src.orchestration.autonomous_factory.build_autonomous_run_controller) -
# there is deliberately no separate code path for manual vs scheduled
# production execution (STEP 10). No interactive prompts anywhere in this
# module - every choice comes from CLI flags/Settings, never a runtime
# input() call.
from __future__ import annotations

import argparse
import asyncio
import sys

from src.config.providers import ProviderConfigError
from src.config.settings import Settings
from src.models.autonomous import AutonomousRunRecord
from src.orchestration.autonomous_factory import build_autonomous_run_controller
from src.orchestration.autonomous_scheduler import AutonomousScheduler

_FAILURE_STATUSES = {"failed"}


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def print_run_record(record: AutonomousRunRecord) -> None:
    print("\n" + "=" * 60)
    print("AUTONOMOUS RUN RESULT")
    print("=" * 60)
    print(f"Run ID:          {record.run_id}")
    print(f"Trigger source:  {record.trigger_source}")
    print(f"Status:          {record.status}")
    print(f"Started at:      {record.started_at}")
    print(f"Finished at:     {record.finished_at}")
    if record.selected_topic:
        print(f"Selected topic:  {record.selected_topic} (planner status: {record.topic_plan_status})")
    if record.pipeline_run_id:
        print(f"Pipeline run id: {record.pipeline_run_id} (status: {record.pipeline_status})")
    if record.compliance_decision:
        print(f"Compliance:      {record.compliance_decision}")
    if record.publishing_video_id:
        print(f"Published video: {record.publishing_video_id} ({record.publishing_video_url})")
    if record.error_message:
        print(f"Error:           {record.error_type}: {record.error_message}")
    for warning in record.warnings:
        print(f"Warning:         {warning}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Orchestration operations entry point")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-once", action="store_true", help="Run one manual autonomous occurrence to completion and exit"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Topic Planner only (no content pipeline, no publishing) and exit",
    )
    return parser.parse_args(argv)


async def _run_manual(dry_run: bool) -> AutonomousRunRecord:
    settings = Settings()
    controller = build_autonomous_run_controller(settings)
    trigger_source = "dry_run" if dry_run else "manual"
    return await controller.run_once(trigger_source, dry_run=dry_run)


async def _run_scheduler_forever() -> None:
    settings = Settings()
    if not settings.autonomous_enabled:
        print("AUTONOMOUS_ENABLED is false - refusing to start the scheduler loop. Use --run-once/--dry-run "
              "for a manual invocation, or set AUTONOMOUS_ENABLED=true to run continuously.")
        sys.exit(1)
    controller = build_autonomous_run_controller(settings)
    scheduler = AutonomousScheduler(controller, settings.autonomous_schedule, settings.autonomous_timezone)
    print(
        f"Autonomous scheduler starting: schedule='{settings.autonomous_schedule}' "
        f"timezone='{settings.autonomous_timezone}' publishing_mode='{settings.autonomous_publishing_mode}'"
    )
    await scheduler.run_forever()


async def main(argv: list[str]) -> None:
    _ensure_utf8_stdout()
    args = _parse_args(argv)

    try:
        if args.run_once or args.dry_run:
            record = await _run_manual(dry_run=args.dry_run)
            print_run_record(record)
            if record.status in _FAILURE_STATUSES:
                sys.exit(1)
        else:
            await _run_scheduler_forever()
    except ProviderConfigError as e:
        print(f"Provider configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
