# Lightweight in-process scheduler (STEP 5) - deliberately NOT Celery/
# Redis/Kafka/APScheduler. Its only job is "when to trigger"; "what a
# trigger does" is entirely AutonomousRunController's job (injected here,
# never reimplemented). Kept swappable: the exact same
# `AutonomousRunController.run_once(...)` call this makes is also what the
# manual `--run-once`/`--dry-run` CLI makes directly, so this scheduler can
# later be replaced outright by cron/systemd/Task Scheduler/a cloud
# scheduler hitting the same callable without any controller change.
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from src.orchestration.autonomous_run_controller import AutonomousRunController
from src.orchestration.cron_schedule import CronSchedule

logger = logging.getLogger(__name__)

SleepFn = Callable[[float], Awaitable[None]]


class AutonomousScheduler:
    """Sleeps until each configured cron occurrence, then calls
    ``AutonomousRunController.run_once("scheduled", scheduled_time=...)``
    exactly once for it.

    Runs strictly sequentially (one occurrence is fully awaited before the
    next is even computed) - this alone prevents the scheduler itself from
    ever launching two overlapping runs, on top of the durable run lock
    inside the controller. A single occurrence's failure (any exception
    escaping ``run_once``, which is not expected but is not trusted either)
    is logged and never stops the loop from computing and waiting for the
    next occurrence (STEP 8: "scheduler failure on one run must not
    permanently stop later scheduled runs").
    """

    def __init__(
        self,
        controller: AutonomousRunController,
        schedule_expression: str,
        timezone_name: str = "UTC",
        sleep_fn: Optional[SleepFn] = None,
    ) -> None:
        self._controller = controller
        self._schedule = CronSchedule(schedule_expression)
        self._tz = ZoneInfo(timezone_name)
        self._sleep = sleep_fn or asyncio.sleep
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask ``run_forever`` to exit after its current wait/run completes."""
        self._stop_requested = True

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    async def run_forever(self, max_iterations: Optional[int] = None) -> None:
        """Loop: compute the next scheduled occurrence, sleep until it,
        trigger the controller once, repeat.

        Args:
            max_iterations: Optional cap on the number of occurrences to
                run before returning - primarily for tests; ``None`` means
                run until ``request_stop()`` is called.
        """
        iterations = 0
        while not self._stop_requested:
            if max_iterations is not None and iterations >= max_iterations:
                return

            now = self._now()
            occurrence = self._schedule.next_occurrence_after(now)
            wait_seconds = max((occurrence - now).total_seconds(), 0.0)
            if wait_seconds > 0:
                await self._sleep(wait_seconds)

            scheduled_time = occurrence.astimezone(timezone.utc).isoformat()
            try:
                await self._controller.run_once("scheduled", scheduled_time=scheduled_time)
            except Exception:
                logger.exception("Autonomous scheduled run failed for occurrence %s", scheduled_time)

            iterations += 1
