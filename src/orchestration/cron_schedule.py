# Minimal standard 5-field cron expression matcher (minute hour day month
# weekday) - stdlib only, no `croniter`/`apscheduler` dependency, per this
# milestone's "no unnecessary frameworks" constraint. Supports the common
# subset needed for a periodic autonomous schedule: "*", "*/n", "a,b,c",
# and "a-b" (each field independently, combinable as "a-b/n" too).
from __future__ import annotations

from datetime import datetime, timedelta
from typing import FrozenSet, Tuple

_FIELD_RANGES: Tuple[Tuple[int, int], ...] = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


class CronScheduleError(Exception):
    """Raised when an ``AUTONOMOUS_SCHEDULE`` expression cannot be parsed."""


def _parse_field(raw: str, low: int, high: int) -> FrozenSet[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise CronScheduleError(f"Empty field component in cron expression segment: '{raw}'")

        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError as e:
                raise CronScheduleError(f"Invalid step '{step_str}' in '{raw}'") from e
            if step <= 0:
                raise CronScheduleError(f"Step must be positive in '{raw}'")

        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start, end = int(start_str), int(end_str)
            except ValueError as e:
                raise CronScheduleError(f"Invalid range '{part}' in '{raw}'") from e
        else:
            try:
                start = end = int(part)
            except ValueError as e:
                raise CronScheduleError(f"Invalid value '{part}' in '{raw}'") from e

        if not (low <= start <= high) or not (low <= end <= high) or start > end:
            raise CronScheduleError(f"Field value out of range [{low}, {high}] in '{raw}'")

        values.update(range(start, end + 1, step))

    return frozenset(values)


class CronSchedule:
    """A parsed 5-field cron expression ("minute hour day month weekday"),
    evaluated against naive datetimes already in the target timezone (the
    caller, ``AutonomousScheduler``, is responsible for timezone conversion
    - this class has no timezone awareness of its own).
    """

    def __init__(self, expression: str) -> None:
        fields = expression.strip().split()
        if len(fields) != 5:
            raise CronScheduleError(
                f"Cron expression must have exactly 5 fields (minute hour day month weekday), got: '{expression}'"
            )
        self.expression = expression
        self._minutes, self._hours, self._days, self._months, self._weekdays = (
            _parse_field(field, low, high) for field, (low, high) in zip(fields, _FIELD_RANGES)
        )

    def matches(self, moment: datetime) -> bool:
        # Python's Monday=0..Sunday=6 vs cron's Sunday=0..Saturday=6.
        cron_weekday = (moment.weekday() + 1) % 7
        return (
            moment.minute in self._minutes
            and moment.hour in self._hours
            and moment.day in self._days
            and moment.month in self._months
            and cron_weekday in self._weekdays
        )

    def next_occurrence_after(self, after: datetime, max_scan_minutes: int = 60 * 24 * 366) -> datetime:
        """The next minute-aligned moment strictly after ``after`` that
        matches this schedule. Minute-granularity linear scan - simple and
        fast enough for a schedule interval measured in minutes/hours; not
        intended for sub-minute cadences."""
        candidate = (after + timedelta(minutes=1)).replace(second=0, microsecond=0)
        for _ in range(max_scan_minutes):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise CronScheduleError(f"No occurrence of '{self.expression}' found within {max_scan_minutes} minutes")
