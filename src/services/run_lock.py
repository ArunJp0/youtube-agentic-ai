# Durable, file-based run lock preventing two autonomous pipeline
# executions from running concurrently (STEP 6). Deliberately NOT an
# in-memory boolean - a plain flag would not survive the scheduler process
# restarting, and would not let a second process (e.g. a manual
# `--run-once` invoked while a scheduled run is in flight) see the lock at
# all. Deliberately NOT a third-party dependency (no `filelock` in
# pyproject.toml) - a single exclusively-created marker file, atomic via
# os.O_CREAT | os.O_EXCL, is enough for this project's single-host,
# file-based architecture and matches its existing "atomic file write"
# idiom used throughout src/services/*_store.py.
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

DEFAULT_LOCK_PATH = os.path.join("output", "autonomous_runs", ".run.lock")


class RunLockError(Exception):
    """Raised by acquire() when the lock is already held by a live (non-stale) run."""


class RunLock:
    """A single exclusively-created lock file at ``lock_path``.

    Usage::

        lock = RunLock(lock_path, timeout_seconds=settings.autonomous_lock_timeout)
        lock.acquire()
        try:
            ...
        finally:
            lock.release()

    A lock older than ``timeout_seconds`` is considered stale (e.g. the
    process that held it crashed without releasing it) and is
    automatically recovered by the next ``acquire()`` call.
    """

    def __init__(self, lock_path: str = DEFAULT_LOCK_PATH, timeout_seconds: int = 10800) -> None:
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self._held = False

    def _read_lock_age_seconds(self) -> Optional[float]:
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            acquired_at = datetime.fromisoformat(data["acquired_at"])
            return (datetime.now(timezone.utc) - acquired_at).total_seconds()
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            # An unreadable/corrupt lock file is treated as stale - it can
            # never be legitimately renewed, so it must not block forever.
            return None

    def _try_remove_if_stale(self) -> None:
        age = self._read_lock_age_seconds()
        if age is None or age > self.timeout_seconds:
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

    def acquire(self) -> None:
        """Acquire the lock, recovering a stale lock first if present.

        Raises:
            RunLockError: If the lock is currently held by a live run.
        """
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)

        if os.path.exists(self.lock_path):
            self._try_remove_if_stale()

        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise RunLockError(f"Run lock '{self.lock_path}' is already held by a live run") from e

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"pid": os.getpid(), "acquired_at": datetime.now(timezone.utc).isoformat()},
                f,
            )
        self._held = True

    def release(self) -> None:
        """Release the lock if this instance holds it. Safe to call even if
        acquire() was never called or already failed."""
        if not self._held:
            return
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        self._held = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
