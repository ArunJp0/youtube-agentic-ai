# Autonomous run record persistence - mirrors PublishingRecordStore/
# ProvenanceManifestStore exactly: atomic writes (temp file + os.replace),
# one JSON file per record, keyed by the autonomous run's own run_id.
#
# list_by_scheduled_time() is the STEP 9 idempotency lookup: before starting
# work for a given scheduled occurrence, AutonomousRunController checks
# whether a terminal (non-"started") record already exists for that exact
# scheduled_time, and skips re-running it if so.
from __future__ import annotations

import glob
import json
import os
import tempfile
from typing import List, Optional

from pydantic import ValidationError

from src.models.autonomous import AutonomousRunRecord

DEFAULT_AUTONOMOUS_RUN_DIR = os.path.join("output", "autonomous_runs")


class AutonomousRunStoreError(Exception):
    """Raised when an autonomous run record file exists but cannot be read
    or parsed (corrupt/malformed) - distinct from "no such record", which
    read() reports as ``None``, never an error."""


class AutonomousRunStore:
    """Reads/writes ``AutonomousRunRecord`` entries, one JSON file per
    autonomous run (``<output_dir>/<run_id>.json``)."""

    def __init__(self, output_dir: str = DEFAULT_AUTONOMOUS_RUN_DIR) -> None:
        self.output_dir = output_dir

    def write(self, record: AutonomousRunRecord) -> str:
        """Atomically write ``record`` to ``<output_dir>/<run_id>.json``,
        overwriting any prior record for the same run_id (a run's record is
        written multiple times over its lifecycle: started -> terminal).

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = os.path.join(self.output_dir, f"{record.run_id}.json")

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-autonomousrun-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def read(self, run_id: str) -> Optional[AutonomousRunRecord]:
        """Read the record for ``run_id``, or ``None`` if it doesn't exist.

        Raises:
            AutonomousRunStoreError: If a file exists at that path but is
                corrupt/unparseable.
        """
        path = os.path.join(self.output_dir, f"{run_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AutonomousRunRecord.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise AutonomousRunStoreError(f"Autonomous run record at '{path}' is corrupt/unreadable: {e}") from e

    def list_recent(self, limit: Optional[int] = None) -> List[AutonomousRunRecord]:
        """Return persisted records, most-recently-started first. A
        corrupt/unreadable record is skipped (never raised)."""
        if not os.path.isdir(self.output_dir):
            return []

        records: List[AutonomousRunRecord] = []
        for path in glob.glob(os.path.join(self.output_dir, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(AutonomousRunRecord.model_validate(data))
            except (OSError, json.JSONDecodeError, ValidationError):
                continue

        records.sort(key=lambda r: r.started_at or "", reverse=True)
        return records[:limit] if limit is not None else records

    def find_by_scheduled_time(self, scheduled_time: str) -> Optional[AutonomousRunRecord]:
        """Return the most recent record for the given scheduled occurrence
        (STEP 9 idempotency key), or ``None`` if this occurrence has never
        been recorded. Corrupt records are skipped."""
        matches = [r for r in self.list_recent() if r.scheduled_time == scheduled_time]
        return matches[0] if matches else None
