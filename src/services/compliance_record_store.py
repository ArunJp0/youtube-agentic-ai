# Compliance result persistence boundary. Mirrors
# src/services/provenance_store.py exactly: the ONLY place that reads or
# writes a persisted ComplianceResult, atomic writes, one JSON file per run.
#
# Persists a small durable envelope (PersistedComplianceRecord: run_id,
# parent_run_id, attempt_number, evaluated_at, plus the full
# ComplianceResult) so a later consumer - a remediation audit, or a human -
# can tell exactly which candidate/revision was evaluated and when, not
# just the bare decision. ``read()`` stays backward-compatible: it always
# returns a plain ComplianceResult (unwrapping the envelope), exactly as
# before this milestone, so YouTubeUploadAgent/upload_artifact_discovery.py
# need no changes, and it transparently reads pre-existing legacy files
# that predate the envelope (a bare ComplianceResult with no wrapper) the
# same way. Use ``read_envelope()`` to get the full envelope, including
# remediation lineage.
#
# compliance_node (src/workflows/pipeline_graph.py) writes here for every
# PASS/REVIEW/BLOCK decision, not just PASS - the WRITE side used to not be
# called from pipeline code at all (a deliberately deferred gap from an
# earlier milestone); this milestone closes that gap.
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from src.models.compliance import ComplianceResult
from src.models.remediation import PersistedComplianceRecord

DEFAULT_COMPLIANCE_RECORD_DIR = os.path.join("output", "compliance")


class ComplianceRecordStoreError(Exception):
    """Raised when a compliance record file exists but cannot be read or
    parsed (corrupt/malformed) - distinct from a record simply not
    existing (which read()/read_envelope() report as ``None``, never an
    error)."""


class ComplianceRecordStore:
    """Reads/writes persisted compliance records, one JSON file per run,
    named after the run's own stable identifier - same naming convention
    as ``ProvenanceManifestStore``."""

    def __init__(self, output_dir: str = DEFAULT_COMPLIANCE_RECORD_DIR) -> None:
        self.output_dir = output_dir

    def write(
        self,
        run_id: str,
        result: ComplianceResult,
        *,
        parent_run_id: Optional[str] = None,
        attempt_number: int = 0,
        evaluated_at: Optional[str] = None,
    ) -> str:
        """Atomically write the durable envelope for ``result`` to
        ``<output_dir>/<run_id>.json``, overwriting any existing record for
        the same run_id.

        Args:
            run_id: This exact candidate's own run identifier
            result: The genuine ComplianceResult this candidate produced
            parent_run_id: The original run this is a remediation attempt
                of, or None if this is not a remediation attempt (attempt 0)
            attempt_number: 0 for an original (non-remediated) evaluation,
                1+ for a remediation attempt
            evaluated_at: ISO 8601 UTC timestamp; defaults to now

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = os.path.join(self.output_dir, f"{run_id}.json")

        payload = {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "attempt_number": attempt_number,
            "evaluated_at": evaluated_at or datetime.now(timezone.utc).isoformat(),
            "result": result.model_dump(mode="json"),
        }

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-compliance-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def read(self, run_id: str) -> Optional[ComplianceResult]:
        """Read the persisted ComplianceResult for ``run_id``, or ``None``
        if none exists - a valid, expected state, never an error.

        Raises:
            ComplianceRecordStoreError: If a file exists at that path but
                is corrupt/unparseable.
        """
        record = self.read_envelope(run_id)
        return record.result if record is not None else None

    def read_envelope(self, run_id: str) -> Optional[PersistedComplianceRecord]:
        """Read the full persisted envelope (run/parent/attempt/timestamp +
        the ComplianceResult) for ``run_id``, or ``None`` if none exists.

        Transparently upgrades a legacy file (written before this
        milestone - a bare ComplianceResult with no envelope wrapper) into
        an envelope with ``attempt_number=0``/``parent_run_id=None`` and an
        empty ``evaluated_at`` - never an error, since that file is still a
        genuine, valid compliance record, just from before this field set
        existed.

        Raises:
            ComplianceRecordStoreError: If a file exists at that path but
                is corrupt/unparseable.
        """
        path = os.path.join(self.output_dir, f"{run_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ComplianceRecordStoreError(f"Compliance record at '{path}' is corrupt/unreadable: {e}") from e

        try:
            if isinstance(data, dict) and "result" in data and "attempt_number" in data:
                return PersistedComplianceRecord.model_validate(data)
            # Legacy format: a bare ComplianceResult with no envelope.
            return PersistedComplianceRecord(
                run_id=run_id,
                parent_run_id=None,
                attempt_number=0,
                evaluated_at="",
                result=ComplianceResult.model_validate(data),
            )
        except ValidationError as e:
            raise ComplianceRecordStoreError(f"Compliance record at '{path}' is corrupt/unreadable: {e}") from e
