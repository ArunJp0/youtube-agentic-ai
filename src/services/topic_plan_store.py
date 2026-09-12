# Topic Planner decision persistence - mirrors ProvenanceManifestStore/
# ComplianceRecordStore/PublishingRecordStore exactly: atomic writes (temp
# file + os.replace), one JSON file per record. Unlike those stores,
# there's no run_id yet at planning time (planning happens BEFORE
# Research), so each record is keyed by its own selection timestamp plus a
# short random suffix for uniqueness.
from __future__ import annotations

import glob
import json
import os
import tempfile
import uuid
from typing import List, Optional

from pydantic import ValidationError

from src.models.topic_planner import TopicSelectionResult

DEFAULT_TOPIC_PLAN_DIR = os.path.join("output", "topic_plans")


class TopicPlanStoreError(Exception):
    """Raised when a topic plan record file exists but cannot be read or
    parsed (corrupt/malformed) - distinct from "no records yet", which is
    simply an empty list, never an error."""


class TopicPlanStore:
    """Reads/writes persisted TopicSelectionResult records, one JSON file
    per planning decision - the durable trail of what was selected, when,
    and why (see TopicSelectionResult), and a secondary duplicate-check
    signal alongside the primary provenance-based history."""

    def __init__(self, output_dir: str = DEFAULT_TOPIC_PLAN_DIR) -> None:
        self.output_dir = output_dir

    def write(self, result: TopicSelectionResult) -> str:
        """Atomically write ``result`` to a new, uniquely-named file.

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_slug = (result.selected_at or "").replace(":", "").replace("-", "").rstrip("Z") or "unknown-time"
        filename = f"{timestamp_slug}-{uuid.uuid4().hex[:8]}.json"
        final_path = os.path.join(self.output_dir, filename)

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-topicplan-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def list_recent(self, limit: Optional[int] = None) -> List[TopicSelectionResult]:
        """Return persisted records, most-recently-selected first.

        A corrupt/unreadable record is skipped (never raised) - history
        lookups degrade gracefully rather than failing planning entirely.
        """
        if not os.path.isdir(self.output_dir):
            return []

        results: List[TopicSelectionResult] = []
        for path in glob.glob(os.path.join(self.output_dir, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(TopicSelectionResult.model_validate(data))
            except (OSError, json.JSONDecodeError, ValidationError):
                continue

        results.sort(key=lambda r: r.selected_at or "", reverse=True)
        return results[:limit] if limit is not None else results

    def read(self, path: str) -> Optional[TopicSelectionResult]:
        """Read one specific record by its exact file path, or ``None`` if
        it doesn't exist.

        Raises:
            TopicPlanStoreError: If the file exists but is corrupt/unparseable.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TopicSelectionResult.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise TopicPlanStoreError(f"Topic plan record at '{path}' is corrupt/unreadable: {e}") from e
