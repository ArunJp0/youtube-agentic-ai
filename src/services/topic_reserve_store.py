# Qualified Topic Reserve persistence (TIER 3 of the multi-tier content-
# continuity strategy - see src.orchestration.topic_continuity_orchestrator).
#
# Mirrors TopicPlanStore's atomic-write pattern exactly (temp file +
# os.replace, one JSON file per record), with one addition: an entry needs
# an in-place UPDATE once consumed (mark_consumed), achieved with the
# identical temp+replace technique targeting the entry's own existing
# filename - which is why, unlike TopicPlanStore, each file here is keyed by
# the entry's own stable entry_id rather than by timestamp.
from __future__ import annotations

import glob
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Set

from pydantic import ValidationError

from src.models.topic_continuity import QualifiedTopicReserveEntry

DEFAULT_TOPIC_RESERVE_DIR = os.path.join("output", "topic_reserve")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QualifiedTopicReserveStoreError(Exception):
    """Raised when a reserve entry file exists but cannot be read/parsed
    (corrupt/malformed) - distinct from "no entries yet", which is simply
    an empty list, never an error."""


class QualifiedTopicReserveStore:
    """Reads/writes persisted QualifiedTopicReserveEntry records, one JSON
    file per entry (keyed by entry_id) - the durable TIER 3 reserve of
    previously-qualified evergreen topics a future run can consume without
    re-running Topic Planner discovery.
    """

    def __init__(self, output_dir: str = DEFAULT_TOPIC_RESERVE_DIR) -> None:
        self.output_dir = output_dir

    def _path_for(self, entry_id: str) -> str:
        return os.path.join(self.output_dir, f"{entry_id}.json")

    def write(self, entry: QualifiedTopicReserveEntry) -> str:
        """Atomically write ``entry`` to its own file (create, or full
        overwrite when the same entry_id already exists) - the shared
        primitive ``add()`` and ``mark_consumed()`` both use.

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = self._path_for(entry.entry_id)

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-topicreserve-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def add(
        self,
        topic: str,
        topic_source: Optional[str] = None,
        category: Optional[str] = None,
        qualification_provider: Optional[str] = None,
    ) -> QualifiedTopicReserveEntry:
        """Persist a new AVAILABLE reserve entry for ``topic``.

        Callers must only call this for a topic that has genuinely just
        passed ``ResearchAgent.research()`` - see
        ``QualifiedTopicReserveEntry``'s docstring; this store never itself
        validates research quality.
        """
        entry = QualifiedTopicReserveEntry(
            entry_id=uuid.uuid4().hex,
            topic=topic,
            topic_source=topic_source,
            category=category,
            qualified_at=_now_iso(),
            qualification_provider=qualification_provider,
            state="available",
        )
        self.write(entry)
        return entry

    def read_entry(self, entry_id: str) -> Optional[QualifiedTopicReserveEntry]:
        """Read one specific entry by its entry_id, or ``None`` if it
        doesn't exist.

        Raises:
            QualifiedTopicReserveStoreError: If the entry file exists but is
                corrupt/unparseable.
        """
        path = self._path_for(entry_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QualifiedTopicReserveEntry.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise QualifiedTopicReserveStoreError(f"Reserve entry at '{path}' is corrupt/unreadable: {e}") from e

    def list_all(self) -> List[QualifiedTopicReserveEntry]:
        """Return every persisted entry (available and consumed), oldest-
        qualified first. A corrupt/unreadable record is skipped (never
        raised) - reserve lookups degrade gracefully rather than blocking
        an autonomous run."""
        if not os.path.isdir(self.output_dir):
            return []

        entries: List[QualifiedTopicReserveEntry] = []
        for path in glob.glob(os.path.join(self.output_dir, "*.json")):
            if os.path.basename(path).startswith(".tmp-"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(QualifiedTopicReserveEntry.model_validate(data))
            except (OSError, json.JSONDecodeError, ValidationError):
                continue

        entries.sort(key=lambda e: e.qualified_at or "")
        return entries

    def list_available(self) -> List[QualifiedTopicReserveEntry]:
        """Every AVAILABLE (never-consumed) entry, oldest-qualified first."""
        return [e for e in self.list_all() if e.state == "available"]

    def get_next_available(self, exclude_topics: Optional[Set[str]] = None) -> Optional[QualifiedTopicReserveEntry]:
        """The oldest-qualified AVAILABLE entry whose topic isn't already in
        ``exclude_topics`` (already tried this run) - never returns a
        consumed entry (see ``mark_consumed`` - a consumed reserve topic is
        never reused)."""
        exclude_topics = exclude_topics or set()
        for entry in self.list_available():
            if entry.topic not in exclude_topics:
                return entry
        return None

    def mark_consumed(self, entry_id: str, reason: Optional[str] = None) -> Optional[QualifiedTopicReserveEntry]:
        """Atomically mark one entry CONSUMED so ``get_next_available`` can
        never return it again. A no-op returning ``None`` if the entry
        doesn't exist (idempotent-safe against a repeated call).

        Raises:
            QualifiedTopicReserveStoreError: If the entry file exists but is
                corrupt/unparseable.
        """
        entry = self.read_entry(entry_id)
        if entry is None:
            return None

        if entry.state == "consumed":
            return entry  # already consumed - never re-consumed/re-timestamped

        entry.state = "consumed"
        entry.consumed_at = _now_iso()
        entry.consumed_reason = reason
        self.write(entry)
        return entry
