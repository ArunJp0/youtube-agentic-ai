# Publishing record persistence boundary - the idempotency guard preventing
# an accidental duplicate YouTube upload of the same run. Mirrors
# src/services/provenance_store.py exactly: the ONLY place that reads or
# writes a publishing record, atomic writes, one JSON file per run.
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from pydantic import ValidationError

from src.models.youtube_upload import PublishingRecord

DEFAULT_PUBLISHING_RECORD_DIR = os.path.join("output", "publishing")


class PublishingRecordStoreError(Exception):
    """Raised when a publishing record file exists but cannot be read or
    parsed (corrupt/malformed) - distinct from a record simply not
    existing (which read() reports as ``None``, never an error)."""


class PublishingRecordStore:
    """Reads/writes ``PublishingRecord`` entries, one JSON file per run
    (``<output_dir>/<run_id>.json``). A run with an existing record has
    already been successfully published - see YouTubeUploadAgent, which
    checks this before ever calling YouTubeClient.insert_video."""

    def __init__(self, output_dir: str = DEFAULT_PUBLISHING_RECORD_DIR) -> None:
        self.output_dir = output_dir

    def write(self, record: PublishingRecord) -> str:
        """Atomically write ``record`` to ``<output_dir>/<run_id>.json``.

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = os.path.join(self.output_dir, f"{record.run_id}.json")

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-publishing-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def read(self, run_id: str) -> Optional[PublishingRecord]:
        """Read the publishing record for ``run_id``, or ``None`` if this
        run has never been published - a valid, expected state.

        Raises:
            PublishingRecordStoreError: If a file exists at that path but
                is corrupt/unparseable.
        """
        path = os.path.join(self.output_dir, f"{run_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PublishingRecord.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise PublishingRecordStoreError(f"Publishing record at '{path}' is corrupt/unreadable: {e}") from e
