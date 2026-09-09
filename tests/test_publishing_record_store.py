# Tests for the publishing-record persistence boundary
# (src/services/publishing_record_store.py). Local filesystem only.
from __future__ import annotations

import glob
import os

import pytest

from src.models.youtube_upload import PublishingRecord
from src.services.publishing_record_store import PublishingRecordStore, PublishingRecordStoreError


def _record(run_id="run-1", **overrides) -> PublishingRecord:
    defaults = dict(
        run_id=run_id,
        video_id="vid123",
        video_url="https://www.youtube.com/watch?v=vid123",
        channel_id="UC1",
        channel_title="My Channel",
        uploaded_at="2027-01-01T00:00:00+00:00",
        privacy_status="private",
        thumbnail_set=True,
    )
    defaults.update(overrides)
    return PublishingRecord(**defaults)


class TestRoundTrip:
    def test_write_then_read_round_trip(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        record = _record()
        path = store.write(record)

        assert os.path.exists(path)
        restored = store.read(record.run_id)
        assert restored == record

    def test_read_nonexistent_returns_none(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        assert store.read("no-such-run") is None

    def test_write_creates_output_directory(self, tmp_path) -> None:
        nested = tmp_path / "does" / "not" / "exist"
        store = PublishingRecordStore(output_dir=str(nested))
        store.write(_record())
        assert nested.exists()


class TestAtomicPersistence:
    def test_no_leftover_temp_files_after_write(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        store.write(_record())
        leftover = glob.glob(os.path.join(str(tmp_path), ".tmp-publishing-*"))
        assert leftover == []

    def test_overwrite_replaces_existing_record(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        store.write(_record(video_id="first"))
        store.write(_record(video_id="second"))
        restored = store.read("run-1")
        assert restored.video_id == "second"


class TestCorruptRecord:
    def test_malformed_json_raises_store_error(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        (tmp_path / "broken-run.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(PublishingRecordStoreError):
            store.read("broken-run")

    def test_missing_vs_corrupt_distinguishable(self, tmp_path) -> None:
        store = PublishingRecordStore(output_dir=str(tmp_path))
        assert store.read("truly-missing") is None
        (tmp_path / "corrupt.json").write_text("not json", encoding="utf-8")
        with pytest.raises(PublishingRecordStoreError):
            store.read("corrupt")
