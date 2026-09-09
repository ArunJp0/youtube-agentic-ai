# Tests for the compliance-result persistence boundary
# (src/services/compliance_record_store.py). Local filesystem only.
from __future__ import annotations

import glob
import os

import pytest

from src.models.compliance import ComplianceResult
from src.services.compliance_record_store import ComplianceRecordStore, ComplianceRecordStoreError


def _result(topic="Why do humans dream?", decision="PASS", **overrides) -> ComplianceResult:
    defaults = dict(success=True, topic=topic, publish_decision=decision, risk_level="low")
    defaults.update(overrides)
    return ComplianceResult(**defaults)


class TestRoundTrip:
    def test_write_then_read_round_trip(self, tmp_path) -> None:
        store = ComplianceRecordStore(output_dir=str(tmp_path))
        result = _result()
        path = store.write("run-1", result)

        assert os.path.exists(path)
        restored = store.read("run-1")
        assert restored == result

    def test_read_nonexistent_returns_none(self, tmp_path) -> None:
        store = ComplianceRecordStore(output_dir=str(tmp_path))
        assert store.read("no-such-run") is None

    def test_review_and_block_decisions_round_trip(self, tmp_path) -> None:
        store = ComplianceRecordStore(output_dir=str(tmp_path))
        store.write("run-review", _result(decision="REVIEW"))
        store.write("run-block", _result(decision="BLOCK"))

        assert store.read("run-review").publish_decision == "REVIEW"
        assert store.read("run-block").publish_decision == "BLOCK"


class TestAtomicPersistence:
    def test_no_leftover_temp_files_after_write(self, tmp_path) -> None:
        store = ComplianceRecordStore(output_dir=str(tmp_path))
        store.write("run-1", _result())
        leftover = glob.glob(os.path.join(str(tmp_path), ".tmp-compliance-*"))
        assert leftover == []


class TestCorruptRecord:
    def test_malformed_json_raises_store_error(self, tmp_path) -> None:
        store = ComplianceRecordStore(output_dir=str(tmp_path))
        (tmp_path / "broken-run.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ComplianceRecordStoreError):
            store.read("broken-run")
