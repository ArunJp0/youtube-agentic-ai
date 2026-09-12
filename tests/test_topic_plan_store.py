# Tests for the Topic Planner decision persistence boundary
# (src/services/topic_plan_store.py). Local filesystem only.
from __future__ import annotations

import glob
import os

import pytest

from src.models.topic_planner import TopicSelectionResult
from src.services.topic_plan_store import TopicPlanStore, TopicPlanStoreError


def _result(topic="Why do cats purr?", selected_at="2026-09-10T06:00:00+00:00", **overrides) -> TopicSelectionResult:
    defaults = dict(success=True, status="selected", selected_topic=topic, score=0.8, selected_at=selected_at)
    defaults.update(overrides)
    return TopicSelectionResult(**defaults)


class TestRoundTrip:
    def test_write_then_read_round_trip(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        result = _result()
        path = store.write(result)

        assert os.path.exists(path)
        restored = store.read(path)
        assert restored == result

    def test_read_nonexistent_returns_none(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        assert store.read(os.path.join(str(tmp_path), "no-such-file.json")) is None

    def test_failure_result_round_trips_too(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        result = TopicSelectionResult(success=False, status="no_candidates", selected_at="2026-09-10T06:00:00+00:00")
        path = store.write(result)
        assert store.read(path) == result


class TestListRecent:
    def test_empty_store_returns_empty_list(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        assert store.list_recent() == []

    def test_most_recent_first(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        store.write(_result(topic="Topic A", selected_at="2026-09-10T06:00:00+00:00"))
        store.write(_result(topic="Topic B", selected_at="2026-09-10T08:00:00+00:00"))
        store.write(_result(topic="Topic C", selected_at="2026-09-10T07:00:00+00:00"))

        recent = store.list_recent()
        assert [r.selected_topic for r in recent] == ["Topic B", "Topic C", "Topic A"]

    def test_limit_caps_results(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        for i in range(5):
            store.write(_result(topic=f"Topic {i}", selected_at=f"2026-09-10T0{i}:00:00+00:00"))

        assert len(store.list_recent(limit=2)) == 2

    def test_corrupt_record_skipped_not_raised(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        store.write(_result())
        (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

        recent = store.list_recent()
        assert len(recent) == 1


class TestAtomicPersistence:
    def test_no_leftover_temp_files_after_write(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        store.write(_result())
        leftover = glob.glob(os.path.join(str(tmp_path), ".tmp-topicplan-*"))
        assert leftover == []

    def test_two_writes_produce_two_distinct_files(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        store.write(_result(topic="Topic A"))
        store.write(_result(topic="Topic B"))
        files = glob.glob(os.path.join(str(tmp_path), "*.json"))
        assert len(files) == 2


class TestCorruptRecord:
    def test_malformed_json_raises_store_error_on_direct_read(self, tmp_path) -> None:
        store = TopicPlanStore(output_dir=str(tmp_path))
        broken_path = str(tmp_path / "broken.json")
        (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(TopicPlanStoreError):
            store.read(broken_path)
