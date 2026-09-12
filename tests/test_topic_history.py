# Tests for Topic Planner's produced-run history reader
# (src/services/topic_history.py) - reuses ProvenanceManifestStore
# read-only, never modifies it.
from __future__ import annotations

from src.models.provenance import ProvenanceManifest
from src.services.provenance_store import ProvenanceManifestStore
from src.services.topic_history import load_recent_produced_topics


def _manifest(run_id: str, topic: str, created_at: str) -> ProvenanceManifest:
    return ProvenanceManifest(
        run_id=run_id, topic=topic, final_video_path=f"output/video/{run_id}.mp4", created_at=created_at
    )


class TestLoadRecentProducedTopics:
    def test_empty_directory_returns_empty_list(self, tmp_path) -> None:
        assert load_recent_produced_topics(provenance_dir=str(tmp_path)) == []

    def test_missing_directory_returns_empty_list(self, tmp_path) -> None:
        assert load_recent_produced_topics(provenance_dir=str(tmp_path / "does-not-exist")) == []

    def test_reads_topic_and_created_at(self, tmp_path) -> None:
        store = ProvenanceManifestStore(str(tmp_path))
        store.write(_manifest("run-1", "Why do humans dream?", "2026-09-08T23:08:17+00:00"))

        history = load_recent_produced_topics(provenance_dir=str(tmp_path))
        assert history == [("Why do humans dream?", "2026-09-08T23:08:17+00:00")]

    def test_most_recent_first(self, tmp_path) -> None:
        store = ProvenanceManifestStore(str(tmp_path))
        store.write(_manifest("run-a", "Topic A", "2026-09-08T10:00:00+00:00"))
        store.write(_manifest("run-b", "Topic B", "2026-09-09T10:00:00+00:00"))
        store.write(_manifest("run-c", "Topic C", "2026-09-07T10:00:00+00:00"))

        history = load_recent_produced_topics(provenance_dir=str(tmp_path))
        assert [topic for topic, _ in history] == ["Topic B", "Topic A", "Topic C"]

    def test_limit_caps_results(self, tmp_path) -> None:
        store = ProvenanceManifestStore(str(tmp_path))
        for i in range(5):
            store.write(_manifest(f"run-{i}", f"Topic {i}", f"2026-09-0{i+1}T10:00:00+00:00"))

        assert len(load_recent_produced_topics(provenance_dir=str(tmp_path), limit=2)) == 2

    def test_corrupt_manifest_skipped_not_raised(self, tmp_path) -> None:
        store = ProvenanceManifestStore(str(tmp_path))
        store.write(_manifest("run-good", "Good Topic", "2026-09-08T10:00:00+00:00"))
        (tmp_path / "run-broken.json").write_text("{not valid json", encoding="utf-8")

        history = load_recent_produced_topics(provenance_dir=str(tmp_path))
        assert history == [("Good Topic", "2026-09-08T10:00:00+00:00")]
