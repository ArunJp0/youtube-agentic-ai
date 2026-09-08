# Tests for the provenance manifest persistence boundary
# (src/services/provenance_store.py). Local filesystem only (tmp_path) -
# no network involved.
from __future__ import annotations

import glob
import os

import pytest

from src.models.provenance import BGMProvenance, ProvenanceManifest, VisualAssetProvenance
from src.services.provenance_store import ProvenanceManifestStore, ProvenanceStoreError


def _manifest(run_id="why-do-humans-dream-8baeee8d", video_path=None, **overrides) -> ProvenanceManifest:
    defaults = dict(
        run_id=run_id,
        topic="Why do humans dream?",
        final_video_path=video_path or f"output/video/{run_id}-captioned-bgm.mp4",
        created_at="2026-01-01T00:00:00+00:00",
        visual_assets=[VisualAssetProvenance(section_index=0, section_heading="Intro", provider="pexels")],
        bgm=BGMProvenance(track_id="calm-music", title="Calm Music", source="s", license_type="l"),
    )
    defaults.update(overrides)
    return ProvenanceManifest(**defaults)


class TestWriteAndRead:
    def test_write_then_read_round_trip(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        manifest = _manifest()
        path = store.write(manifest)

        assert os.path.exists(path)
        restored = store.read(manifest.run_id)
        assert restored == manifest

    def test_read_nonexistent_returns_none(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        assert store.read("no-such-run") is None

    def test_write_creates_output_directory(self, tmp_path) -> None:
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        store = ProvenanceManifestStore(output_dir=str(nested))
        store.write(_manifest())
        assert nested.exists()

    def test_write_overwrites_existing_manifest_for_same_run(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        store.write(_manifest(topic="First topic"))
        store.write(_manifest(topic="Second topic"))
        restored = store.read("why-do-humans-dream-8baeee8d")
        assert restored.topic == "Second topic"

    def test_no_leftover_temp_files_after_write(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        store.write(_manifest())
        leftover_temp_files = glob.glob(os.path.join(str(tmp_path), ".tmp-provenance-*"))
        assert leftover_temp_files == []


class TestFindForVideo:
    def test_finds_manifest_for_original_video(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        store.write(_manifest(run_id="why-do-humans-dream-8baeee8d"))
        found = store.find_for_video("output/video/why-do-humans-dream-8baeee8d.mp4")
        assert found is not None
        assert found.run_id == "why-do-humans-dream-8baeee8d"

    def test_finds_manifest_regardless_of_captioned_or_bgm_suffix(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        store.write(_manifest(run_id="why-do-humans-dream-8baeee8d"))

        assert store.find_for_video("output/video/why-do-humans-dream-8baeee8d-captioned.mp4") is not None
        assert store.find_for_video("output/video/why-do-humans-dream-8baeee8d-captioned-bgm.mp4") is not None

    def test_multiple_runs_of_same_topic_select_correct_manifest(self, tmp_path) -> None:
        """Two separate runs of the same topic get different hashes in their
        video filenames - each must resolve to its own distinct manifest,
        never the other run's."""
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        store.write(_manifest(run_id="why-do-humans-dream-aaaaaaaa", topic="Run A"))
        store.write(_manifest(run_id="why-do-humans-dream-bbbbbbbb", topic="Run B"))

        found_a = store.find_for_video("output/video/why-do-humans-dream-aaaaaaaa-captioned-bgm.mp4")
        found_b = store.find_for_video("output/video/why-do-humans-dream-bbbbbbbb-captioned-bgm.mp4")

        assert found_a.topic == "Run A"
        assert found_b.topic == "Run B"
        assert found_a.run_id != found_b.run_id

    def test_legacy_video_with_no_manifest_returns_none(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        found = store.find_for_video("output/video/some-old-run-deadbeef-captioned-bgm.mp4")
        assert found is None


class TestCorruptManifest:
    def test_malformed_json_raises_store_error(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        (tmp_path / "broken-run.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ProvenanceStoreError):
            store.read("broken-run")

    def test_valid_json_but_wrong_shape_raises_store_error(self, tmp_path) -> None:
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        (tmp_path / "wrong-shape.json").write_text('{"unexpected": "shape"}', encoding="utf-8")
        with pytest.raises(ProvenanceStoreError):
            store.read("wrong-shape")

    def test_corrupt_manifest_distinguishable_from_missing(self, tmp_path) -> None:
        """A missing manifest returns None (a valid state); a corrupt one
        raises - callers must never confuse the two."""
        store = ProvenanceManifestStore(output_dir=str(tmp_path))
        assert store.read("truly-missing") is None

        (tmp_path / "corrupt.json").write_text("not json at all", encoding="utf-8")
        with pytest.raises(ProvenanceStoreError):
            store.read("corrupt")
