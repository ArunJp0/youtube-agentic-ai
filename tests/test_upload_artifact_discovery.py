# Tests for standalone Upload Agent artifact discovery
# (src/services/upload_artifact_discovery.py). Local filesystem (tmp_path)
# only - no network involved.
from __future__ import annotations

import json
import os

from src.models.compliance import ComplianceResult
from src.models.provenance import BGMProvenance, ProvenanceManifest, ThumbnailProvenance, VisualAssetProvenance
from src.services.compliance_record_store import ComplianceRecordStore
from src.services.provenance_store import ProvenanceManifestStore
from src.services.upload_artifact_discovery import discover_latest_run_artifacts


def _write_video(video_dir, run_id: str, suffix: str = "-captioned-bgm.mp4") -> str:
    path = video_dir / f"{run_id}{suffix}"
    path.write_bytes(b"FAKE MP4 BYTES")
    return str(path)


def _write_metadata(metadata_dir, filename: str, topic: str, title: str) -> str:
    path = metadata_dir / filename
    path.write_text(json.dumps({"topic": topic, "title": title, "description": "d", "tags": []}), encoding="utf-8")
    return str(path)


def _write_thumbnail(thumbnail_dir, filename: str) -> str:
    path = thumbnail_dir / filename
    path.write_bytes(b"FAKE JPEG BYTES")
    return str(path)


def _manifest(run_id: str, video_path: str, topic: str, thumbnail_output_path: str) -> ProvenanceManifest:
    return ProvenanceManifest(
        run_id=run_id,
        topic=topic,
        final_video_path=video_path,
        created_at="2027-01-01T00:00:00+00:00",
        visual_assets=[VisualAssetProvenance(section_index=0, section_heading="S", provider="pexels", source_url="https://pexels.com/1")],
        thumbnail=ThumbnailProvenance(provider="pexels", source_url="https://pexels.com/2", output_path=thumbnail_output_path),
        bgm=BGMProvenance(track_id="calm-music", title="Calm Music", source="s", license_type="l"),
    )


def _compliance(topic: str, decision: str = "PASS") -> ComplianceResult:
    return ComplianceResult(success=True, topic=topic, publish_decision=decision, risk_level="low")


class TestBasicDiscovery:
    def test_no_video_returns_none(self, tmp_path) -> None:
        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_output_dir=str(tmp_path / "video"),
            metadata_output_dir=str(tmp_path / "metadata"), thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result is None

    def test_run_id_derived_from_video_filename(self, tmp_path) -> None:
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        video_path = _write_video(video_dir, "why-do-humans-dream-aaaaaaaa")

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path,
            metadata_output_dir=str(tmp_path / "metadata"), thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.run_id == "why-do-humans-dream-aaaaaaaa"
        assert result.final_video_path == video_path


class TestThumbnailAssociation:
    def test_exact_thumbnail_via_provenance_manifest(self, tmp_path) -> None:
        video_dir, thumb_dir, prov_dir = tmp_path / "video", tmp_path / "thumb", tmp_path / "prov"
        video_dir.mkdir(); thumb_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")
        thumb_path = _write_thumbnail(thumb_dir, "exact-thumb.jpg")
        prov_store = ProvenanceManifestStore(output_dir=str(prov_dir))
        prov_store.write(_manifest("run-a", video_path, "Why do humans dream?", thumb_path))

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path,
            metadata_output_dir=str(tmp_path / "metadata"), thumbnail_output_dir=str(thumb_dir),
            provenance_manifest_store=prov_store, compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.thumbnail_path == thumb_path
        assert result.thumbnail_exact_match is True

    def test_fallback_thumbnail_when_no_manifest_warns(self, tmp_path) -> None:
        video_dir, thumb_dir = tmp_path / "video", tmp_path / "thumb"
        video_dir.mkdir(); thumb_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")
        fallback_thumb = _write_thumbnail(thumb_dir, "latest.jpg")

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path,
            metadata_output_dir=str(tmp_path / "metadata"), thumbnail_output_dir=str(thumb_dir),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.thumbnail_path == fallback_thumb
        assert result.thumbnail_exact_match is False
        assert any("no exact thumbnail association" in w.lower() for w in result.warnings)


class TestMetadataAssociation:
    def test_metadata_matched_by_topic(self, tmp_path) -> None:
        video_dir, metadata_dir = tmp_path / "video", tmp_path / "metadata"
        video_dir.mkdir(); metadata_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")
        _write_metadata(metadata_dir, "unrelated-title.json", "A different topic", "Unrelated Title")
        _write_metadata(metadata_dir, "matching-title.json", "Why do humans dream?", "Why Do We Dream?")

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path, metadata_output_dir=str(metadata_dir),
            thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.metadata_result is not None
        assert result.metadata_result.title == "Why Do We Dream?"

    def test_no_topic_match_warns_and_returns_none_metadata(self, tmp_path) -> None:
        video_dir, metadata_dir = tmp_path / "video", tmp_path / "metadata"
        video_dir.mkdir(); metadata_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")
        _write_metadata(metadata_dir, "unrelated.json", "A different topic", "Unrelated Title")

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path, metadata_output_dir=str(metadata_dir),
            thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.metadata_result is None
        assert any("no metadata json" in w.lower() for w in result.warnings)


class TestComplianceAssociation:
    def test_compliance_found_via_exact_run_id(self, tmp_path) -> None:
        video_dir, comp_dir = tmp_path / "video", tmp_path / "comp"
        video_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")
        comp_store = ComplianceRecordStore(output_dir=str(comp_dir))
        comp_store.write("run-a", _compliance("Why do humans dream?", "PASS"))

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path, metadata_output_dir=str(tmp_path / "metadata"),
            thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=comp_store,
        )
        assert result.compliance_result is not None
        assert result.compliance_result.publish_decision == "PASS"

    def test_missing_compliance_warns(self, tmp_path) -> None:
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        video_path = _write_video(video_dir, "run-a")

        result = discover_latest_run_artifacts(
            "Why do humans dream?", video_path=video_path, metadata_output_dir=str(tmp_path / "metadata"),
            thumbnail_output_dir=str(tmp_path / "thumb"),
            provenance_manifest_store=ProvenanceManifestStore(output_dir=str(tmp_path / "prov")),
            compliance_record_store=ComplianceRecordStore(output_dir=str(tmp_path / "comp")),
        )
        assert result.compliance_result is None
        assert any("no persisted compliance result" in w.lower() for w in result.warnings)


class TestNoCrossRunMismatch:
    def test_two_runs_never_mix_provenance_or_compliance(self, tmp_path) -> None:
        """Two separate runs (different run_ids) each get their own
        provenance/thumbnail/compliance data - selecting run B's video must
        never pull in run A's associated data."""
        video_dir, thumb_dir, prov_dir, comp_dir = (
            tmp_path / "video", tmp_path / "thumb", tmp_path / "prov", tmp_path / "comp"
        )
        video_dir.mkdir(); thumb_dir.mkdir()
        prov_store = ProvenanceManifestStore(output_dir=str(prov_dir))
        comp_store = ComplianceRecordStore(output_dir=str(comp_dir))

        video_a = _write_video(video_dir, "run-aaaaaaaa")
        thumb_a = _write_thumbnail(thumb_dir, "thumb-a.jpg")
        prov_store.write(_manifest("run-aaaaaaaa", video_a, "Topic A", thumb_a))
        comp_store.write("run-aaaaaaaa", _compliance("Topic A", "BLOCK"))

        video_b = _write_video(video_dir, "run-bbbbbbbb")
        thumb_b = _write_thumbnail(thumb_dir, "thumb-b.jpg")
        prov_store.write(_manifest("run-bbbbbbbb", video_b, "Topic B", thumb_b))
        comp_store.write("run-bbbbbbbb", _compliance("Topic B", "PASS"))

        result_b = discover_latest_run_artifacts(
            "Topic B", video_path=video_b, metadata_output_dir=str(tmp_path / "metadata"),
            thumbnail_output_dir=str(thumb_dir), provenance_manifest_store=prov_store, compliance_record_store=comp_store,
        )

        assert result_b.run_id == "run-bbbbbbbb"
        assert result_b.thumbnail_path == thumb_b
        assert result_b.compliance_result.publish_decision == "PASS"
        # Never run A's blocked decision or thumbnail.
        assert result_b.thumbnail_path != thumb_a
        assert result_b.compliance_result.publish_decision != "BLOCK"
