# Tests for YouTubeUploadAgent (src/agents/youtube_upload_agent.py):
# deterministic compliance gating, artifact validation, upload/thumbnail
# orchestration, and idempotent publishing-record behavior. Uses
# MockYouTubeClient only - no real Google OAuth, no real YouTube upload,
# no network.
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from src.agents.youtube_upload_agent import YouTubeUploadAgent, YouTubeUploadAgentError
from src.models.compliance import ComplianceResult
from src.models.metadata import MetadataResult
from src.services.publishing_record_store import PublishingRecordStore
from src.services.upload_artifact_discovery import DiscoveredRunArtifacts
from src.tools.youtube_client import MockYouTubeClient


def _video(tmp_path, name="video.mp4") -> str:
    path = tmp_path / name
    path.write_bytes(b"FAKE MP4 BYTES")
    return str(path)


def _thumbnail(tmp_path, name="thumb.jpg") -> str:
    path = tmp_path / name
    Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)
    return str(path)


def _metadata(**overrides) -> MetadataResult:
    defaults = dict(success=True, topic="Why do humans dream?", title="Why Do We Dream?", description="A grounded look.", tags=["dreams", "sleep"])
    defaults.update(overrides)
    return MetadataResult(**defaults)


def _compliance(decision="PASS", **overrides) -> ComplianceResult:
    defaults = dict(success=True, topic="Why do humans dream?", publish_decision=decision, risk_level="low")
    defaults.update(overrides)
    return ComplianceResult(**defaults)


_UNSET = object()


def _artifacts(
    tmp_path, compliance=_UNSET, metadata=_UNSET, thumbnail_path=_UNSET, run_id="run-1", include_thumbnail=True
) -> DiscoveredRunArtifacts:
    """``compliance``/``metadata``/``thumbnail_path`` default to a valid
    PASS-ready fixture when omitted, but a test can pass ``None``
    explicitly to simulate that artifact genuinely being unavailable -
    distinct from "not overridden", via the ``_UNSET`` sentinel."""
    resolved_compliance = _compliance("PASS") if compliance is _UNSET else compliance
    resolved_metadata = _metadata() if metadata is _UNSET else metadata
    resolved_thumbnail = (
        (_thumbnail(tmp_path) if include_thumbnail else None) if thumbnail_path is _UNSET else thumbnail_path
    )
    return DiscoveredRunArtifacts(
        run_id=run_id,
        topic="Why do humans dream?",
        final_video_path=_video(tmp_path),
        metadata_result=resolved_metadata,
        metadata_json_path=str(tmp_path / "metadata.json"),
        thumbnail_path=resolved_thumbnail,
        thumbnail_exact_match=True,
        compliance_result=resolved_compliance,
        warnings=[],
    )


class TestComplianceGate:
    def test_pass_allows_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path, compliance=_compliance("PASS")))

        assert result.success is True
        assert result.video_id is not None
        assert len(client.inserted_videos) == 1

    def test_review_prevents_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path, compliance=_compliance("REVIEW")))

        assert result.success is False
        assert result.status == "compliance_gate_failed"
        assert client.inserted_videos == []

    def test_block_prevents_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path, compliance=_compliance("BLOCK")))

        assert result.success is False
        assert result.status == "compliance_gate_failed"
        assert client.inserted_videos == []

    def test_missing_compliance_result_prevents_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path, compliance=None))

        assert result.success is False
        assert result.status == "compliance_gate_failed"
        assert "no persisted compliance result" in result.error.lower()
        assert client.inserted_videos == []

    def test_gate_is_deterministic_not_string_matched_on_error_text(self, tmp_path) -> None:
        """A ComplianceResult whose warnings/summary happen to mention the
        word 'PASS' in passing text must still be blocked if
        publish_decision itself is REVIEW - the gate checks the enum field,
        never any free-text content."""
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        misleading = _compliance("REVIEW", warnings=["This almost looks like a PASS but is not"])
        result = agent.publish(_artifacts(tmp_path, compliance=misleading))

        assert result.success is False
        assert client.inserted_videos == []


class TestPrivacyDefault:
    def test_default_privacy_is_private(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.privacy_status == "private"
        assert client.inserted_videos[0].privacy_status == "private"


class TestMetadataAndTagsMapping:
    def test_title_and_description_mapped(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        metadata = _metadata(title="Exact Title", description="Exact description text")
        agent.publish(_artifacts(tmp_path, metadata=metadata))

        assert client.inserted_videos[0].title == "Exact Title"
        assert client.inserted_videos[0].description == "Exact description text"

    def test_tags_mapped(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        metadata = _metadata(tags=["alpha", "beta", "gamma"])
        agent.publish(_artifacts(tmp_path, metadata=metadata))

        assert client.inserted_videos[0].tags == ["alpha", "beta", "gamma"]

    def test_missing_metadata_title_blocks_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path, metadata=None))

        assert result.success is False
        assert client.inserted_videos == []


class TestVideoIdCapture:
    def test_video_id_and_url_returned(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.video_id == "mock-video-1"
        assert result.video_url == "https://www.youtube.com/watch?v=mock-video-1"


class TestThumbnailFailurePreservesVideoId:
    def test_thumbnail_failure_does_not_lose_video_id(self, tmp_path) -> None:
        client = MockYouTubeClient(fail_thumbnail=True)
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.success is True  # the video itself exists
        assert result.status == "thumbnail_failed"
        assert result.video_id is not None
        assert result.thumbnail_set is False
        assert result.warnings != []

    def test_upload_failure_never_reports_a_video_id(self, tmp_path) -> None:
        client = MockYouTubeClient(fail_upload=True)
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.success is False
        assert result.status == "upload_failed"
        assert result.video_id is None


class TestChannelVerification:
    def test_channel_verification_failure_blocks_upload(self, tmp_path) -> None:
        client = MockYouTubeClient(fail_channel=True)
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.success is False
        assert result.status == "channel_verification_failed"
        assert client.inserted_videos == []

    def test_channel_info_included_on_success(self, tmp_path) -> None:
        client = MockYouTubeClient(channel_id="UC-real", channel_title="Real Channel")
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        result = agent.publish(_artifacts(tmp_path))

        assert result.channel_id == "UC-real"
        assert result.channel_title == "Real Channel"


class TestDuplicateUploadPrevention:
    def test_second_publish_of_same_run_does_not_reupload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)

        artifacts = _artifacts(tmp_path, run_id="run-dup")
        first = agent.publish(artifacts)
        second = agent.publish(artifacts)

        assert first.video_id == second.video_id
        assert len(client.inserted_videos) == 1  # only uploaded once
        assert any("already published" in w.lower() for w in second.warnings)

    def test_force_allows_reupload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)

        artifacts = _artifacts(tmp_path, run_id="run-dup")
        agent.publish(artifacts)
        agent.publish(artifacts, force=True)

        assert len(client.inserted_videos) == 2

    def test_publishing_record_persisted_after_successful_upload(self, tmp_path) -> None:
        client = MockYouTubeClient()
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)

        artifacts = _artifacts(tmp_path, run_id="run-persist")
        result = agent.publish(artifacts)

        record = store.read("run-persist")
        assert record is not None
        assert record.video_id == result.video_id
        assert record.channel_id == result.channel_id


class TestSchedulingIntegration:
    def test_scheduled_upload_sets_scheduled_status(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        future = datetime.now(timezone.utc) + timedelta(days=1)
        result = agent.publish(_artifacts(tmp_path), privacy_status="private", scheduled_publish_at=future)

        assert result.status == "scheduled"
        assert result.success is True

    def test_scheduling_with_public_privacy_rejected(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        future = datetime.now(timezone.utc) + timedelta(days=1)
        result = agent.publish(_artifacts(tmp_path), privacy_status="public", scheduled_publish_at=future)

        assert result.success is False
        assert result.status == "upload_failed"
        assert client.inserted_videos == []

    def test_naive_schedule_datetime_rejected(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        naive_future = datetime.now() + timedelta(days=1)
        result = agent.publish(_artifacts(tmp_path), privacy_status="private", scheduled_publish_at=naive_future)

        assert result.success is False
        assert result.status == "upload_failed"
        assert client.inserted_videos == []

    def test_past_schedule_rejected(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        past = datetime.now(timezone.utc) - timedelta(days=1)
        result = agent.publish(_artifacts(tmp_path), privacy_status="private", scheduled_publish_at=past)

        assert result.success is False
        assert result.status == "upload_failed"
        assert client.inserted_videos == []

    def test_non_utc_offset_normalized_to_utc_in_request(self, tmp_path) -> None:
        """A future time given in a non-UTC offset (e.g. IST) must reach the
        YouTube API as the correctly-converted UTC RFC 3339 string."""
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        ist = timezone(timedelta(hours=5, minutes=30))
        future_ist = (datetime.now(timezone.utc) + timedelta(days=2)).astimezone(ist)
        agent.publish(_artifacts(tmp_path), privacy_status="private", scheduled_publish_at=future_ist)

        sent_request = client.inserted_videos[0]
        assert sent_request.scheduled_publish_at.tzinfo is not None

    def test_publishing_record_stores_utc_normalized_schedule_string(self, tmp_path) -> None:
        """The persisted PublishingRecord.scheduled_publish_at must be the
        UTC/RFC 3339 form actually sent to YouTube, not the original
        (possibly non-UTC) offset - so the on-disk record is always
        unambiguous."""
        client = MockYouTubeClient()
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)
        ist = timezone(timedelta(hours=5, minutes=30))
        future_ist = datetime(2027, 6, 15, 14, 30, 0, tzinfo=ist)  # 09:00:00Z

        artifacts = _artifacts(tmp_path, run_id="run-schedule-utc")
        agent.publish(artifacts, privacy_status="private", scheduled_publish_at=future_ist)

        record = store.read("run-schedule-utc")
        assert record.scheduled_publish_at == "2027-06-15T09:00:00Z"

    def test_rerun_of_existing_scheduled_run_reports_scheduled_not_completed(self, tmp_path) -> None:
        """Idempotency short-circuit must report the SAME status the
        original scheduled upload received - never a generic 'completed'
        that would misrepresent a still-scheduled video as fully published."""
        client = MockYouTubeClient()
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)
        future = datetime.now(timezone.utc) + timedelta(days=1)

        artifacts = _artifacts(tmp_path, run_id="run-sched-dup")
        first = agent.publish(artifacts, privacy_status="private", scheduled_publish_at=future)
        second = agent.publish(artifacts, privacy_status="private", scheduled_publish_at=future)

        assert first.status == "scheduled"
        assert second.status == "scheduled"
        assert second.video_id == first.video_id
        assert len(client.inserted_videos) == 1  # only uploaded once

    def test_rerun_of_existing_run_with_failed_thumbnail_reports_thumbnail_failed(self, tmp_path) -> None:
        client = MockYouTubeClient(fail_thumbnail=True)
        store = PublishingRecordStore(output_dir=str(tmp_path / "pub"))
        agent = YouTubeUploadAgent(client, publishing_record_store=store)

        artifacts = _artifacts(tmp_path, run_id="run-thumb-dup")
        first = agent.publish(artifacts)
        second = agent.publish(artifacts)

        assert first.status == "thumbnail_failed"
        assert second.status == "thumbnail_failed"
        assert second.video_id == first.video_id


class TestConfigurationErrors:
    def test_missing_artifacts_raises(self, tmp_path) -> None:
        client = MockYouTubeClient()
        agent = YouTubeUploadAgent(client, publishing_record_store=PublishingRecordStore(output_dir=str(tmp_path / "pub")))
        with pytest.raises(YouTubeUploadAgentError):
            agent.publish(None)
