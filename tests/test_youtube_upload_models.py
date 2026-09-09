# Tests for YouTube upload typed models (src/models/youtube_upload.py).
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.youtube_upload import DEFAULT_PRIVACY_STATUS, UploadRequest, UploadResult, YouTubeChannelInfo


class TestPrivacyDefault:
    def test_default_privacy_is_private(self) -> None:
        assert DEFAULT_PRIVACY_STATUS == "private"

    def test_upload_request_defaults_to_private(self) -> None:
        request = UploadRequest(video_path="v.mp4", title="T")
        assert request.privacy_status == "private"


class TestScheduledPublishAtValidation:
    def test_timezone_aware_datetime_accepted(self) -> None:
        request = UploadRequest(
            video_path="v.mp4", title="T", scheduled_publish_at=datetime(2027, 1, 1, tzinfo=timezone.utc)
        )
        assert request.scheduled_publish_at.tzinfo is not None

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UploadRequest(video_path="v.mp4", title="T", scheduled_publish_at=datetime(2027, 1, 1))

    def test_none_is_allowed(self) -> None:
        request = UploadRequest(video_path="v.mp4", title="T", scheduled_publish_at=None)
        assert request.scheduled_publish_at is None


class TestYouTubeChannelInfo:
    def test_construction(self) -> None:
        channel = YouTubeChannelInfo(channel_id="UC1", channel_title="My Channel")
        assert channel.channel_id == "UC1"


class TestUploadResult:
    def test_defaults(self) -> None:
        result = UploadResult(success=False, status="not_started")
        assert result.video_id is None
        assert result.thumbnail_set is False
        assert result.warnings == []

    def test_video_id_preserved_on_thumbnail_failed_status(self) -> None:
        result = UploadResult(success=True, status="thumbnail_failed", video_id="v123", thumbnail_set=False)
        assert result.video_id == "v123"
        assert result.success is True
