# Tests for deterministic pre-flight upload validation
# (src/services/youtube_upload_validation.py). No network involved.
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from src.services.youtube_upload_validation import (
    UploadValidationError,
    format_publish_at,
    validate_scheduling,
    validate_thumbnail_file,
    validate_video_file,
)


class TestValidateVideoFile:
    def test_existing_nonempty_video_passes(self, tmp_path) -> None:
        path = tmp_path / "video.mp4"
        path.write_bytes(b"FAKE MP4")
        validate_video_file(str(path))  # should not raise

    def test_missing_path_raises(self) -> None:
        with pytest.raises(UploadValidationError):
            validate_video_file(None)

    def test_nonexistent_file_raises(self, tmp_path) -> None:
        with pytest.raises(UploadValidationError):
            validate_video_file(str(tmp_path / "nope.mp4"))

    def test_empty_file_raises(self, tmp_path) -> None:
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        with pytest.raises(UploadValidationError):
            validate_video_file(str(path))


class TestValidateThumbnailFile:
    def test_valid_1280x720_image_passes(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)
        validate_thumbnail_file(str(path))  # should not raise

    def test_missing_path_raises(self) -> None:
        with pytest.raises(UploadValidationError):
            validate_thumbnail_file(None)

    def test_nonexistent_file_raises(self, tmp_path) -> None:
        with pytest.raises(UploadValidationError):
            validate_thumbnail_file(str(tmp_path / "nope.jpg"))

    def test_corrupt_file_raises(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        path.write_bytes(b"not an image")
        with pytest.raises(UploadValidationError):
            validate_thumbnail_file(str(path))

    def test_wrong_dimensions_raises(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        Image.new("RGB", (640, 480), (10, 20, 30)).save(path)
        with pytest.raises(UploadValidationError):
            validate_thumbnail_file(str(path))

    def test_oversized_file_raises(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "thumb.jpg"
        Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)
        monkeypatch.setattr("src.services.youtube_upload_validation.MAX_THUMBNAIL_BYTES", 10)
        with pytest.raises(UploadValidationError):
            validate_thumbnail_file(str(path))


class TestValidateScheduling:
    def test_none_scheduled_time_is_always_ok(self) -> None:
        validate_scheduling("public", None)  # should not raise

    def test_future_private_scheduled_time_ok(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(days=1)
        validate_scheduling("private", future)  # should not raise

    def test_scheduling_requires_private(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(days=1)
        with pytest.raises(UploadValidationError, match="private"):
            validate_scheduling("public", future)

        with pytest.raises(UploadValidationError, match="private"):
            validate_scheduling("unlisted", future)

    def test_scheduling_requires_future_time(self) -> None:
        past = datetime.now(timezone.utc) - timedelta(days=1)
        with pytest.raises(UploadValidationError, match="future"):
            validate_scheduling("private", past)

    def test_scheduling_rejects_exact_now(self) -> None:
        now = datetime.now(timezone.utc)
        with pytest.raises(UploadValidationError):
            validate_scheduling("private", now, now=now)

    def test_naive_datetime_rejected(self) -> None:
        naive = datetime.now() + timedelta(days=1)
        with pytest.raises(UploadValidationError, match="timezone-aware"):
            validate_scheduling("private", naive)

    def test_injected_now_used_for_comparison(self) -> None:
        fixed_now = datetime(2027, 1, 1, tzinfo=timezone.utc)
        future = datetime(2027, 1, 2, tzinfo=timezone.utc)
        validate_scheduling("private", future, now=fixed_now)  # should not raise

        past_relative = datetime(2026, 12, 31, tzinfo=timezone.utc)
        with pytest.raises(UploadValidationError):
            validate_scheduling("private", past_relative, now=fixed_now)


class TestFormatPublishAt:
    def test_utc_datetime_formatting(self) -> None:
        dt = datetime(2027, 3, 15, 9, 30, 0, tzinfo=timezone.utc)
        assert format_publish_at(dt) == "2027-03-15T09:30:00Z"

    def test_non_utc_datetime_converted_to_utc(self) -> None:
        from datetime import timezone as tz

        offset = tz(timedelta(hours=5, minutes=30))  # IST
        dt = datetime(2027, 3, 15, 15, 0, 0, tzinfo=offset)
        assert format_publish_at(dt) == "2027-03-15T09:30:00Z"
