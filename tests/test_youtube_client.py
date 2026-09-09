# Tests for the YouTube Data API client abstraction
# (src/tools/youtube_client.py). GoogleYouTubeClient is tested with an
# injected fake API resource (no real googleapiclient network calls);
# MockYouTubeClient is tested directly. No real YouTube upload anywhere.
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from googleapiclient.errors import HttpError

from src.models.youtube_upload import UploadRequest
from src.tools.youtube_client import GoogleYouTubeClient, MockYouTubeClient, YouTubeClientError


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = "error"


def _http_error(status: int) -> HttpError:
    return HttpError(resp=_FakeResp(status), content=b'{"error": "simulated"}')


class FakeChannelsResource:
    def __init__(self, items) -> None:
        self._items = items

    def list(self, part, mine):
        return self

    def execute(self):
        return {"items": self._items}


class FakeChannelsResourceRaising:
    def __init__(self, error: HttpError) -> None:
        self._error = error

    def list(self, part, mine):
        return self

    def execute(self):
        raise self._error


class FakeInsertRequest:
    """Simulates googleapiclient's resumable-upload request object: each
    next_chunk() call either raises (to test retry) or returns a final
    response after ``fail_times`` retryable failures."""

    def __init__(self, video_id: str, fail_times: int = 0, error_status: int = 503, permanent_error: HttpError | None = None):
        self.video_id = video_id
        self.fail_times = fail_times
        self.error_status = error_status
        self.permanent_error = permanent_error
        self.calls = 0

    def next_chunk(self):
        self.calls += 1
        if self.permanent_error is not None:
            raise self.permanent_error
        if self.calls <= self.fail_times:
            raise _http_error(self.error_status)
        return (None, {"id": self.video_id})


class FakeVideosResource:
    def __init__(self, insert_request: FakeInsertRequest, captured: dict) -> None:
        self._insert_request = insert_request
        self._captured = captured

    def insert(self, part, body, media_body):
        self._captured["part"] = part
        self._captured["body"] = body
        self._captured["media_body"] = media_body
        return self._insert_request


class FakeThumbnailsResource:
    def __init__(self, captured: dict, error: HttpError | None = None) -> None:
        self._captured = captured
        self._error = error

    def set(self, videoId, media_body):
        self._captured["video_id"] = videoId
        self._captured["media_body"] = media_body
        return self

    def execute(self):
        if self._error is not None:
            raise self._error
        return {}


class FakeYouTubeService:
    def __init__(self, channels=None, videos=None, thumbnails=None) -> None:
        self._channels = channels
        self._videos = videos
        self._thumbnails = thumbnails

    def channels(self):
        return self._channels

    def videos(self):
        return self._videos

    def thumbnails(self):
        return self._thumbnails


def _request(video_path: str, **overrides) -> UploadRequest:
    defaults = dict(video_path=video_path, title="T", description="D", tags=["a", "b"])
    defaults.update(overrides)
    return UploadRequest(**defaults)


def _video_file(tmp_path) -> str:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"FAKE MP4 BYTES")
    return str(path)


class TestChannelVerification:
    def test_returns_authenticated_channel(self) -> None:
        service = FakeYouTubeService(
            channels=FakeChannelsResource([{"id": "UC123", "snippet": {"title": "My Channel"}}])
        )
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        channel = client.get_authenticated_channel()
        assert channel.channel_id == "UC123"
        assert channel.channel_title == "My Channel"

    def test_no_channel_found_raises(self) -> None:
        service = FakeYouTubeService(channels=FakeChannelsResource([]))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        with pytest.raises(YouTubeClientError):
            client.get_authenticated_channel()

    def test_api_error_raises_client_error(self) -> None:
        service = FakeYouTubeService(channels=FakeChannelsResourceRaising(_http_error(403)))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        with pytest.raises(YouTubeClientError):
            client.get_authenticated_channel()


class TestVideoInsertRequestMapping:
    def test_request_body_mapping(self, tmp_path) -> None:
        captured: dict = {}
        insert_request = FakeInsertRequest(video_id="vid123")
        service = FakeYouTubeService(videos=FakeVideosResource(insert_request, captured))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        request = _request(_video_file(tmp_path), title="My Title", description="My Desc", tags=["x", "y"], privacy_status="private")
        video_id = client.insert_video(request)

        assert video_id == "vid123"
        assert captured["body"]["snippet"]["title"] == "My Title"
        assert captured["body"]["snippet"]["description"] == "My Desc"
        assert captured["body"]["snippet"]["tags"] == ["x", "y"]
        assert captured["body"]["status"]["privacyStatus"] == "private"

    def test_missing_video_file_raises(self) -> None:
        service = FakeYouTubeService(videos=FakeVideosResource(FakeInsertRequest("v"), {}))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        request = _request("/does/not/exist.mp4")
        with pytest.raises(YouTubeClientError):
            client.insert_video(request)

    def test_scheduled_publish_at_included_when_set(self, tmp_path) -> None:
        captured: dict = {}
        insert_request = FakeInsertRequest(video_id="vid123")
        service = FakeYouTubeService(videos=FakeVideosResource(insert_request, captured))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        scheduled = datetime(2027, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        request = _request(_video_file(tmp_path), privacy_status="private", scheduled_publish_at=scheduled)
        client.insert_video(request)

        assert captured["body"]["status"]["publishAt"] == "2027-01-01T09:00:00Z"


class TestTransientRetry:
    def test_retries_on_retryable_status_then_succeeds(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.tools.youtube_client.time.sleep", lambda _: None)
        insert_request = FakeInsertRequest(video_id="vid123", fail_times=2, error_status=503)
        service = FakeYouTubeService(videos=FakeVideosResource(insert_request, {}))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        video_id = client.insert_video(_request(_video_file(tmp_path)))
        assert video_id == "vid123"
        assert insert_request.calls == 3  # 2 failures + 1 success

    def test_gives_up_after_max_retries(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.tools.youtube_client.time.sleep", lambda _: None)
        insert_request = FakeInsertRequest(video_id="vid123", fail_times=999, error_status=503)
        service = FakeYouTubeService(videos=FakeVideosResource(insert_request, {}))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        with pytest.raises(YouTubeClientError):
            client.insert_video(_request(_video_file(tmp_path)))


class TestPermanentErrorNoRetry:
    def test_auth_permission_error_not_retried(self, tmp_path, monkeypatch) -> None:
        sleep_calls = []
        monkeypatch.setattr("src.tools.youtube_client.time.sleep", lambda s: sleep_calls.append(s))
        insert_request = FakeInsertRequest(video_id="vid123", permanent_error=_http_error(403))
        service = FakeYouTubeService(videos=FakeVideosResource(insert_request, {}))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        with pytest.raises(YouTubeClientError):
            client.insert_video(_request(_video_file(tmp_path)))
        assert insert_request.calls == 1  # no retry attempted at all
        assert sleep_calls == []


class TestThumbnailSet:
    def test_thumbnail_set_request(self, tmp_path) -> None:
        captured: dict = {}
        thumb_path = tmp_path / "thumb.jpg"
        thumb_path.write_bytes(b"FAKE JPEG BYTES")
        service = FakeYouTubeService(thumbnails=FakeThumbnailsResource(captured))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)

        client.set_thumbnail("vid123", str(thumb_path))
        assert captured["video_id"] == "vid123"

    def test_missing_thumbnail_file_raises(self) -> None:
        service = FakeYouTubeService(thumbnails=FakeThumbnailsResource({}))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        with pytest.raises(YouTubeClientError):
            client.set_thumbnail("vid123", "/does/not/exist.jpg")

    def test_api_failure_raises(self, tmp_path) -> None:
        thumb_path = tmp_path / "thumb.jpg"
        thumb_path.write_bytes(b"FAKE JPEG BYTES")
        service = FakeYouTubeService(thumbnails=FakeThumbnailsResource({}, error=_http_error(500)))
        client = GoogleYouTubeClient(credentials=None, youtube_service=service)
        with pytest.raises(YouTubeClientError):
            client.set_thumbnail("vid123", str(thumb_path))


class TestMockYouTubeClient:
    def test_insert_records_request_and_returns_id(self, tmp_path) -> None:
        client = MockYouTubeClient()
        request = _request(_video_file(tmp_path))
        video_id = client.insert_video(request)
        assert video_id
        assert client.inserted_videos == [request]

    def test_channel_info(self) -> None:
        client = MockYouTubeClient(channel_id="c1", channel_title="Ch")
        channel = client.get_authenticated_channel()
        assert channel.channel_id == "c1"
        assert channel.channel_title == "Ch"

    def test_failure_flags(self, tmp_path) -> None:
        client = MockYouTubeClient(fail_channel=True, fail_upload=True, fail_thumbnail=True)
        with pytest.raises(YouTubeClientError):
            client.get_authenticated_channel()
        with pytest.raises(YouTubeClientError):
            client.insert_video(_request(_video_file(tmp_path)))
        with pytest.raises(YouTubeClientError):
            client.set_thumbnail("v1", "thumb.jpg")
