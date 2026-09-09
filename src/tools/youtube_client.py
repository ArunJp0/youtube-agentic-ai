# YouTube Data API v3 client abstraction. Mirrors every other provider
# abstraction in this project (MediaProvider, VoiceProvider, ...): the
# application (YouTubeUploadAgent) only ever depends on this interface,
# never on googleapiclient directly, so a mock can stand in for tests/dry
# runs and the real client is swappable without touching agent code.
#
# Deliberately split into separate insert_video/set_thumbnail calls (rather
# than one combined "publish" method) so a thumbnail failure can be
# reported without ever losing/hiding the already-created video_id - see
# YouTubeUploadAgent.
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from src.models.youtube_upload import UploadRequest, YouTubeChannelInfo
from src.services.youtube_upload_validation import format_publish_at

# Transient server-side statuses worth a bounded retry - never authentication
# (401/403), not-found, or malformed-request statuses, which retrying can
# never fix.
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_MAX_UPLOAD_RETRIES = 5
_RETRY_BACKOFF_SECONDS = 2.0


class YouTubeClientError(Exception):
    """Raised for a genuine, non-retryable (or retry-exhausted) YouTube Data
    API failure - authentication/permission errors, invalid metadata,
    quota exhaustion, missing files, or a permanent API validation error.
    Never raised for a transient error that a bounded retry already
    resolved.
    """


class YouTubeClient(ABC):
    """Abstract base class for YouTube Data API v3 access."""

    @abstractmethod
    def get_authenticated_channel(self) -> YouTubeChannelInfo:
        """Return the actual channel the current OAuth credentials are
        authorized for - never a manually typed/assumed value.

        Raises:
            YouTubeClientError: If the channel cannot be verified.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_video(self, request: UploadRequest) -> str:
        """Upload the video file and metadata, returning the new video_id.

        Raises:
            YouTubeClientError: If the upload fails (never partially
                returns a video_id on failure).
        """
        raise NotImplementedError

    @abstractmethod
    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> None:
        """Set the custom thumbnail for an already-uploaded video.

        Raises:
            YouTubeClientError: If setting the thumbnail fails - the caller
                is responsible for preserving ``video_id`` regardless (see
                YouTubeUploadAgent).
        """
        raise NotImplementedError


class GoogleYouTubeClient(YouTubeClient):
    """Real YouTube Data API v3 client, backed by Google's official
    googleapiclient library. Never re-implements OAuth itself - callers
    supply already-obtained credentials (see src.tools.youtube_oauth)."""

    def __init__(self, credentials: Any, youtube_service: Optional[Resource] = None) -> None:
        """Initialize the client.

        Args:
            credentials: Valid google.oauth2.credentials.Credentials
            youtube_service: Optional pre-built API resource (dependency
                injection point for tests) - if omitted, built for real via
                ``googleapiclient.discovery.build``.
        """
        self._youtube = youtube_service or build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def get_authenticated_channel(self) -> YouTubeChannelInfo:
        try:
            response = self._youtube.channels().list(part="snippet", mine=True).execute()
        except HttpError as e:
            raise YouTubeClientError(f"Failed to verify the authenticated YouTube channel: {e}") from e

        items = response.get("items") or []
        if not items:
            raise YouTubeClientError("No YouTube channel found for the authenticated account")

        channel = items[0]
        return YouTubeChannelInfo(channel_id=channel["id"], channel_title=channel["snippet"]["title"])

    def insert_video(self, request: UploadRequest) -> str:
        if not os.path.exists(request.video_path):
            raise YouTubeClientError(f"Video file not found: {request.video_path}")

        status_body = {"privacyStatus": request.privacy_status, "selfDeclaredMadeForKids": False}
        if request.scheduled_publish_at is not None:
            status_body["publishAt"] = format_publish_at(request.scheduled_publish_at)

        body = {
            "snippet": {
                "title": request.title,
                "description": request.description,
                "tags": request.tags,
                "categoryId": request.category_id,
            },
            "status": status_body,
        }

        media = MediaFileUpload(request.video_path, chunksize=-1, resumable=True, mimetype="video/*")
        api_request = self._youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        attempt = 0
        while response is None:
            try:
                _, response = api_request.next_chunk()
            except HttpError as e:
                status = getattr(getattr(e, "resp", None), "status", None)
                if status in _RETRYABLE_STATUS_CODES and attempt < _MAX_UPLOAD_RETRIES:
                    attempt += 1
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise YouTubeClientError(f"Video upload failed: {e}") from e

        return response["id"]

    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> None:
        if not os.path.exists(thumbnail_path):
            raise YouTubeClientError(f"Thumbnail file not found: {thumbnail_path}")
        try:
            self._youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        except HttpError as e:
            raise YouTubeClientError(f"Setting thumbnail failed: {e}") from e


class MockYouTubeClient(YouTubeClient):
    """In-memory test/dry-run double - no network, no googleapiclient
    involved. Records every call so tests can assert exactly what was
    requested."""

    def __init__(
        self,
        channel_id: str = "mock-channel-id",
        channel_title: str = "Mock Channel",
        fail_channel: bool = False,
        fail_upload: bool = False,
        fail_thumbnail: bool = False,
    ) -> None:
        self.channel_id = channel_id
        self.channel_title = channel_title
        self.fail_channel = fail_channel
        self.fail_upload = fail_upload
        self.fail_thumbnail = fail_thumbnail
        self.inserted_videos: List[UploadRequest] = []
        self.thumbnails_set: List[Tuple[str, str]] = []

    def get_authenticated_channel(self) -> YouTubeChannelInfo:
        if self.fail_channel:
            raise YouTubeClientError("simulated channel verification failure")
        return YouTubeChannelInfo(channel_id=self.channel_id, channel_title=self.channel_title)

    def insert_video(self, request: UploadRequest) -> str:
        if self.fail_upload:
            raise YouTubeClientError("simulated video upload failure")
        self.inserted_videos.append(request)
        return f"mock-video-{len(self.inserted_videos)}"

    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> None:
        if self.fail_thumbnail:
            raise YouTubeClientError("simulated thumbnail set failure")
        self.thumbnails_set.append((video_id, thumbnail_path))
