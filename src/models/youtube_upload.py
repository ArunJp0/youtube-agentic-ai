# YouTube Upload Agent data models.
#
# The uploader is a pre-publishing SAFETY-GATED action, not a bare API
# wrapper: it never uploads without a confirmed Compliance PASS (see
# src.agents.youtube_upload_agent), and it never silently loses a
# successfully-created YouTube video ID just because a later step (setting
# the thumbnail, scheduling) fails - see UploadStatus below.
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

PrivacyStatus = Literal["private", "unlisted", "public"]

DEFAULT_PRIVACY_STATUS: PrivacyStatus = "private"

# "Education" - this project's own stated focus (see CLAUDE.md: "educational
# and interesting knowledge content"). Callers may override per-request.
DEFAULT_CATEGORY_ID = "27"

# youtube.upload alone covers videos.insert and thumbnails.set. Verified
# empirically (real OAuth run) that YouTube's channels.list(mine=True) -
# required for the mandatory pre-upload channel-identity check - returns
# "insufficient authentication scopes" on upload-only credentials, so
# youtube.readonly is added as the narrowest additional scope that grants
# read access to channel data without adding any extra write permission
# beyond what youtube.upload already grants. Still never the broad
# read-write youtube scope.
YOUTUBE_UPLOAD_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

UploadStatus = Literal[
    "not_started",
    "auth_failed",
    "channel_verification_failed",
    "compliance_gate_failed",
    "upload_failed",
    "video_uploaded",
    "thumbnail_failed",
    "private_uploaded",
    "scheduled",
    "completed",
]


class UploadRequest(BaseModel):
    """Everything needed to publish one already-produced video to YouTube.

    Never constructed from a bare human-typed channel name - the
    authenticated OAuth channel (see YouTubeChannelInfo) is always the
    source of truth for *where* this uploads to.
    """

    video_path: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=100, description="YouTube title cap is 100 characters")
    description: str = Field(default="", max_length=5000, description="YouTube description cap is 5000 characters")
    tags: List[str] = Field(default_factory=list)
    thumbnail_path: Optional[str] = Field(default=None)
    privacy_status: PrivacyStatus = Field(default=DEFAULT_PRIVACY_STATUS)
    scheduled_publish_at: Optional[datetime] = Field(
        default=None, description="Must be an offset-aware (timezone-aware) datetime - never an ambiguous local time"
    )
    category_id: str = Field(default=DEFAULT_CATEGORY_ID)

    @field_validator("scheduled_publish_at")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError(
                "scheduled_publish_at must be timezone-aware (offset-aware) - "
                "an ambiguous local timestamp is never accepted"
            )
        return value


class YouTubeChannelInfo(BaseModel):
    """The actual authenticated channel, as confirmed by the OAuth-authorized
    account itself - never a manually typed/assumed value."""

    channel_id: str = Field(min_length=1)
    channel_title: str = Field(min_length=1)


class UploadResult(BaseModel):
    """Structured output of one publish attempt.

    ``video_id`` is preserved on every outcome from ``video_uploaded``
    onward, even if a later step (thumbnail, scheduling confirmation)
    fails - a thumbnail failure must never look like the video itself was
    lost.
    """

    success: bool = Field(description="Whether the video itself now exists on YouTube (thumbnail issues don't unset this)")
    status: UploadStatus
    video_id: Optional[str] = Field(default=None)
    video_url: Optional[str] = Field(default=None)
    channel_id: Optional[str] = Field(default=None)
    channel_title: Optional[str] = Field(default=None)
    privacy_status: Optional[str] = Field(default=None, description="The privacy state YouTube actually reports")
    scheduled_publish_at: Optional[datetime] = Field(default=None)
    thumbnail_set: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


class PublishingRecord(BaseModel):
    """Local, machine-readable record that a run was already published -
    the idempotency guard preventing an accidental duplicate upload of the
    same run."""

    schema_version: int = Field(default=1)
    run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    video_url: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    channel_title: str = Field(min_length=1)
    uploaded_at: str = Field(min_length=1, description="ISO 8601 UTC timestamp this record was written")
    privacy_status: str = Field(min_length=1)
    thumbnail_set: bool = Field(default=False)
    scheduled_publish_at: Optional[str] = Field(default=None)
