# Deterministic pre-flight validation for YouTube uploads - the primary
# safety layer that keeps a malformed request from ever reaching the real
# API. Nothing here calls the network.
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from src.services.thumbnail_validation import ThumbnailValidationError, validate_output_image

# The YouTube Data API's own documented thumbnails.set size limit.
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024


class UploadValidationError(Exception):
    """Raised for a request that must never reach the YouTube API at all -
    a missing/invalid file, or a scheduling rule violation."""


def validate_video_file(video_path: Optional[str]) -> None:
    if not video_path or not os.path.exists(video_path):
        raise UploadValidationError(f"Video file not found: {video_path}")
    if os.path.getsize(video_path) == 0:
        raise UploadValidationError(f"Video file is empty: {video_path}")


def validate_thumbnail_file(thumbnail_path: Optional[str]) -> None:
    if not thumbnail_path or not os.path.exists(thumbnail_path):
        raise UploadValidationError(f"Thumbnail file not found: {thumbnail_path}")
    try:
        validate_output_image(thumbnail_path)
    except ThumbnailValidationError as e:
        raise UploadValidationError(f"Thumbnail file is invalid: {e}") from e
    size = os.path.getsize(thumbnail_path)
    if size > MAX_THUMBNAIL_BYTES:
        raise UploadValidationError(
            f"Thumbnail file is {size} bytes, exceeding the YouTube Data API's {MAX_THUMBNAIL_BYTES}-byte limit"
        )


def validate_scheduling(
    privacy_status: str, scheduled_publish_at: Optional[datetime], now: Optional[datetime] = None
) -> None:
    """Enforce YouTube's own scheduling rules deterministically, before any
    API call: a scheduled video must stay private until its publish time,
    and that time must be a real, timezone-aware, future moment - never an
    ambiguous local timestamp silently assumed to mean something."""
    if scheduled_publish_at is None:
        return
    if scheduled_publish_at.tzinfo is None:
        raise UploadValidationError("scheduled_publish_at must be timezone-aware (offset-aware)")
    if privacy_status != "private":
        raise UploadValidationError("Scheduled publishing requires privacy_status='private' until the publish time")
    current = now or datetime.now(timezone.utc)
    if scheduled_publish_at <= current:
        raise UploadValidationError("scheduled_publish_at must be in the future")


def format_publish_at(scheduled_publish_at: datetime) -> str:
    """YouTube's required RFC 3339 UTC format for ``status.publishAt`` -
    the single source of truth for this formatting, reused by
    GoogleYouTubeClient so the request body and any test/log output can
    never drift apart."""
    return scheduled_publish_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
