# YouTube Upload Agent: publishes one already-produced, already-reviewed
# run to YouTube. Deterministic orchestration only (no LLM call anywhere in
# this agent) - named "Agent" to match this milestone's own naming and
# because it is intended to eventually become a pipeline stage exactly like
# ComplianceAgent/ThumbnailAgent/MetadataAgent, not because it reasons.
#
# Never uploads without a confirmed Compliance PASS (checked deterministically
# against ComplianceResult.publish_decision, never a human-readable string
# or terminal log), never re-runs Research/Script/Voice/Visuals/Captions/
# BGM/Metadata/Thumbnail/Compliance to obtain its inputs (see
# src.services.upload_artifact_discovery), and never silently re-uploads an
# already-published run (see src.services.publishing_record_store).
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.models.youtube_upload import (
    DEFAULT_PRIVACY_STATUS,
    PrivacyStatus,
    PublishingRecord,
    UploadRequest,
    UploadResult,
    UploadStatus,
)
from src.services.publishing_record_store import PublishingRecordStore
from src.services.upload_artifact_discovery import DiscoveredRunArtifacts
from src.services.youtube_upload_validation import (
    UploadValidationError,
    format_publish_at,
    validate_scheduling,
    validate_thumbnail_file,
    validate_video_file,
)
from src.tools.youtube_client import YouTubeClient, YouTubeClientError

_VIDEO_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"


class YouTubeUploadAgentError(Exception):
    """Raised only for configuration/programmer errors (e.g. missing
    artifacts object). Auth/channel/upload/thumbnail failures are NEVER
    raised - they are captured in the returned UploadResult so callers
    always get a structured result back.
    """


class YouTubeUploadAgent:
    """Publishes one already-discovered run's artifacts to YouTube, gated
    strictly on a confirmed Compliance PASS."""

    def __init__(
        self,
        youtube_client: YouTubeClient,
        publishing_record_store: Optional[PublishingRecordStore] = None,
    ) -> None:
        """Initialize the Upload Agent.

        Args:
            youtube_client: YouTubeClient implementation (real or mock) -
                dependency injection, exactly like every other provider in
                this project.
            publishing_record_store: Idempotency store; defaults to the
                real ``output/publishing/`` location.
        """
        self.youtube_client = youtube_client
        self.publishing_record_store = publishing_record_store or PublishingRecordStore()

    def publish(
        self,
        artifacts: DiscoveredRunArtifacts,
        privacy_status: PrivacyStatus = DEFAULT_PRIVACY_STATUS,
        scheduled_publish_at: Optional[datetime] = None,
        force: bool = False,
    ) -> UploadResult:
        """Publish ``artifacts`` to YouTube.

        Never raises for auth/channel/validation/upload/thumbnail
        failures - those are captured in the returned UploadResult. Only
        raises YouTubeUploadAgentError for a missing ``artifacts`` object.

        Args:
            artifacts: Already-discovered, paired run artifacts (see
                ``src.services.upload_artifact_discovery``) - never
                reconstructed/rerun here.
            privacy_status: Requested privacy - defaults to 'private'.
            scheduled_publish_at: Optional timezone-aware future publish
                time; requires ``privacy_status='private'``.
            force: If True, uploads again even if this run already has a
                publishing record (normally refused - see the idempotency
                check below).

        Returns:
            Structured UploadResult describing the outcome.
        """
        if artifacts is None:
            raise YouTubeUploadAgentError("artifacts is required")

        if not force:
            existing = self.publishing_record_store.read(artifacts.run_id)
            if existing is not None:
                # Report the SAME status this run would have originally
                # received (completed / private_uploaded / scheduled) -
                # never a hardcoded "completed" that would misreport an
                # existing scheduled upload as already fully published.
                existing_had_thumbnail_request = artifacts.thumbnail_path is not None
                return UploadResult(
                    success=True,
                    status=self._derive_status(
                        wanted_thumbnail=existing_had_thumbnail_request,
                        thumbnail_set=existing.thumbnail_set,
                        is_scheduled=existing.scheduled_publish_at is not None,
                        privacy_status=existing.privacy_status,
                    ),
                    video_id=existing.video_id,
                    video_url=existing.video_url,
                    channel_id=existing.channel_id,
                    channel_title=existing.channel_title,
                    privacy_status=existing.privacy_status,
                    thumbnail_set=existing.thumbnail_set,
                    warnings=[
                        f"Run '{artifacts.run_id}' was already published as {existing.video_id} - "
                        "skipping re-upload (pass force=True to upload again anyway)"
                    ],
                )

        # Deterministic compliance gate - never a human-readable-string or
        # terminal-log check. A missing compliance result is treated
        # exactly like REVIEW/BLOCK: upload must not occur.
        compliance = artifacts.compliance_result
        if compliance is None or compliance.publish_decision != "PASS":
            reason = "no persisted compliance result is available for this run" if compliance is None else (
                f"the compliance decision for this run is {compliance.publish_decision}, not PASS"
            )
            return UploadResult(
                success=False,
                status="compliance_gate_failed",
                error=f"Upload blocked: {reason}",
            )

        if artifacts.metadata_result is None or not (artifacts.metadata_result.title or "").strip():
            return UploadResult(success=False, status="upload_failed", error="No metadata (title) available for this run")

        try:
            validate_video_file(artifacts.final_video_path)
            if artifacts.thumbnail_path:
                validate_thumbnail_file(artifacts.thumbnail_path)
            validate_scheduling(privacy_status, scheduled_publish_at)
        except UploadValidationError as e:
            return UploadResult(success=False, status="upload_failed", error=str(e))

        try:
            channel = self.youtube_client.get_authenticated_channel()
        except YouTubeClientError as e:
            return UploadResult(success=False, status="channel_verification_failed", error=str(e))

        request = UploadRequest(
            video_path=artifacts.final_video_path,
            title=artifacts.metadata_result.title,
            description=artifacts.metadata_result.description or "",
            tags=artifacts.metadata_result.tags,
            thumbnail_path=artifacts.thumbnail_path,
            privacy_status=privacy_status,
            scheduled_publish_at=scheduled_publish_at,
        )

        try:
            video_id = self.youtube_client.insert_video(request)
        except YouTubeClientError as e:
            return UploadResult(
                success=False,
                status="upload_failed",
                channel_id=channel.channel_id,
                channel_title=channel.channel_title,
                error=str(e),
            )

        # The video now genuinely exists on YouTube - video_id must never
        # be lost from here on, even if the thumbnail step below fails.
        video_url = _VIDEO_URL_TEMPLATE.format(video_id=video_id)
        warnings = []
        thumbnail_set = False
        if artifacts.thumbnail_path:
            try:
                self.youtube_client.set_thumbnail(video_id, artifacts.thumbnail_path)
                thumbnail_set = True
            except YouTubeClientError as e:
                warnings.append(f"Video uploaded successfully (video_id={video_id}), but setting the thumbnail failed: {e}")

        status = self._derive_status(
            wanted_thumbnail=artifacts.thumbnail_path is not None,
            thumbnail_set=thumbnail_set,
            is_scheduled=scheduled_publish_at is not None,
            privacy_status=privacy_status,
        )

        record = PublishingRecord(
            run_id=artifacts.run_id,
            video_id=video_id,
            video_url=video_url,
            channel_id=channel.channel_id,
            channel_title=channel.channel_title,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            privacy_status=privacy_status,
            thumbnail_set=thumbnail_set,
            # Stored in the same UTC/RFC 3339 form actually sent to the
            # YouTube API (format_publish_at) - never the original,
            # possibly non-UTC offset - so the persisted record is always
            # unambiguous and consistent with what YouTube itself received.
            scheduled_publish_at=format_publish_at(scheduled_publish_at) if scheduled_publish_at else None,
        )
        try:
            self.publishing_record_store.write(record)
        except OSError as e:
            warnings.append(f"Video uploaded successfully (video_id={video_id}), but the publishing record could not be saved: {e}")

        return UploadResult(
            success=True,
            status=status,
            video_id=video_id,
            video_url=video_url,
            channel_id=channel.channel_id,
            channel_title=channel.channel_title,
            privacy_status=privacy_status,
            scheduled_publish_at=scheduled_publish_at,
            thumbnail_set=thumbnail_set,
            warnings=warnings,
        )

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _derive_status(
        wanted_thumbnail: bool, thumbnail_set: bool, is_scheduled: bool, privacy_status: str
    ) -> UploadStatus:
        """The single place that decides which of the three "video exists
        on YouTube" outcomes (thumbnail_failed / scheduled / private_uploaded
        / completed) applies - reused for both a fresh upload and the
        idempotency short-circuit, so a rerun of an already-published run
        always reports the SAME status a first-time caller would have seen,
        never a hardcoded generic value.

        A missing/failed thumbnail always takes priority (the most
        actionable partial-failure signal) over scheduled/privacy status.
        """
        if wanted_thumbnail and not thumbnail_set:
            return "thumbnail_failed"
        if is_scheduled:
            return "scheduled"
        if privacy_status == "private":
            return "private_uploaded"
        return "completed"
