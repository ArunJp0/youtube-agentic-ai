# Standalone live demo/CLI for the YouTube Upload Agent.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope.
#
# Reuses the most recently produced final video and its paired metadata/
# thumbnail/provenance/compliance artifacts (see
# src.services.upload_artifact_discovery) - Research/Script/Voice/Visuals/
# Captions/BGM/Metadata/Thumbnail/Compliance are never re-run.
#
# Three distinct, deliberately separate actions - never combined into one
# default behavior, so nothing ever uploads by accident:
#   --dry-run       (default): discover + validate + report only, no OAuth,
#                    no upload.
#   --authenticate  : run/reuse OAuth and verify the authenticated channel
#                    only - no upload.
#   --execute       : perform the real upload (requires an explicit
#                    --privacy; defaults to 'private'). Never the default.
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from src.agents.youtube_upload_agent import YouTubeUploadAgent, YouTubeUploadAgentError
from src.config.settings import Settings
from src.models.youtube_upload import DEFAULT_PRIVACY_STATUS, YOUTUBE_UPLOAD_SCOPES
from src.services.upload_artifact_discovery import DiscoveredRunArtifacts, discover_latest_run_artifacts
from src.services.youtube_upload_validation import UploadValidationError, format_publish_at, validate_scheduling
from src.tools.youtube_client import GoogleYouTubeClient, YouTubeClientError
from src.tools.youtube_oauth import YouTubeAuthError, get_credentials

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _print_discovery(artifacts: "DiscoveredRunArtifacts | None") -> None:
    print("\n" + "=" * 60)
    print("ARTIFACT DISCOVERY")
    print("=" * 60)
    if artifacts is None:
        print("No final video found under output/video/ - run the pipeline demo first.")
        return

    print(f"Run ID:       {artifacts.run_id}")
    print(f"Topic:        {artifacts.topic}")
    print(f"Final video:  {artifacts.final_video_path}")

    if artifacts.metadata_result:
        print(f"Metadata:     {artifacts.metadata_json_path}")
        print(f"   Title:     {artifacts.metadata_result.title}")
    else:
        print("Metadata:     NOT FOUND")

    if artifacts.thumbnail_path:
        association = "exact (via provenance manifest)" if artifacts.thumbnail_exact_match else "best-effort (latest file by modification time)"
        print(f"Thumbnail:    {artifacts.thumbnail_path} [{association}]")
    else:
        print("Thumbnail:    NOT FOUND")

    if artifacts.compliance_result:
        c = artifacts.compliance_result
        print(f"Compliance:   {c.publish_decision} (risk: {c.risk_level}, warnings: {len(c.warnings)}, blockers: {len(c.blockers)})")
    else:
        print("Compliance:   NOT FOUND - no persisted compliance result exists for this run")

    if artifacts.warnings:
        print(f"\nDiscovery warnings ({len(artifacts.warnings)}):")
        for warning in artifacts.warnings:
            print(f"   - {warning}")


def _parse_schedule_arg(value: str | None) -> datetime | None:
    """Parse ``--schedule``'s ISO 8601 string into a timezone-aware
    datetime, or exit cleanly with a clear message - never a raw
    traceback, and never a silently-assumed timezone. Only actual parsing
    (naive-vs-aware) happens here; the real validation rules (future,
    privacy='private') live centrally in validate_scheduling and are never
    duplicated here."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        print(f"Error: --schedule value '{value}' is not a valid ISO 8601 timestamp.")
        sys.exit(1)
    if parsed.tzinfo is None:
        print("Error: --schedule must be an offset-aware ISO 8601 timestamp (include a UTC offset, e.g. +00:00).")
        sys.exit(1)
    return parsed


def _print_schedule_preview(privacy: str, schedule: datetime | None) -> bool:
    """Print the requested local time / normalized UTC publish time and
    validate it via the same validate_scheduling used for a real upload.
    Returns True if the schedule (or lack of one) is valid."""
    if schedule is None:
        print("Scheduling:   none requested (immediate upload)")
        return True

    print(f"Requested schedule (as given): {schedule.isoformat()}")
    print(f"Requested schedule timezone:   {schedule.tzinfo}")
    print(f"Normalized UTC publish time:   {format_publish_at(schedule)}")
    try:
        validate_scheduling(privacy, schedule)
    except UploadValidationError as e:
        print(f"Scheduling validation: INVALID - {e}")
        return False
    print("Scheduling validation: OK (timezone-aware, in the future, privacy is 'private')")
    return True


def run_dry_run(
    topic: str, video_path: str | None = None, privacy: str = DEFAULT_PRIVACY_STATUS, schedule: datetime | None = None
) -> DiscoveredRunArtifacts | None:
    """Discover artifacts, report the compliance-gate outcome and planned
    upload/scheduling configuration, and perform NO OAuth and NO upload."""
    settings = Settings()
    artifacts = discover_latest_run_artifacts(topic, video_path=video_path)
    _print_discovery(artifacts)

    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY (no OAuth performed, no upload attempted)")
    print("=" * 60)
    if artifacts is None:
        print("Cannot proceed: no final video found.")
        return None

    compliance_ok = artifacts.compliance_result is not None and artifacts.compliance_result.publish_decision == "PASS"
    print(f"Compliance gate: {'WOULD ALLOW UPLOAD (PASS)' if compliance_ok else 'WOULD BLOCK UPLOAD'}")
    print(f"Planned privacy: {privacy}")
    _print_schedule_preview(privacy, schedule)
    print(
        f"OAuth client secret: {settings.youtube_oauth_client_secret_path} "
        f"(exists: {os.path.exists(settings.youtube_oauth_client_secret_path)})"
    )
    print(
        f"OAuth token:         {settings.youtube_oauth_token_path} "
        f"(exists: {os.path.exists(settings.youtube_oauth_token_path)})"
    )
    return artifacts


def run_authenticate() -> None:
    """Run (or reuse) the OAuth flow and report the verified authenticated
    channel - no upload performed."""
    settings = Settings()
    print("Requesting YouTube OAuth credentials (a browser window will open if no valid token is stored)...")
    try:
        credentials = get_credentials(
            settings.youtube_oauth_client_secret_path,
            settings.youtube_oauth_token_path,
            YOUTUBE_UPLOAD_SCOPES,
        )
    except YouTubeAuthError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    print(f"Token stored at: {settings.youtube_oauth_token_path}")

    client = GoogleYouTubeClient(credentials)
    try:
        channel = client.get_authenticated_channel()
    except YouTubeClientError as e:
        print(f"Channel verification failed: {e}")
        sys.exit(1)

    print("\nAuthenticated channel:")
    print(f"   Channel title: {channel.channel_title}")
    print(f"   Channel ID:    {channel.channel_id}")


def run_execute(topic: str, privacy: str, schedule: datetime | None, force: bool, video_path: str | None = None) -> None:
    settings = Settings()
    artifacts = discover_latest_run_artifacts(topic, video_path=video_path)
    _print_discovery(artifacts)
    if artifacts is None:
        sys.exit(1)

    if not _print_schedule_preview(privacy, schedule):
        sys.exit(1)

    try:
        credentials = get_credentials(
            settings.youtube_oauth_client_secret_path,
            settings.youtube_oauth_token_path,
            YOUTUBE_UPLOAD_SCOPES,
        )
    except YouTubeAuthError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    client = GoogleYouTubeClient(credentials)
    agent = YouTubeUploadAgent(client)
    try:
        result = agent.publish(artifacts, privacy_status=privacy, scheduled_publish_at=schedule, force=force)
    except YouTubeUploadAgentError as e:
        print(f"{e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("UPLOAD RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Status:  {result.status}")
    if result.video_id:
        print(f"Video ID:  {result.video_id}")
        print(f"Video URL: {result.video_url}")
    if result.channel_title:
        print(f"Channel:   {result.channel_title} ({result.channel_id})")
    print(f"Privacy:   {result.privacy_status}")
    if result.scheduled_publish_at:
        print(f"Scheduled publish (UTC): {format_publish_at(result.scheduled_publish_at)}")
    print(f"Thumbnail set: {result.thumbnail_set}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    if result.error:
        print(f"Error: {result.error}")

    if not result.success:
        sys.exit(1)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone YouTube Upload Agent demo")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC)
    parser.add_argument("--dry-run", action="store_true", help="Discover/validate only (default behavior)")
    parser.add_argument("--authenticate", action="store_true", help="Run/verify OAuth + channel identity only")
    parser.add_argument("--execute", action="store_true", help="Perform the real upload - never the default")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"], default=DEFAULT_PRIVACY_STATUS)
    parser.add_argument("--schedule", default=None, help="ISO 8601 offset-aware timestamp, e.g. 2026-09-15T09:00:00+00:00")
    parser.add_argument("--force", action="store_true", help="Upload again even if this run was already published")
    parser.add_argument("--video-path", default=None, help="Explicit video path override (default: latest final video)")
    return parser.parse_args(argv)


def main() -> None:
    _ensure_utf8_stdout()
    args = _parse_args(sys.argv[1:])

    if args.execute:
        schedule = _parse_schedule_arg(args.schedule)
        run_execute(args.topic, args.privacy, schedule, args.force, video_path=args.video_path)
    elif args.authenticate:
        run_authenticate()
    else:
        schedule = _parse_schedule_arg(args.schedule)
        run_dry_run(args.topic, video_path=args.video_path, privacy=args.privacy, schedule=schedule)


if __name__ == "__main__":
    main()
