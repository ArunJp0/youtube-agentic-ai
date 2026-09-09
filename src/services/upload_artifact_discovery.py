# Standalone artifact discovery for the YouTube Upload Agent - pairs a
# completed run's final video with its matching metadata, thumbnail,
# provenance, and (if persisted) compliance result, preferring EXACT
# run/video association over "newest arbitrary file" wherever a real
# association actually exists on disk today.
#
# What's exact vs. best-effort, and why:
#   - final video: the video the caller asked for (explicit path) or the
#     most recently produced one - this IS the run being discovered.
#   - run_id: derived deterministically from the video's own filename via
#     the same original_base_name helper every other stage already uses -
#     exact by construction.
#   - provenance manifest / thumbnail / compliance record: looked up BY
#     that exact run_id (ProvenanceManifestStore/ComplianceRecordStore key
#     their files by run_id) - exact, never a filename guess.
#   - metadata: MetadataAgent's JSON artifact is named after the generated
#     title/hook, not the run_id (no exact on-disk link exists anywhere in
#     this project today - a genuine pre-existing gap, not introduced
#     here). This module picks the metadata JSON whose own recorded
#     ``topic`` field matches the run's topic (from the provenance
#     manifest, or the topic argument) with the most recent modification
#     time among matches - a verified, non-exact association - and adds an
#     explicit warning if no topic-verified match was found, so a caller
#     never mistakes an unverified guess for a confirmed pairing.
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from src.agents.metadata_agent import DEFAULT_METADATA_OUTPUT_DIR
from src.agents.thumbnail_agent import DEFAULT_THUMBNAIL_OUTPUT_DIR
from src.models.compliance import ComplianceResult
from src.models.metadata import MetadataResult
from src.services.artifact_discovery import find_latest_file, find_latest_final_video
from src.services.compliance_record_store import ComplianceRecordStore, ComplianceRecordStoreError
from src.services.provenance_store import ProvenanceManifestStore, ProvenanceStoreError
from src.services.script_context_reconstruction import original_base_name
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR


@dataclass
class DiscoveredRunArtifacts:
    """Everything the Upload Agent needs, paired to one specific run."""

    run_id: str
    topic: str
    final_video_path: str
    metadata_result: Optional[MetadataResult] = None
    metadata_json_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    thumbnail_exact_match: bool = False
    compliance_result: Optional[ComplianceResult] = None
    warnings: List[str] = field(default_factory=list)


def _load_metadata_result(metadata_json_path: str) -> Optional[MetadataResult]:
    try:
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # The written artifact intentionally omits a few in-memory-only fields
    # (success, llm_provider/model) - a JSON file only ever exists if
    # generation succeeded, so those are filled in here rather than
    # re-derived. Mirrors compliance_demo.py's identical reconstruction.
    return MetadataResult(success=True, output_path=metadata_json_path, **data)


def _find_topic_matching_metadata(metadata_output_dir: str, topic: str) -> tuple[Optional[str], Optional[MetadataResult]]:
    """Among all metadata JSON files, return the most-recently-modified one
    whose own recorded ``topic`` field equals ``topic`` exactly - a
    verified association, not a filename guess."""
    if not os.path.isdir(metadata_output_dir):
        return None, None

    candidates = []
    for path in glob.glob(os.path.join(metadata_output_dir, "*.json")):
        result = _load_metadata_result(path)
        if result is not None and (result.topic or "").strip() == topic.strip():
            candidates.append((path, result))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: os.path.getmtime(item[0]), reverse=True)
    return candidates[0]


def discover_latest_run_artifacts(
    topic: str,
    video_path: Optional[str] = None,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    metadata_output_dir: str = DEFAULT_METADATA_OUTPUT_DIR,
    thumbnail_output_dir: str = DEFAULT_THUMBNAIL_OUTPUT_DIR,
    provenance_manifest_store: Optional[ProvenanceManifestStore] = None,
    compliance_record_store: Optional[ComplianceRecordStore] = None,
) -> Optional[DiscoveredRunArtifacts]:
    """Discover one run's paired artifacts for ``topic``.

    Never combines a video from one run with a provenance/compliance
    record from a different run - both are looked up by the exact run_id
    derived from the selected video's own filename.

    Returns:
        DiscoveredRunArtifacts, or ``None`` if no final video exists at all.
    """
    video_path = video_path or find_latest_final_video(video_output_dir)
    if not video_path:
        return None

    run_id = original_base_name(video_path)
    warnings: List[str] = []

    provenance_manifest_store = provenance_manifest_store or ProvenanceManifestStore()
    try:
        manifest = provenance_manifest_store.find_for_video(video_path)
    except ProvenanceStoreError as e:
        manifest = None
        warnings.append(f"Provenance manifest for run '{run_id}' is corrupt/unreadable: {e}")
    if manifest is None:
        warnings.append(f"No provenance manifest found for run '{run_id}' - visual/thumbnail/BGM provenance unavailable")

    thumbnail_path: Optional[str] = None
    thumbnail_exact_match = False
    if manifest is not None and manifest.thumbnail is not None and manifest.thumbnail.output_path:
        thumbnail_path = manifest.thumbnail.output_path
        thumbnail_exact_match = True
    else:
        fallback_thumbnail = find_latest_file(thumbnail_output_dir, "*.jpg")
        if fallback_thumbnail:
            thumbnail_path = fallback_thumbnail
            warnings.append(
                f"No exact thumbnail association for run '{run_id}' (no provenance manifest) - "
                f"using the most recently modified thumbnail file instead: {fallback_thumbnail}"
            )

    metadata_json_path, metadata_result = _find_topic_matching_metadata(metadata_output_dir, topic)
    if metadata_json_path is None:
        warnings.append(f"No metadata JSON found whose recorded topic matches '{topic}' - metadata unavailable")

    compliance_record_store = compliance_record_store or ComplianceRecordStore()
    try:
        compliance_result = compliance_record_store.read(run_id)
    except ComplianceRecordStoreError as e:
        compliance_result = None
        warnings.append(f"Compliance record for run '{run_id}' is corrupt/unreadable: {e}")
    if compliance_result is None:
        warnings.append(
            f"No persisted compliance result found for run '{run_id}' - upload cannot proceed "
            "without a confirmed PASS (see YouTubeUploadAgent's compliance gate)"
        )

    return DiscoveredRunArtifacts(
        run_id=run_id,
        topic=topic,
        final_video_path=video_path,
        metadata_result=metadata_result,
        metadata_json_path=metadata_json_path,
        thumbnail_path=thumbnail_path,
        thumbnail_exact_match=thumbnail_exact_match,
        compliance_result=compliance_result,
        warnings=warnings,
    )
