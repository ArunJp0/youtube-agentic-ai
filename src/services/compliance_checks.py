# Deterministic compliance checks - the primary safety layer for the
# Copyright / Compliance Agent. Every function here is a pure, centrally
# testable rule: given already-produced artifacts/results, decide a
# ProvenanceCheckResult. Nothing here calls an LLM, downloads anything, or
# invents license/provenance information that isn't already recorded.
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from src.models.compliance import ProvenanceCheckResult, RequiredAttribution
from src.models.metadata import MetadataResult
from src.models.music import BGMTrack
from src.models.provenance import BGMProvenance, ThumbnailProvenance, VisualAssetProvenance
from src.models.thumbnail import ThumbnailResult
from src.services.thumbnail_validation import ThumbnailValidationError, validate_output_image

# The only visual providers this architecture currently knows how to reason
# about the licensing of - anything else is unidentifiable provenance, not
# assumed safe.
KNOWN_VISUAL_PROVIDERS = {"pexels", "mock"}


def check_final_video(video_path: Optional[str]) -> ProvenanceCheckResult:
    """Required publishing artifact: the final video must exist and be non-empty."""
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return ProvenanceCheckResult(
            check_name="final_video_present",
            status="blocker",
            detail=f"Final video artifact is missing or empty: {video_path}",
        )
    return ProvenanceCheckResult(
        check_name="final_video_present", status="ok", detail=f"Final video present: {video_path}"
    )


def check_metadata(metadata_result: Optional[MetadataResult]) -> ProvenanceCheckResult:
    """Required publishing artifact: metadata must exist, have succeeded, and
    have a non-empty title/description."""
    if metadata_result is None or not metadata_result.success:
        return ProvenanceCheckResult(
            check_name="metadata_present", status="blocker", detail="Metadata artifact is missing or generation failed"
        )
    if not (metadata_result.title or "").strip() or not (metadata_result.description or "").strip():
        return ProvenanceCheckResult(
            check_name="metadata_present", status="blocker", detail="Metadata title or description is empty"
        )
    return ProvenanceCheckResult(check_name="metadata_present", status="ok", detail="Metadata title/description present")


def check_thumbnail(thumbnail_path: Optional[str]) -> ProvenanceCheckResult:
    """Required publishing artifact: the thumbnail must exist and be a valid
    1280x720 image (reuses the exact same validator the Thumbnail Agent
    itself uses - no duplicated image-validation logic)."""
    if not thumbnail_path:
        return ProvenanceCheckResult(check_name="thumbnail_present", status="blocker", detail="Thumbnail artifact is missing")
    try:
        width, height = validate_output_image(thumbnail_path)
    except ThumbnailValidationError as e:
        return ProvenanceCheckResult(check_name="thumbnail_present", status="blocker", detail=f"Thumbnail artifact is invalid: {e}")
    return ProvenanceCheckResult(
        check_name="thumbnail_present", status="ok", detail=f"Thumbnail present and valid ({width}x{height})"
    )


def check_content_topic_consistency(
    topic: str, metadata_result: Optional[MetadataResult], thumbnail_result: Optional[ThumbnailResult]
) -> ProvenanceCheckResult:
    """Deterministic cross-artifact consistency check: the topic each
    artifact was actually generated for should match the topic being
    reviewed. A mismatch is a warning (plausible legitimate rewording), not
    a blocker - this is not a semantic content review."""
    mismatches: List[str] = []
    requested = (topic or "").strip()
    if metadata_result is not None and (metadata_result.topic or "").strip():
        if metadata_result.topic.strip() != requested:
            mismatches.append(f"metadata topic '{metadata_result.topic}' differs from the requested topic '{requested}'")
    if thumbnail_result is not None and (thumbnail_result.topic or "").strip():
        if thumbnail_result.topic.strip() != requested:
            mismatches.append(f"thumbnail topic '{thumbnail_result.topic}' differs from the requested topic '{requested}'")
    if mismatches:
        return ProvenanceCheckResult(check_name="content_topic_consistency", status="warning", detail="; ".join(mismatches))
    return ProvenanceCheckResult(
        check_name="content_topic_consistency", status="ok", detail="Topic is consistent across inspected artifacts"
    )


def check_visual_provenance(visual_assets: Optional[List[VisualAssetProvenance]]) -> ProvenanceCheckResult:
    """Every recorded visual asset must carry identifiable provider/source
    information. A missing or empty provenance list (no persisted
    ProvenanceManifest - the pre-manifest legacy-artifact case, or a
    manifest recorded with no visual entries) is reported as 'unavailable',
    never silently treated as clean."""
    if not visual_assets:
        return ProvenanceCheckResult(
            check_name="visual_provenance",
            status="unavailable",
            detail="Visual asset provenance not available for review (no persisted provenance manifest, or it has no recorded visual assets)",
        )

    problems: List[str] = []
    for asset in visual_assets:
        provider = (asset.provider or "").strip().lower()
        if provider not in KNOWN_VISUAL_PROVIDERS:
            problems.append(
                f"section {asset.section_index} asset has an unrecognized/missing provider ('{asset.provider or 'unknown'}')"
            )
        elif provider == "pexels" and not (asset.source_url or "").strip():
            problems.append(f"section {asset.section_index} Pexels asset is missing its source URL")

    if problems:
        return ProvenanceCheckResult(check_name="visual_provenance", status="blocker", detail="; ".join(problems))
    return ProvenanceCheckResult(
        check_name="visual_provenance",
        status="ok",
        detail=f"{len(visual_assets)} visual asset(s) have identifiable provenance",
    )


def check_thumbnail_provenance(thumbnail_provenance: Optional[ThumbnailProvenance]) -> ProvenanceCheckResult:
    """The thumbnail's own source image must carry identifiable provider/
    source information, exactly like a visual asset. A missing record (no
    persisted ProvenanceManifest) is 'unavailable', never silently clean."""
    if thumbnail_provenance is None:
        return ProvenanceCheckResult(
            check_name="thumbnail_provenance",
            status="unavailable",
            detail="Thumbnail source-asset provenance not available for review (no persisted provenance manifest)",
        )

    provider = (thumbnail_provenance.provider or "").strip().lower()
    if provider not in KNOWN_VISUAL_PROVIDERS:
        return ProvenanceCheckResult(
            check_name="thumbnail_provenance",
            status="blocker",
            detail=f"Thumbnail source asset has an unrecognized/missing provider ('{thumbnail_provenance.provider or 'unknown'}')",
        )
    if provider == "pexels" and not (thumbnail_provenance.source_url or "").strip():
        return ProvenanceCheckResult(
            check_name="thumbnail_provenance", status="blocker", detail="Thumbnail Pexels source asset is missing its source URL"
        )
    return ProvenanceCheckResult(
        check_name="thumbnail_provenance", status="ok", detail="Thumbnail source asset has identifiable provenance"
    )


def check_bgm_provenance(
    manifest_bgm: Optional[BGMProvenance], catalog_tracks: Optional[List[BGMTrack]]
) -> Tuple[ProvenanceCheckResult, List[RequiredAttribution], bool]:
    """Verify the recorded BGM track against the CURRENT approved catalog -
    the catalog is always the source of truth for licensing/attribution,
    never the manifest's own snapshot of it (recorded at run time, and may
    have since drifted from the catalog - see the disagreement check below).

    A missing record (no persisted ProvenanceManifest - the pre-manifest
    legacy-artifact case) is reported as 'unavailable', never silently
    treated as clean.

    Returns:
        (check_result, required_attributions, attribution_required)
    """
    if manifest_bgm is None:
        return (
            ProvenanceCheckResult(
                check_name="bgm_provenance",
                status="unavailable",
                detail="BGM selection not available for review (no persisted provenance manifest)",
            ),
            [],
            False,
        )

    catalog_by_id = {t.track_id: t for t in (catalog_tracks or [])}
    canonical = catalog_by_id.get(manifest_bgm.track_id)
    if canonical is None:
        return (
            ProvenanceCheckResult(
                check_name="bgm_provenance",
                status="blocker",
                detail=(
                    f"Recorded BGM track '{manifest_bgm.track_id}' ({manifest_bgm.title}) is not present "
                    "in the current approved catalog"
                ),
            ),
            [],
            manifest_bgm.attribution_required,
        )

    disagreements: List[str] = []
    if canonical.attribution_required != manifest_bgm.attribution_required:
        disagreements.append(
            f"manifest recorded attribution_required={manifest_bgm.attribution_required} at run time, "
            f"but the current catalog now says {canonical.attribution_required}"
        )
    if (canonical.attribution_text or "") != (manifest_bgm.attribution_text or ""):
        disagreements.append("manifest's recorded attribution text differs from the current catalog entry")
    disagreement_note = f" (NOTE - manifest/catalog disagreement: {'; '.join(disagreements)})" if disagreements else ""
    status_when_ok = "warning" if disagreements else "ok"

    if canonical.attribution_required:
        text = (canonical.attribution_text or "").strip()
        if not text:
            return (
                ProvenanceCheckResult(
                    check_name="bgm_provenance",
                    status="blocker",
                    detail=(
                        f"BGM track '{canonical.track_id}' requires attribution but no attribution text is "
                        f"recorded in the current catalog{disagreement_note}"
                    ),
                ),
                [],
                True,
            )
        required = [
            RequiredAttribution(
                asset_type="bgm", asset_id=canonical.track_id, source=canonical.source, attribution_text=text
            )
        ]
        return (
            ProvenanceCheckResult(
                check_name="bgm_provenance",
                status=status_when_ok,
                detail=f"BGM track '{canonical.track_id}' is approved and requires attribution: {text}{disagreement_note}",
            ),
            required,
            True,
        )

    return (
        ProvenanceCheckResult(
            check_name="bgm_provenance",
            status=status_when_ok,
            detail=(
                f"BGM track '{canonical.track_id}' is approved; no attribution required "
                f"({canonical.license_type}){disagreement_note}"
            ),
        ),
        [],
        False,
    )
