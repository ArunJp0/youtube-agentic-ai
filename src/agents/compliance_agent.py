# Copyright / Compliance Agent: orchestrates the two-part compliance
# architecture - deterministic provenance/licensing/artifact checks
# (src/services/compliance_checks.py, the PRIMARY safety layer) and one
# bounded advisory semantic review (ComplianceReviewer, at most one LLM
# call) - into a single typed ComplianceResult, then applies the
# centralized decision rules (src/services/compliance_rules.py) to reach a
# PASS / REVIEW / BLOCK publish decision.
#
# This agent is a pre-publishing SAFETY GATE, not a legal authority - it
# NEVER claims a video is legally or copyright "safe"; see
# src.models.compliance.DISCLAIMER. It reuses whatever real artifacts/
# results are already available (final video, MetadataResult, thumbnail,
# and - when available - a persisted ProvenanceManifest recovering exact
# visual/thumbnail/BGM asset provenance) and never reruns Research/Script/
# Video/Metadata/Thumbnail/BGM itself to obtain them.
#
# Visual/thumbnail/BGM provenance is read exclusively from a
# ProvenanceManifest (see src.models.provenance/src.services.
# provenance_store) - never from raw VisualResult/AudioMixResult, which are
# never persisted to disk and so are never actually available to a
# standalone caller. This agent does not discover the manifest itself
# (that's the caller's job, e.g. src/compliance_demo.py via
# ProvenanceManifestStore.find_for_video) - it only reads whatever manifest
# it's handed, keeping collection/persistence/compliance-reading cleanly
# separated.
from __future__ import annotations

from typing import List, Optional

from src.agents.compliance_reviewer import ComplianceReviewer
from src.llm.provider import LLMProvider
from src.models.compliance import ComplianceResult, DISCLAIMER, ProvenanceCheckResult, SemanticReviewResult
from src.models.metadata import MetadataResult
from src.models.provenance import ProvenanceManifest
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailResult
from src.services.compliance_checks import (
    check_bgm_provenance,
    check_content_topic_consistency,
    check_final_video,
    check_metadata,
    check_thumbnail,
    check_thumbnail_provenance,
    check_visual_provenance,
)
from src.services.compliance_rules import decide_publish_status
from src.tools.music_catalog_provider import MusicCatalogProvider, MusicCatalogProviderError


class ComplianceAgentError(Exception):
    """Raised only for configuration/programmer errors (e.g. a missing topic).

    Deterministic-check failures and semantic-review failures are NEVER
    raised - they are captured as blockers/warnings and an unperformed
    SemanticReviewResult respectively, so callers always get a structured
    ComplianceResult back.
    """


class ComplianceAgent:
    """Produces a pre-publishing ComplianceResult for a video from its
    already-produced real artifacts, using deterministic provenance/
    licensing checks as the primary safety layer plus at most one advisory
    semantic LLM review call."""

    def __init__(
        self,
        music_catalog_provider: MusicCatalogProvider,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        """Initialize the Compliance Agent.

        Args:
            music_catalog_provider: Source of the approved local BGM catalog
                (the same abstraction AudioMixingService already uses) -
                the trusted, CURRENT source of truth for BGM licensing/
                attribution, even when a provenance manifest carries its own
                (possibly stale) snapshot of those fields
            llm_provider: Optional LLMProvider for the single advisory
                semantic review call. If None, semantic review is treated
                as unperformed (never a fabricated PASS) - see
                ``src.services.compliance_rules.decide_publish_status``.
        """
        self.music_catalog_provider = music_catalog_provider
        self.reviewer = ComplianceReviewer(llm_provider) if llm_provider is not None else None

    def review_compliance(
        self,
        topic: str,
        final_video_path: Optional[str],
        metadata_result: Optional[MetadataResult],
        thumbnail_path: Optional[str],
        script: Optional[ScriptResult] = None,
        thumbnail_result: Optional[ThumbnailResult] = None,
        provenance_manifest: Optional[ProvenanceManifest] = None,
    ) -> ComplianceResult:
        """Review a video's publishing package for known compliance risks.

        Never raises for check/review failures - those are captured in the
        returned ComplianceResult. Only raises ComplianceAgentError for
        configuration/programmer errors.

        Args:
            topic: Overall video topic
            final_video_path: Path to the final video file to publish
            metadata_result: Already-generated MetadataResult (never
                regenerated here)
            thumbnail_path: Path to the final thumbnail image file
            script: Structured script, if available - used only as extra
                semantic-review context
            thumbnail_result: Already-generated ThumbnailResult, if
                available - used only as extra semantic-review context
                (hook text/visual concept), never for provenance
            provenance_manifest: Already-persisted ProvenanceManifest for
                this exact run, if one exists - the sole source of visual/
                thumbnail/BGM provenance for this review. A legacy artifact
                predating provenance persistence will have none; that is a
                valid, honestly-reported "unavailable" state, never
                fabricated or inferred from filenames.

        Returns:
            Structured ComplianceResult describing the outcome
        """
        if not topic:
            raise ComplianceAgentError("topic is required")

        checks: List[ProvenanceCheckResult] = []
        blockers: List[str] = []
        warnings: List[str] = []
        artifacts_inspected: List[str] = []

        video_check = check_final_video(final_video_path)
        checks.append(video_check)
        if final_video_path:
            artifacts_inspected.append(f"final video: {final_video_path}")
        self._collect(video_check, blockers, warnings)

        metadata_check = check_metadata(metadata_result)
        checks.append(metadata_check)
        if metadata_result is not None and metadata_result.output_path:
            artifacts_inspected.append(f"metadata JSON: {metadata_result.output_path}")
        self._collect(metadata_check, blockers, warnings)

        thumbnail_check = check_thumbnail(thumbnail_path)
        checks.append(thumbnail_check)
        if thumbnail_path:
            artifacts_inspected.append(f"thumbnail: {thumbnail_path}")
        self._collect(thumbnail_check, blockers, warnings)

        content_check = check_content_topic_consistency(topic, metadata_result, thumbnail_result)
        checks.append(content_check)
        self._collect(content_check, blockers, warnings)

        visual_assets = provenance_manifest.visual_assets if provenance_manifest is not None else None
        visual_check = check_visual_provenance(visual_assets)
        checks.append(visual_check)
        self._collect(visual_check, blockers, warnings)

        manifest_thumbnail = provenance_manifest.thumbnail if provenance_manifest is not None else None
        thumbnail_provenance_check = check_thumbnail_provenance(manifest_thumbnail)
        checks.append(thumbnail_provenance_check)
        self._collect(thumbnail_provenance_check, blockers, warnings)

        manifest_bgm = provenance_manifest.bgm if provenance_manifest is not None else None
        bgm_check, required_attributions, attribution_required = self._check_bgm(manifest_bgm)
        checks.append(bgm_check)
        self._collect(bgm_check, blockers, warnings)

        if provenance_manifest is not None:
            artifacts_inspected.append(f"provenance manifest: run_id={provenance_manifest.run_id}")

        semantic_review = self._run_semantic_review(topic, script, metadata_result, thumbnail_result)

        publish_decision, risk_level = decide_publish_status(blockers, warnings, semantic_review)

        return ComplianceResult(
            success=True,
            topic=topic,
            publish_decision=publish_decision,
            risk_level=risk_level,
            checks=checks,
            warnings=warnings,
            blockers=blockers,
            attribution_required=attribution_required,
            required_attributions=required_attributions,
            provenance_summary=self._build_provenance_summary(checks),
            semantic_review=semantic_review,
            llm_used=semantic_review.performed,
            llm_provider=semantic_review.llm_provider,
            llm_model=semantic_review.llm_model,
            used_fallback_model=semantic_review.used_fallback_model,
            artifacts_inspected=artifacts_inspected,
            disclaimer=DISCLAIMER,
        )

    # ---- helpers ------------------------------------------------------------

    def _check_bgm(self, manifest_bgm):
        if manifest_bgm is None:
            return check_bgm_provenance(None, None)

        try:
            catalog_tracks = self.music_catalog_provider.list_tracks()
        except MusicCatalogProviderError as e:
            detail = f"Could not load the approved BGM catalog to verify the recorded track: {e}"
            return (
                ProvenanceCheckResult(check_name="bgm_provenance", status="blocker", detail=detail),
                [],
                manifest_bgm.attribution_required,
            )

        return check_bgm_provenance(manifest_bgm, catalog_tracks)

    def _run_semantic_review(
        self,
        topic: str,
        script: Optional[ScriptResult],
        metadata_result: Optional[MetadataResult],
        thumbnail_result: Optional[ThumbnailResult],
    ) -> SemanticReviewResult:
        if self.reviewer is None:
            return SemanticReviewResult(performed=False, fallback_reason="No LLM provider configured for semantic review")
        return self.reviewer.review(topic, script, metadata_result, thumbnail_result)

    @staticmethod
    def _collect(check: ProvenanceCheckResult, blockers: List[str], warnings: List[str]) -> None:
        if check.status == "blocker":
            blockers.append(check.detail)
        elif check.status in ("warning", "unavailable"):
            warnings.append(check.detail)

    @staticmethod
    def _build_provenance_summary(checks: List[ProvenanceCheckResult]) -> str:
        return "; ".join(f"{c.check_name}: {c.status}" for c in checks)
