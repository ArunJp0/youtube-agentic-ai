# Standalone live demo for the Copyright / Compliance Agent.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope.
#
# Reuses the most recently produced final video, metadata JSON, and
# thumbnail image, and reconstructs narration context from the video's own
# existing .srt transcript (via the existing shared
# src/services/script_context_reconstruction.py module - no duplicated
# logic), exactly like src/bgm_demo.py / src/metadata_demo.py /
# src/thumbnail_demo.py. Research/Script/Voice/Visual Media/Visual QC/Video
# Assembly/Captions/BGM/Metadata/Thumbnail are never re-run.
#
# Visual/thumbnail/BGM provenance is recovered from a persisted
# ProvenanceManifest (output/provenance/<run_id>.json - see
# src.services.provenance_store), if the video's own run wrote one. Real
# pipeline runs from before that persistence milestone (or any run whose
# manifest write failed) are legacy artifacts with no manifest - this demo
# honestly reports that provenance as unavailable rather than guessing or
# inferring it from filenames, and ComplianceAgent treats it as a warning
# (contributing to REVIEW), never silently as clean.
from __future__ import annotations

import json
import sys
from typing import Optional

from src.agents.compliance_agent import ComplianceAgent, ComplianceAgentError
from src.agents.thumbnail_agent import DEFAULT_THUMBNAIL_OUTPUT_DIR
from src.config.providers import ProviderConfigError, get_llm_provider
from src.config.settings import Settings
from src.models.compliance import ComplianceResult
from src.models.metadata import MetadataResult
from src.models.provenance import ProvenanceManifest
from src.agents.metadata_agent import DEFAULT_METADATA_OUTPUT_DIR
from src.services.artifact_discovery import find_latest_file, find_latest_final_video
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR
from src.services.provenance_store import ProvenanceManifestStore, ProvenanceStoreError
from src.services.script_context_reconstruction import (
    build_context_from_srt,
    build_topic_only_context,
    find_matching_srt,
)
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR
from src.tools.music_catalog_provider import LocalMusicCatalogProvider

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _load_metadata_result(metadata_json_path: Optional[str]) -> Optional[MetadataResult]:
    """Reconstruct a MetadataResult from the JSON artifact MetadataAgent
    already wrote to disk. The artifact intentionally omits a few
    in-memory-only fields (success, llm_provider/model) - a JSON file only
    ever exists if generation succeeded, so those are filled in here rather
    than re-derived; this is a standalone-demo reconstruction, never used
    by real pipeline integration (which has the real MetadataResult in
    PipelineState already)."""
    if not metadata_json_path:
        return None
    try:
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return MetadataResult(success=True, output_path=metadata_json_path, **data)


def _load_provenance_manifest(video_path: str) -> Optional[ProvenanceManifest]:
    """Locate the ProvenanceManifest matching ``video_path``'s own run.

    Returns None either when this video predates provenance persistence (a
    legacy artifact - valid, never fabricated) or when a manifest file
    exists but is corrupt/unreadable - both are honestly reported to the
    caller as "no usable provenance", never raised out of this demo, and a
    corrupt-file note is printed separately so the two cases stay
    distinguishable in the report.
    """
    try:
        return ProvenanceManifestStore().find_for_video(video_path)
    except ProvenanceStoreError as e:
        print(f"      WARNING: {e}")
        return None


def run_compliance_demo(topic: str, video_path: Optional[str] = None) -> ComplianceResult:
    """Run a compliance review for the most recently generated final video
    (or an explicit ``video_path``), reusing its matching metadata JSON and
    thumbnail image, and reconstructing narration context from its existing
    .srt transcript when available.

    Synchronous end to end: ComplianceAgent only calls the synchronous
    LLMProvider.generate_text plus local file I/O - there is no async work
    in this demo.

    Args:
        topic: Overall video topic (used for the semantic review and, when
            no matching .srt exists, as the only script context available)
        video_path: Final video to review; auto-discovered under
            output/video/ if omitted

    Returns:
        ComplianceResult describing the outcome

    Raises:
        ComplianceAgentError: If no source video is available (configuration-
            level, not a review failure)
    """
    print("[1/3] Locating the latest final video, metadata, and thumbnail...")
    video_path = video_path or find_latest_final_video(DEFAULT_VIDEO_OUTPUT_DIR)
    if not video_path:
        raise ComplianceAgentError(
            f"No existing assembled/captioned/mixed MP4 found under {DEFAULT_VIDEO_OUTPUT_DIR}/. "
            "Run the pipeline demo (python -m src.pipeline_demo) at least once first, "
            "or pass an explicit video path."
        )
    print(f"      Video: {video_path}")

    metadata_json_path = find_latest_file(DEFAULT_METADATA_OUTPUT_DIR, "*.json")
    metadata_result = _load_metadata_result(metadata_json_path)
    print(f"      Metadata JSON: {metadata_json_path or '(none found)'}")

    thumbnail_path = find_latest_file(DEFAULT_THUMBNAIL_OUTPUT_DIR, "*.jpg")
    print(f"      Thumbnail: {thumbnail_path or '(none found)'}")

    provenance_manifest = _load_provenance_manifest(video_path)
    if provenance_manifest is not None:
        print(f"      Provenance manifest: output/provenance/{provenance_manifest.run_id}.json")
    else:
        print("      No provenance manifest found for this run - a legacy artifact predating provenance persistence")

    print("\n[2/3] Reconstructing narration context...")
    srt_path = find_matching_srt(video_path, DEFAULT_SUBTITLE_OUTPUT_DIR)
    if srt_path:
        print(f"      Reusing existing narration context from: {srt_path}")
        script_result = build_context_from_srt(topic, srt_path, video_path)
    else:
        print("      No matching .srt transcript found - using topic-only context (Research/Script are NOT re-run)")
        script_result = build_topic_only_context(topic)

    print("\n[3/3] Running deterministic checks and semantic compliance review...")
    settings = Settings()
    llm_provider = get_llm_provider(settings)
    agent = ComplianceAgent(music_catalog_provider=LocalMusicCatalogProvider(), llm_provider=llm_provider)
    return agent.review_compliance(
        topic,
        final_video_path=video_path,
        metadata_result=metadata_result,
        thumbnail_path=thumbnail_path,
        script=script_result,
        thumbnail_result=None,
        provenance_manifest=provenance_manifest,
    )


def print_compliance_result(result: ComplianceResult) -> None:
    """Pretty print a ComplianceResult."""
    print("\n" + "=" * 60)
    print("COPYRIGHT / COMPLIANCE RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error}")
        return

    print(f"\nOverall decision: {result.publish_decision}")
    print(f"Risk level:       {result.risk_level}")
    print(f"\n{result.disclaimer}")

    print("\nDeterministic checks:")
    for check in result.checks:
        print(f"   [{check.status.upper():>10s}] {check.check_name}: {check.detail}")

    print(f"\nAttribution required: {result.attribution_required}")
    if result.required_attributions:
        print("Required attributions:")
        for attribution in result.required_attributions:
            print(
                f"   - ({attribution.asset_type}) {attribution.asset_id} [{attribution.source}]: "
                f"{attribution.attribution_text}"
            )

    if result.semantic_review:
        review = result.semantic_review
        print(f"\nSemantic review performed: {review.performed}")
        if review.performed:
            print(f"Summary: {review.summary or '(no summary)'}")
            if review.findings:
                print("Findings:")
                for finding in review.findings:
                    print(f"   - [{finding.severity}] {finding.category}: {finding.description}")
            else:
                print("Findings: none")
        else:
            print(f"Fallback reason: {review.fallback_reason}")
        print(f"LLM provider: {review.llm_provider} | model: {review.llm_model} | fallback used: {review.used_fallback_model}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   - {warning}")

    if result.blockers:
        print(f"\nBlockers ({len(result.blockers)}):")
        for blocker in result.blockers:
            print(f"   - {blocker}")

    print(f"\nProvenance summary: {result.provenance_summary}")
    print(f"Artifacts inspected: {result.artifacts_inspected}")


def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = run_compliance_demo(topic)
    except (ProviderConfigError, ComplianceAgentError) as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_compliance_result(result)
    if result.publish_decision == "BLOCK":
        sys.exit(1)


if __name__ == "__main__":
    main()
