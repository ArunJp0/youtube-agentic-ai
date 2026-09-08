# Provenance collection - the ONLY place that reads real pipeline results
# (VisualResult/AudioMixResult/ThumbnailResult) to build a ProvenanceManifest.
# Kept separate from persistence (src/services/provenance_store.py) and from
# compliance reading (src/agents/compliance_agent.py): this module only ever
# turns an already-completed PipelineState into a manifest object - it never
# touches the filesystem itself, and it never runs for an incomplete/failed
# run (no fabricated/partial manifests).
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from src.models.provenance import BGMProvenance, ProvenanceManifest, ThumbnailProvenance, VisualAssetProvenance
from src.services.provenance_store import DEFAULT_PROVENANCE_OUTPUT_DIR, ProvenanceManifestStore
from src.services.script_context_reconstruction import original_base_name

if TYPE_CHECKING:
    from src.workflows.pipeline_graph import PipelineState


def build_manifest_from_pipeline_state(state: "PipelineState") -> Optional[ProvenanceManifest]:
    """Build a ProvenanceManifest from a PipelineState's real results, or
    ``None`` if the run's prerequisite stage (BGM/Audio Mixing) didn't
    actually succeed - never fabricates a manifest for a failed/partial run.

    Deliberately checks ``audio_mix_result.success`` directly rather than
    the pipeline's overall ``status`` field: the true prerequisite for
    having valid visual/BGM provenance to record is that BGM mixing itself
    produced the final video, independent of whether a later stage
    (Thumbnail, Compliance) has run yet or ultimately succeeds - this is
    what lets Thumbnail persist the manifest immediately on its own
    success, before the pipeline's final status is known, so Compliance can
    read the exact current run's provenance rather than a previous one.

    Only inspects fields already produced by successful stages:
    ``qc_approved_visual_result`` (the assets actually used in the final
    video, not the raw pre-QC selection), ``audio_mix_result`` (the final
    video path and selected BGM track), and ``thumbnail_result``.
    """
    if state.audio_mix_result is None or not state.audio_mix_result.success:
        return None
    final_video_path = state.audio_mix_result.output_path
    if not final_video_path:
        return None

    visual_assets = []
    if state.qc_approved_visual_result is not None:
        for section in state.qc_approved_visual_result.sections:
            for asset in section.assets:
                if not asset.success:
                    continue
                visual_assets.append(
                    VisualAssetProvenance(
                        section_index=section.section_index,
                        section_heading=section.section_heading,
                        provider=asset.provider,
                        provider_asset_id=asset.provider_asset_id,
                        source_url=asset.source_url,
                        attribution=asset.attribution,
                        local_file_path=asset.local_file_path,
                    )
                )

    thumbnail_provenance = None
    if state.thumbnail_result is not None and state.thumbnail_result.success and state.thumbnail_result.selected_asset:
        source_asset = state.thumbnail_result.selected_asset
        thumbnail_provenance = ThumbnailProvenance(
            provider=source_asset.provider,
            provider_asset_id=source_asset.provider_asset_id,
            source_url=source_asset.source_url,
            attribution=source_asset.attribution,
            output_path=state.thumbnail_result.output_path,
        )

    bgm_provenance = None
    selected_track = state.audio_mix_result.selected_track
    if selected_track is not None:
        bgm_provenance = BGMProvenance(
            track_id=selected_track.track_id,
            title=selected_track.title,
            source=selected_track.source,
            license_type=selected_track.license_type,
            attribution_required=selected_track.attribution_required,
            attribution_text=selected_track.attribution_text,
            local_file_path=selected_track.file_path,
        )

    return ProvenanceManifest(
        run_id=original_base_name(final_video_path),
        topic=state.topic,
        final_video_path=final_video_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        visual_assets=visual_assets,
        thumbnail=thumbnail_provenance,
        bgm=bgm_provenance,
    )


def persist_provenance_if_completed(
    state: "PipelineState", output_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR
) -> Optional[str]:
    """Build and atomically persist a ProvenanceManifest for a PipelineState
    whose BGM/Audio Mixing stage succeeded, returning the written path - or
    ``None`` if that prerequisite wasn't met, or if the write itself failed
    (a filesystem error here never retroactively fails an already-succeeded
    stage; it's best-effort supplementary provenance, not a required
    publishing artifact itself).

    This is the ONE place pipeline code calls to persist provenance - no
    individual agent writes its own provenance file.
    """
    manifest = build_manifest_from_pipeline_state(state)
    if manifest is None:
        return None
    try:
        return ProvenanceManifestStore(output_dir).write(manifest)
    except OSError:
        return None
