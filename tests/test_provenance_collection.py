# Tests for provenance collection (src/services/provenance_collection.py) -
# building/persisting a ProvenanceManifest from a completed PipelineState.
# Local filesystem only (tmp_path) - no network involved.
from __future__ import annotations

import os

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.music import AudioMixResult, BGMTrack
from src.models.thumbnail import ThumbnailResult, ThumbnailSourceAsset
from src.services.provenance_collection import build_manifest_from_pipeline_state, persist_provenance_if_completed
from src.services.provenance_store import ProvenanceManifestStore
from src.workflows.pipeline_graph import PipelineState


def _bgm_track(**overrides) -> BGMTrack:
    defaults = dict(
        track_id="calm-music",
        file_path="assets/bgm/tracks/calm.mp3",
        title="Calm Music",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        attribution_required=False,
        attribution_text=None,
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


def _visual_asset(section_index=0, success=True, provider="pexels") -> MediaAsset:
    return MediaAsset(
        provider=provider,
        provider_asset_id="123",
        source_url="https://pexels.com/video/123",
        attribution="Some Author",
        local_file_path=f"output/media/section-{section_index:02d}-abcd.mp4",
        search_query="q",
        section_index=section_index,
        success=success,
    )


def _completed_state(video_path="output/video/why-do-humans-dream-8baeee8d-captioned-bgm.mp4", **overrides) -> PipelineState:
    defaults = dict(
        topic="Why do humans dream?",
        status="completed",
        audio_mix_result=AudioMixResult(success=True, output_path=video_path, selected_track=_bgm_track()),
        qc_approved_visual_result=VisualResult(
            topic="t",
            provider="pexels",
            success=True,
            sections=[SectionMediaMapping(section_index=0, section_heading="Intro", assets=[_visual_asset()])],
        ),
        thumbnail_result=ThumbnailResult(
            success=True,
            output_path="output/thumbnails/why-do-we-dream.jpg",
            selected_asset=ThumbnailSourceAsset(
                provider="pexels", provider_asset_id="456", source_url="https://pexels.com/photo/456", attribution="Someone"
            ),
        ),
    )
    defaults.update(overrides)
    return PipelineState(**defaults)


class TestBuildManifestFromPipelineState:
    def test_completed_run_produces_manifest(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        assert manifest is not None
        assert manifest.topic == "Why do humans dream?"
        assert manifest.final_video_path == "output/video/why-do-humans-dream-8baeee8d-captioned-bgm.mp4"

    def test_run_id_derived_from_video_base_name(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        assert manifest.run_id == "why-do-humans-dream-8baeee8d"

    def test_not_completed_status_returns_none(self) -> None:
        state = _completed_state(status="failed")
        assert build_manifest_from_pipeline_state(state) is None

    def test_missing_audio_mix_result_returns_none(self) -> None:
        state = _completed_state(audio_mix_result=None)
        assert build_manifest_from_pipeline_state(state) is None

    def test_failed_audio_mix_result_returns_none(self) -> None:
        state = _completed_state(
            audio_mix_result=AudioMixResult(success=False, error="mix failed")
        )
        assert build_manifest_from_pipeline_state(state) is None

    def test_missing_output_path_returns_none(self) -> None:
        state = _completed_state(
            audio_mix_result=AudioMixResult(success=True, output_path=None, selected_track=_bgm_track())
        )
        assert build_manifest_from_pipeline_state(state) is None

    def test_visual_asset_provenance_round_trip(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        assert len(manifest.visual_assets) == 1
        asset = manifest.visual_assets[0]
        assert asset.provider == "pexels"
        assert asset.provider_asset_id == "123"
        assert asset.source_url == "https://pexels.com/video/123"
        assert asset.attribution == "Some Author"
        assert asset.section_index == 0

    def test_failed_visual_assets_excluded(self) -> None:
        state = _completed_state(
            qc_approved_visual_result=VisualResult(
                topic="t",
                provider="pexels",
                success=True,
                sections=[
                    SectionMediaMapping(
                        section_index=0, section_heading="Intro", assets=[_visual_asset(success=False)]
                    )
                ],
            )
        )
        manifest = build_manifest_from_pipeline_state(state)
        assert manifest.visual_assets == []

    def test_missing_qc_approved_visual_result_yields_empty_visual_assets(self) -> None:
        state = _completed_state(qc_approved_visual_result=None)
        manifest = build_manifest_from_pipeline_state(state)
        assert manifest.visual_assets == []

    def test_thumbnail_provenance_round_trip(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        assert manifest.thumbnail is not None
        assert manifest.thumbnail.provider == "pexels"
        assert manifest.thumbnail.provider_asset_id == "456"
        assert manifest.thumbnail.output_path == "output/thumbnails/why-do-we-dream.jpg"

    def test_missing_thumbnail_result_yields_none_thumbnail(self) -> None:
        state = _completed_state(thumbnail_result=None)
        manifest = build_manifest_from_pipeline_state(state)
        assert manifest.thumbnail is None

    def test_bgm_track_id_round_trip(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        assert manifest.bgm is not None
        assert manifest.bgm.track_id == "calm-music"
        assert manifest.bgm.title == "Calm Music"

    def test_attribution_required_track_round_trip(self) -> None:
        state = _completed_state(
            audio_mix_result=AudioMixResult(
                success=True,
                output_path="output/video/why-do-humans-dream-8baeee8d-captioned-bgm.mp4",
                selected_track=_bgm_track(attribution_required=True, attribution_text="Credit: Example"),
            )
        )
        manifest = build_manifest_from_pipeline_state(state)
        assert manifest.bgm.attribution_required is True
        assert manifest.bgm.attribution_text == "Credit: Example"

    def test_no_secrets_or_prompts_in_built_manifest(self) -> None:
        manifest = build_manifest_from_pipeline_state(_completed_state())
        dumped_json = manifest.model_dump_json()
        for forbidden in ("api_key", "GEMINI_API_KEY", "prompt"):
            assert forbidden not in dumped_json


class TestPersistProvenanceIfCompleted:
    def test_writes_manifest_for_completed_run(self, tmp_path) -> None:
        path = persist_provenance_if_completed(_completed_state(), output_dir=str(tmp_path))
        assert path is not None
        assert os.path.exists(path)

    def test_returns_none_for_incomplete_run(self, tmp_path) -> None:
        state = _completed_state(status="failed")
        path = persist_provenance_if_completed(state, output_dir=str(tmp_path))
        assert path is None
        assert os.listdir(tmp_path) == []

    def test_written_manifest_is_discoverable_via_store(self, tmp_path) -> None:
        state = _completed_state()
        persist_provenance_if_completed(state, output_dir=str(tmp_path))
        found = ProvenanceManifestStore(output_dir=str(tmp_path)).find_for_video(state.audio_mix_result.output_path)
        assert found is not None
        assert found.topic == "Why do humans dream?"

    def test_write_failure_does_not_raise(self, tmp_path) -> None:
        """A filesystem error persisting provenance must never raise out of
        this function - it's best-effort supplementary data, not a required
        publishing artifact, and must never retroactively fail an already-
        completed pipeline run."""
        blocking_file = tmp_path / "blocked"
        blocking_file.write_text("not a directory")
        # output_dir points at a path that's actually a file - os.makedirs
        # will fail with an OSError subclass.
        path = persist_provenance_if_completed(_completed_state(), output_dir=str(blocking_file))
        assert path is None
