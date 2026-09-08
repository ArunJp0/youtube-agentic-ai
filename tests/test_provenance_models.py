# Tests for provenance manifest typed models (src/models/provenance.py).
from __future__ import annotations

from src.models.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    BGMProvenance,
    ProvenanceManifest,
    ThumbnailProvenance,
    VisualAssetProvenance,
)


class TestVisualAssetProvenance:
    def test_construction_with_all_fields(self) -> None:
        asset = VisualAssetProvenance(
            section_index=0,
            section_heading="Intro",
            provider="pexels",
            provider_asset_id="123",
            source_url="https://pexels.com/video/123",
            attribution="Some Author",
            local_file_path="output/media/section-01-abcd.mp4",
        )
        assert asset.provider == "pexels"

    def test_provider_optional_for_incomplete_records(self) -> None:
        asset = VisualAssetProvenance(section_index=0, section_heading="Intro")
        assert asset.provider is None


class TestThumbnailProvenance:
    def test_defaults(self) -> None:
        thumb = ThumbnailProvenance()
        assert thumb.provider is None
        assert thumb.output_path is None


class TestBGMProvenance:
    def test_construction(self) -> None:
        bgm = BGMProvenance(
            track_id="calm-music",
            title="Calm Music",
            source="YouTube Audio Library",
            license_type="youtube_audio_library_no_attribution",
            attribution_required=False,
        )
        assert bgm.track_id == "calm-music"
        assert bgm.attribution_text is None


class TestProvenanceManifest:
    def test_serialization_round_trip(self) -> None:
        manifest = ProvenanceManifest(
            run_id="why-do-humans-dream-8baeee8d",
            topic="Why do humans dream?",
            final_video_path="output/video/why-do-humans-dream-8baeee8d-captioned-bgm.mp4",
            created_at="2026-01-01T00:00:00+00:00",
            visual_assets=[VisualAssetProvenance(section_index=0, section_heading="Intro", provider="pexels")],
            thumbnail=ThumbnailProvenance(provider="pexels", source_url="https://pexels.com/x"),
            bgm=BGMProvenance(track_id="calm-music", title="Calm Music", source="s", license_type="l"),
        )
        dumped = manifest.model_dump()
        restored = ProvenanceManifest.model_validate(dumped)

        assert restored == manifest
        assert restored.run_id == manifest.run_id
        assert restored.visual_assets[0].provider == "pexels"
        assert restored.bgm.track_id == "calm-music"

    def test_schema_version_default(self) -> None:
        manifest = ProvenanceManifest(
            run_id="x", topic="t", final_video_path="v.mp4", created_at="2026-01-01T00:00:00+00:00"
        )
        assert manifest.schema_version == PROVENANCE_SCHEMA_VERSION

    def test_defaults_have_no_visual_thumbnail_or_bgm(self) -> None:
        manifest = ProvenanceManifest(
            run_id="x", topic="t", final_video_path="v.mp4", created_at="2026-01-01T00:00:00+00:00"
        )
        assert manifest.visual_assets == []
        assert manifest.thumbnail is None
        assert manifest.bgm is None

    def test_no_secret_or_prompt_fields_exist_on_model(self) -> None:
        """The manifest schema itself must never carry API keys, tokens, or
        raw LLM prompts - a structural guarantee, not just a convention."""
        field_names = set(ProvenanceManifest.model_fields.keys())
        forbidden_substrings = ("key", "token", "secret", "prompt", "credential")
        for name in field_names:
            lowered = name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), f"suspicious field name: {name}"
