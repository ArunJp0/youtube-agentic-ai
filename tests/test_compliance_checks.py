# Tests for deterministic Copyright/Compliance checks
# (src/services/compliance_checks.py). No LLM/network involved.
from __future__ import annotations

import pytest
from PIL import Image

from src.models.metadata import MetadataResult
from src.models.music import BGMTrack
from src.models.provenance import BGMProvenance, ThumbnailProvenance, VisualAssetProvenance
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


def _visual_asset(provider="pexels", source_url="https://pexels.com/x", section_index=0) -> VisualAssetProvenance:
    return VisualAssetProvenance(
        section_index=section_index, section_heading=f"Section {section_index}", provider=provider, source_url=source_url
    )


def _thumbnail_provenance(**overrides) -> ThumbnailProvenance:
    defaults = dict(provider="pexels", provider_asset_id="123", source_url="https://pexels.com/x", attribution="Author")
    defaults.update(overrides)
    return ThumbnailProvenance(**defaults)


def _catalog_track(**overrides) -> BGMTrack:
    defaults = dict(
        track_id="calm-music",
        file_path="tracks/calm.mp3",
        title="Calm Music",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        attribution_required=False,
        attribution_text=None,
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


def _manifest_bgm(**overrides) -> BGMProvenance:
    defaults = dict(
        track_id="calm-music",
        title="Calm Music",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        attribution_required=False,
        attribution_text=None,
    )
    defaults.update(overrides)
    return BGMProvenance(**defaults)


class TestCheckFinalVideo:
    def test_existing_nonempty_video_passes(self, tmp_path) -> None:
        path = tmp_path / "video.mp4"
        path.write_bytes(b"FAKE MP4")
        result = check_final_video(str(path))
        assert result.status == "ok"

    def test_missing_path_blocks(self) -> None:
        result = check_final_video(None)
        assert result.status == "blocker"

    def test_nonexistent_file_blocks(self, tmp_path) -> None:
        result = check_final_video(str(tmp_path / "nope.mp4"))
        assert result.status == "blocker"

    def test_empty_file_blocks(self, tmp_path) -> None:
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        result = check_final_video(str(path))
        assert result.status == "blocker"


class TestCheckMetadata:
    def test_successful_metadata_passes(self) -> None:
        result = check_metadata(MetadataResult(success=True, title="T", description="D"))
        assert result.status == "ok"

    def test_missing_metadata_blocks(self) -> None:
        assert check_metadata(None).status == "blocker"

    def test_failed_metadata_blocks(self) -> None:
        result = check_metadata(MetadataResult(success=False, error="LLM outage"))
        assert result.status == "blocker"

    def test_empty_title_blocks(self) -> None:
        result = check_metadata(MetadataResult(success=True, title="", description="D"))
        assert result.status == "blocker"

    def test_empty_description_blocks(self) -> None:
        result = check_metadata(MetadataResult(success=True, title="T", description=""))
        assert result.status == "blocker"


class TestCheckThumbnail:
    def test_valid_1280x720_image_passes(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)
        result = check_thumbnail(str(path))
        assert result.status == "ok"

    def test_missing_path_blocks(self) -> None:
        assert check_thumbnail(None).status == "blocker"

    def test_wrong_dimensions_blocks(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        Image.new("RGB", (640, 480), (10, 20, 30)).save(path)
        result = check_thumbnail(str(path))
        assert result.status == "blocker"

    def test_corrupt_file_blocks(self, tmp_path) -> None:
        path = tmp_path / "thumb.jpg"
        path.write_bytes(b"not an image")
        result = check_thumbnail(str(path))
        assert result.status == "blocker"


class TestCheckContentTopicConsistency:
    def test_matching_topics_ok(self) -> None:
        metadata = MetadataResult(success=True, topic="Why do humans dream?", title="T", description="D")
        thumbnail = ThumbnailResult(success=True, topic="Why do humans dream?")
        result = check_content_topic_consistency("Why do humans dream?", metadata, thumbnail)
        assert result.status == "ok"

    def test_mismatched_metadata_topic_warns(self) -> None:
        metadata = MetadataResult(success=True, topic="Something else", title="T", description="D")
        result = check_content_topic_consistency("Why do humans dream?", metadata, None)
        assert result.status == "warning"

    def test_none_inputs_ok(self) -> None:
        result = check_content_topic_consistency("Why do humans dream?", None, None)
        assert result.status == "ok"


class TestCheckVisualProvenance:
    def test_none_visual_assets_is_unavailable(self) -> None:
        result = check_visual_provenance(None)
        assert result.status == "unavailable"

    def test_empty_visual_assets_is_unavailable(self) -> None:
        result = check_visual_provenance([])
        assert result.status == "unavailable"

    def test_all_pexels_assets_with_source_url_ok(self) -> None:
        result = check_visual_provenance([_visual_asset(provider="pexels", source_url="https://pexels.com/1")])
        assert result.status == "ok"

    def test_mock_provider_ok(self) -> None:
        result = check_visual_provenance([_visual_asset(provider="mock", source_url=None)])
        assert result.status == "ok"

    def test_unknown_provider_blocks(self) -> None:
        result = check_visual_provenance([_visual_asset(provider="some-random-scraper", source_url="https://example.com")])
        assert result.status == "blocker"

    def test_pexels_asset_missing_source_url_blocks(self) -> None:
        result = check_visual_provenance([_visual_asset(provider="pexels", source_url=None)])
        assert result.status == "blocker"

    def test_incomplete_asset_with_missing_provider_blocks(self) -> None:
        """A recorded entry with no provider at all is incomplete provenance
        - the same 'unrecognized provider' rule catches it, not a crash."""
        result = check_visual_provenance([_visual_asset(provider=None, source_url=None)])
        assert result.status == "blocker"

    def test_multiple_assets_one_bad_still_blocks(self) -> None:
        result = check_visual_provenance(
            [
                _visual_asset(provider="pexels", source_url="https://pexels.com/1", section_index=0),
                _visual_asset(provider="pexels", source_url=None, section_index=1),
            ]
        )
        assert result.status == "blocker"


class TestCheckThumbnailProvenance:
    def test_none_is_unavailable(self) -> None:
        assert check_thumbnail_provenance(None).status == "unavailable"

    def test_valid_pexels_provenance_ok(self) -> None:
        result = check_thumbnail_provenance(_thumbnail_provenance())
        assert result.status == "ok"

    def test_mock_provider_ok(self) -> None:
        result = check_thumbnail_provenance(_thumbnail_provenance(provider="mock", source_url=None))
        assert result.status == "ok"

    def test_unknown_provider_blocks(self) -> None:
        result = check_thumbnail_provenance(_thumbnail_provenance(provider="random-scraper"))
        assert result.status == "blocker"

    def test_pexels_missing_source_url_blocks(self) -> None:
        result = check_thumbnail_provenance(_thumbnail_provenance(provider="pexels", source_url=None))
        assert result.status == "blocker"


class TestCheckBgmProvenance:
    def test_none_manifest_bgm_is_unavailable(self) -> None:
        check, attributions, attribution_required = check_bgm_provenance(None, [_catalog_track()])
        assert check.status == "unavailable"
        assert attributions == []
        assert attribution_required is False

    def test_track_in_catalog_no_attribution_required(self) -> None:
        manifest_bgm = _manifest_bgm()
        check, attributions, attribution_required = check_bgm_provenance(manifest_bgm, [_catalog_track()])
        assert check.status == "ok"
        assert attributions == []
        assert attribution_required is False

    def test_track_missing_from_catalog_blocks(self) -> None:
        manifest_bgm = _manifest_bgm(track_id="not-in-catalog")
        check, attributions, attribution_required = check_bgm_provenance(manifest_bgm, [_catalog_track()])
        assert check.status == "blocker"
        assert attributions == []

    def test_missing_catalog_entirely_blocks(self) -> None:
        manifest_bgm = _manifest_bgm()
        check, attributions, attribution_required = check_bgm_provenance(manifest_bgm, [])
        assert check.status == "blocker"

    def test_attribution_required_with_text_available_ok(self) -> None:
        catalog_track = _catalog_track(attribution_required=True, attribution_text="Music by Example Artist")
        manifest_bgm = _manifest_bgm(attribution_required=True, attribution_text="Music by Example Artist")
        check, attributions, attribution_required = check_bgm_provenance(manifest_bgm, [catalog_track])
        assert check.status == "ok"
        assert attribution_required is True
        assert len(attributions) == 1
        assert attributions[0].attribution_text == "Music by Example Artist"
        assert attributions[0].asset_type == "bgm"

    def test_attribution_required_with_text_missing_blocks(self) -> None:
        catalog_track = _catalog_track(attribution_required=True, attribution_text=None)
        manifest_bgm = _manifest_bgm(attribution_required=True, attribution_text=None)
        check, attributions, attribution_required = check_bgm_provenance(manifest_bgm, [catalog_track])
        assert check.status == "blocker"
        assert attributions == []
        assert attribution_required is True

    def test_catalog_entry_is_trusted_over_manifest_snapshot(self) -> None:
        """Even if the manifest's recorded snapshot claims no attribution
        was required at run time, the CURRENT catalog entry is authoritative."""
        stale_manifest_snapshot = _manifest_bgm(attribution_required=False, attribution_text=None)
        current_catalog_entry = _catalog_track(attribution_required=True, attribution_text="Credit required")
        check, attributions, attribution_required = check_bgm_provenance(stale_manifest_snapshot, [current_catalog_entry])
        assert attribution_required is True
        assert attributions[0].attribution_text == "Credit required"

    def test_manifest_catalog_disagreement_reported_as_warning(self) -> None:
        """The manifest and the current catalog disagreeing (but both still
        resolve to a usable attribution) is a WARNING, not a silent pass and
        not a hard BLOCK - the current catalog value still wins."""
        stale_manifest_snapshot = _manifest_bgm(attribution_required=False, attribution_text=None)
        current_catalog_entry = _catalog_track(attribution_required=True, attribution_text="Credit required")
        check, _, _ = check_bgm_provenance(stale_manifest_snapshot, [current_catalog_entry])
        assert check.status == "warning"
        assert "disagreement" in check.detail.lower()

    def test_no_disagreement_when_manifest_matches_catalog(self) -> None:
        manifest_bgm = _manifest_bgm(attribution_required=True, attribution_text="Credit required")
        catalog_track = _catalog_track(attribution_required=True, attribution_text="Credit required")
        check, _, _ = check_bgm_provenance(manifest_bgm, [catalog_track])
        assert check.status == "ok"
        assert "disagreement" not in check.detail.lower()
