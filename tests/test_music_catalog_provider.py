# Tests for MusicCatalogProvider implementations. LocalMusicCatalogProvider
# reads real local JSON/files (no network); MockMusicCatalogProvider is
# in-memory only.
from __future__ import annotations

import json
import os

import pytest

from src.models.music import BGMTrack
from src.tools.music_catalog_provider import (
    LocalMusicCatalogProvider,
    MockMusicCatalogProvider,
    MusicCatalogProviderError,
)


def _entry(**overrides) -> dict:
    defaults = dict(
        track_id="calm-piano-01",
        file_path="tracks/calm-piano-01.mp3",
        title="Calm Piano",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        mood_tags=["calm", "subtle"],
        energy_level="low",
        instrumental=True,
    )
    defaults.update(overrides)
    return defaults


class TestLocalMusicCatalogProviderMissingOrInvalid:
    def test_missing_catalog_file_raises(self, tmp_path) -> None:
        provider = LocalMusicCatalogProvider(catalog_path=str(tmp_path / "does-not-exist.json"))
        with pytest.raises(MusicCatalogProviderError, match="not found"):
            provider.list_tracks()

    def test_non_json_file_raises(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text("not valid json", encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))
        with pytest.raises(MusicCatalogProviderError):
            provider.list_tracks()

    def test_non_array_json_raises(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))
        with pytest.raises(MusicCatalogProviderError, match="JSON array"):
            provider.list_tracks()

    def test_invalid_entry_raises_with_details(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps([_entry(license_type="")]), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))
        with pytest.raises(MusicCatalogProviderError, match="Invalid BGM catalog entry"):
            provider.list_tracks()

    def test_missing_required_field_raises(self, tmp_path) -> None:
        entry = _entry()
        del entry["source"]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps([entry]), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))
        with pytest.raises(MusicCatalogProviderError):
            provider.list_tracks()


class TestLocalMusicCatalogProviderValidCatalog:
    def test_empty_catalog_returns_empty_list(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text("[]", encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))
        assert provider.list_tracks() == []

    def test_valid_entries_parsed_as_bgm_tracks(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps([_entry(), _entry(track_id="track-2")]), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))

        tracks = provider.list_tracks()

        assert len(tracks) == 2
        assert all(isinstance(t, BGMTrack) for t in tracks)
        assert {t.track_id for t in tracks} == {"calm-piano-01", "track-2"}

    def test_relative_file_path_resolved_against_catalog_directory(self, tmp_path) -> None:
        nested_dir = tmp_path / "bgm"
        nested_dir.mkdir()
        catalog_path = nested_dir / "catalog.json"
        catalog_path.write_text(json.dumps([_entry(file_path="tracks/calm-piano-01.mp3")]), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))

        [track] = provider.list_tracks()

        assert os.path.isabs(track.file_path)
        assert track.file_path.replace("\\", "/").endswith("bgm/tracks/calm-piano-01.mp3")

    def test_absolute_file_path_left_unchanged(self, tmp_path) -> None:
        absolute_audio_path = str(tmp_path / "somewhere-else" / "track.mp3")
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps([_entry(file_path=absolute_audio_path)]), encoding="utf-8")
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))

        [track] = provider.list_tracks()

        assert track.file_path == absolute_audio_path

    def test_attribution_metadata_preserved(self, tmp_path) -> None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(
            json.dumps([_entry(attribution_required=True, attribution_text="Music by Example Artist")]),
            encoding="utf-8",
        )
        provider = LocalMusicCatalogProvider(catalog_path=str(catalog_path))

        [track] = provider.list_tracks()

        assert track.attribution_required is True
        assert track.attribution_text == "Music by Example Artist"


class TestMockMusicCatalogProvider:
    def test_default_is_empty(self) -> None:
        assert MockMusicCatalogProvider().list_tracks() == []

    def test_returns_configured_tracks(self) -> None:
        track = BGMTrack(
            track_id="t1", file_path="t1.mp3", title="T1", source="Test", license_type="test_license"
        )
        provider = MockMusicCatalogProvider(tracks=[track])
        assert provider.list_tracks() == [track]

    def test_returns_a_copy_not_the_same_list(self) -> None:
        track = BGMTrack(
            track_id="t1", file_path="t1.mp3", title="T1", source="Test", license_type="test_license"
        )
        provider = MockMusicCatalogProvider(tracks=[track])
        result = provider.list_tracks()
        result.append(track)
        assert len(provider.list_tracks()) == 1
