# Tests for BGM/music data models (BGMTrack, MusicPlan, AudioMixResult).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.music import AudioMixResult, BGMTrack, MusicPlan


def _track(**overrides) -> BGMTrack:
    defaults = dict(
        track_id="calm-piano-01",
        file_path="tracks/calm-piano-01.mp3",
        title="Calm Piano",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


class TestBGMTrackValidation:
    def test_minimal_valid_track(self) -> None:
        track = _track()
        assert track.instrumental is True
        assert track.attribution_required is False
        assert track.energy_level == "medium"
        assert track.mood_tags == []

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            BGMTrack(track_id="x", file_path="x.mp3", title="X", source="X")  # missing license_type

    def test_empty_track_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _track(track_id="")

    def test_invalid_energy_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            _track(energy_level="extreme")

    def test_attribution_metadata_round_trips(self) -> None:
        track = _track(attribution_required=True, attribution_text="Music by Example Artist")
        assert track.attribution_required is True
        assert track.attribution_text == "Music by Example Artist"

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValidationError):
            _track(duration_seconds=-1.0)


class TestMusicPlanValidation:
    def test_minimal_valid_plan(self) -> None:
        plan = MusicPlan(topic="dreams", primary_mood="calm", used_semantic_planning=True)
        assert plan.energy_level == "medium"
        assert plan.avoid_styles == []
        assert plan.fallback_reason is None

    def test_missing_primary_mood_raises(self) -> None:
        with pytest.raises(ValidationError):
            MusicPlan(topic="dreams", primary_mood="", used_semantic_planning=True)

    def test_invalid_energy_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            MusicPlan(topic="dreams", primary_mood="calm", energy_level="extreme", used_semantic_planning=True)

    def test_fallback_plan_shape(self) -> None:
        plan = MusicPlan(
            topic="dreams",
            primary_mood="neutral",
            used_semantic_planning=False,
            fallback_reason="LLM outage",
        )
        assert plan.used_semantic_planning is False
        assert plan.fallback_reason == "LLM outage"


class TestAudioMixResultValidation:
    def test_minimal_failure_result(self) -> None:
        result = AudioMixResult(success=False, error="no track available")
        assert result.output_path is None
        assert result.selected_track is None
        assert result.warnings == []

    def test_success_result_carries_plan_and_track(self) -> None:
        track = _track()
        plan = MusicPlan(topic="dreams", primary_mood="calm", used_semantic_planning=True)
        result = AudioMixResult(
            success=True,
            output_path="output/video/x-bgm.mp4",
            source_video_path="output/video/x-captioned.mp4",
            music_plan=plan,
            selected_track=track,
            bgm_gain_db=-24.0,
            ducking_used=True,
            looped=False,
            source_duration_seconds=100.0,
            output_duration_seconds=100.1,
        )
        assert result.selected_track.track_id == "calm-piano-01"
        assert result.music_plan.primary_mood == "calm"

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValidationError):
            AudioMixResult(success=True, source_duration_seconds=-5.0)
