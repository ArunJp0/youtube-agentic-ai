# Tests for MusicSelectionService: deterministic ranking of the approved
# local BGM catalog against a MusicPlan. No LLM/network involved.
from __future__ import annotations

from src.models.music import BGMTrack, MusicPlan
from src.services.music_selection_service import MusicSelectionService


def _plan(**overrides) -> MusicPlan:
    defaults = dict(topic="dreams", primary_mood="calm", used_semantic_planning=True)
    defaults.update(overrides)
    return MusicPlan(**defaults)


def _track(**overrides) -> BGMTrack:
    defaults = dict(
        track_id="track-1",
        file_path="track-1.mp3",
        title="Track One",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


class TestEmptyCatalog:
    def test_empty_catalog_returns_none_with_warning(self) -> None:
        track, warnings = MusicSelectionService().select_track(_plan(), [])
        assert track is None
        assert any("empty" in w.lower() for w in warnings)


class TestVocalTrackExclusion:
    def test_vocal_track_excluded_by_default(self) -> None:
        vocal = _track(track_id="vocal-1", instrumental=False, mood_tags=["calm"])
        instrumental = _track(track_id="instrumental-1", instrumental=True, mood_tags=["calm"])
        track, _ = MusicSelectionService().select_track(_plan(primary_mood="calm"), [vocal, instrumental])
        assert track.track_id == "instrumental-1"

    def test_all_vocal_catalog_warns_and_returns_none_without_fallback(self) -> None:
        vocal = _track(track_id="vocal-1", instrumental=False)
        track, warnings = MusicSelectionService().select_track(_plan(), [vocal])
        assert track is None
        assert any("instrumental" in w.lower() for w in warnings)


class TestAvoidStylesFiltering:
    def test_track_matching_avoid_style_via_mood_tag_is_excluded(self) -> None:
        aggressive = _track(track_id="aggressive-1", mood_tags=["aggressive", "loud"])
        calm = _track(track_id="calm-1", mood_tags=["calm", "subtle"])
        plan = _plan(avoid_styles=["aggressive"])

        track, _ = MusicSelectionService().select_track(plan, [aggressive, calm])

        assert track.track_id == "calm-1"

    def test_track_matching_avoid_style_via_genre_is_excluded(self) -> None:
        rock = _track(track_id="rock-1", genre="rock", mood_tags=["energetic"])
        ambient = _track(track_id="ambient-1", genre="ambient", mood_tags=["calm"])
        plan = _plan(avoid_styles=["rock"])

        track, _ = MusicSelectionService().select_track(plan, [rock, ambient])

        assert track.track_id == "ambient-1"


class TestMoodEnergyRanking:
    def test_higher_mood_overlap_wins(self) -> None:
        strong_match = _track(track_id="strong", mood_tags=["thoughtful", "calm"])
        weak_match = _track(track_id="weak", mood_tags=["upbeat"])
        plan = _plan(primary_mood="thoughtful", secondary_mood="calm")

        track, _ = MusicSelectionService().select_track(plan, [weak_match, strong_match])

        assert track.track_id == "strong"

    def test_genre_match_contributes_to_score(self) -> None:
        genre_match = _track(track_id="genre-match", genre="ambient", mood_tags=[])
        no_genre_match = _track(track_id="no-genre-match", genre="rock", mood_tags=[])
        plan = _plan(preferred_genres=["ambient"])

        track, _ = MusicSelectionService().select_track(plan, [no_genre_match, genre_match])

        assert track.track_id == "genre-match"

    def test_exact_energy_match_preferred_over_distant_energy(self) -> None:
        exact = _track(track_id="exact", energy_level="low", mood_tags=[])
        distant = _track(track_id="distant", energy_level="high", mood_tags=[])
        plan = _plan(energy_level="low")

        track, _ = MusicSelectionService().select_track(plan, [distant, exact])

        assert track.track_id == "exact"

    def test_neutral_subtle_requirement_favors_subtle_tagged_track(self) -> None:
        subtle = _track(track_id="subtle", mood_tags=["subtle"])
        plain = _track(track_id="plain", mood_tags=[])
        plan = _plan(requires_neutral_subtle=True)

        track, _ = MusicSelectionService().select_track(plan, [plain, subtle])

        assert track.track_id == "subtle"

    def test_tie_broken_deterministically_by_track_id(self) -> None:
        a = _track(track_id="a-track", mood_tags=[])
        b = _track(track_id="b-track", mood_tags=[])
        plan = _plan()

        track, _ = MusicSelectionService().select_track(plan, [b, a])

        assert track.track_id == "a-track"


class TestNeutralFallbackTrack:
    def test_fallback_track_used_when_nothing_eligible(self) -> None:
        only_track = _track(track_id="rejected", mood_tags=["aggressive"])
        fallback = _track(track_id="neutral-fallback", mood_tags=["aggressive"])
        plan = _plan(avoid_styles=["aggressive"])
        service = MusicSelectionService(fallback_track_id="neutral-fallback")

        track, warnings = service.select_track(plan, [only_track, fallback])

        assert track.track_id == "neutral-fallback"
        assert any("fallback" in w.lower() for w in warnings)

    def test_no_fallback_configured_returns_none(self) -> None:
        only_track = _track(track_id="rejected", mood_tags=["aggressive"])
        plan = _plan(avoid_styles=["aggressive"])

        track, warnings = MusicSelectionService().select_track(plan, [only_track])

        assert track is None
        assert any("no fallback" in w.lower() for w in warnings)

    def test_unknown_fallback_track_id_returns_none(self) -> None:
        only_track = _track(track_id="rejected", mood_tags=["aggressive"])
        plan = _plan(avoid_styles=["aggressive"])
        service = MusicSelectionService(fallback_track_id="does-not-exist")

        track, _ = service.select_track(plan, [only_track])

        assert track is None


class TestDeterministicSelection:
    def test_same_inputs_produce_same_selection_regardless_of_order(self) -> None:
        a = _track(track_id="a", mood_tags=["calm"])
        b = _track(track_id="b", mood_tags=["calm"])
        c = _track(track_id="c", mood_tags=["upbeat"])
        plan = _plan(primary_mood="calm")
        service = MusicSelectionService()

        first, _ = service.select_track(plan, [a, b, c])
        second, _ = service.select_track(plan, [c, b, a])

        assert first.track_id == second.track_id == "a"
