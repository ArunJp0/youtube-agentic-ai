# Tests for AudioMixingService (mood planning -> track selection -> mixing
# orchestration). Uses fake assembler/catalog provider only - no real
# network, Gemini, or FFmpeg process.
from __future__ import annotations

import os

import pytest

from src.models.music import BGMTrack
from src.models.script import ScriptResult, ScriptSection
from src.models.video import VideoAssemblyResult
from src.services.audio_mixing_service import (
    DEFAULT_BGM_GAIN_DB,
    DEFAULT_FADE_OUT_SECONDS,
    OUTPUT_SUFFIX,
    AudioMixingService,
    AudioMixingServiceError,
)
from src.services.music_selection_service import MusicSelectionService
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.music_catalog_provider import MockMusicCatalogProvider


class FakeAssembler(VideoAssembler):
    """Test double: records mix_background_audio calls and writes tiny
    placeholder files instead of running real FFmpeg."""

    def __init__(
        self,
        video_duration: float = 100.0,
        track_duration: float = 30.0,
        output_duration: float | None = None,
        fail_probe_video: bool = False,
        fail_probe_track: bool = False,
        fail_probe_output: bool = False,
        fail_mix: bool = False,
        write_output: bool = True,
    ) -> None:
        self.video_duration = video_duration
        self.track_duration = track_duration
        self.output_duration = output_duration if output_duration is not None else video_duration
        self.fail_probe_video = fail_probe_video
        self.fail_probe_track = fail_probe_track
        self.fail_probe_output = fail_probe_output
        self.fail_mix = fail_mix
        self.write_output = write_output
        self.mix_calls: list[dict] = []

    def probe_duration_seconds(self, media_path: str) -> float:
        ext = os.path.splitext(media_path)[1].lower()
        if ext in {".mp3", ".wav", ".m4a"}:
            if self.fail_probe_track:
                raise VideoAssemblerError("simulated track probe failure")
            return self.track_duration
        if media_path.endswith(f"{OUTPUT_SUFFIX}.mp4"):
            if self.fail_probe_output:
                raise VideoAssemblerError("simulated output probe failure")
            return self.output_duration
        if self.fail_probe_video:
            raise VideoAssemblerError("simulated video probe failure")
        return self.video_duration

    def build_section_clip(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Audio Mixing Service tests")

    def concatenate_and_mux_audio(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Audio Mixing Service tests")

    def extract_frames(self, *args, **kwargs):
        raise NotImplementedError("not exercised by Audio Mixing Service tests")

    def burn_subtitles(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Audio Mixing Service tests")

    def mix_background_audio(
        self,
        input_video_path,
        music_path,
        output_path,
        target_duration_seconds,
        music_gain_db,
        fade_in_seconds,
        fade_out_seconds,
        use_ducking=True,
    ) -> None:
        self.mix_calls.append(
            {
                "input_video_path": input_video_path,
                "music_path": music_path,
                "output_path": output_path,
                "target_duration_seconds": target_duration_seconds,
                "music_gain_db": music_gain_db,
                "fade_in_seconds": fade_in_seconds,
                "fade_out_seconds": fade_out_seconds,
                "use_ducking": use_ducking,
            }
        )
        if self.fail_mix:
            raise VideoAssemblerError("simulated ffmpeg mix failure")
        if self.write_output:
            with open(output_path, "wb") as f:
                f.write(b"FAKE MIXED VIDEO")


def _script() -> ScriptResult:
    return ScriptResult(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[ScriptSection(heading="REM Sleep", narration="Dreams occur during REM sleep.")],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )


def _video_result(tmp_path, success=True, missing=False, output_path=None) -> VideoAssemblyResult:
    path = output_path or str(tmp_path / "video-captioned.mp4")
    if success and not missing:
        with open(path, "wb") as f:
            f.write(b"FAKE SOURCE VIDEO")
    return VideoAssemblyResult(success=success, output_path=path if success else None, format="mp4")


def _track(tmp_path, track_id="track-1", write_file=True, **overrides) -> BGMTrack:
    file_path = str(tmp_path / f"{track_id}.mp3")
    if write_file:
        with open(file_path, "wb") as f:
            f.write(b"FAKE TRACK AUDIO")
    defaults = dict(
        track_id=track_id,
        file_path=file_path,
        title="Track",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        mood_tags=["neutral", "calm", "subtle"],
        instrumental=True,
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


class TestConfigurationErrors:
    def test_positive_gain_raises_on_construction(self, tmp_path) -> None:
        assembler = FakeAssembler()
        catalog = MockMusicCatalogProvider(tracks=[_track(tmp_path)])
        with pytest.raises(AudioMixingServiceError, match="bgm_gain_db"):
            AudioMixingService(catalog_provider=catalog, assembler=assembler, bgm_gain_db=5.0)

    @pytest.mark.asyncio
    async def test_none_script_raises(self, tmp_path) -> None:
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(), assembler=FakeAssembler()
        )
        video_result = _video_result(tmp_path)
        with pytest.raises(AudioMixingServiceError, match="required"):
            await service.generate_mix("dreams", None, video_result)


class TestSourceVideoValidation:
    @pytest.mark.asyncio
    async def test_unsuccessful_video_result_returns_failure(self, tmp_path) -> None:
        service = AudioMixingService(catalog_provider=MockMusicCatalogProvider(), assembler=FakeAssembler())
        video_result = _video_result(tmp_path, success=False)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "not successful" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_source_video_file_returns_failure(self, tmp_path) -> None:
        service = AudioMixingService(catalog_provider=MockMusicCatalogProvider(), assembler=FakeAssembler())
        video_result = _video_result(tmp_path, success=True, missing=True)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "not found" in result.error.lower()


class TestCatalogAndSelectionFailures:
    @pytest.mark.asyncio
    async def test_empty_catalog_returns_failure_with_music_plan(self, tmp_path) -> None:
        service = AudioMixingService(catalog_provider=MockMusicCatalogProvider(tracks=[]), assembler=FakeAssembler())
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "no suitable" in result.error.lower()
        assert result.music_plan is not None
        assert result.selected_track is None

    @pytest.mark.asyncio
    async def test_no_eligible_track_returns_failure(self, tmp_path) -> None:
        vocal_only = _track(tmp_path, track_id="vocal-1", instrumental=False)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[vocal_only]), assembler=FakeAssembler()
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert result.selected_track is None

    @pytest.mark.asyncio
    async def test_selected_track_file_missing_returns_failure(self, tmp_path) -> None:
        missing_track = _track(tmp_path, track_id="missing-1", write_file=False)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[missing_track]), assembler=FakeAssembler()
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "not found" in result.error.lower()
        assert result.selected_track.track_id == "missing-1"


class TestDurationProbingAndMixingFailures:
    @pytest.mark.asyncio
    async def test_video_duration_probe_failure_returns_failure(self, tmp_path) -> None:
        assembler = FakeAssembler(fail_probe_video=True)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "probe" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ffmpeg_mix_failure_returns_failure_and_preserves_source(self, tmp_path) -> None:
        assembler = FakeAssembler(fail_mix=True)
        video_result = _video_result(tmp_path)
        original_bytes = open(video_result.output_path, "rb").read()
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "mixing failed" in result.error.lower()
        assert open(video_result.output_path, "rb").read() == original_bytes

    @pytest.mark.asyncio
    async def test_missing_output_file_after_mix_returns_failure(self, tmp_path) -> None:
        assembler = FakeAssembler(write_output=False)
        video_result = _video_result(tmp_path)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is False
        assert "missing" in result.error.lower()


class TestSuccessfulMix:
    @pytest.mark.asyncio
    async def test_success_result_has_expected_fields(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=100.0, track_duration=30.0)
        video_result = _video_result(tmp_path)
        track = _track(tmp_path)
        service = AudioMixingService(catalog_provider=MockMusicCatalogProvider(tracks=[track]), assembler=assembler)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is True
        assert result.output_path == video_result.output_path.replace(".mp4", f"{OUTPUT_SUFFIX}.mp4")
        assert result.source_video_path == video_result.output_path
        assert result.selected_track.track_id == track.track_id
        assert result.bgm_gain_db == DEFAULT_BGM_GAIN_DB
        assert result.ducking_used is True
        assert result.source_duration_seconds == 100.0
        assert os.path.exists(result.output_path)

    @pytest.mark.asyncio
    async def test_deterministic_fallback_plan_used_when_no_llm_provider(self, tmp_path) -> None:
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=FakeAssembler()
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.music_plan.used_semantic_planning is False

    @pytest.mark.asyncio
    async def test_track_shorter_than_video_is_marked_looped(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=100.0, track_duration=20.0)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.looped is True

    @pytest.mark.asyncio
    async def test_track_longer_than_video_is_not_looped(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=30.0, track_duration=180.0)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.looped is False

    @pytest.mark.asyncio
    async def test_track_exact_duration_match_is_not_looped(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=60.0, track_duration=60.0)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.looped is False

    @pytest.mark.asyncio
    async def test_fade_out_capped_to_half_source_duration(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=4.0)  # shorter than 2x default fade-out (3.0s)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        await service.generate_mix("dreams", _script(), video_result)

        assert assembler.mix_calls[0]["fade_out_seconds"] == pytest.approx(2.0)
        assert assembler.mix_calls[0]["fade_out_seconds"] < DEFAULT_FADE_OUT_SECONDS

    @pytest.mark.asyncio
    async def test_ducking_flag_passed_through_when_disabled(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]),
            assembler=assembler,
            use_ducking=False,
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.ducking_used is False
        assert assembler.mix_calls[0]["use_ducking"] is False

    @pytest.mark.asyncio
    async def test_custom_gain_passed_through(self, tmp_path) -> None:
        assembler = FakeAssembler()
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]),
            assembler=assembler,
            bgm_gain_db=-18.0,
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.bgm_gain_db == -18.0
        assert assembler.mix_calls[0]["music_gain_db"] == -18.0

    @pytest.mark.asyncio
    async def test_source_video_never_modified_on_success(self, tmp_path) -> None:
        assembler = FakeAssembler()
        video_result = _video_result(tmp_path)
        original_bytes = open(video_result.output_path, "rb").read()
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )

        await service.generate_mix("dreams", _script(), video_result)

        assert open(video_result.output_path, "rb").read() == original_bytes

    @pytest.mark.asyncio
    async def test_output_duration_drift_beyond_tolerance_adds_warning(self, tmp_path) -> None:
        assembler = FakeAssembler(video_duration=100.0, output_duration=105.0)
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[_track(tmp_path)]), assembler=assembler
        )
        video_result = _video_result(tmp_path)

        result = await service.generate_mix("dreams", _script(), video_result)

        assert any("differs from source" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_selection_warnings_propagated_on_success(self, tmp_path) -> None:
        only_track = _track(tmp_path, track_id="rejected", mood_tags=["aggressive"])
        fallback_track = _track(tmp_path, track_id="fallback-track", mood_tags=["aggressive"])
        service = AudioMixingService(
            catalog_provider=MockMusicCatalogProvider(tracks=[only_track, fallback_track]),
            assembler=FakeAssembler(),
            selection_service=MusicSelectionService(fallback_track_id="fallback-track"),
        )
        video_result = _video_result(tmp_path)

        # Force an avoid_styles rejection by using a script whose deterministic
        # fallback plan always flags "aggressive" as an avoid style.
        result = await service.generate_mix("dreams", _script(), video_result)

        assert result.success is True
        assert result.selected_track.track_id == "fallback-track"
        assert any("fallback" in w.lower() for w in result.warnings)
