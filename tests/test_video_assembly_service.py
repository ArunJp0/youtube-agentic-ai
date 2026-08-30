# Tests for VideoAssemblyService (timing calculation, section ordering,
# error propagation). Uses a fake VideoAssembler - no real FFmpeg process
# or network calls.
from __future__ import annotations

import os

import pytest

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.models.voice import VoiceResult
from src.services.video_assembly_service import VideoAssemblyService, VideoAssemblyServiceError
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError


class FakeVideoAssembler(VideoAssembler):
    """Test double: records calls and writes tiny placeholder files instead
    of running real FFmpeg."""

    def __init__(self, audio_duration: float = 30.0, fail_at: str | None = None) -> None:
        self.audio_duration = audio_duration
        self.fail_at = fail_at
        self.build_calls: list[dict] = []
        self.concat_call: dict | None = None

    def probe_duration_seconds(self, media_path: str) -> float:
        if self.fail_at == "probe":
            raise VideoAssemblerError("simulated probe failure")
        return self.audio_duration

    def build_section_clip(self, input_path, output_path, target_duration_seconds, width, height, fps, is_video):
        if self.fail_at == "build":
            raise VideoAssemblerError("simulated build failure")
        self.build_calls.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "target_duration_seconds": target_duration_seconds,
                "width": width,
                "height": height,
                "fps": fps,
                "is_video": is_video,
            }
        )
        with open(output_path, "wb") as f:
            f.write(b"FAKE CLIP")

    def concatenate_and_mux_audio(self, section_clip_paths, audio_path, output_path, tmp_dir):
        if self.fail_at == "concat":
            raise VideoAssemblerError("simulated concat failure")
        self.concat_call = {
            "section_clip_paths": list(section_clip_paths),
            "audio_path": audio_path,
            "output_path": output_path,
        }
        with open(output_path, "wb") as f:
            f.write(b"FAKE VIDEO")


def _sample_script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(heading="Section One", narration="This is a short section about dreams."),
            ScriptSection(
                heading="Section Two",
                narration="This is a much longer section about memory consolidation and brain "
                "activity during REM sleep cycles at night for adults everywhere.",
            ),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _successful_voice_result(tmp_path) -> VoiceResult:
    audio_path = str(tmp_path / "narration.mp3")
    with open(audio_path, "wb") as f:
        f.write(b"FAKE AUDIO")
    return VoiceResult(
        audio_file_path=audio_path, provider="mock", voice_name="v", format="mp3", success=True
    )


def _successful_visual_result(tmp_path, num_sections: int = 2) -> VisualResult:
    sections = []
    for i in range(num_sections):
        media_path = str(tmp_path / f"section-{i + 1}.mp4")
        with open(media_path, "wb") as f:
            f.write(b"FAKE MEDIA")
        sections.append(
            SectionMediaMapping(
                section_index=i,
                section_heading=f"Section {i + 1}",
                search_query="q",
                assets=[
                    MediaAsset(
                        provider="mock",
                        asset_type="video",
                        local_file_path=media_path,
                        search_query="q",
                        section_index=i,
                        success=True,
                    )
                ],
            )
        )
    return VisualResult(topic="Why do humans dream?", provider="mock", sections=sections, success=True)


class TestVideoAssemblyServiceValidation:
    @pytest.mark.asyncio
    async def test_missing_required_inputs_raises(self, tmp_path) -> None:
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path))
        with pytest.raises(VideoAssemblyServiceError):
            await service.assemble_video(
                None, _successful_voice_result(tmp_path), _successful_visual_result(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_failed_voice_result_returns_clean_failure(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = VoiceResult(provider="mock", voice_name="v", format="mp3", success=False, error="tts broke")
        visual_result = _successful_visual_result(tmp_path)
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "VoiceResult" in result.error

    @pytest.mark.asyncio
    async def test_failed_visual_result_returns_clean_failure(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = VisualResult(topic="Dreams", provider="mock", success=False, error="pexels broke")
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "VisualResult" in result.error

    @pytest.mark.asyncio
    async def test_missing_audio_file_on_disk_returns_clean_failure(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = VoiceResult(
            audio_file_path=str(tmp_path / "missing.mp3"),
            provider="mock",
            voice_name="v",
            format="mp3",
            success=True,
        )
        visual_result = _successful_visual_result(tmp_path)
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_media_file_on_disk_returns_clean_failure(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = VisualResult(
            topic="Dreams",
            provider="mock",
            success=True,
            sections=[
                SectionMediaMapping(
                    section_index=0,
                    section_heading="Section One",
                    search_query="q",
                    assets=[
                        MediaAsset(
                            provider="mock",
                            asset_type="video",
                            local_file_path=str(tmp_path / "gone1.mp4"),
                            search_query="q",
                            section_index=0,
                            success=True,
                        )
                    ],
                ),
                SectionMediaMapping(
                    section_index=1,
                    section_heading="Section Two",
                    search_query="q",
                    assets=[
                        MediaAsset(
                            provider="mock",
                            asset_type="video",
                            local_file_path=str(tmp_path / "gone2.mp4"),
                            search_query="q",
                            section_index=1,
                            success=True,
                        )
                    ],
                ),
            ],
        )
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_section_media_mapping_returns_clean_failure(self, tmp_path) -> None:
        """VisualResult overall success=True but one section has no
        successful asset (e.g. only 1 of 2 sections got media)."""
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        media_path = str(tmp_path / "section-1.mp4")
        with open(media_path, "wb") as f:
            f.write(b"FAKE MEDIA")
        visual_result = VisualResult(
            topic="Dreams",
            provider="mock",
            success=True,
            sections=[
                SectionMediaMapping(
                    section_index=0,
                    section_heading="Section One",
                    search_query="q",
                    assets=[
                        MediaAsset(
                            provider="mock",
                            asset_type="video",
                            local_file_path=media_path,
                            search_query="q",
                            section_index=0,
                            success=True,
                        )
                    ],
                ),
                SectionMediaMapping(
                    section_index=1,
                    section_heading="Section Two",
                    search_query="q",
                    assets=[MediaAsset(provider="mock", search_query="q", section_index=1, success=False, error="no asset")],
                ),
            ],
        )
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "Missing media" in result.error

    @pytest.mark.asyncio
    async def test_no_sections_returns_clean_failure(self, tmp_path) -> None:
        script = _sample_script(sections=[])
        voice_result = _successful_voice_result(tmp_path)
        visual_result = VisualResult(topic="Dreams", provider="mock", success=True, sections=[])
        service = VideoAssemblyService(assembler=FakeVideoAssembler(), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False


class TestVideoAssemblyServiceTiming:
    def test_durations_proportional_to_narration_length(self) -> None:
        sections = [
            ScriptSection(heading="A", narration="one two three four"),
            ScriptSection(
                heading="B", narration="one two three four five six seven eight ten eleven twelve"
            ),
        ]
        durations = VideoAssemblyService.calculate_section_durations(sections, 40.0)
        assert len(durations) == 2
        assert durations[1] > durations[0]
        assert sum(durations) == pytest.approx(40.0, abs=0.01)

    def test_durations_sum_exactly_matches_total_despite_rounding(self) -> None:
        sections = [ScriptSection(heading=f"S{i}", narration="word " * (i + 1)) for i in range(5)]
        durations = VideoAssemblyService.calculate_section_durations(sections, 123.456)
        assert sum(durations) == pytest.approx(123.456, abs=0.001)

    def test_single_section_gets_full_duration(self) -> None:
        sections = [ScriptSection(heading="Only", narration="Some narration text here.")]
        durations = VideoAssemblyService.calculate_section_durations(sections, 50.0)
        assert durations == pytest.approx([50.0])

    def test_equal_length_sections_get_equal_durations(self) -> None:
        sections = [
            ScriptSection(heading="A", narration="one two three four"),
            ScriptSection(heading="B", narration="five six seven eight"),
        ]
        durations = VideoAssemblyService.calculate_section_durations(sections, 20.0)
        assert durations[0] == pytest.approx(10.0)
        assert durations[1] == pytest.approx(10.0)


class TestVideoAssemblyServiceAssembly:
    @pytest.mark.asyncio
    async def test_successful_assembly_flow(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(audio_duration=30.0)
        service = VideoAssemblyService(assembler=assembler, output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is True
        assert result.output_path is not None
        assert os.path.exists(result.output_path)
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30
        assert result.format == "mp4"
        assert result.video_codec == "h264"
        assert result.audio_codec == "aac"
        assert result.section_count == 2
        assert len(result.section_durations_seconds) == 2
        assert sum(result.section_durations_seconds) == pytest.approx(30.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_sections_processed_and_concatenated_in_script_order(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(audio_duration=30.0)
        service = VideoAssemblyService(assembler=assembler, output_dir=str(tmp_path / "out"))

        await service.assemble_video(script, voice_result, visual_result)

        assert len(assembler.build_calls) == 2
        assert assembler.build_calls[0]["input_path"].endswith("section-1.mp4")
        assert assembler.build_calls[1]["input_path"].endswith("section-2.mp4")
        clip_paths = assembler.concat_call["section_clip_paths"]
        assert "section-01" in clip_paths[0]
        assert "section-02" in clip_paths[1]

    @pytest.mark.asyncio
    async def test_probe_failure_propagates_as_clean_result(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(fail_at="probe")
        service = VideoAssemblyService(assembler=assembler, output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "probe" in result.error.lower()

    @pytest.mark.asyncio
    async def test_build_failure_propagates_as_clean_result(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(audio_duration=30.0, fail_at="build")
        service = VideoAssemblyService(assembler=assembler, output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "simulated build failure" in result.error

    @pytest.mark.asyncio
    async def test_concat_failure_propagates_as_clean_result(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(audio_duration=30.0, fail_at="concat")
        service = VideoAssemblyService(assembler=assembler, output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.success is False
        assert "simulated concat failure" in result.error

    @pytest.mark.asyncio
    async def test_output_dir_created_if_missing(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        nested_dir = str(tmp_path / "nested" / "video")
        service = VideoAssemblyService(assembler=FakeVideoAssembler(audio_duration=30.0), output_dir=nested_dir)

        result = await service.assemble_video(script, voice_result, visual_result)

        assert os.path.isdir(nested_dir)
        assert result.output_path.startswith(nested_dir)

    @pytest.mark.asyncio
    async def test_output_filename_derived_from_topic(self, tmp_path) -> None:
        script = _sample_script(topic="Why do humans dream?")
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        service = VideoAssemblyService(assembler=FakeVideoAssembler(audio_duration=30.0), output_dir=str(tmp_path / "out"))

        result = await service.assemble_video(script, voice_result, visual_result)

        assert "why-do-humans-dream" in os.path.basename(result.output_path)
        assert result.output_path.endswith(".mp4")

    @pytest.mark.asyncio
    async def test_custom_resolution_and_fps_are_used(self, tmp_path) -> None:
        script = _sample_script()
        voice_result = _successful_voice_result(tmp_path)
        visual_result = _successful_visual_result(tmp_path)
        assembler = FakeVideoAssembler(audio_duration=10.0)
        service = VideoAssemblyService(
            assembler=assembler, output_dir=str(tmp_path / "out"), width=1280, height=720, fps=24
        )

        result = await service.assemble_video(script, voice_result, visual_result)

        assert result.width == 1280
        assert result.height == 720
        assert result.fps == 24
        assert assembler.build_calls[0]["width"] == 1280
        assert assembler.build_calls[0]["fps"] == 24
