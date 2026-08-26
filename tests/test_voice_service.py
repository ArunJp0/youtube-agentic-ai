# Tests for VoiceService (narration extraction/ordering + generation).
# Uses MockVoiceProvider only - no real TTS/network calls.
from __future__ import annotations

import os

import pytest

from src.models.script import ScriptResult, ScriptSection
from src.services.voice_service import VoiceService, VoiceServiceError
from src.tools.voice_provider import MockVoiceProvider, VoiceProvider


def _sample_script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK_TEXT",
        introduction="INTRO_TEXT",
        sections=[
            ScriptSection(heading="Heading One", narration="SECTION_ONE_TEXT"),
            ScriptSection(heading="Heading Two", narration="SECTION_TWO_TEXT"),
        ],
        conclusion="CONCLUSION_TEXT",
        call_to_action="CTA_TEXT",
        sources=["https://en.wikipedia.org/wiki/Dream"],
        script_notes="INTERNAL_RESEARCH_NOTES_SHOULD_NOT_BE_NARRATED",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


class ExplodingVoiceProvider(VoiceProvider):
    """Test double whose synthesize always raises, to simulate TTS failure."""

    @property
    def name(self) -> str:
        return "exploding"

    @property
    def output_format(self) -> str:
        return "mp3"

    async def synthesize(self, text: str, voice_name: str, output_path: str):
        raise RuntimeError("simulated TTS outage")


class TestExtractNarration:
    """Tests for narration text extraction and ordering."""

    def test_narration_order(self) -> None:
        script = _sample_script()
        narration = VoiceService.extract_narration(script)

        expected_order = [
            "HOOK_TEXT",
            "INTRO_TEXT",
            "SECTION_ONE_TEXT",
            "SECTION_TWO_TEXT",
            "CONCLUSION_TEXT",
            "CTA_TEXT",
        ]
        positions = [narration.index(part) for part in expected_order]
        assert positions == sorted(positions)  # each appears, in this exact order

    def test_narration_excludes_metadata(self) -> None:
        script = _sample_script()
        narration = VoiceService.extract_narration(script)

        assert "en.wikipedia.org" not in narration
        assert "INTERNAL_RESEARCH_NOTES_SHOULD_NOT_BE_NARRATED" not in narration
        assert "Heading One" not in narration
        assert "Heading Two" not in narration

    def test_controlled_unique_lines_each_appear_exactly_once(self) -> None:
        """Regression test for the reported repeated-narration bug.

        Confirms VoiceService's own assembly logic is not the source of
        duplication: with six distinct, controlled narration lines, each
        must appear in the final narration text exactly once, in order.
        """
        script = _sample_script(
            hook="Hook line",
            introduction="Introduction line",
            sections=[
                ScriptSection(heading="Section One", narration="Section one line"),
                ScriptSection(heading="Section Two", narration="Section two line"),
            ],
            conclusion="Conclusion line",
            call_to_action="CTA line",
        )

        narration = VoiceService.extract_narration(script)

        lines = [
            "Hook line",
            "Introduction line",
            "Section one line",
            "Section two line",
            "Conclusion line",
            "CTA line",
        ]
        for line in lines:
            assert narration.count(line) == 1, f"expected exactly one occurrence of {line!r}"

        # And in the required playback order.
        positions = [narration.index(line) for line in lines]
        assert positions == sorted(positions)

    def test_final_validation_drops_exact_duplicate_section(self) -> None:
        """Final safety-net dedup: even if a ScriptResult somehow contains an
        exact-duplicate section (bypassing ScriptAgent's own check), the
        final narration sent to TTS must not repeat it."""
        script = _sample_script(
            sections=[
                ScriptSection(heading="A", narration="This exact sentence repeats."),
                ScriptSection(heading="B", narration="This exact sentence repeats."),
            ]
        )
        narration = VoiceService.extract_narration(script)
        assert narration.count("This exact sentence repeats.") == 1

    def test_final_validation_drops_near_duplicate_section(self) -> None:
        """Reproduces the actual reported bug at the VoiceService layer:
        sections sharing the same boilerplate body but a different lead-in
        clause must be collapsed to one occurrence in the final narration."""
        boilerplate = (
            ", here are the key findings: the topic involves multiple "
            "interconnected aspects studied extensively across many fields."
        )
        script = _sample_script(
            sections=[
                ScriptSection(heading="A", narration="Based on point A" + boilerplate),
                ScriptSection(heading="B", narration="Based on point B" + boilerplate),
                ScriptSection(heading="C", narration="Based on point C" + boilerplate),
            ]
        )
        narration = VoiceService.extract_narration(script)
        assert narration.count(boilerplate) == 1
        assert "Based on point A" in narration
        assert "Based on point B" not in narration
        assert "Based on point C" not in narration

    def test_final_validation_keeps_legitimately_related_sections(self) -> None:
        """Related-but-distinct sections must not be dropped just for
        sharing a topic - only strongly similar/duplicate text is removed."""
        script = _sample_script(
            sections=[
                ScriptSection(
                    heading="Timing",
                    narration="REM sleep is when the brain is highly active and most vivid dreaming occurs.",
                ),
                ScriptSection(
                    heading="Memory",
                    narration="During sleep, the brain replays and strengthens memories from the day.",
                ),
            ]
        )
        narration = VoiceService.extract_narration(script)
        assert "REM sleep is when the brain is highly active" in narration
        assert "the brain replays and strengthens memories" in narration

    def test_final_validation_each_field_appears_exactly_once_despite_duplicate_section(
        self,
    ) -> None:
        """hook/introduction/conclusion/call_to_action must each still appear
        exactly once in the final narration, even when sections contain a
        duplicate that gets dropped."""
        script = _sample_script(
            hook="Hook line",
            introduction="Introduction line",
            sections=[
                ScriptSection(heading="A", narration="Duplicated content here."),
                ScriptSection(heading="B", narration="Duplicated content here."),
            ],
            conclusion="Conclusion line",
            call_to_action="CTA line",
        )
        narration = VoiceService.extract_narration(script)

        for line in ["Hook line", "Introduction line", "Conclusion line", "CTA line"]:
            assert narration.count(line) == 1
        assert narration.count("Duplicated content here.") == 1

    def test_narration_excludes_visual_notes(self) -> None:
        script = _sample_script(
            sections=[
                ScriptSection(
                    heading="H",
                    narration="SECTION_TEXT",
                    visual_notes="B-ROLL_NOTE_SHOULD_NOT_BE_NARRATED",
                )
            ]
        )
        narration = VoiceService.extract_narration(script)
        assert "B-ROLL_NOTE_SHOULD_NOT_BE_NARRATED" not in narration
        assert "SECTION_TEXT" in narration

    def test_narration_with_no_sections(self) -> None:
        script = _sample_script(sections=[])
        narration = VoiceService.extract_narration(script)
        assert "HOOK_TEXT" in narration
        assert "CTA_TEXT" in narration

    def test_narration_empty_when_all_fields_blank(self) -> None:
        # Bypass validation (ScriptResult normally requires non-empty text)
        # to exercise VoiceService's defensive empty-narration handling.
        script = ScriptResult.model_construct(
            topic="T",
            video_title="T",
            hook="   ",
            introduction="",
            sections=[],
            conclusion="",
            call_to_action="",
            estimated_duration_seconds=0.0,
            sources=[],
            script_notes=None,
        )
        assert VoiceService.extract_narration(script) == ""


class TestVoiceServiceGenerateVoice:
    """Tests for VoiceService.generate_voice using MockVoiceProvider."""

    @pytest.mark.asyncio
    async def test_generate_voice_success(self, tmp_path) -> None:
        provider = MockVoiceProvider(fixed_duration_seconds=12.3)
        service = VoiceService(
            voice_provider=provider, voice_name="en-US-AriaNeural", output_dir=str(tmp_path)
        )

        result = await service.generate_voice(_sample_script())

        assert result.success is True
        assert result.error is None
        assert result.provider == "mock"
        assert result.voice_name == "en-US-AriaNeural"
        assert result.format == "mp3"
        assert result.duration_seconds == 12.3
        assert result.audio_file_path is not None
        assert os.path.exists(result.audio_file_path)

    @pytest.mark.asyncio
    async def test_generate_voice_creates_nested_output_dir(self, tmp_path) -> None:
        nested_dir = str(tmp_path / "nested" / "audio")
        provider = MockVoiceProvider()
        service = VoiceService(voice_provider=provider, voice_name="v", output_dir=nested_dir)

        result = await service.generate_voice(_sample_script())

        assert os.path.isdir(nested_dir)
        assert result.audio_file_path.startswith(nested_dir)

    @pytest.mark.asyncio
    async def test_generate_voice_none_script_raises(self, tmp_path) -> None:
        service = VoiceService(
            voice_provider=MockVoiceProvider(), voice_name="v", output_dir=str(tmp_path)
        )
        with pytest.raises(VoiceServiceError, match="ScriptResult is required"):
            await service.generate_voice(None)

    @pytest.mark.asyncio
    async def test_generate_voice_provider_failure_captured_in_result(self, tmp_path) -> None:
        service = VoiceService(
            voice_provider=ExplodingVoiceProvider(), voice_name="v", output_dir=str(tmp_path)
        )

        result = await service.generate_voice(_sample_script())

        assert result.success is False
        assert result.audio_file_path is None
        assert result.duration_seconds is None
        assert "simulated TTS outage" in result.error
        assert result.provider == "exploding"

    @pytest.mark.asyncio
    async def test_generate_voice_empty_narration_handled(self, tmp_path) -> None:
        script = ScriptResult.model_construct(
            topic="T",
            video_title="T",
            hook="",
            introduction="",
            sections=[],
            conclusion="",
            call_to_action="",
            estimated_duration_seconds=0.0,
            sources=[],
            script_notes=None,
        )
        provider = MockVoiceProvider()
        service = VoiceService(voice_provider=provider, voice_name="v", output_dir=str(tmp_path))

        result = await service.generate_voice(script)

        assert result.success is False
        assert result.error == "No narration text to synthesize"
        assert provider.calls == []  # provider must never be called

    @pytest.mark.asyncio
    async def test_generate_voice_passes_narration_to_provider(self, tmp_path) -> None:
        provider = MockVoiceProvider()
        service = VoiceService(voice_provider=provider, voice_name="v", output_dir=str(tmp_path))

        await service.generate_voice(_sample_script())

        assert len(provider.calls) == 1
        sent_text = provider.calls[0]["text"]
        assert "HOOK_TEXT" in sent_text
        assert "en.wikipedia.org" not in sent_text
