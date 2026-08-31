# Tests for the Research -> Script -> Voice -> Visual Media -> Video
# Assembly pipeline workflow (LangGraph). All tests use mock providers/fake
# assembler only - no real network/API/FFmpeg calls.
from __future__ import annotations

import os

import pytest

from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.media import VisualResult
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.media_provider import MediaProvider, MockMediaProvider
from src.tools.search_provider import MockSearchProvider, SearchProvider
from src.tools.voice_provider import MockVoiceProvider, VoiceProvider
from src.workflows.pipeline_graph import PipelineState, build_pipeline_graph, run_pipeline

TEST_VOICE_NAME = "test-voice"


class EmptySearchProvider(SearchProvider):
    """Test double: search always returns no results."""

    async def search(self, query: str, num_results: int = 5):
        return []


class ExplodingLLMProvider(LLMProvider):
    """Test double: generate_text always raises, to simulate LLM failure."""

    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


class ResearchMockWithVariedSections(LLMProvider):
    """Delegates to MockLLMProvider for Research Agent prompts (unchanged,
    realistic behavior), but returns genuinely distinct, non-boilerplate
    content for ScriptAgent's per-section prompts.

    The pipeline shares one LLMProvider between both agents. Plain
    MockLLMProvider's generic fallback embeds only ~50 characters of the
    prompt into ~300 characters of fixed boilerplate, so it can't tell two
    different section points apart enough to clear ScriptAgent's near-
    duplicate threshold - unsuitable for testing a full pipeline with real
    section-distinctness requirements.
    """

    def __init__(self) -> None:
        self._mock = MockLLMProvider()

    def generate_text(self, prompt: str) -> str:
        if "Point to expand on: '" in prompt:
            point = prompt.split("Point to expand on: '", 1)[1].split("'.", 1)[0]
            return f"{point}. A distinct detail worth covering on its own."
        return self._mock.generate_text(prompt)


class ExplodingVoiceProvider(VoiceProvider):
    """Test double: synthesize always raises, to simulate TTS failure."""

    @property
    def name(self) -> str:
        return "exploding"

    @property
    def output_format(self) -> str:
        return "mp3"

    async def synthesize(self, text: str, voice_name: str, output_path: str):
        raise RuntimeError("simulated TTS outage")


class ExplodingMediaProvider(MediaProvider):
    """Test double: search always raises, to simulate media provider failure."""

    @property
    def name(self) -> str:
        return "exploding"

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        raise RuntimeError("simulated media provider outage")

    async def download(self, candidate, output_path: str) -> None:
        raise RuntimeError("should never be called")


class FakeVideoAssembler(VideoAssembler):
    """Test double: records calls and writes tiny placeholder files instead
    of running real FFmpeg, so the pipeline's video_assembly stage can be
    exercised without any real encoding process."""

    def __init__(self, audio_duration: float = 30.0, fail: bool = False) -> None:
        self.audio_duration = audio_duration
        self.fail = fail
        self.build_calls: list[dict] = []
        self.assemble_calls: list[dict] = []

    def probe_duration_seconds(self, media_path: str) -> float:
        if self.fail:
            raise VideoAssemblerError("simulated ffmpeg outage")
        return self.audio_duration

    def build_section_clip(self, input_path, output_path, target_duration_seconds, width, height, fps, is_video):
        self.build_calls.append({"input_path": input_path, "target_duration_seconds": target_duration_seconds})
        with open(output_path, "wb") as f:
            f.write(b"FAKE CLIP")

    def concatenate_and_mux_audio(self, section_clip_paths, audio_path, output_path, tmp_dir):
        self.assemble_calls.append(
            {"section_clip_paths": list(section_clip_paths), "audio_path": audio_path}
        )
        with open(output_path, "wb") as f:
            f.write(b"FAKE VIDEO")

    def extract_frames(self, input_path, timestamps_seconds, output_dir, basename):
        raise NotImplementedError("not exercised by pipeline workflow tests")


class TestPipelineWorkflow:
    """Tests for the combined Research -> Script -> Voice -> Visual Media ->
    Video Assembly LangGraph pipeline."""

    @pytest.fixture
    def providers(self, tmp_path):
        # All output dirs under tmp_path so tests never write into the real
        # project output/ directories.
        return (
            MockSearchProvider(),
            ResearchMockWithVariedSections(),
            MockVoiceProvider(),
            MockMediaProvider(),
            FakeVideoAssembler(),
            str(tmp_path / "audio"),
            str(tmp_path / "media"),
            str(tmp_path / "video"),
        )

    @staticmethod
    def _run(providers, topic="Why do humans dream?", **overrides):
        search_provider, llm_provider, voice_provider, media_provider, assembler, voice_dir, media_dir, video_dir = providers
        return run_pipeline(
            topic,
            overrides.get("search_provider", search_provider),
            overrides.get("llm_provider", llm_provider),
            overrides.get("voice_provider", voice_provider),
            TEST_VOICE_NAME,
            overrides.get("media_provider", media_provider),
            overrides.get("assembler", assembler),
            voice_dir,
            media_dir,
            video_dir,
        )

    @pytest.mark.asyncio
    async def test_pipeline_builds_and_compiles(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, assembler, voice_dir, media_dir, video_dir = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, assembler,
            voice_dir, media_dir, video_dir,
        )
        assert graph is not None
        compiled = graph.compile()
        assert compiled is not None

    # ---- A. successful full orchestration ---------------------------------

    @pytest.mark.asyncio
    async def test_pipeline_success_end_to_end(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state, PipelineState)
        assert state.status == "completed"
        assert state.error is None
        assert state.topic == "Why do humans dream?"

    @pytest.mark.asyncio
    async def test_research_result_is_structured(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.research_result, ResearchResult)
        assert state.research_result.topic == "Why do humans dream?"
        assert len(state.research_result.summary) > 0

    @pytest.mark.asyncio
    async def test_script_result_is_structured_and_derived_from_research(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.script_result, ScriptResult)
        assert state.script_result.topic == state.research_result.topic
        assert len(state.script_result.sections) > 0

    @pytest.mark.asyncio
    async def test_voice_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        _, _, voice_provider, _, _, _, _, _ = providers
        state = await self._run(providers)

        assert isinstance(state.voice_result, VoiceResult)
        assert state.voice_result.success is True
        assert state.voice_result.audio_file_path is not None
        assert len(voice_provider.calls) == 1
        assert state.script_result.hook in voice_provider.calls[0]["text"]

    @pytest.mark.asyncio
    async def test_visual_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.visual_result, VisualResult)
        assert state.visual_result.success is True
        assert len(state.visual_result.sections) == len(state.script_result.sections)

    @pytest.mark.asyncio
    async def test_visual_media_uses_visual_context_planner_with_safe_fallback(self, providers) -> None:
        """The pipeline wires a VisualContextPlanner (one shared LLMProvider
        call) into the media stage. Since the mock LLM never returns valid
        JSON, this exercises - and must not break on - the planner's own
        deterministic fallback path."""
        state = await self._run(providers)

        assert state.visual_result.semantic_planning_used is False
        assert state.visual_result.semantic_planning_fallback_reason is not None
        assert state.status == "completed"

    # ---- B. VideoAssemblyResult stored in final state ----------------------

    @pytest.mark.asyncio
    async def test_video_assembly_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.video_assembly_result, VideoAssemblyResult)
        assert state.video_assembly_result.success is True
        assert state.video_assembly_result.output_path is not None
        assert os.path.exists(state.video_assembly_result.output_path)
        assert state.video_assembly_result.section_count == len(state.script_result.sections)

    # ---- C. Video Assembly receives the expected earlier-stage results -----

    @pytest.mark.asyncio
    async def test_video_assembly_receives_expected_inputs(self, providers) -> None:
        _, _, _, _, assembler, _, _, _ = providers
        state = await self._run(providers)

        # The assembler was called once per section, with the exact media
        # files VisualMediaService downloaded for those sections.
        expected_paths = {
            mapping.assets[0].local_file_path for mapping in state.visual_result.sections
        }
        actual_paths = {call["input_path"] for call in assembler.build_calls}
        assert actual_paths == expected_paths

        # The assembler received the exact narration audio VoiceService produced.
        assert assembler.assemble_calls[0]["audio_path"] == state.voice_result.audio_file_path

        # Section durations sum to the (fake) narration audio duration.
        assert sum(c["target_duration_seconds"] for c in assembler.build_calls) == pytest.approx(
            assembler.audio_duration, abs=0.01
        )

    # ---- D-G. Video Assembly not called on earlier-stage failure -----------

    @pytest.mark.asyncio
    async def test_video_assembly_not_called_when_research_fails(self, providers) -> None:
        _, _, voice_provider, media_provider, assembler, _, _, _ = providers
        state = await self._run(providers, topic="")

        assert state.status == "failed"
        assert state.research_result is None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_result is None
        assert state.video_assembly_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []

    @pytest.mark.asyncio
    async def test_video_assembly_not_called_when_script_fails(self, providers) -> None:
        search_provider, _, voice_provider, media_provider, assembler, _, _, _ = providers
        state = await self._run(providers, llm_provider=ExplodingLLMProvider())

        assert state.status == "failed"
        assert state.script_result is None
        assert state.video_assembly_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_video_assembly_not_called_when_voice_fails(self, providers) -> None:
        _, _, _, media_provider, assembler, _, _, _ = providers
        state = await self._run(providers, voice_provider=ExplodingVoiceProvider())

        assert state.status == "failed"
        assert "Voice generation failed" in state.error
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is False
        assert state.visual_result is None
        assert state.video_assembly_result is None
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_video_assembly_not_called_when_visual_media_fails(self, providers) -> None:
        _, _, _, _, assembler, _, _, _ = providers
        state = await self._run(providers, media_provider=ExplodingMediaProvider())

        assert state.status == "failed"
        assert "Visual media generation failed" in state.error
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_result.success is False
        assert state.video_assembly_result is None
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []

    # ---- H. Video Assembly failure surfaced correctly ----------------------

    @pytest.mark.asyncio
    async def test_video_assembly_failure_is_surfaced_cleanly(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))

        assert state.status == "failed"
        assert state.error is not None
        assert "Video assembly failed" in state.error
        # Earlier stages' results are preserved, not discarded.
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_result.success is True
        # VideoAssemblyService never raises for processing failures - it
        # returns a structured failed VideoAssemblyResult.
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is False
        assert "simulated ffmpeg outage" in state.video_assembly_result.error

    # ---- I. status only "completed" when video assembly succeeds -----------

    @pytest.mark.asyncio
    async def test_status_is_completed_only_when_video_assembly_succeeds(self, providers) -> None:
        success_state = await self._run(providers)
        assert success_state.status == "completed"
        assert success_state.video_assembly_result.success is True

        failure_state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))
        assert failure_state.status == "failed"

    # ---- J. existing behavior preserved -------------------------------------

    @pytest.mark.asyncio
    async def test_no_search_results_still_completes_through_video_assembly(self, providers) -> None:
        """MockSearchProvider returning [] is a valid (if sparse) research result,
        not an error - the pipeline should still complete all the way through
        video assembly."""
        state = await self._run(providers, topic="obscure topic", search_provider=EmptySearchProvider())

        assert state.research_result is not None
        assert state.research_result.sources == []
        assert state.status == "completed"
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True

    @pytest.mark.asyncio
    async def test_pipeline_state_defaults(self) -> None:
        state = PipelineState()
        assert state.topic == ""
        assert state.research_result is None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_result is None
        assert state.video_assembly_result is None
        assert state.status == "pending"
        assert state.error is None

    @pytest.mark.asyncio
    async def test_pipeline_multiple_runs_are_independent(self, providers) -> None:
        state_a = await self._run(providers, topic="Topic A")
        state_b = await self._run(providers, topic="Topic B")

        assert state_a.topic == "Topic A"
        assert state_b.topic == "Topic B"
        assert state_a.research_result.topic != state_b.research_result.topic
