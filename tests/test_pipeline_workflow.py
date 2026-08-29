# Tests for the Research -> Script -> Voice -> Visual Media pipeline
# workflow (LangGraph). All tests use mock providers only - no real
# network/API calls.
from __future__ import annotations

import pytest

from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.media import VisualResult
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.voice import VoiceResult
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


class TestPipelineWorkflow:
    """Tests for the combined Research -> Script -> Voice -> Visual Media LangGraph pipeline."""

    @pytest.fixture
    def providers(self, tmp_path):
        # voice_output_dir/media_output_dir under tmp_path so tests never
        # write into the real project output/ directories.
        return (
            MockSearchProvider(),
            ResearchMockWithVariedSections(),
            MockVoiceProvider(),
            MockMediaProvider(),
            str(tmp_path / "audio"),
            str(tmp_path / "media"),
        )

    @pytest.mark.asyncio
    async def test_pipeline_builds_and_compiles(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, voice_dir, media_dir
        )
        assert graph is not None
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_pipeline_success_end_to_end(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert isinstance(state, PipelineState)
        assert state.status == "completed"
        assert state.error is None
        assert state.topic == "Why do humans dream?"

    @pytest.mark.asyncio
    async def test_research_result_is_structured(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert isinstance(state.research_result, ResearchResult)
        assert state.research_result.topic == "Why do humans dream?"
        assert len(state.research_result.summary) > 0

    @pytest.mark.asyncio
    async def test_script_result_is_structured_and_derived_from_research(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert isinstance(state.script_result, ScriptResult)
        # Script topic must match the research topic it was generated from.
        assert state.script_result.topic == state.research_result.topic
        assert len(state.script_result.sections) > 0

    @pytest.mark.asyncio
    async def test_script_sources_match_research_sources(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "How does photosynthesis work?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert [str(s) for s in state.script_result.sources] == [
            str(s) for s in state.research_result.sources
        ]

    @pytest.mark.asyncio
    async def test_voice_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        """ScriptResult must be passed directly to VoiceService, and the
        resulting VoiceResult must end up in the final pipeline state."""
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert isinstance(state.voice_result, VoiceResult)
        assert state.voice_result.success is True
        assert state.voice_result.provider == "mock"
        assert state.voice_result.voice_name == TEST_VOICE_NAME
        assert state.voice_result.audio_file_path is not None
        # The provider must have received the ScriptResult's narration, not
        # something re-derived independently.
        assert len(voice_provider.calls) == 1
        assert state.script_result.hook in voice_provider.calls[0]["text"]

    @pytest.mark.asyncio
    async def test_visual_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        """The same ScriptResult must be passed directly to
        VisualMediaService (not regenerated), and the resulting VisualResult
        must end up in the final pipeline state alongside VoiceResult."""
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert isinstance(state.voice_result, VoiceResult)
        assert isinstance(state.visual_result, VisualResult)
        assert state.visual_result.success is True
        assert state.visual_result.provider == "mock"
        assert state.visual_result.topic == state.script_result.topic
        assert len(state.visual_result.sections) == len(state.script_result.sections)
        for section_mapping in state.visual_result.sections:
            assert section_mapping.assets[0].success is True

    @pytest.mark.asyncio
    async def test_empty_topic_fails_research_and_skips_all_later_stages(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "", search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, voice_dir, media_dir
        )

        assert state.status == "failed"
        assert state.error is not None
        assert state.research_result is None
        assert state.script_result is None  # script node must not have run
        assert state.voice_result is None  # voice node must not have run
        assert state.visual_result is None  # media node must not have run
        assert voice_provider.calls == []  # confirms voice was never invoked
        assert media_provider.calls == []  # confirms media was never invoked

    @pytest.mark.asyncio
    async def test_no_search_results_still_completes_through_media(self, providers) -> None:
        """MockSearchProvider returning [] is a valid (if sparse) research result,
        not an error - the pipeline should still complete through scripting,
        voice, and visual media."""
        _, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "obscure topic",
            EmptySearchProvider(),
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert state.research_result is not None
        assert state.research_result.sources == []
        assert state.status == "completed"
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_result.success is True

    @pytest.mark.asyncio
    async def test_script_stage_failure_is_reported_and_skips_voice_and_media(self, providers) -> None:
        search_provider, _, voice_provider, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            ExplodingLLMProvider(),
            voice_provider,
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        # Research itself calls the LLM too, so the exploding provider fails
        # at the research stage; either way the pipeline must fail cleanly
        # and neither voice nor visual media generation must ever run.
        assert state.status == "failed"
        assert state.error is not None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []

    @pytest.mark.asyncio
    async def test_voice_stage_failure_is_surfaced_and_skips_media(self, providers) -> None:
        """Research and Script succeed, but the VoiceProvider fails - the
        failure must be surfaced in pipeline state without crashing, visual
        media generation must never run, and the failure must not be
        mistaken for a research/script/media failure."""
        search_provider, llm_provider, _, media_provider, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            ExplodingVoiceProvider(),
            TEST_VOICE_NAME,
            media_provider,
            voice_dir,
            media_dir,
        )

        assert state.status == "failed"
        assert state.error is not None
        assert "Voice generation failed" in state.error
        assert state.research_result is not None  # earlier stages' results are preserved
        assert state.script_result is not None
        # VoiceService never raises for synthesis failures - it returns a
        # structured failed VoiceResult, which the pipeline must preserve
        # rather than discard.
        assert state.voice_result is not None
        assert state.voice_result.success is False
        assert "simulated TTS outage" in state.voice_result.error
        # Media must never have been reached.
        assert state.visual_result is None
        assert media_provider.calls == []

    @pytest.mark.asyncio
    async def test_media_stage_failure_is_surfaced_cleanly(self, providers) -> None:
        """Research, Script, and Voice all succeed, but the MediaProvider
        fails - the failure must be surfaced in pipeline state without
        crashing, and earlier-stage results must be preserved."""
        search_provider, llm_provider, voice_provider, _, voice_dir, media_dir = providers
        state = await run_pipeline(
            "Why do humans dream?",
            search_provider,
            llm_provider,
            voice_provider,
            TEST_VOICE_NAME,
            ExplodingMediaProvider(),
            voice_dir,
            media_dir,
        )

        assert state.status == "failed"
        assert state.error is not None
        assert "Visual media generation failed" in state.error
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        # VisualMediaService never raises for search/download failures - it
        # returns a structured failed VisualResult, which the pipeline must
        # preserve rather than discard.
        assert state.visual_result is not None
        assert state.visual_result.success is False

    @pytest.mark.asyncio
    async def test_pipeline_state_defaults(self) -> None:
        state = PipelineState()
        assert state.topic == ""
        assert state.research_result is None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_result is None
        assert state.status == "pending"
        assert state.error is None

    @pytest.mark.asyncio
    async def test_pipeline_multiple_runs_are_independent(self, providers) -> None:
        search_provider, llm_provider, voice_provider, media_provider, voice_dir, media_dir = providers
        state_a = await run_pipeline(
            "Topic A", search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, voice_dir, media_dir
        )
        state_b = await run_pipeline(
            "Topic B", search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, voice_dir, media_dir
        )

        assert state_a.topic == "Topic A"
        assert state_b.topic == "Topic B"
        assert state_a.research_result.topic != state_b.research_result.topic
