# Tests for the Research -> Script -> Voice -> Visual Media -> Visual QC ->
# Video Assembly -> Subtitle/Caption pipeline workflow (LangGraph). All
# tests use mock providers/fake assembler/fake evaluator/fake transcription
# only - no real network/API/FFmpeg/Whisper calls.
from __future__ import annotations

import os

import pytest

from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.captions import CaptionResult
from src.models.media import VisualResult
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.models.visual_qc import RawAssetVerdict, VisualQCResult
from src.models.voice import VoiceResult
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.media_provider import MediaProvider, MockMediaProvider
from src.tools.search_provider import MockSearchProvider, SearchProvider
from src.tools.transcription_provider import MockTranscriptionProvider, TranscriptionProviderError
from src.tools.visual_relevance_evaluator import MockVisualRelevanceEvaluator, VisualRelevanceEvaluator
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
    of running real FFmpeg, so the pipeline's visual_qc/video_assembly/
    captions stages can be exercised without any real encoding/extraction/
    subtitle-burning process."""

    def __init__(
        self, audio_duration: float = 30.0, fail: bool = False, fail_burn_subtitles: bool = False
    ) -> None:
        self.audio_duration = audio_duration
        self.fail = fail
        self.fail_burn_subtitles = fail_burn_subtitles
        self.build_calls: list[dict] = []
        self.assemble_calls: list[dict] = []
        self.extract_frame_calls: list[dict] = []
        self.burn_subtitle_calls: list[dict] = []

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
        self.extract_frame_calls.append({"input_path": input_path, "timestamps_seconds": list(timestamps_seconds)})
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for index, _ in enumerate(timestamps_seconds):
            path = os.path.join(output_dir, f"{basename}-{index + 1:02d}.jpg")
            with open(path, "wb") as f:
                f.write(b"FAKE FRAME")
            paths.append(path)
        return paths

    def burn_subtitles(self, input_video_path, srt_path, output_path, force_style=None):
        self.burn_subtitle_calls.append(
            {"input_video_path": input_video_path, "srt_path": srt_path, "output_path": output_path}
        )
        if self.fail_burn_subtitles:
            raise VideoAssemblerError("simulated ffmpeg subtitle burn failure")
        with open(output_path, "wb") as f:
            f.write(b"FAKE CAPTIONED VIDEO")

    def mix_background_audio(self, *args, **kwargs):
        raise NotImplementedError("not exercised by pipeline workflow tests")


class ExplodingVisualRelevanceEvaluator(VisualRelevanceEvaluator):
    """Test double: evaluate_section always raises, to simulate a Gemini
    vision outage - exercises VisualQCService's metadata-fallback path."""

    @property
    def name(self) -> str:
        return "exploding"

    async def evaluate_section(self, context):
        raise RuntimeError("simulated vision QC outage")


class AlwaysMisleadingEvaluator(VisualRelevanceEvaluator):
    """Test double: every asset (including every bounded-replacement
    attempt) is flagged misleading, regardless of its id - guarantees QC
    exhausts replacement and ends with rejected_count > 0, so pipeline
    tests can exercise the hard-QC-failure path deterministically."""

    @property
    def name(self) -> str:
        return "always-misleading"

    async def evaluate_section(self, context):
        return [
            RawAssetVerdict(
                asset_id=asset.asset_id, relevance_score=0.9, misleading_or_conflicting=True, reason="always misleading"
            )
            for asset in context.assets
        ]


class FirstAttemptWeakEvaluator(VisualRelevanceEvaluator):
    """Test double: the first evaluate_section call for a given section
    marks its asset(s) weak (replacement recommended); every later call for
    that same section (i.e. the replacement re-check) approves. Lets
    pipeline tests deterministically exercise "QC replaced the asset, and
    the replacement changed what reaches Video Assembly"."""

    def __init__(self) -> None:
        self.seen_sections: set = set()

    @property
    def name(self) -> str:
        return "first-weak"

    async def evaluate_section(self, context):
        already_seen = context.section_index in self.seen_sections
        self.seen_sections.add(context.section_index)
        score = 0.95 if already_seen else 0.2
        reason = "great replacement" if already_seen else "weak first try"
        return [RawAssetVerdict(asset_id=a.asset_id, relevance_score=score, reason=reason) for a in context.assets]


class TestPipelineWorkflow:
    """Tests for the combined Research -> Script -> Voice -> Visual Media ->
    Visual QC -> Video Assembly -> Subtitle/Caption LangGraph pipeline."""

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
            MockVisualRelevanceEvaluator(default_score=0.9),
            MockTranscriptionProvider(),
            str(tmp_path / "audio"),
            str(tmp_path / "media"),
            str(tmp_path / "video"),
            str(tmp_path / "subtitles"),
        )

    @staticmethod
    def _run(providers, topic="Why do humans dream?", **overrides):
        (
            search_provider,
            llm_provider,
            voice_provider,
            media_provider,
            assembler,
            visual_relevance_evaluator,
            transcription_provider,
            voice_dir,
            media_dir,
            video_dir,
            subtitle_dir,
        ) = providers
        return run_pipeline(
            topic,
            overrides.get("search_provider", search_provider),
            overrides.get("llm_provider", llm_provider),
            overrides.get("voice_provider", voice_provider),
            TEST_VOICE_NAME,
            overrides.get("media_provider", media_provider),
            overrides.get("assembler", assembler),
            overrides.get("visual_relevance_evaluator", visual_relevance_evaluator),
            overrides.get("transcription_provider", transcription_provider),
            voice_dir,
            media_dir,
            video_dir,
            subtitle_dir,
        )

    @pytest.mark.asyncio
    async def test_pipeline_builds_and_compiles(self, providers) -> None:
        (
            search_provider, llm_provider, voice_provider, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        ) = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, voice_dir, media_dir, video_dir, subtitle_dir,
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
        _, _, voice_provider, _, _, _, _, _, _, _, _ = providers
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

    # ---- B. VisualQCResult stored in final state ---------------------------

    @pytest.mark.asyncio
    async def test_visual_qc_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.visual_qc_result, VisualQCResult)
        assert state.visual_qc_result.success is True
        assert state.visual_qc_result.total_assets_checked > 0
        assert state.visual_qc_result.rejected_count == 0
        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_qc_approved_visual_result_stored_alongside_original(self, providers) -> None:
        """The pre-QC VisualResult (visual_result) must never be silently
        replaced - the post-QC one is a separate field."""
        state = await self._run(providers)

        assert isinstance(state.qc_approved_visual_result, VisualResult)
        assert state.visual_result is not None
        assert state.qc_approved_visual_result is not state.visual_result

    # ---- C. VideoAssemblyResult stored in final state ----------------------

    @pytest.mark.asyncio
    async def test_video_assembly_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.video_assembly_result, VideoAssemblyResult)
        assert state.video_assembly_result.success is True
        assert state.video_assembly_result.output_path is not None
        assert os.path.exists(state.video_assembly_result.output_path)
        assert state.video_assembly_result.section_count == len(state.script_result.sections)

    # ---- D. Video Assembly receives the QC-approved media mapping ----------

    @pytest.mark.asyncio
    async def test_video_assembly_receives_qc_approved_inputs(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers)

        # The assembler was called with the exact media files from the
        # post-QC qc_approved_visual_result - not any other object.
        expected_paths = {
            asset.local_file_path
            for mapping in state.qc_approved_visual_result.sections
            for asset in mapping.assets
        }
        actual_paths = {call["input_path"] for call in assembler.build_calls}
        assert actual_paths == expected_paths

        # The assembler received the exact narration audio VoiceService produced.
        assert assembler.assemble_calls[0]["audio_path"] == state.voice_result.audio_file_path

        # Section durations sum to the (fake) narration audio duration.
        assert sum(c["target_duration_seconds"] for c in assembler.build_calls) == pytest.approx(
            assembler.audio_duration, abs=0.01
        )

    @pytest.mark.asyncio
    async def test_qc_replacement_changes_media_passed_to_video_assembly(self, providers) -> None:
        """When Visual QC replaces a weak asset, Video Assembly must
        receive the replacement - not the originally-selected asset."""
        _, _, _, _, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=FirstAttemptWeakEvaluator())

        assert state.status == "completed"
        assert state.visual_qc_result.replaced_count == len(state.script_result.sections)

        original_ids = {
            asset.provider_asset_id for m in state.visual_result.sections for asset in m.assets
        }
        approved_ids = {
            asset.provider_asset_id for m in state.qc_approved_visual_result.sections for asset in m.assets
        }
        assert original_ids != approved_ids

        # Video Assembly built clips from the replacement paths, not the
        # originally-selected ones.
        original_paths = {a.local_file_path for m in state.visual_result.sections for a in m.assets}
        assembled_paths = {c["input_path"] for c in assembler.build_calls}
        assert assembled_paths.isdisjoint(original_paths)

    # ---- E. Visual QC not called on earlier-stage failure ------------------

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_research_fails(self, providers) -> None:
        _, _, voice_provider, media_provider, assembler, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers, topic="")

        assert state.status == "failed"
        assert state.research_result is None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_result is None
        assert state.visual_qc_result is None
        assert state.qc_approved_visual_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []
        assert assembler.extract_frame_calls == []
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_script_fails(self, providers) -> None:
        _, _, voice_provider, media_provider, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers, llm_provider=ExplodingLLMProvider())

        assert state.status == "failed"
        assert state.script_result is None
        assert state.visual_qc_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_voice_fails(self, providers) -> None:
        _, _, _, media_provider, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers, voice_provider=ExplodingVoiceProvider())

        assert state.status == "failed"
        assert "Voice generation failed" in state.error
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is False
        assert state.visual_result is None
        assert state.visual_qc_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_visual_media_fails(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers, media_provider=ExplodingMediaProvider())

        assert state.status == "failed"
        assert "Visual media generation failed" in state.error
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_result.success is False
        assert state.visual_qc_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []
        assert assembler.extract_frame_calls == []

    # ---- F. QC failure/fallback policy -------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_fallback_approval_continues_pipeline(self, providers) -> None:
        """A vision-evaluator outage must not fail the pipeline - QC falls
        back to the metadata filter's prior approval and continues."""
        state = await self._run(providers, visual_relevance_evaluator=ExplodingVisualRelevanceEvaluator())

        assert state.status == "completed"
        assert state.visual_qc_result.success is True
        assert state.visual_qc_result.fallback_used is True
        assert state.visual_qc_result.rejected_count == 0
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True

        all_assets = [a for s in state.visual_qc_result.sections for a in s.assets]
        assert all(a.evaluation_source == "metadata_fallback" for a in all_assets)

    @pytest.mark.asyncio
    async def test_qc_fallback_reason_preserved_in_state(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=ExplodingVisualRelevanceEvaluator())

        assert state.visual_qc_result.fallback_reason is not None
        assert "simulated vision QC outage" in state.visual_qc_result.fallback_reason

    @pytest.mark.asyncio
    async def test_video_assembly_not_called_after_hard_qc_failure(self, providers) -> None:
        """An asset still flagged misleading after bounded replacement is
        exhausted must stop the pipeline before Video Assembly - never
        reaching the final video (or captions)."""
        _, _, _, _, assembler, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert "rejected" in state.error.lower()
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_rejected_assets_not_passed_to_video_assembly(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.visual_qc_result.rejected_count > 0
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_earlier_stage_results_preserved_after_hard_qc_failure(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_result.success is True
        # QC's own results are preserved for inspection even on hard failure.
        assert state.visual_qc_result is not None
        assert state.qc_approved_visual_result is not None

    # ---- G. Video Assembly not called on its own upstream failure ----------

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
        assert state.visual_qc_result is not None
        assert state.visual_qc_result.success is True
        assert state.qc_approved_visual_result is not None
        # VideoAssemblyService never raises for processing failures - it
        # returns a structured failed VideoAssemblyResult.
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is False
        assert "simulated ffmpeg outage" in state.video_assembly_result.error
        # Captions never run on a failed/missing assembled video.
        assert state.caption_result is None

    # ---- H. status only "completed" when captioning succeeds after assembly -

    @pytest.mark.asyncio
    async def test_status_is_completed_only_when_captioning_succeeds(self, providers) -> None:
        success_state = await self._run(providers)
        assert success_state.status == "completed"
        assert success_state.video_assembly_result.success is True
        assert success_state.caption_result.success is True

        failure_state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))
        assert failure_state.status == "failed"

    @pytest.mark.asyncio
    async def test_status_is_failed_not_completed_after_hard_qc_failure(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())
        assert state.status == "failed"
        assert state.status != "completed"

    # ---- I. existing behavior preserved -------------------------------------

    @pytest.mark.asyncio
    async def test_no_search_results_still_completes_through_captions(self, providers) -> None:
        """MockSearchProvider returning [] is a valid (if sparse) research result,
        not an error - the pipeline should still complete all the way through
        captioning."""
        state = await self._run(providers, topic="obscure topic", search_provider=EmptySearchProvider())

        assert state.research_result is not None
        assert state.research_result.sources == []
        assert state.status == "completed"
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert state.caption_result is not None
        assert state.caption_result.success is True

    @pytest.mark.asyncio
    async def test_pipeline_state_defaults(self) -> None:
        state = PipelineState()
        assert state.topic == ""
        assert state.research_result is None
        assert state.script_result is None
        assert state.voice_result is None
        assert state.visual_plan is None
        assert state.visual_result is None
        assert state.visual_qc_result is None
        assert state.qc_approved_visual_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert state.status == "pending"
        assert state.error is None

    @pytest.mark.asyncio
    async def test_pipeline_multiple_runs_are_independent(self, providers) -> None:
        state_a = await self._run(providers, topic="Topic A")
        state_b = await self._run(providers, topic="Topic B")

        assert state_a.topic == "Topic A"
        assert state_b.topic == "Topic B"
        assert state_a.research_result.topic != state_b.research_result.topic

    # ---- J. CaptionResult stored / correct inputs ---------------------------

    @pytest.mark.asyncio
    async def test_caption_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.caption_result, CaptionResult)
        assert state.caption_result.success is True
        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_caption_service_receives_correct_voice_audio_path(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers)

        assert transcription_provider.calls == [state.voice_result.audio_file_path]

    @pytest.mark.asyncio
    async def test_caption_service_receives_correct_video_assembly_path(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _ = providers
        state = await self._run(providers)

        assert len(assembler.burn_subtitle_calls) == 1
        assert assembler.burn_subtitle_calls[0]["input_video_path"] == state.video_assembly_result.output_path

    @pytest.mark.asyncio
    async def test_original_video_assembly_result_preserved_after_captioning(self, providers) -> None:
        state = await self._run(providers)

        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert os.path.exists(state.video_assembly_result.output_path)

    @pytest.mark.asyncio
    async def test_captioned_output_stored_separately_from_original(self, providers) -> None:
        state = await self._run(providers)

        assert state.caption_result.captioned_video_path is not None
        assert state.caption_result.captioned_video_path != state.video_assembly_result.output_path
        assert os.path.exists(state.caption_result.captioned_video_path)
        assert state.caption_result.srt_path is not None
        assert os.path.exists(state.caption_result.srt_path)

    # ---- K. Caption node not called on earlier failure -----------------------

    @pytest.mark.asyncio
    async def test_caption_node_not_called_when_video_assembly_fails(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))

        assert state.status == "failed"
        assert state.caption_result is None
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_caption_node_not_called_after_hard_qc_failure(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert state.caption_result is None
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_caption_node_not_called_when_research_fails(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _ = providers
        state = await self._run(providers, topic="")

        assert state.caption_result is None
        assert transcription_provider.calls == []

    # ---- L. Caption failure behavior -----------------------------------------

    @pytest.mark.asyncio
    async def test_caption_burn_failure_marks_pipeline_failed(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.status == "failed"
        assert "Caption generation failed" in state.error

    @pytest.mark.asyncio
    async def test_caption_burn_failure_preserves_original_mp4(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert os.path.exists(state.video_assembly_result.output_path)

    @pytest.mark.asyncio
    async def test_caption_burn_failure_preserves_earlier_results_and_stores_failed_caption_result(
        self, providers
    ) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.visual_result is not None
        assert state.visual_qc_result is not None
        assert state.qc_approved_visual_result is not None
        assert state.video_assembly_result is not None
        assert state.caption_result is not None
        assert state.caption_result.success is False
        assert state.caption_result.error is not None

    @pytest.mark.asyncio
    async def test_transcription_failure_marks_pipeline_failed_and_preserves_original_mp4(self, providers) -> None:
        failing_provider = MockTranscriptionProvider(raise_error=TranscriptionProviderError("engine crashed"))
        state = await self._run(providers, transcription_provider=failing_provider)

        assert state.status == "failed"
        assert "Caption generation failed" in state.error
        assert state.caption_result is not None
        assert state.caption_result.success is False
        assert state.video_assembly_result.success is True
        assert os.path.exists(state.video_assembly_result.output_path)
