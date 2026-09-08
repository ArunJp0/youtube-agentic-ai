# Tests for the Research -> Script -> Voice -> Visual Media -> Visual QC ->
# Video Assembly -> Subtitle/Caption -> BGM/Audio Mixing pipeline workflow
# (LangGraph). All tests use mock providers/fake assembler/fake evaluator/
# fake transcription/mock BGM catalog only - no real network/API/FFmpeg/
# Whisper/Gemini calls.
from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.captions import CaptionResult
from src.models.media import VisualResult
from src.models.metadata import MetadataResult
from src.models.music import AudioMixResult, BGMTrack
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailResult
from src.models.video import VideoAssemblyResult
from src.models.visual_qc import RawAssetVerdict, VisualQCResult
from src.models.voice import VoiceResult
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.media_provider import MediaProvider, MockMediaProvider
from src.tools.music_catalog_provider import MockMusicCatalogProvider
from src.tools.search_provider import MockSearchProvider, SearchProvider
from src.tools.transcription_provider import MockTranscriptionProvider, TranscriptionProviderError
from src.tools.visual_relevance_evaluator import MockVisualRelevanceEvaluator, VisualRelevanceEvaluator
from src.tools.voice_provider import MockVoiceProvider, VoiceProvider
from src.workflows.pipeline_graph import PipelineState, build_pipeline_graph, run_pipeline

TEST_VOICE_NAME = "test-voice"

# MusicContextPlanner's prompt always opens with this line (see
# src/agents/music_context_planner.py) - a unique marker that lets test
# doubles distinguish the BGM mood-planning call from Research/Script
# prompts sharing the same LLMProvider.
_MUSIC_PROMPT_MARKER = "BACKGROUND MUSIC CHARACTERISTICS"

# MetadataAgent's prompt always mentions this phrase (see
# src/agents/metadata_agent.py) - a unique marker that lets test doubles
# distinguish the metadata-generation call from Research/Script/BGM
# prompts sharing the same LLMProvider.
_METADATA_PROMPT_MARKER = "YouTube upload metadata"

# ThumbnailPlanner's prompt always opens with this phrase (see
# src/agents/thumbnail_planner.py) - a unique marker that lets test doubles
# distinguish the thumbnail-planning call from Research/Script/BGM/Metadata
# prompts sharing the same LLMProvider.
_THUMBNAIL_PROMPT_MARKER = "planning a YouTube thumbnail CONCEPT"


def _default_thumbnail_plan_json(**overrides) -> str:
    """A valid ThumbnailPlanner response payload."""
    payload = {
        "hook_text": "WHY DO WE DREAM",
        "visual_concept": "A sleeping person with dream imagery",
        "search_query": "person sleeping dreaming",
        "mood": "curious",
        "subject": "a sleeping person",
        "composition": "subject_left",
        "text_position": "right",
        "avoid_concepts": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _default_metadata_json(**overrides) -> str:
    """A valid MetadataAgent response payload - deliberately omits
    chapter_labels so MetadataAgent's own section-heading fallback fills
    every chapter title, regardless of how many real sections the default
    fixture's mock Research->Script pipeline happens to produce."""
    payload = {
        "title": "Why Do Humans Dream? The Science Explained",
        "description": "A grounded look at the real science of dreaming, covering what current research says.",
        "seo_summary": "Learn what science says about why humans dream.",
        "tags": ["dreams", "sleep science"],
        "hashtags": ["#dreams", "#sleep"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class EmptySearchProvider(SearchProvider):
    """Test double: search always returns no results."""

    async def search(self, query: str, num_results: int = 5):
        return []


class RecordingSearchProvider(SearchProvider):
    """Wraps MockSearchProvider and records every query - lets tests prove
    Research is never re-triggered by a later stage (e.g. BGM mood
    planning), since ResearchAgent is the only caller of SearchProvider."""

    def __init__(self) -> None:
        self._delegate = MockSearchProvider()
        self.calls: list[str] = []

    async def search(self, query: str, num_results: int = 5):
        self.calls.append(query)
        return await self._delegate.search(query, num_results)


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

    Never returns valid JSON for MusicContextPlanner's mood-planning
    prompt, so the default pipeline fixture exercises BGM's deterministic
    fallback path (see MusicPlanningLLMProvider below for the successful-
    semantic-planning counterpart). DOES return a valid MetadataAgent
    response by default (metadata generation has no deterministic content
    fallback, so the default fixture must produce something usable for
    every "full pipeline success" test to mean anything).
    """

    def __init__(self) -> None:
        self._mock = MockLLMProvider()

    def generate_text(self, prompt: str) -> str:
        if "Point to expand on: '" in prompt:
            point = prompt.split("Point to expand on: '", 1)[1].split("'.", 1)[0]
            return f"{point}. A distinct detail worth covering on its own."
        if _METADATA_PROMPT_MARKER in prompt:
            return _default_metadata_json()
        return self._mock.generate_text(prompt)


class RecordingLLMProvider(LLMProvider):
    """Wraps ResearchMockWithVariedSections and records every prompt seen -
    lets tests inspect exactly what was sent to the shared LLMProvider
    (e.g. to confirm the BGM mood-planning prompt reflects the real,
    already-produced ScriptResult) without needing a fully scripted fake."""

    def __init__(self) -> None:
        self._delegate = ResearchMockWithVariedSections()
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._delegate.generate_text(prompt)


class MusicPlanningLLMProvider(LLMProvider):
    """Delegates Research/Script prompts to ResearchMockWithVariedSections
    (unchanged, realistic behavior), but returns a valid structured
    MusicPlan JSON response for MusicContextPlanner's distinctive prompt -
    exercising the successful semantic mood-planning path deterministically."""

    def __init__(self) -> None:
        self._delegate = ResearchMockWithVariedSections()
        self.music_plan_calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        if _MUSIC_PROMPT_MARKER in prompt:
            self.music_plan_calls.append(prompt)
            return json.dumps(
                {
                    "primary_mood": "thoughtful",
                    "secondary_mood": "calm",
                    "energy_level": "low",
                    "preferred_genres": ["ambient"],
                    "preferred_instrumentation": ["piano"],
                    "avoid_styles": ["aggressive"],
                    "requires_neutral_subtle": True,
                    "reasoning_summary": "test reasoning",
                }
            )
        return self._delegate.generate_text(prompt)


class ConfigurableMetadataLLMProvider(LLMProvider):
    """Delegates Research/Script/BGM prompts to ResearchMockWithVariedSections
    unchanged, but lets a test control exactly what MetadataAgent's prompt
    receives (a custom payload, or a raised error) without touching any
    earlier stage - isolates metadata-only success/failure scenarios."""

    def __init__(self, metadata_payload: dict | None = None, metadata_error: Exception | None = None) -> None:
        self._delegate = ResearchMockWithVariedSections()
        self.metadata_payload = metadata_payload
        self.metadata_error = metadata_error
        self.metadata_calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        if _METADATA_PROMPT_MARKER in prompt:
            self.metadata_calls.append(prompt)
            if self.metadata_error is not None:
                raise self.metadata_error
            return _default_metadata_json(**(self.metadata_payload or {}))
        return self._delegate.generate_text(prompt)


class ConfigurableThumbnailLLMProvider(LLMProvider):
    """Delegates Research/Script/BGM/Metadata prompts to
    ResearchMockWithVariedSections unchanged (including its default valid
    MetadataAgent response), but lets a test control exactly what
    ThumbnailPlanner's prompt receives (a custom payload, or a raised
    error) without touching any earlier stage - isolates thumbnail-only
    success/failure/hook-quality scenarios."""

    def __init__(self, thumbnail_payload: dict | None = None, thumbnail_error: Exception | None = None) -> None:
        self._delegate = ResearchMockWithVariedSections()
        self.thumbnail_payload = thumbnail_payload
        self.thumbnail_error = thumbnail_error
        self.thumbnail_calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        if _THUMBNAIL_PROMPT_MARKER in prompt:
            self.thumbnail_calls.append(prompt)
            if self.thumbnail_error is not None:
                raise self.thumbnail_error
            return _default_thumbnail_plan_json(**(self.thumbnail_payload or {}))
        return self._delegate.generate_text(prompt)


class RecordingMetadataAndThumbnailLLMProvider(LLMProvider):
    """Lets a test set a distinctive metadata title and records every
    prompt seen - proving the Thumbnail stage's prompt reflects that exact
    in-memory MetadataResult.title rather than anything reloaded from disk
    (no ScriptResult/MetadataResult is ever reloaded from JSON by the real
    pipeline - that reconstruction fallback exists only for standalone
    demos, which have no PipelineState to read a real result from)."""

    def __init__(self, metadata_title: str) -> None:
        self._delegate = ResearchMockWithVariedSections()
        self.metadata_title = metadata_title
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if _METADATA_PROMPT_MARKER in prompt:
            return _default_metadata_json(title=self.metadata_title)
        return self._delegate.generate_text(prompt)


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


class ThumbnailCapableMediaProvider(MediaProvider):
    """Wraps MockMediaProvider but writes a real, valid, small Pillow image
    on download instead of MockMediaProvider's raw placeholder bytes.

    The full pipeline's Thumbnail stage (unlike the FFmpeg-based Visual
    Media/Video Assembly stages, which never inspect file content because
    FakeVideoAssembler fakes encoding entirely) decodes the downloaded
    image with real Pillow - so the default pipeline fixture needs a
    media provider whose "downloaded" files are genuinely openable images.
    """

    def __init__(self, **kwargs) -> None:
        self._delegate = MockMediaProvider(**kwargs)

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def calls(self):
        return self._delegate.calls

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        return await self._delegate.search(query, prefer_video, max_results)

    async def download(self, candidate, output_path: str) -> None:
        self._delegate.calls.append(("download", candidate.download_url))
        Image.new("RGB", (1920, 1080), (100, 120, 140)).save(output_path, format="JPEG")


class FailThumbnailSearchMediaProvider(ThumbnailCapableMediaProvider):
    """Behaves exactly like ThumbnailCapableMediaProvider (Visual Media's
    own per-section searches succeed normally with real downloadable
    images), except a search whose query is the raw, unmodified topic
    string always fails - the Thumbnail stage's own deterministic-fallback
    search query (see build_deterministic_thumbnail_plan) is always the
    topic verbatim, letting this test double fail ONLY the Thumbnail
    stage's image search without touching Visual Media's earlier, already-
    succeeded search calls."""

    def __init__(self, topic: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.topic = topic

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        if query == self.topic:
            raise RuntimeError("simulated thumbnail image search outage")
        return await super().search(query, prefer_video, max_results)


class FakeVideoAssembler(VideoAssembler):
    """Test double: records calls and writes tiny placeholder files instead
    of running real FFmpeg, so the pipeline's visual_qc/video_assembly/
    captions/bgm stages can be exercised without any real encoding/
    extraction/subtitle-burning/audio-mixing process."""

    def __init__(
        self,
        audio_duration: float = 30.0,
        fail: bool = False,
        fail_burn_subtitles: bool = False,
        fail_mix_background_audio: bool = False,
    ) -> None:
        self.audio_duration = audio_duration
        self.fail = fail
        self.fail_burn_subtitles = fail_burn_subtitles
        self.fail_mix_background_audio = fail_mix_background_audio
        self.build_calls: list[dict] = []
        self.assemble_calls: list[dict] = []
        self.extract_frame_calls: list[dict] = []
        self.burn_subtitle_calls: list[dict] = []
        self.mix_background_audio_calls: list[dict] = []

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
        self.mix_background_audio_calls.append(
            {
                "input_video_path": input_video_path,
                "music_path": music_path,
                "output_path": output_path,
                "target_duration_seconds": target_duration_seconds,
                "music_gain_db": music_gain_db,
                "use_ducking": use_ducking,
            }
        )
        if self.fail_mix_background_audio:
            raise VideoAssemblerError("simulated ffmpeg BGM mixing failure")
        with open(output_path, "wb") as f:
            f.write(b"FAKE BGM MIXED VIDEO")


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


def _music_catalog_provider(tmp_path, track_id: str = "calm-test-track", write_file: bool = True) -> MockMusicCatalogProvider:
    """One approved, instrumental, neutral-tagged test track - passes the
    deterministic fallback MusicPlan's avoid_styles filter by construction
    (see src/services/music_planning.py's FALLBACK_AVOID_STYLES)."""
    track_path = tmp_path / "bgm-tracks" / f"{track_id}.mp3"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    if write_file:
        track_path.write_bytes(b"FAKE BGM TRACK AUDIO")
    return MockMusicCatalogProvider(
        tracks=[
            BGMTrack(
                track_id=track_id,
                file_path=str(track_path),
                title="Calm Test Track",
                source="Test Fixture",
                license_type="test_license",
                instrumental=True,
                mood_tags=["calm", "neutral", "subtle"],
                energy_level="low",
            )
        ]
    )


class TestPipelineWorkflow:
    """Tests for the combined Research -> Script -> Voice -> Visual Media ->
    Visual QC -> Video Assembly -> Subtitle/Caption -> BGM/Audio Mixing
    LangGraph pipeline."""

    @pytest.fixture
    def providers(self, tmp_path):
        # All output dirs under tmp_path so tests never write into the real
        # project output/ directories.
        return (
            MockSearchProvider(),
            ResearchMockWithVariedSections(),
            MockVoiceProvider(),
            ThumbnailCapableMediaProvider(),
            FakeVideoAssembler(),
            MockVisualRelevanceEvaluator(default_score=0.9),
            MockTranscriptionProvider(),
            _music_catalog_provider(tmp_path),
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
            music_catalog_provider,
            voice_dir,
            media_dir,
            video_dir,
            subtitle_dir,
        ) = providers
        # Derived from voice_dir (rather than adding more fixture elements) so
        # every existing positional-unpacking of `providers` throughout this
        # file keeps working unchanged - still always under tmp_path, never
        # the real project output/ directory. metadata_dir/provenance_dir
        # matter just as much as thumbnail_dir here: without them,
        # MetadataAgent/the provenance manifest writer would otherwise fall
        # back to their real DEFAULT_*_OUTPUT_DIR constants and write test
        # artifacts into the actual project output/ directories.
        thumbnail_dir = os.path.join(os.path.dirname(voice_dir), "thumbnails")
        metadata_dir = os.path.join(os.path.dirname(voice_dir), "metadata")
        provenance_dir = os.path.join(os.path.dirname(voice_dir), "provenance")
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
            overrides.get("music_catalog_provider", music_catalog_provider),
            voice_dir,
            media_dir,
            video_dir,
            subtitle_dir,
            overrides.get("thumbnail_output_dir", thumbnail_dir),
            overrides.get("metadata_output_dir", metadata_dir),
            overrides.get("provenance_output_dir", provenance_dir),
        )

    @pytest.mark.asyncio
    async def test_pipeline_builds_and_compiles(self, providers) -> None:
        (
            search_provider, llm_provider, voice_provider, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        ) = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
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
        _, _, voice_provider, _, _, _, _, _, _, _, _, _ = providers
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
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
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
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
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
        _, _, voice_provider, media_provider, assembler, _, transcription_provider, _, _, _, _, _ = providers
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
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []
        assert assembler.extract_frame_calls == []
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_script_fails(self, providers) -> None:
        _, _, voice_provider, media_provider, assembler, _, _, _, _, _, _, _ = providers
        state = await self._run(providers, llm_provider=ExplodingLLMProvider())

        assert state.status == "failed"
        assert state.script_result is None
        assert state.visual_qc_result is None
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert voice_provider.calls == []
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_voice_fails(self, providers) -> None:
        _, _, _, media_provider, assembler, _, _, _, _, _, _, _ = providers
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
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert media_provider.calls == []
        assert assembler.build_calls == []

    @pytest.mark.asyncio
    async def test_visual_qc_not_called_when_visual_media_fails(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
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
        assert state.audio_mix_result is None
        assert state.metadata_result is None
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
        reaching the final video (or captions, or BGM)."""
        _, _, _, _, assembler, _, transcription_provider, _, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert "rejected" in state.error.lower()
        assert state.video_assembly_result is None
        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert assembler.build_calls == []
        assert assembler.assemble_calls == []
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_rejected_assets_not_passed_to_video_assembly(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
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
        # Captions/BGM never run on a failed/missing assembled video.
        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None

    # ---- H. status only "completed" when metadata generation succeeds after BGM -

    @pytest.mark.asyncio
    async def test_status_is_completed_only_when_metadata_generation_succeeds(self, providers) -> None:
        success_state = await self._run(providers)
        assert success_state.status == "completed"
        assert success_state.video_assembly_result.success is True
        assert success_state.caption_result.success is True
        assert success_state.audio_mix_result.success is True
        assert success_state.metadata_result.success is True

        failure_state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))
        assert failure_state.status == "failed"

    @pytest.mark.asyncio
    async def test_bgm_success_alone_no_longer_marks_pipeline_completed(self, providers) -> None:
        """BGM succeeding is necessary but not sufficient - a metadata
        failure after a successful BGM mix must NOT be reported as
        "completed"; the pipeline's final deliverable now includes metadata."""
        state = await self._run(
            providers, llm_provider=ConfigurableMetadataLLMProvider(metadata_error=RuntimeError("metadata outage"))
        )

        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True
        assert state.status == "failed"
        assert state.status != "completed"

    @pytest.mark.asyncio
    async def test_status_is_failed_not_completed_after_hard_qc_failure(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())
        assert state.status == "failed"
        assert state.status != "completed"

    # ---- I. existing behavior preserved -------------------------------------

    @pytest.mark.asyncio
    async def test_no_search_results_still_completes_through_metadata(self, providers) -> None:
        """MockSearchProvider returning [] is a valid (if sparse) research result,
        not an error - the pipeline should still complete all the way through
        metadata generation."""
        state = await self._run(providers, topic="obscure topic", search_provider=EmptySearchProvider())

        assert state.research_result is not None
        assert state.research_result.sources == []
        assert state.status == "completed"
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert state.caption_result is not None
        assert state.caption_result.success is True
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True
        assert state.metadata_result is not None
        assert state.metadata_result.success is True

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
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert state.metadata_result is None
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
        _, _, _, _, _, _, transcription_provider, _, _, _, _, _ = providers
        state = await self._run(providers)

        assert transcription_provider.calls == [state.voice_result.audio_file_path]

    @pytest.mark.asyncio
    async def test_caption_service_receives_correct_video_assembly_path(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
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
        _, _, _, _, _, _, transcription_provider, _, _, _, _, _ = providers
        state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))

        assert state.status == "failed"
        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_caption_node_not_called_after_hard_qc_failure(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _, _ = providers
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert transcription_provider.calls == []

    @pytest.mark.asyncio
    async def test_caption_node_not_called_when_research_fails(self, providers) -> None:
        _, _, _, _, _, _, transcription_provider, _, _, _, _, _ = providers
        state = await self._run(providers, topic="")

        assert state.caption_result is None
        assert state.audio_mix_result is None
        assert state.metadata_result is None
        assert transcription_provider.calls == []

    # ---- L. Caption failure behavior -----------------------------------------

    @pytest.mark.asyncio
    async def test_caption_burn_failure_marks_pipeline_failed(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.status == "failed"
        assert "Caption generation failed" in state.error
        assert state.audio_mix_result is None
        assert state.metadata_result is None

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
        assert state.audio_mix_result is None
        assert state.metadata_result is None

    # ---- M. AudioMixResult stored / correct inputs ---------------------------

    @pytest.mark.asyncio
    async def test_audio_mix_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.audio_mix_result, AudioMixResult)
        assert state.audio_mix_result.success is True
        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_bgm_node_receives_existing_script_result_from_state(self, providers) -> None:
        """The BGM mood-planning prompt must reflect the SAME ScriptResult
        the pipeline actually produced earlier - proving the existing
        PipelineState.script_result is what's passed through, not a
        freshly (re)generated one."""
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, llm_provider=recording_llm)

        music_prompts = [c for c in recording_llm.calls if _MUSIC_PROMPT_MARKER in c]
        assert len(music_prompts) == 1
        assert state.script_result.video_title in music_prompts[0]
        assert state.script_result.sections[0].heading in music_prompts[0]

    @pytest.mark.asyncio
    async def test_bgm_node_receives_correct_captioned_mp4(self, providers) -> None:
        _, _, _, _, assembler, _, _, _, _, _, _, _ = providers
        state = await self._run(providers)

        assert len(assembler.mix_background_audio_calls) == 1
        assert (
            assembler.mix_background_audio_calls[0]["input_video_path"]
            == state.caption_result.captioned_video_path
        )
        # BGM mixes onto the captioned MP4, never the pre-caption assembly.
        assert (
            assembler.mix_background_audio_calls[0]["input_video_path"]
            != state.video_assembly_result.output_path
        )

    @pytest.mark.asyncio
    async def test_no_research_or_script_regeneration_inside_bgm_node(self, providers) -> None:
        recording_search = RecordingSearchProvider()
        await self._run(providers, search_provider=recording_search)

        # ResearchAgent is the only caller of SearchProvider.search() in the
        # whole pipeline - if BGM (or anything else) re-ran Research, this
        # would be > 1.
        assert len(recording_search.calls) == 1

    @pytest.mark.asyncio
    async def test_caption_result_remains_preserved_after_bgm_stage(self, providers) -> None:
        state = await self._run(providers)

        assert state.caption_result is not None
        assert state.caption_result.success is True
        assert os.path.exists(state.caption_result.captioned_video_path)

    # ---- N. Semantic MusicPlan / deterministic fallback ----------------------

    @pytest.mark.asyncio
    async def test_successful_semantic_music_plan_path(self, providers) -> None:
        llm_provider = MusicPlanningLLMProvider()
        state = await self._run(providers, llm_provider=llm_provider)

        assert len(llm_provider.music_plan_calls) == 1
        assert state.audio_mix_result.success is True
        assert state.audio_mix_result.music_plan.used_semantic_planning is True
        assert state.audio_mix_result.music_plan.primary_mood == "thoughtful"
        assert state.audio_mix_result.music_plan.fallback_reason is None

    @pytest.mark.asyncio
    async def test_semantic_planner_failure_falls_back_and_pipeline_still_succeeds(self, providers) -> None:
        """The default fixture's LLM never returns valid JSON for the BGM
        mood-planning prompt - MusicContextPlanner must fall back to the
        deterministic MusicPlan, and mixing/the pipeline must still succeed."""
        state = await self._run(providers)

        assert state.status == "completed"
        assert state.audio_mix_result.success is True
        assert state.audio_mix_result.music_plan.used_semantic_planning is False
        assert state.audio_mix_result.music_plan.fallback_reason is not None

    @pytest.mark.asyncio
    async def test_fallback_used_recorded_in_music_plan(self, providers) -> None:
        state = await self._run(providers)

        plan = state.audio_mix_result.music_plan
        assert plan.used_semantic_planning is False
        assert plan.fallback_reason is not None
        assert plan.primary_mood  # fallback plan is still fully usable

    @pytest.mark.asyncio
    async def test_correct_approved_track_passed_to_audio_mixing_service(self, providers) -> None:
        state = await self._run(providers)

        assert state.audio_mix_result.selected_track is not None
        assert state.audio_mix_result.selected_track.track_id == "calm-test-track"
        assert state.audio_mix_result.selected_track.instrumental is True

    @pytest.mark.asyncio
    async def test_final_mixed_mp4_becomes_final_pipeline_output(self, providers) -> None:
        state = await self._run(providers)

        assert state.audio_mix_result.output_path is not None
        assert os.path.exists(state.audio_mix_result.output_path)
        assert state.audio_mix_result.output_path != state.caption_result.captioned_video_path

    # ---- O. BGM node not called after earlier hard failure -------------------

    @pytest.mark.asyncio
    async def test_bgm_node_not_called_when_caption_fails(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.status == "failed"
        assert state.audio_mix_result is None
        assert state.metadata_result is None

    @pytest.mark.asyncio
    async def test_bgm_node_not_called_after_hard_qc_failure(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert state.audio_mix_result is None
        assert state.metadata_result is None

    @pytest.mark.asyncio
    async def test_bgm_node_not_called_when_video_assembly_fails(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail=True))

        assert state.status == "failed"
        assert state.audio_mix_result is None
        assert state.metadata_result is None

    # ---- P. BGM failure behavior ----------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_music_catalog_marks_pipeline_failed(self, providers) -> None:
        state = await self._run(providers, music_catalog_provider=MockMusicCatalogProvider(tracks=[]))

        assert state.status == "failed"
        assert "BGM mixing failed" in state.error
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is False

    @pytest.mark.asyncio
    async def test_missing_music_file_marks_pipeline_failed(self, providers, tmp_path) -> None:
        missing_catalog = _music_catalog_provider(tmp_path, track_id="missing-track", write_file=False)
        state = await self._run(providers, music_catalog_provider=missing_catalog)

        assert state.status == "failed"
        assert "BGM mixing failed" in state.error
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is False
        assert state.audio_mix_result.selected_track is not None

    @pytest.mark.asyncio
    async def test_mixing_ffmpeg_failure_marks_pipeline_failed(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_mix_background_audio=True))

        assert state.status == "failed"
        assert "BGM mixing failed" in state.error
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is False

    @pytest.mark.asyncio
    async def test_bgm_failure_preserves_captioned_mp4(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_mix_background_audio=True))

        assert state.caption_result is not None
        assert state.caption_result.success is True
        assert os.path.exists(state.caption_result.captioned_video_path)

    @pytest.mark.asyncio
    async def test_earlier_stage_state_preserved_after_bgm_failure(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_mix_background_audio=True))

        assert state.status == "failed"
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_qc_result is not None
        assert state.qc_approved_visual_result is not None
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert state.caption_result is not None
        assert state.caption_result.success is True

    # ---- Q. Metadata: stored / correct inputs / routing -----------------------

    @pytest.mark.asyncio
    async def test_metadata_node_exists_and_pipeline_builds(self, providers) -> None:
        (
            search_provider, llm_provider, voice_provider, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        ) = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        )
        compiled = graph.compile()
        assert "metadata" in compiled.get_graph().nodes

    @pytest.mark.asyncio
    async def test_metadata_result_is_structured_and_stored_in_final_state(self, providers) -> None:
        state = await self._run(providers)

        assert isinstance(state.metadata_result, MetadataResult)
        assert state.metadata_result.success is True
        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_metadata_runs_only_after_bgm_succeeds(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider()
        state = await self._run(providers, llm_provider=provider)

        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True
        assert len(provider.metadata_calls) == 1
        assert state.metadata_result is not None
        assert state.metadata_result.success is True

    @pytest.mark.asyncio
    async def test_metadata_receives_original_topic(self, providers) -> None:
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, topic="Why do humans dream?", llm_provider=recording_llm)

        metadata_prompts = [c for c in recording_llm.calls if _METADATA_PROMPT_MARKER in c]
        assert len(metadata_prompts) == 1
        assert state.topic in metadata_prompts[0]

    @pytest.mark.asyncio
    async def test_metadata_receives_actual_multi_section_script_result(self, providers) -> None:
        """The metadata prompt must reflect the SAME multi-section
        ScriptResult the pipeline actually produced - proving the real
        PipelineState.script_result is used, not a reconstructed one."""
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, llm_provider=recording_llm)

        assert len(state.script_result.sections) >= 2  # a real multi-section script, not a synthetic 1-section stand-in

        metadata_prompts = [c for c in recording_llm.calls if _METADATA_PROMPT_MARKER in c]
        assert len(metadata_prompts) == 1
        for section in state.script_result.sections:
            assert section.heading in metadata_prompts[0]
            assert section.narration in metadata_prompts[0]

    @pytest.mark.asyncio
    async def test_metadata_does_not_reconstruct_script_from_srt(self, providers) -> None:
        """No script-context-reconstruction placeholder text should ever
        appear in pipeline-integrated metadata output - that fallback only
        exists for the standalone demo, which has no PipelineState."""
        state = await self._run(providers)

        assert "reconstructed for standalone validation" not in state.metadata_result.title.lower()
        assert "reconstructed for standalone validation" not in state.metadata_result.description.lower()

    @pytest.mark.asyncio
    async def test_metadata_receives_final_mixed_video_duration(self, providers) -> None:
        state = await self._run(providers)

        assert state.metadata_result.duration_seconds == state.audio_mix_result.output_duration_seconds

    @pytest.mark.asyncio
    async def test_no_research_or_script_rerun_for_metadata(self, providers) -> None:
        recording_search = RecordingSearchProvider()
        await self._run(providers, search_provider=recording_search)

        # ResearchAgent is the only caller of SearchProvider.search() in the
        # whole pipeline - if metadata generation re-ran Research, this
        # would be > 1.
        assert len(recording_search.calls) == 1

    @pytest.mark.asyncio
    async def test_metadata_agent_called_exactly_once(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider()
        await self._run(providers, llm_provider=provider)

        assert len(provider.metadata_calls) == 1

    # ---- R. Metadata not called before/without BGM success --------------------

    @pytest.mark.asyncio
    async def test_metadata_not_called_when_caption_fails(self, providers) -> None:
        state = await self._run(providers, assembler=FakeVideoAssembler(fail_burn_subtitles=True))

        assert state.status == "failed"
        assert state.metadata_result is None

    @pytest.mark.asyncio
    async def test_metadata_not_called_after_hard_qc_failure(self, providers) -> None:
        state = await self._run(providers, visual_relevance_evaluator=AlwaysMisleadingEvaluator())

        assert state.status == "failed"
        assert state.metadata_result is None

    @pytest.mark.asyncio
    async def test_metadata_not_called_when_bgm_mixing_fails(self, providers) -> None:
        state = await self._run(providers, music_catalog_provider=MockMusicCatalogProvider(tracks=[]))

        assert state.status == "failed"
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is False
        assert state.metadata_result is None

    # ---- S. Metadata failure behavior ------------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_llm_failure_marks_pipeline_failed(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider(metadata_error=RuntimeError("simulated metadata LLM outage"))
        state = await self._run(providers, llm_provider=provider)

        assert state.status == "failed"
        assert "Metadata generation failed" in state.error
        assert state.metadata_result is not None
        assert state.metadata_result.success is False

    @pytest.mark.asyncio
    async def test_metadata_failure_preserves_final_mixed_video(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider(metadata_error=RuntimeError("outage"))
        state = await self._run(providers, llm_provider=provider)

        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True
        assert os.path.exists(state.audio_mix_result.output_path)

    @pytest.mark.asyncio
    async def test_metadata_failure_preserves_earlier_pipeline_outputs(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider(metadata_error=RuntimeError("outage"))
        state = await self._run(providers, llm_provider=provider)

        assert state.status == "failed"
        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_qc_result is not None
        assert state.qc_approved_visual_result is not None
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert state.caption_result is not None
        assert state.caption_result.success is True
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True

    # ---- T. Deterministic chapter generation from the real multi-section script -

    @pytest.mark.asyncio
    async def test_real_multi_section_script_produces_chapter_candidates(self, providers) -> None:
        state = await self._run(providers)

        assert state.metadata_result.chapters_available is True
        assert len(state.metadata_result.chapters) == len(state.script_result.sections)

    @pytest.mark.asyncio
    async def test_first_chapter_starts_at_zero(self, providers) -> None:
        state = await self._run(providers)

        assert state.metadata_result.chapters[0].timestamp_seconds == 0.0
        assert state.metadata_result.chapters[0].timestamp_text == "0:00"

    @pytest.mark.asyncio
    async def test_chapter_timestamps_strictly_increasing(self, providers) -> None:
        state = await self._run(providers)

        timestamps = [c.timestamp_seconds for c in state.metadata_result.chapters]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)

    @pytest.mark.asyncio
    async def test_chapter_timestamps_within_final_duration(self, providers) -> None:
        state = await self._run(providers)

        duration = state.audio_mix_result.output_duration_seconds
        assert all(c.timestamp_seconds < duration for c in state.metadata_result.chapters)

    @pytest.mark.asyncio
    async def test_llm_does_not_control_raw_chapter_timestamps(self, providers) -> None:
        """The metadata prompt must tell the LLM timestamps are already
        fixed when chapters are being requested - proving the LLM is only
        ever asked for labels, never timestamps."""
        recording_llm = RecordingLLMProvider()
        await self._run(providers, llm_provider=recording_llm)

        metadata_prompts = [c for c in recording_llm.calls if _METADATA_PROMPT_MARKER in c]
        assert "ALREADY FIXED" in metadata_prompts[0]

    @pytest.mark.asyncio
    async def test_missing_chapter_label_falls_back_to_section_heading(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider(metadata_payload={"chapter_labels": ["Only One Real Label"]})
        state = await self._run(providers, llm_provider=provider)

        assert state.metadata_result.chapters_available is True
        assert state.metadata_result.chapters[0].title == "Only One Real Label"
        # Every later chapter falls back to its own section's real heading.
        for index in range(1, len(state.metadata_result.chapters)):
            assert state.metadata_result.chapters[index].title == state.script_result.sections[index].heading

    # ---- U. Metadata JSON artifact ---------------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_json_artifact_written_to_disk(self, providers) -> None:
        state = await self._run(providers)

        assert state.metadata_result.output_path is not None
        assert os.path.exists(state.metadata_result.output_path)
        with open(state.metadata_result.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["title"] == state.metadata_result.title
        assert data["chapters_available"] is True
        assert len(data["chapters"]) == len(state.script_result.sections)


class TestThumbnailPipelineIntegration:
    """Tests for the Thumbnail Agent's integration as the final stage of the
    LangGraph pipeline (Metadata -> Thumbnail -> END). Reuses the same
    TestPipelineWorkflow fixture/_run helper shape - all mock/fake/local
    providers only, no real Gemini/Pexels/FFmpeg/Whisper calls."""

    providers = TestPipelineWorkflow.providers
    _run = staticmethod(TestPipelineWorkflow._run)

    # ---- A. Node exists / routing -------------------------------------------

    @pytest.mark.asyncio
    async def test_thumbnail_node_exists_and_pipeline_builds(self, providers) -> None:
        (
            search_provider, llm_provider, voice_provider, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        ) = providers
        graph = build_pipeline_graph(
            search_provider, llm_provider, voice_provider, TEST_VOICE_NAME, media_provider, assembler,
            visual_relevance_evaluator, transcription_provider, music_catalog_provider,
            voice_dir, media_dir, video_dir, subtitle_dir,
        )
        compiled = graph.compile()
        assert "thumbnail" in compiled.get_graph().nodes

    @pytest.mark.asyncio
    async def test_thumbnail_runs_after_successful_metadata(self, providers) -> None:
        state = await self._run(providers)

        assert state.metadata_result is not None
        assert state.metadata_result.success is True
        assert state.thumbnail_result is not None
        assert isinstance(state.thumbnail_result, ThumbnailResult)
        assert state.thumbnail_result.success is True
        assert state.status == "completed"
        # Output validation (exact 1280x720) is preserved unchanged.
        assert state.thumbnail_result.width == 1280
        assert state.thumbnail_result.height == 720
        assert state.thumbnail_result.output_path is not None
        assert os.path.exists(state.thumbnail_result.output_path)

    @pytest.mark.asyncio
    async def test_thumbnail_not_called_when_metadata_fails(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider(metadata_error=RuntimeError("simulated metadata outage"))
        state = await self._run(providers, llm_provider=provider)

        assert state.status == "failed"
        assert state.metadata_result is not None
        assert state.metadata_result.success is False
        assert state.thumbnail_result is None

    @pytest.mark.asyncio
    async def test_thumbnail_not_called_when_bgm_fails(self, providers) -> None:
        """Thumbnail must never run before Metadata - proven here via an
        earlier-stage (BGM) failure, which must never let the graph reach
        metadata or thumbnail at all."""
        state = await self._run(providers, music_catalog_provider=MockMusicCatalogProvider(tracks=[]))

        assert state.status == "failed"
        assert state.metadata_result is None
        assert state.thumbnail_result is None

    # ---- B. Inputs reused from real pipeline state ---------------------------

    @pytest.mark.asyncio
    async def test_thumbnail_receives_original_topic(self, providers) -> None:
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, topic="Why do humans dream?", llm_provider=recording_llm)

        thumbnail_prompts = [c for c in recording_llm.calls if _THUMBNAIL_PROMPT_MARKER in c]
        assert len(thumbnail_prompts) == 1
        assert state.topic in thumbnail_prompts[0]

    @pytest.mark.asyncio
    async def test_thumbnail_receives_actual_script_result(self, providers) -> None:
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, llm_provider=recording_llm)

        assert len(state.script_result.sections) >= 2  # a real multi-section script, not a synthetic stand-in

        thumbnail_prompts = [c for c in recording_llm.calls if _THUMBNAIL_PROMPT_MARKER in c]
        assert len(thumbnail_prompts) == 1
        for section in state.script_result.sections:
            assert section.heading in thumbnail_prompts[0]
            assert section.narration in thumbnail_prompts[0]

    @pytest.mark.asyncio
    async def test_thumbnail_receives_actual_metadata_result(self, providers) -> None:
        recording_llm = RecordingLLMProvider()
        state = await self._run(providers, llm_provider=recording_llm)

        thumbnail_prompts = [c for c in recording_llm.calls if _THUMBNAIL_PROMPT_MARKER in c]
        assert len(thumbnail_prompts) == 1
        assert state.metadata_result.title in thumbnail_prompts[0]
        assert state.metadata_result.seo_summary in thumbnail_prompts[0]

    @pytest.mark.asyncio
    async def test_thumbnail_uses_in_memory_metadata_not_reloaded_from_json(self, providers) -> None:
        """A distinctive, never-persisted-elsewhere title proves the
        Thumbnail stage's prompt reflects state.metadata_result.title
        directly - not anything reconstructed/reloaded from a JSON file
        (the real pipeline never reloads metadata from disk; that fallback
        only exists for standalone demo tooling)."""
        distinctive_title = "UNIQUE-METADATA-TITLE-Q7F2"
        provider = RecordingMetadataAndThumbnailLLMProvider(metadata_title=distinctive_title)
        state = await self._run(providers, llm_provider=provider)

        assert state.metadata_result.title == distinctive_title
        thumbnail_prompts = [c for c in provider.calls if _THUMBNAIL_PROMPT_MARKER in c]
        assert len(thumbnail_prompts) == 1
        assert distinctive_title in thumbnail_prompts[0]

    @pytest.mark.asyncio
    async def test_metadata_agent_not_rerun_for_thumbnail(self, providers) -> None:
        provider = ConfigurableMetadataLLMProvider()
        state = await self._run(providers, llm_provider=provider)

        assert state.thumbnail_result is not None
        assert state.thumbnail_result.success is True
        assert len(provider.metadata_calls) == 1

    @pytest.mark.asyncio
    async def test_no_research_or_script_rerun_for_thumbnail(self, providers) -> None:
        recording_search = RecordingSearchProvider()
        await self._run(providers, search_provider=recording_search)

        # ResearchAgent is the only caller of SearchProvider.search() in the
        # whole pipeline - if thumbnail generation re-ran Research, this
        # would be > 1.
        assert len(recording_search.calls) == 1

    # ---- C. Provider reuse (dependency injection, no duplicates) -------------

    @pytest.mark.asyncio
    async def test_thumbnail_agent_planner_called_exactly_once(self, providers) -> None:
        recording_llm = RecordingLLMProvider()
        await self._run(providers, llm_provider=recording_llm)

        thumbnail_prompts = [c for c in recording_llm.calls if _THUMBNAIL_PROMPT_MARKER in c]
        assert len(thumbnail_prompts) == 1

    @pytest.mark.asyncio
    async def test_shared_llm_provider_reused_for_thumbnail(self, providers) -> None:
        """The same LLMProvider instance already used by Research/Script/
        BGM/Metadata must be the one Thumbnail planning calls too - no
        second provider/API key path."""
        recording_llm = RecordingLLMProvider()
        await self._run(providers, llm_provider=recording_llm)

        assert any(_METADATA_PROMPT_MARKER in c for c in recording_llm.calls)
        assert any(_THUMBNAIL_PROMPT_MARKER in c for c in recording_llm.calls)

    @pytest.mark.asyncio
    async def test_shared_media_provider_reused_for_thumbnail(self, providers) -> None:
        """The same MediaProvider instance already used by Visual Media must
        be the one Thumbnail image search/download uses too - no second
        Pexels/HTTP implementation."""
        _, _, _, media_provider, _, _, _, _, _, _, _, _ = providers
        state = await self._run(providers)

        assert state.thumbnail_result.success is True
        search_queries = [call[1] for call in media_provider.calls if call[0] == "search"]
        # Thumbnail's own deterministic-fallback search query is the raw
        # topic (see build_deterministic_thumbnail_plan) - distinct from
        # Visual Media's per-section queries, proving a real extra call was
        # made on the SAME shared provider instance, not a duplicate one.
        assert search_queries.count(state.topic) == 1

    # ---- D. Routing / final status --------------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_success_alone_no_longer_completes_pipeline(self, providers) -> None:
        """Metadata succeeding is necessary but not sufficient - a thumbnail
        failure after successful metadata must NOT be reported as
        "completed"; the pipeline's final deliverable now includes the
        thumbnail."""
        state = await self._run(providers, media_provider=FailThumbnailSearchMediaProvider(topic="Why do humans dream?"))

        assert state.metadata_result is not None
        assert state.metadata_result.success is True
        assert state.status == "failed"
        assert state.status != "completed"

    @pytest.mark.asyncio
    async def test_thumbnail_failure_marks_pipeline_failed(self, providers) -> None:
        state = await self._run(providers, media_provider=FailThumbnailSearchMediaProvider(topic="Why do humans dream?"))

        assert state.status == "failed"
        assert "Thumbnail generation failed" in state.error
        assert state.thumbnail_result is not None
        assert state.thumbnail_result.success is False

    @pytest.mark.asyncio
    async def test_thumbnail_failure_preserves_final_bgm_video(self, providers) -> None:
        state = await self._run(providers, media_provider=FailThumbnailSearchMediaProvider(topic="Why do humans dream?"))

        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True
        assert os.path.exists(state.audio_mix_result.output_path)

    @pytest.mark.asyncio
    async def test_thumbnail_failure_preserves_metadata_result(self, providers) -> None:
        state = await self._run(providers, media_provider=FailThumbnailSearchMediaProvider(topic="Why do humans dream?"))

        assert state.metadata_result is not None
        assert state.metadata_result.success is True
        assert state.metadata_result.output_path is not None
        assert os.path.exists(state.metadata_result.output_path)

    @pytest.mark.asyncio
    async def test_thumbnail_failure_preserves_earlier_results(self, providers) -> None:
        state = await self._run(providers, media_provider=FailThumbnailSearchMediaProvider(topic="Why do humans dream?"))

        assert state.research_result is not None
        assert state.script_result is not None
        assert state.voice_result is not None
        assert state.voice_result.success is True
        assert state.visual_result is not None
        assert state.visual_qc_result is not None
        assert state.qc_approved_visual_result is not None
        assert state.video_assembly_result is not None
        assert state.video_assembly_result.success is True
        assert state.caption_result is not None
        assert state.caption_result.success is True
        assert state.audio_mix_result is not None
        assert state.audio_mix_result.success is True

    # ---- E. Approved hook-quality policy / deterministic fallback -----------

    @pytest.mark.asyncio
    async def test_approved_hook_quality_policy_remains_active(self, providers) -> None:
        """An ambiguous isolated-statistic hook from the LLM must still be
        deterministically replaced inside the real pipeline - the approved
        hook-quality guard is not bypassed by integration."""
        provider = ConfigurableThumbnailLLMProvider(thumbnail_payload={"hook_text": "Two Hours Every Night"})
        state = await self._run(providers, llm_provider=provider)

        assert state.thumbnail_result.success is True
        assert state.thumbnail_result.plan.hook_text != "Two Hours Every Night"
        assert any("isolated statistic" in w.lower() for w in state.thumbnail_result.warnings)

    @pytest.mark.asyncio
    async def test_deterministic_hook_fallback_remains_active(self, providers) -> None:
        """The default fixture's LLM never returns valid JSON for the
        thumbnail-planning prompt - ThumbnailPlanner must fall back to the
        deterministic plan, and the pipeline must still complete."""
        state = await self._run(providers)

        assert state.status == "completed"
        assert state.thumbnail_result.success is True
        assert state.thumbnail_result.plan.used_semantic_planning is False
        assert state.thumbnail_result.plan.fallback_reason is not None

    @pytest.mark.asyncio
    async def test_one_semantic_llm_call_maximum_for_thumbnail(self, providers) -> None:
        provider = ConfigurableThumbnailLLMProvider(thumbnail_payload={"hook_text": "Two Hours Every Night"})
        await self._run(providers, llm_provider=provider)

        assert len(provider.thumbnail_calls) == 1

    # ---- F. Pipeline demo stage labels -----------------------------------------

    def test_pipeline_demo_stage_labels_include_thumbnail(self) -> None:
        from src.pipeline_demo import _STAGE_LABELS

        assert len(_STAGE_LABELS) == 10
        assert _STAGE_LABELS["thumbnail"] == "[10/10] Thumbnail"
