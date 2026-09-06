# Full Research -> Script -> Voice -> Visual Media -> Visual QC -> Video
# Assembly -> Subtitle/Caption -> BGM/Audio Mixing pipeline using LangGraph.
#
# This module does not implement any research, scripting, voice-synthesis,
# visual-media, QC-evaluation, video-encoding, transcription, subtitle-
# rendering, or music-planning/selection/mixing logic itself - it only
# wires the existing ResearchAgent, ScriptAgent, VoiceService,
# VisualMediaService, VisualQCService, VideoAssemblyService,
# CaptionService, and AudioMixingService together into a single LangGraph
# state machine, passing each stage's output directly into the next
# stage's input.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent, ScriptAgentError
from src.agents.visual_context_planner import VisualContextPlanner
from src.models.captions import CaptionResult
from src.models.media import VisualResult
from src.models.music import AudioMixResult
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.models.visual_plan import VisualPlan
from src.models.visual_qc import VisualQCResult
from src.models.voice import VoiceResult
from src.services.audio_mixing_service import AudioMixingService, AudioMixingServiceError
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR, CaptionService, CaptionServiceError
from src.services.video_assembly_service import (
    DEFAULT_VIDEO_OUTPUT_DIR,
    VideoAssemblyService,
    VideoAssemblyServiceError,
)
from src.services.visual_media_service import (
    DEFAULT_MEDIA_OUTPUT_DIR,
    VisualMediaService,
    VisualMediaServiceError,
)
from src.services.visual_qc_service import VisualQCService, VisualQCServiceError
from src.services.voice_service import DEFAULT_OUTPUT_DIR, VoiceService, VoiceServiceError
from src.tools.ffmpeg_video_assembler import VideoAssembler
from src.tools.media_provider import MediaProvider
from src.tools.music_catalog_provider import MusicCatalogProvider
from src.tools.transcription_provider import TranscriptionProvider
from src.tools.visual_relevance_evaluator import VisualRelevanceEvaluator
from src.tools.voice_provider import VoiceProvider


def _captioned_video_result(caption_result: CaptionResult) -> VideoAssemblyResult:
    """Adapt CaptionService's captioned MP4 into the VideoAssemblyResult
    shape AudioMixingService already expects as its source video.

    BGM is mixed onto the captioned output (what viewers will actually
    see), not the pre-caption assembly - the same file the standalone
    ``bgm_demo.py`` targets for real validation. This is a thin adapter,
    not new business logic: AudioMixingService itself only reads
    ``output_path``/``success`` off whatever VideoAssemblyResult-shaped
    object it's given.
    """
    return VideoAssemblyResult(
        success=True,
        output_path=caption_result.captioned_video_path,
        duration_seconds=caption_result.captioned_duration_seconds,
        format="mp4",
    )


@dataclass
class PipelineState:
    """Shared state for the Research -> Script -> Voice -> Visual Media ->
    Visual QC -> Video Assembly pipeline."""

    topic: str = ""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    voice_result: Optional[VoiceResult] = None
    visual_plan: Optional[VisualPlan] = None
    visual_result: Optional[VisualResult] = None
    visual_qc_result: Optional[VisualQCResult] = None
    # The VisualResult Video Assembly actually consumes: visual_result
    # (above) is never overwritten/silently replaced - it stays exactly
    # what VisualMediaService produced, pre-QC - so this field is always
    # populated separately once Visual QC completes acceptably, and it's
    # the only one that may differ from visual_result (when QC replaced a
    # weak/misleading asset).
    qc_approved_visual_result: Optional[VisualResult] = None
    video_assembly_result: Optional[VideoAssemblyResult] = None
    caption_result: Optional[CaptionResult] = None
    # AudioMixResult already carries the mood plan (.music_plan), the
    # selected track (.selected_track), and the final mixed MP4 path
    # (.output_path) - no separate top-level fields for those, consistent
    # with how VisualQCResult/CaptionResult are each the single source of
    # truth for their own stage's structured data.
    audio_mix_result: Optional[AudioMixResult] = None
    # pending -> researching -> researched -> scripted -> voiced ->
    # visualized -> qc_passed -> assembled -> captioned -> completed -> failed
    status: str = "pending"
    error: Optional[str] = None


def build_pipeline_graph(
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    visual_relevance_evaluator: VisualRelevanceEvaluator,
    transcription_provider: TranscriptionProvider,
    music_catalog_provider: MusicCatalogProvider,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    subtitle_output_dir: str = DEFAULT_SUBTITLE_OUTPUT_DIR,
) -> StateGraph:
    """Build the LangGraph state machine chaining Research -> Script -> Voice
    -> Visual Media -> Visual QC -> Video Assembly -> Subtitle/Caption ->
    BGM/Audio Mixing.

    Reuses the existing ResearchAgent, ScriptAgent, VoiceService,
    VisualMediaService, VisualQCService, VideoAssemblyService,
    CaptionService, and AudioMixingService as-is (no duplicated business
    logic, no reimplemented FFmpeg calls, no re-implemented QC evaluation/
    replacement, no re-implemented transcription/SRT/subtitle-rendering or
    music-planning/selection/mixing logic); this graph only wires their
    existing async interfaces together and shares one PipelineState across
    all eight. VoiceService, VisualMediaService, VisualQCService,
    VideoAssemblyService, CaptionService, and AudioMixingService all remain
    deterministic orchestration here - each is invoked directly, not
    treated as a reasoning agent (semantic judgment stays inside
    VisualContextPlanner/VisualQCService's injected evaluator and
    AudioMixingService's injected MusicContextPlanner; speech-to-text stays
    inside CaptionService's injected transcription provider).
    VideoAssemblyService receives the exact ScriptResult/VoiceResult
    already produced earlier in this same run, and the post-QC
    ``qc_approved_visual_result`` (never the raw, pre-QC ``visual_result``).
    CaptionService receives the exact VoiceResult/VideoAssemblyResult
    already produced earlier in this same run. AudioMixingService receives
    the exact ScriptResult already produced earlier in this same run
    (Research/Script are never re-run for BGM mood planning) and the
    captioned MP4 CaptionService just produced - nothing is regenerated,
    re-synthesized, re-downloaded, or re-assembled.

    Graph structure:
        START → research ─(ok)─→ script ─(ok)─→ voice ─(ok)─→ media ─(ok)─→ visual_qc ─(ok)─→ video_assembly ─(ok)─→ captions ─(ok)─→ bgm → END
                    │                  │               │              │                │                  │                  │
                    └──────(error)─────┴──────(error)───┴───(error)────┴────(error)─────┴──────(error)─────┴──────(error)─────┴──────(error)──→ END

    Args:
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research, Script,
            Visual Context Planning, Visual QC, and BGM mood planning
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation for the Visual Media Service
        assembler: VideoAssembler implementation, shared by Visual QC (frame
            extraction/probing), the Video Assembly Service (encoding), the
            Caption Service (subtitle burning), and Audio Mixing (background
            music mixing)
        visual_relevance_evaluator: VisualRelevanceEvaluator implementation
            for the Visual QC Service
        transcription_provider: TranscriptionProvider implementation for
            the Caption Service
        music_catalog_provider: MusicCatalogProvider implementation (the
            approved BGM catalog) for the Audio Mixing Service
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into
        subtitle_output_dir: Directory the Caption Service writes .srt files into

    Returns:
        StateGraph ready to be ``.compile()``d
    """
    research_agent = ResearchAgent(search_provider=search_provider, llm_provider=llm_provider)
    script_agent = ScriptAgent(llm_provider=llm_provider)
    voice_service = VoiceService(
        voice_provider=voice_provider, voice_name=voice_name, output_dir=voice_output_dir
    )
    # Reuses the same LLMProvider Research/Script already depend on - one
    # extra call per pipeline run (for the whole script at once), not a new
    # provider/setting. If that call fails, VisualContextPlanner falls back
    # to the deterministic query-generation path on its own.
    visual_planner = VisualContextPlanner(llm_provider=llm_provider)
    visual_service = VisualMediaService(
        media_provider=media_provider, visual_planner=visual_planner, output_dir=media_output_dir
    )
    # visual_media_service=visual_service lets Visual QC request bounded
    # replacements through VisualMediaService's own existing selection
    # logic (see visual_qc_node) - VisualQCService never re-implements
    # candidate search itself.
    visual_qc_service = VisualQCService(
        evaluator=visual_relevance_evaluator, assembler=assembler, visual_media_service=visual_service
    )
    video_service = VideoAssemblyService(assembler=assembler, output_dir=video_output_dir)
    # Shares the same VideoAssembler as Visual QC/Video Assembly (subtitle
    # burning reuses its burn_subtitles method) - no second FFmpeg wrapper.
    caption_service = CaptionService(
        transcription_provider=transcription_provider, assembler=assembler, output_dir=subtitle_output_dir
    )
    # Reuses the same LLMProvider (MusicContextPlanner's single optional
    # mood-planning call per run) and the same VideoAssembler (background-
    # audio mixing reuses its mix_background_audio method) - no new
    # provider/setting and no second FFmpeg wrapper.
    audio_mixing_service = AudioMixingService(
        catalog_provider=music_catalog_provider, assembler=assembler, llm_provider=llm_provider
    )

    async def research_node(state: PipelineState) -> dict:
        try:
            result = await research_agent.research(state.topic)
            return {"research_result": result, "status": "researched", "error": None}
        except ResearchAgentError as e:
            return {"research_result": None, "status": "failed", "error": f"Research failed: {e}"}
        except Exception as e:
            return {
                "research_result": None,
                "status": "failed",
                "error": f"Unexpected research error: {e}",
            }

    async def script_node(state: PipelineState) -> dict:
        try:
            result = await script_agent.generate_script(state.research_result)
            return {"script_result": result, "status": "scripted", "error": None}
        except ScriptAgentError as e:
            return {"script_result": None, "status": "failed", "error": f"Script generation failed: {e}"}
        except Exception as e:
            return {
                "script_result": None,
                "status": "failed",
                "error": f"Unexpected script error: {e}",
            }

    async def voice_node(state: PipelineState) -> dict:
        try:
            result = await voice_service.generate_voice(state.script_result)
        except VoiceServiceError as e:
            return {"voice_result": None, "status": "failed", "error": f"Voice generation failed: {e}"}
        except Exception as e:
            return {
                "voice_result": None,
                "status": "failed",
                "error": f"Unexpected voice error: {e}",
            }

        if not result.success:
            # VoiceService never raises for synthesis failures - it reports
            # them in VoiceResult.error instead. Surface that in pipeline
            # state (and keep the VoiceResult itself for inspection).
            return {
                "voice_result": result,
                "status": "failed",
                "error": f"Voice generation failed: {result.error}",
            }

        return {"voice_result": result, "status": "voiced", "error": None}

    async def media_node(state: PipelineState) -> dict:
        try:
            # The same ScriptResult produced by the script stage is passed
            # straight through - the script is never regenerated or rewritten.
            # VoiceResult.duration_seconds (the real narration audio length)
            # is the upstream timing input for visual planning; fall back to
            # the script's own deterministic estimate only if a voice
            # provider couldn't report a duration.
            narration_duration = (
                state.voice_result.duration_seconds or state.script_result.estimated_duration_seconds
            )
            # Built once and threaded through (via visual_plan=) so Visual
            # QC can reuse the exact same plan later - one Gemini planning
            # call per run, never a second one for QC.
            plan = visual_service.build_plan(state.script_result)
            result = await visual_service.generate_visuals(
                state.script_result, narration_duration, visual_plan=plan
            )
        except VisualMediaServiceError as e:
            return {
                "visual_plan": None,
                "visual_result": None,
                "status": "failed",
                "error": f"Visual media generation failed: {e}",
            }
        except Exception as e:
            return {
                "visual_plan": None,
                "visual_result": None,
                "status": "failed",
                "error": f"Unexpected visual media error: {e}",
            }

        if not result.success:
            # VisualMediaService never raises for search/download failures -
            # it reports them in VisualResult.error instead. Surface that in
            # pipeline state (and keep the VisualResult itself for inspection).
            return {
                "visual_plan": plan,
                "visual_result": result,
                "status": "failed",
                "error": f"Visual media generation failed: {result.error}",
            }

        return {"visual_plan": plan, "visual_result": result, "status": "visualized", "error": None}

    async def visual_qc_node(state: PipelineState) -> dict:
        try:
            # The exact VisualPlan/VisualResult already produced by the
            # media stage are passed straight through - nothing is
            # re-planned or re-selected except bounded, QC-driven
            # replacements (via VisualMediaService.acquire_replacement_asset,
            # VisualQCService's own existing mechanism - not duplicated here).
            qc_result, qc_visual_result = await visual_qc_service.run_qc(
                state.topic, state.script_result, state.visual_plan, state.visual_result
            )
        except VisualQCServiceError as e:
            return {
                "visual_qc_result": None,
                "qc_approved_visual_result": None,
                "status": "failed",
                "error": f"Visual QC failed: {e}",
            }
        except Exception as e:
            return {
                "visual_qc_result": None,
                "qc_approved_visual_result": None,
                "status": "failed",
                "error": f"Unexpected visual QC error: {e}",
            }

        if not qc_result.success:
            # VisualQCService itself couldn't run (e.g. it was handed an
            # already-unsuccessful VisualResult) - surfaced in
            # VisualQCResult.error. Keep both results for inspection.
            return {
                "visual_qc_result": qc_result,
                "qc_approved_visual_result": qc_visual_result,
                "status": "failed",
                "error": f"Visual QC failed: {qc_result.error}",
            }

        if qc_result.rejected_count > 0:
            # At least one asset is still flagged misleading/conflicting
            # after bounded replacement was exhausted - required media
            # could not be safely approved or replaced. Stop before Video
            # Assembly rather than risk a misleading clip reaching the
            # final video; earlier-stage results are preserved below.
            return {
                "visual_qc_result": qc_result,
                "qc_approved_visual_result": qc_visual_result,
                "status": "failed",
                "error": (
                    f"Visual QC rejected {qc_result.rejected_count} asset(s) as misleading "
                    "with no safe replacement available"
                ),
            }

        return {
            "visual_qc_result": qc_result,
            "qc_approved_visual_result": qc_visual_result,
            "status": "qc_passed",
            "error": None,
        }

    async def video_assembly_node(state: PipelineState) -> dict:
        try:
            # ScriptResult/VoiceResult already produced earlier in this run
            # are passed straight through, and Video Assembly consumes the
            # post-QC qc_approved_visual_result (never the raw visual_result)
            # - nothing is regenerated, re-synthesized, or re-downloaded.
            result = await video_service.assemble_video(
                state.script_result, state.voice_result, state.qc_approved_visual_result
            )
        except VideoAssemblyServiceError as e:
            return {
                "video_assembly_result": None,
                "status": "failed",
                "error": f"Video assembly failed: {e}",
            }
        except Exception as e:
            return {
                "video_assembly_result": None,
                "status": "failed",
                "error": f"Unexpected video assembly error: {e}",
            }

        if not result.success:
            # VideoAssemblyService never raises for FFmpeg/processing
            # failures - it reports them in VideoAssemblyResult.error
            # instead. Surface that in pipeline state (and keep the
            # VideoAssemblyResult itself for inspection).
            return {
                "video_assembly_result": result,
                "status": "failed",
                "error": f"Video assembly failed: {result.error}",
            }

        return {"video_assembly_result": result, "status": "assembled", "error": None}

    async def caption_node(state: PipelineState) -> dict:
        try:
            # The exact VoiceResult/VideoAssemblyResult already produced
            # earlier in this run are passed straight through - narration
            # is never regenerated and video is never reassembled. Caption
            # timing comes entirely from CaptionService's own transcription
            # of the real narration audio, not from any script-derived estimate.
            result = await caption_service.generate_captions(state.voice_result, state.video_assembly_result)
        except CaptionServiceError as e:
            return {"caption_result": None, "status": "failed", "error": f"Caption generation failed: {e}"}
        except Exception as e:
            return {
                "caption_result": None,
                "status": "failed",
                "error": f"Unexpected caption error: {e}",
            }

        if not result.success:
            # CaptionService never raises for transcription/rendering
            # failures - it reports them in CaptionResult.error instead.
            # Surface that in pipeline state (and keep the CaptionResult
            # itself for inspection); the original assembled MP4 is
            # untouched regardless (CaptionService never writes to it).
            return {
                "caption_result": result,
                "status": "failed",
                "error": f"Caption generation failed: {result.error}",
            }

        return {"caption_result": result, "status": "captioned", "error": None}

    async def bgm_node(state: PipelineState) -> dict:
        try:
            # The exact ScriptResult already produced earlier in this run is
            # passed straight through for mood planning - Research/Script
            # are never re-run just to get BGM context (unlike the
            # standalone demo, which has no PipelineState to read a
            # ScriptResult from and must reconstruct one from an existing
            # .srt transcript instead). BGM mixes onto the captioned MP4
            # CaptionService just produced, not the pre-caption assembly.
            result = await audio_mixing_service.generate_mix(
                state.topic, state.script_result, _captioned_video_result(state.caption_result)
            )
        except AudioMixingServiceError as e:
            return {"audio_mix_result": None, "status": "failed", "error": f"BGM mixing failed: {e}"}
        except Exception as e:
            return {
                "audio_mix_result": None,
                "status": "failed",
                "error": f"Unexpected BGM mixing error: {e}",
            }

        if not result.success:
            # AudioMixingService never raises for catalog/selection/mixing
            # failures - it reports them in AudioMixResult.error instead
            # (including a semantic mood-planning failure, which is NOT a
            # mixing failure: MusicContextPlanner already fell back to a
            # deterministic MusicPlan internally and mixing still would
            # have been attempted - see AudioMixResult.music_plan.fallback_reason
            # for that case). Surface the actual failure here; the
            # captioned MP4 is untouched regardless (AudioMixingService
            # never writes to it, always to a new copy).
            return {
                "audio_mix_result": result,
                "status": "failed",
                "error": f"BGM mixing failed: {result.error}",
            }

        return {"audio_mix_result": result, "status": "completed", "error": None}

    def route_after_research(state: PipelineState) -> str:
        """Only proceed to scripting if research actually produced a result."""
        return "script" if state.research_result is not None else END

    def route_after_script(state: PipelineState) -> str:
        """Only proceed to voice generation if scripting actually succeeded."""
        return "voice" if state.script_result is not None else END

    def route_after_voice(state: PipelineState) -> str:
        """Only proceed to visual media generation if voice actually succeeded."""
        return "media" if state.voice_result is not None and state.voice_result.success else END

    def route_after_media(state: PipelineState) -> str:
        """Only proceed to Visual QC if visual media actually succeeded."""
        return (
            "visual_qc"
            if state.visual_result is not None and state.visual_result.success
            else END
        )

    def route_after_visual_qc(state: PipelineState) -> str:
        """Only proceed to video assembly once Visual QC has approved (or
        safely replaced) every section's media - never after a hard QC
        failure (see visual_qc_node's rejected_count policy above)."""
        return "video_assembly" if state.status == "qc_passed" else END

    def route_after_video_assembly(state: PipelineState) -> str:
        """Only proceed to captioning if video assembly actually succeeded -
        the caption node must never run on a missing/failed assembled MP4."""
        return (
            "captions"
            if state.video_assembly_result is not None and state.video_assembly_result.success
            else END
        )

    def route_after_captions(state: PipelineState) -> str:
        """Only proceed to BGM mixing if captioning actually succeeded - the
        bgm node must never run on a missing/failed captioned MP4."""
        return "bgm" if state.caption_result is not None and state.caption_result.success else END

    graph = StateGraph(PipelineState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("voice", voice_node)
    graph.add_node("media", media_node)
    graph.add_node("visual_qc", visual_qc_node)
    graph.add_node("video_assembly", video_assembly_node)
    graph.add_node("captions", caption_node)
    graph.add_node("bgm", bgm_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"script": "script", END: END})
    graph.add_conditional_edges("script", route_after_script, {"voice": "voice", END: END})
    graph.add_conditional_edges("voice", route_after_voice, {"media": "media", END: END})
    graph.add_conditional_edges("media", route_after_media, {"visual_qc": "visual_qc", END: END})
    graph.add_conditional_edges(
        "visual_qc", route_after_visual_qc, {"video_assembly": "video_assembly", END: END}
    )
    graph.add_conditional_edges(
        "video_assembly", route_after_video_assembly, {"captions": "captions", END: END}
    )
    graph.add_conditional_edges("captions", route_after_captions, {"bgm": "bgm", END: END})
    graph.set_finish_point("bgm")

    return graph


async def run_pipeline(
    topic: str,
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    visual_relevance_evaluator: VisualRelevanceEvaluator,
    transcription_provider: TranscriptionProvider,
    music_catalog_provider: MusicCatalogProvider,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    subtitle_output_dir: str = DEFAULT_SUBTITLE_OUTPUT_DIR,
) -> PipelineState:
    """Run the full Research -> Script -> Voice -> Visual Media -> Visual QC
    -> Video Assembly -> Subtitle/Caption -> BGM/Audio Mixing pipeline and
    return the final state.

    Unlike the individual research/script workflow convenience functions
    (which raise on failure), this returns the full PipelineState so callers
    can inspect ``status``/``error`` directly, including on partial failure
    (e.g. every stage through captioning succeeded but BGM mixing failed).

    Each stage only runs if the previous one succeeded - a failure at any
    stage short-circuits the rest of the pipeline and it never silently
    continues (see route_after_research/route_after_script/route_after_voice/
    route_after_media/route_after_visual_qc/route_after_video_assembly/
    route_after_captions). Visual QC itself fails the pipeline (status
    "failed", Video Assembly never runs) if any asset is still flagged
    misleading/conflicting after bounded replacement is exhausted - see
    visual_qc_node. Pipeline status only becomes "completed" once BGM
    mixing itself succeeds and a final captioned-and-mixed MP4 exists; a
    semantic mood-planning failure inside that stage does not itself fail
    the pipeline (AudioMixingService falls back to a deterministic
    MusicPlan and mixing proceeds) - only an actual mixing/selection/
    catalog failure does. The captioned MP4 (``caption_result``) and every
    earlier-stage result are preserved unchanged either way.

    Args:
        topic: Research topic to investigate, script, narrate, illustrate, and assemble
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research, Script,
            Visual Context Planning, Visual QC, and BGM mood planning
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation for the Visual Media Service
        assembler: VideoAssembler implementation, shared by Visual QC, the
            Video Assembly Service, the Caption Service, and Audio Mixing
        visual_relevance_evaluator: VisualRelevanceEvaluator implementation
            for the Visual QC Service
        transcription_provider: TranscriptionProvider implementation for
            the Caption Service
        music_catalog_provider: MusicCatalogProvider implementation (the
            approved BGM catalog) for the Audio Mixing Service
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into
        subtitle_output_dir: Directory the Caption Service writes .srt files into

    Returns:
        Final PipelineState (check ``.status``/``.error`` for outcome)
    """
    graph = build_pipeline_graph(
        search_provider,
        llm_provider,
        voice_provider,
        voice_name,
        media_provider,
        assembler,
        visual_relevance_evaluator,
        transcription_provider,
        music_catalog_provider,
        voice_output_dir,
        media_output_dir,
        video_output_dir,
        subtitle_output_dir,
    ).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    raw_result = await graph.ainvoke(initial_state)

    return PipelineState(
        topic=raw_result.get("topic", topic),
        research_result=raw_result.get("research_result"),
        script_result=raw_result.get("script_result"),
        voice_result=raw_result.get("voice_result"),
        visual_plan=raw_result.get("visual_plan"),
        visual_result=raw_result.get("visual_result"),
        visual_qc_result=raw_result.get("visual_qc_result"),
        qc_approved_visual_result=raw_result.get("qc_approved_visual_result"),
        video_assembly_result=raw_result.get("video_assembly_result"),
        caption_result=raw_result.get("caption_result"),
        audio_mix_result=raw_result.get("audio_mix_result"),
        status=raw_result.get("status", "unknown"),
        error=raw_result.get("error"),
    )
