# Full Research -> Script -> Voice -> Visual Media -> Video Assembly
# pipeline using LangGraph.
#
# This module does not implement any research, scripting, voice-synthesis,
# visual-media, or video-encoding logic itself - it only wires the existing
# ResearchAgent, ScriptAgent, VoiceService, VisualMediaService, and
# VideoAssemblyService together into a single LangGraph state machine,
# passing each stage's output directly into the next stage's input.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent, ScriptAgentError
from src.agents.visual_context_planner import VisualContextPlanner
from src.models.media import VisualResult
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
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
from src.services.voice_service import DEFAULT_OUTPUT_DIR, VoiceService, VoiceServiceError
from src.tools.ffmpeg_video_assembler import VideoAssembler
from src.tools.media_provider import MediaProvider
from src.tools.voice_provider import VoiceProvider


@dataclass
class PipelineState:
    """Shared state for the Research -> Script -> Voice -> Visual Media ->
    Video Assembly pipeline."""

    topic: str = ""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    voice_result: Optional[VoiceResult] = None
    visual_result: Optional[VisualResult] = None
    video_assembly_result: Optional[VideoAssemblyResult] = None
    # pending -> researching -> researched -> scripted -> voiced -> visualized -> completed -> failed
    status: str = "pending"
    error: Optional[str] = None


def build_pipeline_graph(
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
) -> StateGraph:
    """Build the LangGraph state machine chaining Research -> Script -> Voice
    -> Visual Media -> Video Assembly.

    Reuses the existing ResearchAgent, ScriptAgent, VoiceService,
    VisualMediaService, and VideoAssemblyService as-is (no duplicated
    business logic, no reimplemented FFmpeg calls); this graph only wires
    their existing async interfaces together and shares one PipelineState
    across all five. VoiceService, VisualMediaService, and
    VideoAssemblyService all remain deterministic services here - each is
    invoked directly, not treated as a reasoning agent. VideoAssemblyService
    receives the exact ScriptResult/VoiceResult/VisualResult already
    produced earlier in this same run - nothing is regenerated or re-downloaded.

    Graph structure:
        START → research ─(ok)─→ script ─(ok)─→ voice ─(ok)─→ media ─(ok)─→ video_assembly → END
                    │                  │               │              │
                    └──────(error)─────┴──────(error)───┴───(error)────┴──────(error)──→ END

    Args:
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research and Script agents
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation for the Visual Media Service
        assembler: VideoAssembler implementation for the Video Assembly Service
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into

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
    video_service = VideoAssemblyService(assembler=assembler, output_dir=video_output_dir)

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
            result = await visual_service.generate_visuals(state.script_result, narration_duration)
        except VisualMediaServiceError as e:
            return {
                "visual_result": None,
                "status": "failed",
                "error": f"Visual media generation failed: {e}",
            }
        except Exception as e:
            return {
                "visual_result": None,
                "status": "failed",
                "error": f"Unexpected visual media error: {e}",
            }

        if not result.success:
            # VisualMediaService never raises for search/download failures -
            # it reports them in VisualResult.error instead. Surface that in
            # pipeline state (and keep the VisualResult itself for inspection).
            return {
                "visual_result": result,
                "status": "failed",
                "error": f"Visual media generation failed: {result.error}",
            }

        return {"visual_result": result, "status": "visualized", "error": None}

    async def video_assembly_node(state: PipelineState) -> dict:
        try:
            # The exact ScriptResult/VoiceResult/VisualResult already
            # produced earlier in this run are passed straight through -
            # nothing is regenerated, re-synthesized, or re-downloaded.
            result = await video_service.assemble_video(
                state.script_result, state.voice_result, state.visual_result
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

        return {"video_assembly_result": result, "status": "completed", "error": None}

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
        """Only proceed to video assembly if visual media actually succeeded."""
        return (
            "video_assembly"
            if state.visual_result is not None and state.visual_result.success
            else END
        )

    graph = StateGraph(PipelineState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("voice", voice_node)
    graph.add_node("media", media_node)
    graph.add_node("video_assembly", video_assembly_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"script": "script", END: END})
    graph.add_conditional_edges("script", route_after_script, {"voice": "voice", END: END})
    graph.add_conditional_edges("voice", route_after_voice, {"media": "media", END: END})
    graph.add_conditional_edges(
        "media", route_after_media, {"video_assembly": "video_assembly", END: END}
    )
    graph.set_finish_point("video_assembly")

    return graph


async def run_pipeline(
    topic: str,
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
) -> PipelineState:
    """Run the full Research -> Script -> Voice -> Visual Media -> Video
    Assembly pipeline and return the final state.

    Unlike the individual research/script workflow convenience functions
    (which raise on failure), this returns the full PipelineState so callers
    can inspect ``status``/``error`` directly, including on partial failure
    (e.g. research, scripting, voice, and visual media all succeeded but
    video assembly failed).

    Each stage only runs if the previous one succeeded - a failure at any
    stage short-circuits the rest of the pipeline and it never silently
    continues (see route_after_research/route_after_script/route_after_voice/
    route_after_media). Pipeline status only becomes "completed" once video
    assembly itself succeeds and a real final MP4 exists.

    Args:
        topic: Research topic to investigate, script, narrate, illustrate, and assemble
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research and Script agents
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation for the Visual Media Service
        assembler: VideoAssembler implementation for the Video Assembly Service
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into

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
        voice_output_dir,
        media_output_dir,
        video_output_dir,
    ).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    raw_result = await graph.ainvoke(initial_state)

    return PipelineState(
        topic=raw_result.get("topic", topic),
        research_result=raw_result.get("research_result"),
        script_result=raw_result.get("script_result"),
        voice_result=raw_result.get("voice_result"),
        visual_result=raw_result.get("visual_result"),
        video_assembly_result=raw_result.get("video_assembly_result"),
        status=raw_result.get("status", "unknown"),
        error=raw_result.get("error"),
    )
