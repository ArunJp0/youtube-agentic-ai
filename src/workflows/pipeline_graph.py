# Full Research -> Script -> Voice pipeline workflow using LangGraph.
#
# This module does not implement any research, scripting, or voice-synthesis
# logic itself - it only wires the existing ResearchAgent, ScriptAgent, and
# VoiceService together into a single LangGraph state machine, passing each
# stage's output directly into the next stage's input.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent, ScriptAgentError
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.voice import VoiceResult
from src.services.voice_service import DEFAULT_OUTPUT_DIR, VoiceService, VoiceServiceError
from src.tools.voice_provider import VoiceProvider


@dataclass
class PipelineState:
    """Shared state for the Research -> Script -> Voice pipeline workflow."""

    topic: str = ""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    voice_result: Optional[VoiceResult] = None
    # pending -> researching -> researched -> scripted -> completed -> failed
    status: str = "pending"
    error: Optional[str] = None


def build_pipeline_graph(
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
) -> StateGraph:
    """Build the LangGraph state machine chaining Research -> Script -> Voice.

    Reuses the existing ResearchAgent, ScriptAgent, and VoiceService as-is
    (no duplicated research/scripting/synthesis logic); this graph only
    wires their existing async interfaces together and shares one
    PipelineState across all three. VoiceService remains a deterministic
    service here - it is invoked directly, not treated as a reasoning agent.

    Graph structure:
        START → research ─(success)─→ script ─(success)─→ voice → END
                    │                       │
                    └──────(error)──────────┴──────(error)──────→ END

    Args:
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research and Script agents
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        voice_output_dir: Directory the Voice Service writes audio files into

    Returns:
        StateGraph ready to be ``.compile()``d
    """
    research_agent = ResearchAgent(search_provider=search_provider, llm_provider=llm_provider)
    script_agent = ScriptAgent(llm_provider=llm_provider)
    voice_service = VoiceService(
        voice_provider=voice_provider, voice_name=voice_name, output_dir=voice_output_dir
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

        return {"voice_result": result, "status": "completed", "error": None}

    def route_after_research(state: PipelineState) -> str:
        """Only proceed to scripting if research actually produced a result."""
        return "script" if state.research_result is not None else END

    def route_after_script(state: PipelineState) -> str:
        """Only proceed to voice generation if scripting actually succeeded."""
        return "voice" if state.script_result is not None else END

    graph = StateGraph(PipelineState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("voice", voice_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"script": "script", END: END})
    graph.add_conditional_edges("script", route_after_script, {"voice": "voice", END: END})
    graph.set_finish_point("voice")

    return graph


async def run_pipeline(
    topic: str,
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
) -> PipelineState:
    """Run the full Research -> Script -> Voice pipeline and return the final state.

    Unlike the individual research/script workflow convenience functions
    (which raise on failure), this returns the full PipelineState so callers
    can inspect ``status``/``error`` directly, including on partial failure
    (e.g. research and scripting succeeded but voice synthesis failed).

    Voice generation only runs if scripting succeeded, which itself only
    runs if research succeeded - a failure at any stage short-circuits the
    rest of the pipeline (see route_after_research/route_after_script).

    Args:
        topic: Research topic to investigate, script, and narrate
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research and Script agents
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        voice_output_dir: Directory the Voice Service writes audio files into

    Returns:
        Final PipelineState (check ``.status``/``.error`` for outcome)
    """
    graph = build_pipeline_graph(
        search_provider, llm_provider, voice_provider, voice_name, voice_output_dir
    ).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    raw_result = await graph.ainvoke(initial_state)

    return PipelineState(
        topic=raw_result.get("topic", topic),
        research_result=raw_result.get("research_result"),
        script_result=raw_result.get("script_result"),
        voice_result=raw_result.get("voice_result"),
        status=raw_result.get("status", "unknown"),
        error=raw_result.get("error"),
    )
