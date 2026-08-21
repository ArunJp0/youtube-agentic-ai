# Full Research -> Script pipeline workflow using LangGraph.
#
# This module does not implement any research or scripting logic itself -
# it only wires the existing ResearchAgent and ScriptAgent together into a
# single LangGraph state machine, passing ResearchResult from the research
# node directly into the script node.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent, ScriptAgentError
from src.models.research import ResearchResult
from src.models.script import ScriptResult


@dataclass
class PipelineState:
    """Shared state for the Research -> Script pipeline workflow."""

    topic: str = ""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    status: str = "pending"  # pending -> researching -> researched -> scripting -> completed -> failed
    error: Optional[str] = None


def build_pipeline_graph(search_provider, llm_provider) -> StateGraph:
    """Build the LangGraph state machine chaining Research Agent -> Script Agent.

    Reuses the existing ResearchAgent and ScriptAgent as-is (no duplicated
    research/scripting logic); this graph only wires their existing async
    interfaces together and shares one PipelineState across both.

    Graph structure:
        START → research ─(success)─→ script → END
                    │
                    └──────(error)───────────→ END

    Args:
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by both agents

    Returns:
        StateGraph ready to be ``.compile()``d
    """
    research_agent = ResearchAgent(search_provider=search_provider, llm_provider=llm_provider)
    script_agent = ScriptAgent(llm_provider=llm_provider)

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
            return {"script_result": result, "status": "completed", "error": None}
        except ScriptAgentError as e:
            return {"script_result": None, "status": "failed", "error": f"Script generation failed: {e}"}
        except Exception as e:
            return {
                "script_result": None,
                "status": "failed",
                "error": f"Unexpected script error: {e}",
            }

    def route_after_research(state: PipelineState) -> str:
        """Only proceed to scripting if research actually produced a result."""
        return "script" if state.research_result is not None else END

    graph = StateGraph(PipelineState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"script": "script", END: END})
    graph.set_finish_point("script")

    return graph


async def run_pipeline(topic: str, search_provider, llm_provider) -> PipelineState:
    """Run the full Research -> Script pipeline and return the final state.

    Unlike the individual research/script workflow convenience functions
    (which raise on failure), this returns the full PipelineState so callers
    can inspect ``status``/``error`` directly, including on partial failure
    (e.g. research succeeded but scripting failed).

    Args:
        topic: Research topic to investigate and script
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by both agents

    Returns:
        Final PipelineState (check ``.status``/``.error`` for outcome)
    """
    graph = build_pipeline_graph(search_provider, llm_provider).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    raw_result = await graph.ainvoke(initial_state)

    return PipelineState(
        topic=raw_result.get("topic", topic),
        research_result=raw_result.get("research_result"),
        script_result=raw_result.get("script_result"),
        status=raw_result.get("status", "unknown"),
        error=raw_result.get("error"),
    )
