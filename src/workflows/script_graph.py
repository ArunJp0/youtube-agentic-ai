# Script workflow using LangGraph
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import StateGraph

from src.agents.script import ScriptAgent, ScriptAgentError
from src.models.research import ResearchResult
from src.models.script import ScriptResult


@dataclass
class ScriptState:
    """State for the script workflow."""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    error: Optional[str] = None


def create_script_agent(llm_provider) -> ScriptAgent:
    """Factory to create a configured Script Agent."""
    return ScriptAgent(llm_provider=llm_provider)


def build_script_graph(llm_provider) -> StateGraph:
    """Build the LangGraph state machine for the script workflow.

    Graph structure:
        START → script → END

    Args:
        llm_provider: Implementation of LLMProvider

    Returns:
        Compiled StateGraph
    """
    agent = create_script_agent(llm_provider)

    async def script_node(state: ScriptState) -> dict:
        """LangGraph node that executes script generation."""
        try:
            result = await agent.generate_script(state.research_result)
            return {"script_result": result, "error": None}
        except ScriptAgentError as e:
            return {"script_result": None, "error": str(e)}
        except Exception as e:
            return {"script_result": None, "error": f"Unexpected error: {e}"}

    graph = StateGraph(ScriptState)

    # Add the script node with the agent injected (closure captures agent)
    graph.add_node("script", script_node)

    # Define edges - single node workflow
    graph.set_entry_point("script")
    graph.set_finish_point("script")

    return graph


async def run_script_workflow(research_result: ResearchResult, llm_provider) -> ScriptResult:
    """Convenience function to run the script workflow synchronously.

    Args:
        research_result: Completed ResearchResult from the Research Agent
        llm_provider: LLMProvider implementation

    Returns:
        ScriptResult from the workflow
    """
    graph = build_script_graph(llm_provider).compile()

    initial_state = ScriptState(research_result=research_result)

    result = await graph.ainvoke(initial_state)

    if result.get("error"):
        raise ScriptAgentError(result["error"])

    return result["script_result"]
