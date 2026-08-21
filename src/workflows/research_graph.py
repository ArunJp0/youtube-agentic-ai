# Research workflow using LangGraph
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import asyncio

from langgraph.graph import StateGraph

from src.agents.research import ResearchAgent, ResearchAgentError
from src.models.research import ResearchResult


@dataclass
class ResearchState:
    """State for the research workflow."""
    topic: str = ""
    research_result: Optional[ResearchResult] = None
    error: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3


def create_research_agent(
    search_provider,
    llm_provider,
) -> ResearchAgent:
    """Factory to create a configured Research Agent."""
    return ResearchAgent(
        search_provider=search_provider,
        llm_provider=llm_provider,
        max_sources=5,
        timeout_seconds=30.0,
    )


def build_research_graph(
    search_provider,
    llm_provider,
) -> StateGraph:
    """Build the LangGraph state machine for research workflow.

    Graph structure:
        START → research → END

    Args:
        search_provider: Implementation of SearchProvider
        llm_provider: Implementation of LLMProvider

    Returns:
        Compiled StateGraph
    """
    agent = create_research_agent(search_provider, llm_provider)

    async def research_node(state: ResearchState) -> dict:
        """LangGraph node that executes research."""
        try:
            result = await agent.research(state.topic)
            return {
                "research_result": result,
                "error": None,
                "iteration": state.iteration + 1,
            }
        except ResearchAgentError as e:
            return {
                "research_result": None,
                "error": str(e),
                "iteration": state.iteration + 1,
            }
        except Exception as e:
            return {
                "research_result": None,
                "error": f"Unexpected error: {e}",
                "iteration": state.iteration + 1,
            }

    graph = StateGraph(ResearchState)

    # Add the research node with the agent injected (closure captures agent)
    graph.add_node("research", research_node)

    # Define edges - single node workflow
    graph.set_entry_point("research")
    graph.set_finish_point("research")

    return graph


# Pre-compiled graph for convenience
_research_graph = None


def get_research_graph(search_provider, llm_provider) -> StateGraph:
    """Get or create the research graph (singleton pattern)."""
    global _research_graph
    if _research_graph is None:
        graph = build_research_graph(search_provider, llm_provider)
        _research_graph = graph.compile()
    return _research_graph


async def run_research_workflow(topic: str, search_provider, llm_provider) -> ResearchResult:
    """Convenience function to run research synchronously.

    Args:
        topic: Research topic
        search_provider: SearchProvider implementation
        llm_provider: LLMProvider implementation

    Returns:
        ResearchResult from the workflow
    """
    graph = build_research_graph(search_provider, llm_provider).compile()

    initial_state = ResearchState(topic=topic)

    result = await graph.ainvoke(initial_state)

    if result.get("error"):
        raise ResearchAgentError(result["error"])

    return result["research_result"]