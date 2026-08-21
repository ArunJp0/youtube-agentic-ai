# Tests for Script Workflow (LangGraph)
from __future__ import annotations

import pytest

from src.agents.script import ScriptAgentError
from src.llm.mock import MockLLMProvider
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.workflows.script_graph import ScriptState, build_script_graph, run_script_workflow


def _sample_research() -> ResearchResult:
    return ResearchResult(
        topic="Why do humans dream?",
        summary="Dreams occur during REM sleep and help consolidate memories.",
        key_points=["Dreams occur during REM sleep", "Dreaming aids memory consolidation"],
        sources=["https://en.wikipedia.org/wiki/Dream"],
    )


class TestScriptWorkflow:
    """Tests for LangGraph script workflow. All LLM calls are mocked."""

    @pytest.fixture
    def llm_provider(self) -> MockLLMProvider:
        return MockLLMProvider()

    @pytest.mark.asyncio
    async def test_workflow_builds_successfully(self, llm_provider) -> None:
        graph = build_script_graph(llm_provider)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_workflow_compiles_successfully(self, llm_provider) -> None:
        compiled = build_script_graph(llm_provider).compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_run_script_workflow_success(self, llm_provider) -> None:
        research = _sample_research()
        result = await run_script_workflow(research, llm_provider)

        assert isinstance(result, ScriptResult)
        assert result.topic == research.topic
        assert len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_workflow_returns_structured_data(self, llm_provider) -> None:
        research = _sample_research()
        result = await run_script_workflow(research, llm_provider)

        assert isinstance(result.video_title, str)
        assert isinstance(result.hook, str)
        assert isinstance(result.sections, list)
        for section in result.sections:
            assert hasattr(section, "narration")
            assert hasattr(section, "estimated_duration_seconds")

    @pytest.mark.asyncio
    async def test_workflow_missing_research_raises(self, llm_provider) -> None:
        with pytest.raises(ScriptAgentError):
            await run_script_workflow(None, llm_provider)

    @pytest.mark.asyncio
    async def test_graph_direct_invocation(self, llm_provider) -> None:
        research = _sample_research()
        graph = build_script_graph(llm_provider).compile()

        initial_state = ScriptState(research_result=research)
        result_state = await graph.ainvoke(initial_state)

        assert "script_result" in result_state or "error" in result_state
        if result_state.get("script_result"):
            assert isinstance(result_state["script_result"], ScriptResult)

    @pytest.mark.asyncio
    async def test_workflow_multiple_runs(self, llm_provider) -> None:
        research = _sample_research()
        results = []
        for _ in range(3):
            result = await run_script_workflow(research, llm_provider)
            results.append(result)
        assert len(results) == 3
        assert all(isinstance(r, ScriptResult) for r in results)
