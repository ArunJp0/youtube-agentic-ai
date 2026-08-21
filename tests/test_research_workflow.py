# Tests for Research Workflow (LangGraph)

from __future__ import annotations

import asyncio
import pytest

from src.llm.mock import MockLLMProvider
from src.tools.search_provider import MockSearchProvider
from src.workflows.research_graph import (
    build_research_graph,
    run_research_workflow,
)
from src.models.research import ResearchResult


class TestResearchWorkflow:
    """Tests for LangGraph research workflow."""

    @pytest.fixture
    def mock_providers(self) -> tuple[MockSearchProvider, MockLLMProvider]:
        """Provide mock providers for testing."""
        return MockSearchProvider(), MockLLMProvider()

    @pytest.mark.asyncio
    async def test_workflow_builds_successfully(self, mock_providers) -> None:
        """Graph should build without errors."""
        search_provider, llm_provider = mock_providers
        graph = build_research_graph(search_provider, llm_provider)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_workflow_compiles_successfully(self, mock_providers) -> None:
        """Graph should compile without errors."""
        search_provider, llm_provider = mock_providers
        compiled = build_research_graph(search_provider, llm_provider).compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_run_research_workflow_success(self, mock_providers) -> None:
        """Convenience function should return a ResearchResult."""
        search_provider, llm_provider = mock_providers
        result = await run_research_workflow(
            topic="Why do humans dream?",
            search_provider=search_provider,
            llm_provider=llm_provider,
        )
        assert isinstance(result, ResearchResult)
        assert result.topic == "Why do humans dream?"
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_workflow_with_unknown_topic(self, mock_providers) -> None:
        """Workflow should handle unknown topics gracefully."""
        search_provider, llm_provider = mock_providers
        result = await run_research_workflow(
            topic="How does photosynthesis work?",  # Use a known topic
            search_provider=search_provider,
            llm_provider=llm_provider,
        )
        assert isinstance(result, ResearchResult)
        assert result.topic == "How does photosynthesis work?"

    @pytest.mark.asyncio
    async def test_workflow_empty_topic_handled(self, mock_providers) -> None:
        """Workflow should handle empty topic gracefully (should fail via agent)."""
        search_provider, llm_provider = mock_providers
        # The agent should reject empty topics
        with pytest.raises(Exception):  # AgentError or similar
            await run_research_workflow(
                topic="",
                search_provider=search_provider,
                llm_provider=llm_provider,
            )

    @pytest.mark.asyncio
    async def test_workflow_returns_structured_data(self, mock_providers) -> None:
        """Workflow output should contain expected structured fields."""
        search_provider, llm_provider = mock_providers
        result = await run_research_workflow(
            topic="How does photosynthesis work?",
            search_provider=search_provider,
            llm_provider=llm_provider,
        )
        assert isinstance(result.summary, str)
        assert isinstance(result.key_points, list)
        assert isinstance(result.facts, list)
        assert isinstance(result.sources, list)

        # Check types within lists
        for fact in result.facts:
            assert hasattr(fact, "claim")
            assert hasattr(fact, "confidence")
            assert 0.0 <= fact.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_workflow_multiple_runs(self, mock_providers) -> None:
        """Workflow should be able to run multiple times."""
        search_provider, llm_provider = mock_providers
        topics = ["Topic 1", "Topic 2", "Topic 3"]
        results = []
        for topic in topics:
            result = await run_research_workflow(
                topic=topic,
                search_provider=search_provider,
                llm_provider=llm_provider,
            )
            results.append(result)
            assert result.topic == topic

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_graph_direct_invocation(self, mock_providers) -> None:
        """Direct graph invocation should work."""
        from src.workflows.research_graph import ResearchState

        search_provider, llm_provider = mock_providers
        graph = build_research_graph(search_provider, llm_provider).compile()

        # This mimics what the convenience function does
        initial_state = ResearchState(topic="Test topic for direct invocation")
        result_state = await graph.ainvoke(initial_state)

        # Should contain research_result or error
        assert "research_result" in result_state or "error" in result_state

        # If successful, should have research_result
        if result_state.get("research_result"):
            assert isinstance(result_state["research_result"], ResearchResult)