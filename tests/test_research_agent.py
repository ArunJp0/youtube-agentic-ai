# Tests for Research Agent

from __future__ import annotations

import asyncio
import pytest

from src.agents.research import ResearchAgent, ResearchAgentError
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import MockSearchProvider
from src.models.research import ResearchResult, ResearchFact


class TestResearchAgent:
    """Tests for ResearchAgent functionality."""

    @pytest.fixture
    def mock_providers(self) -> tuple[MockSearchProvider, MockLLMProvider]:
        """Provide mock providers for testing."""
        return MockSearchProvider(), MockLLMProvider()

    @pytest.fixture
    def research_agent(self, mock_providers) -> ResearchAgent:
        """Provide a ResearchAgent with mock providers."""
        search_provider, llm_provider = mock_providers
        return ResearchAgent(
            search_provider=search_provider,
            llm_provider=llm_provider,
            max_sources=3,
        )

    @pytest.mark.asyncio
    async def test_research_agent_initialization(self, mock_providers) -> None:
        """ResearchAgent should be created with providers."""
        search_provider, llm_provider = mock_providers
        agent = ResearchAgent(
            search_provider=search_provider,
            llm_provider=llm_provider,
        )
        assert agent.search_provider == search_provider
        assert agent.llm_provider == llm_provider

    @pytest.mark.asyncio
    async def test_research_empty_topic(self, research_agent) -> None:
        """Empty topic should raise ResearchAgentError."""
        with pytest.raises(ResearchAgentError, match="Topic cannot be empty"):
            await research_agent.research("")

    @pytest.mark.asyncio
    async def test_research_only_spaces_topic(self, research_agent) -> None:
        """Whitespace-only topic should raise ResearchAgentError."""
        with pytest.raises(ResearchAgentError, match="Topic cannot be empty"):
            await research_agent.research("   ")

    @pytest.mark.asyncio
    async def test_research_successful(self, research_agent) -> None:
        """Successful research should return a ResearchResult."""
        result = await research_agent.research("Why do humans dream?")
        assert isinstance(result, ResearchResult)
        assert result.topic == "Why do humans dream?"
        assert len(result.summary) > 0
        assert isinstance(result.key_points, list)
        assert isinstance(result.facts, list)
        assert isinstance(result.sources, list)

    @pytest.mark.asyncio
    async def test_research_structured_output(self, research_agent) -> None:
        """Result should contain properly structured fields."""
        result = await research_agent.research("Dreams")
        assert isinstance(result.summary, str)
        assert isinstance(result.key_points, list)
        for point in result.key_points:
            assert isinstance(point, str)

        # Check facts structure
        for fact in result.facts:
            assert isinstance(fact, ResearchFact)
            assert isinstance(fact.claim, str)
            assert 0.0 <= fact.confidence <= 1.0
            if fact.source:
                assert isinstance(fact.source, str)

    @pytest.mark.asyncio
    async def test_research_sources_extracted(self, research_agent) -> None:
        """Sources should be extracted from search results."""
        result = await research_agent.research("How does photosynthesis work?")
        assert len(result.sources) > 0
        # Should contain known URLs from mock data
        sources_str = [str(s) for s in result.sources]
        assert any("nature.com" in s for s in sources_str)

    @pytest.mark.asyncio
    async def test_research_fact_parsing(self, research_agent) -> None:
        """Facts should be parsed from LLM response."""
        result = await research_agent.research("Test topic")
        # With mock LLM, should get several facts
        assert len(result.facts) >= 1
        # Check fact structure
        fact = result.facts[0]
        assert fact.claim
        assert 0.0 <= fact.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_research_key_points_parsing(self, research_agent) -> None:
        """Key points should be parsed from LLM response."""
        result = await research_agent.research("Test topic")
        # Mock LLM should return key points
        assert len(result.key_points) >= 1
        # Check they're strings
        for point in result.key_points:
            assert isinstance(point, str)
            assert len(point) > 0

    @pytest.mark.asyncio
    async def test_research_notes_generation(self, research_agent) -> None:
        """Research notes should be generated."""
        result = await research_agent.research("Test topic")
        # Should either be None or a string
        assert result.research_notes is None or isinstance(result.research_notes, str)

    @pytest.mark.asyncio
    async def test_research_max_sources_limit(self) -> None:
        """Agent should respect max_sources parameter."""
        search_provider = MockSearchProvider()
        llm_provider = MockLLMProvider()
        agent = ResearchAgent(
            search_provider=search_provider,
            llm_provider=llm_provider,
            max_sources=2,  # Limit to 2
        )
        result = await agent.research("Test topic")
        # Should have at most 2 sources from our mock
        assert len(result.sources) <= 2

    @pytest.mark.asyncio
    async def test_research_agent_repr(self, research_agent) -> None:
        """Agent should have a reasonable string representation."""
        repr_str = repr(research_agent)
        assert "ResearchAgent" in repr_str