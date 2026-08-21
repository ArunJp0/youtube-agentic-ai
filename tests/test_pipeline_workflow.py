# Tests for the Research -> Script pipeline workflow (LangGraph).
# All tests use mock providers only - no real network/API calls.
from __future__ import annotations

import pytest

from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.tools.search_provider import MockSearchProvider, SearchProvider
from src.workflows.pipeline_graph import PipelineState, build_pipeline_graph, run_pipeline


class EmptySearchProvider(SearchProvider):
    """Test double: search always returns no results."""

    async def search(self, query: str, num_results: int = 5):
        return []


class ExplodingLLMProvider(LLMProvider):
    """Test double: generate_text always raises, to simulate LLM failure."""

    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


class TestPipelineWorkflow:
    """Tests for the combined Research -> Script LangGraph pipeline."""

    @pytest.fixture
    def providers(self) -> tuple[MockSearchProvider, MockLLMProvider]:
        return MockSearchProvider(), MockLLMProvider()

    @pytest.mark.asyncio
    async def test_pipeline_builds_and_compiles(self, providers) -> None:
        search_provider, llm_provider = providers
        graph = build_pipeline_graph(search_provider, llm_provider)
        assert graph is not None
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_pipeline_success_end_to_end(self, providers) -> None:
        search_provider, llm_provider = providers
        state = await run_pipeline("Why do humans dream?", search_provider, llm_provider)

        assert isinstance(state, PipelineState)
        assert state.status == "completed"
        assert state.error is None
        assert state.topic == "Why do humans dream?"

    @pytest.mark.asyncio
    async def test_research_result_is_structured(self, providers) -> None:
        search_provider, llm_provider = providers
        state = await run_pipeline("Why do humans dream?", search_provider, llm_provider)

        assert isinstance(state.research_result, ResearchResult)
        assert state.research_result.topic == "Why do humans dream?"
        assert len(state.research_result.summary) > 0

    @pytest.mark.asyncio
    async def test_script_result_is_structured_and_derived_from_research(self, providers) -> None:
        search_provider, llm_provider = providers
        state = await run_pipeline("Why do humans dream?", search_provider, llm_provider)

        assert isinstance(state.script_result, ScriptResult)
        # Script topic must match the research topic it was generated from.
        assert state.script_result.topic == state.research_result.topic
        assert len(state.script_result.sections) > 0

    @pytest.mark.asyncio
    async def test_script_sources_match_research_sources(self, providers) -> None:
        search_provider, llm_provider = providers
        state = await run_pipeline("How does photosynthesis work?", search_provider, llm_provider)

        assert [str(s) for s in state.script_result.sources] == [
            str(s) for s in state.research_result.sources
        ]

    @pytest.mark.asyncio
    async def test_empty_topic_fails_research_and_skips_script(self) -> None:
        search_provider, llm_provider = MockSearchProvider(), MockLLMProvider()
        state = await run_pipeline("", search_provider, llm_provider)

        assert state.status == "failed"
        assert state.error is not None
        assert state.research_result is None
        assert state.script_result is None  # script node must not have run

    @pytest.mark.asyncio
    async def test_no_search_results_still_completes_with_empty_research(self) -> None:
        """MockSearchProvider returning [] is a valid (if sparse) research result,
        not an error - the pipeline should still complete through scripting."""
        llm_provider = MockLLMProvider()
        state = await run_pipeline("obscure topic", EmptySearchProvider(), llm_provider)

        assert state.research_result is not None
        assert state.research_result.sources == []
        assert state.status == "completed"
        assert state.script_result is not None

    @pytest.mark.asyncio
    async def test_script_stage_failure_is_reported_without_crashing(self) -> None:
        search_provider = MockSearchProvider()
        state = await run_pipeline("Why do humans dream?", search_provider, ExplodingLLMProvider())

        # Research itself calls the LLM too, so the exploding provider fails
        # at the research stage; either way the pipeline must fail cleanly.
        assert state.status == "failed"
        assert state.error is not None
        assert state.script_result is None

    @pytest.mark.asyncio
    async def test_pipeline_state_defaults(self) -> None:
        state = PipelineState()
        assert state.topic == ""
        assert state.research_result is None
        assert state.script_result is None
        assert state.status == "pending"
        assert state.error is None

    @pytest.mark.asyncio
    async def test_pipeline_multiple_runs_are_independent(self, providers) -> None:
        search_provider, llm_provider = providers
        state_a = await run_pipeline("Topic A", search_provider, llm_provider)
        state_b = await run_pipeline("Topic B", search_provider, llm_provider)

        assert state_a.topic == "Topic A"
        assert state_b.topic == "Topic B"
        assert state_a.research_result.topic != state_b.research_result.topic
