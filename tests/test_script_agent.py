# Tests for Script Agent
from __future__ import annotations

import pytest

from src.agents.script import ScriptAgent, ScriptAgentError
from src.llm.mock import MockLLMProvider
from src.llm.provider import LLMProvider
from src.models.research import ResearchFact, ResearchResult
from src.models.script import ScriptResult, ScriptSection


class ExplodingLLMProvider(LLMProvider):
    """Test double whose generate_text always raises, to simulate LLM failure."""

    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


def _sample_research(
    key_points=None,
    facts=None,
    sources=None,
    research_notes=None,
) -> ResearchResult:
    return ResearchResult(
        topic="Why do humans dream?",
        summary="Dreams occur during REM sleep and help consolidate memories.",
        key_points=key_points
        if key_points is not None
        else [
            "Dreams occur mainly during REM sleep",
            "Dreaming helps consolidate memories",
            "Most adults dream for about two hours a night",
        ],
        facts=facts if facts is not None else [ResearchFact(claim="REM sleep involves rapid eye movement", source="Sleep Foundation", confidence=0.9)],
        sources=sources if sources is not None else ["https://en.wikipedia.org/wiki/Dream"],
        research_notes=research_notes,
    )


class TestScriptAgent:
    """Tests for ScriptAgent functionality. All LLM calls are mocked."""

    @pytest.fixture
    def llm_provider(self) -> MockLLMProvider:
        return MockLLMProvider()

    @pytest.fixture
    def script_agent(self, llm_provider) -> ScriptAgent:
        return ScriptAgent(llm_provider=llm_provider, max_sections=5)

    def test_script_agent_initialization(self, llm_provider) -> None:
        agent = ScriptAgent(llm_provider=llm_provider)
        assert agent.llm_provider == llm_provider
        assert agent.max_sections > 0

    @pytest.mark.asyncio
    async def test_generate_script_none_research_raises(self, script_agent) -> None:
        with pytest.raises(ScriptAgentError, match="ResearchResult is required"):
            await script_agent.generate_script(None)

    @pytest.mark.asyncio
    async def test_generate_script_empty_summary_raises(self, script_agent) -> None:
        research = ResearchResult(topic="Dreams", summary="   ")
        with pytest.raises(ScriptAgentError, match="non-empty summary"):
            await script_agent.generate_script(research)

    @pytest.mark.asyncio
    async def test_generate_script_successful(self, script_agent) -> None:
        research = _sample_research()
        result = await script_agent.generate_script(research)

        assert isinstance(result, ScriptResult)
        assert result.topic == research.topic
        assert len(result.video_title) > 0
        assert len(result.hook) > 0
        assert len(result.introduction) > 0
        assert len(result.conclusion) > 0
        assert len(result.call_to_action) > 0
        assert len(result.sections) > 0
        assert result.estimated_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_generate_script_sections_structure(self, script_agent) -> None:
        research = _sample_research()
        result = await script_agent.generate_script(research)

        for section in result.sections:
            assert isinstance(section, ScriptSection)
            assert len(section.heading) > 0
            assert len(section.narration) > 0
            assert section.estimated_duration_seconds >= 0.0
            assert isinstance(section.source_refs, list)

    @pytest.mark.asyncio
    async def test_generate_script_one_section_per_key_point(self, script_agent) -> None:
        research = _sample_research(
            key_points=["Point A", "Point B"],
        )
        result = await script_agent.generate_script(research)
        assert len(result.sections) == 2

    @pytest.mark.asyncio
    async def test_generate_script_sections_capped_at_max_sections(self, llm_provider) -> None:
        agent = ScriptAgent(llm_provider=llm_provider, max_sections=2)
        research = _sample_research(
            key_points=["Point A", "Point B", "Point C", "Point D"],
        )
        result = await agent.generate_script(research)
        assert len(result.sections) == 2

    @pytest.mark.asyncio
    async def test_generate_script_no_key_points_falls_back_to_overview(self, script_agent) -> None:
        research = _sample_research(key_points=[])
        result = await script_agent.generate_script(research)
        assert len(result.sections) == 1
        assert result.sections[0].heading == "Overview"

    @pytest.mark.asyncio
    async def test_generate_script_sources_propagated(self, script_agent) -> None:
        research = _sample_research(sources=["https://en.wikipedia.org/wiki/Dream", "https://en.wikipedia.org/wiki/REM_sleep"])
        result = await script_agent.generate_script(research)

        assert len(result.sources) == 2
        source_strs = [str(s) for s in result.sources]
        assert any("Dream" in s for s in source_strs)

        for section in result.sections:
            assert section.source_refs == [str(s) for s in research.sources]

    @pytest.mark.asyncio
    async def test_generate_script_notes_carried_over_from_research(self, script_agent) -> None:
        research = _sample_research(research_notes="Some gaps remain in the literature.")
        result = await script_agent.generate_script(research)
        assert result.script_notes == "Some gaps remain in the literature."

    @pytest.mark.asyncio
    async def test_generate_script_llm_failure_wrapped(self) -> None:
        agent = ScriptAgent(llm_provider=ExplodingLLMProvider())
        research = _sample_research()
        with pytest.raises(ScriptAgentError, match="LLM processing failed"):
            await agent.generate_script(research)

    @pytest.mark.asyncio
    async def test_generate_script_empty_research_result(self, script_agent) -> None:
        """A research result with no key points/facts/sources should still produce a script."""
        research = ResearchResult(
            topic="Obscure topic",
            summary="No sources found for this topic.",
            key_points=[],
            facts=[],
            sources=[],
        )
        result = await script_agent.generate_script(research)
        assert isinstance(result, ScriptResult)
        assert len(result.sections) == 1
        assert result.sources == []

    def test_script_agent_repr(self, script_agent) -> None:
        repr_str = repr(script_agent)
        assert "ScriptAgent" in repr_str
