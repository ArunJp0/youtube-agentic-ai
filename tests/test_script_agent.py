# Tests for Script Agent
from __future__ import annotations

import pytest

from src.agents.script import ScriptAgent, ScriptAgentError
from src.llm.provider import LLMProvider
from src.models.research import ResearchFact, ResearchResult
from src.models.script import ScriptResult, ScriptSection


class ExplodingLLMProvider(LLMProvider):
    """Test double whose generate_text always raises, to simulate LLM failure."""

    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


class RepetitiveSectionLLMProvider(LLMProvider):
    """Test double that can return identical narration for multiple section
    prompts, reproducing the real repeated-narration bug seen when a
    provider's response doesn't vary per key point (e.g. a static mock
    response_map, or a degenerate real LLM response)."""

    def __init__(self, section_responses: dict[str, str] | None = None, default_response: str = "Same repeated narration text.") -> None:
        self.section_responses = section_responses or {}
        self.default_response = default_response
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "Point to expand on:" in prompt:
            for point_marker, response in self.section_responses.items():
                if point_marker in prompt:
                    return response
            return self.default_response
        if "video title" in prompt.lower():
            return "Some Title"
        if "HOOK" in prompt:
            return "Some hook."
        if "INTRODUCTION" in prompt:
            return "Some introduction."
        if "CONCLUSION" in prompt:
            return "Some conclusion."
        if "call-to-action" in prompt:
            return "Some CTA."
        return "Generic fallback text."


class VariedSectionLLMProvider(LLMProvider):
    """Test double representative of a competent LLM (e.g. real Gemini, as
    observed in production): produces genuinely distinct, non-boilerplate
    narration for each different section point.

    Deliberately NOT MockLLMProvider: MockLLMProvider's generic fallback
    embeds only the prompt's first ~50 characters into ~300 characters of
    fixed boilerplate, so any two different prompts score ~0.93-0.95
    similarity under difflib - too similar to reliably test content-
    distinctness requirements. Used as the default ScriptAgent test
    fixture provider.
    """

    def generate_text(self, prompt: str) -> str:
        if "Point to expand on: '" in prompt:
            point = prompt.split("Point to expand on: '", 1)[1].split("'.", 1)[0]
            # Keep the fixed wrapper text short relative to the point text -
            # a long fixed wrapper (even a "varying" one) can itself push
            # the similarity ratio between two different points above the
            # near-duplicate threshold, since most of the string would be
            # identical regardless of the point.
            return f"{point}. A distinct detail worth covering on its own."
        if "video title" in prompt.lower():
            return "An Engaging Video Title"
        if "HOOK" in prompt:
            return "Here's a hook that grabs attention right away."
        if "INTRODUCTION" in prompt:
            return "Here's an introduction that sets up what this video covers."
        if "CONCLUSION" in prompt:
            return "Here's a conclusion that wraps everything up."
        if "call-to-action" in prompt:
            return "Please like and subscribe for more like this."
        return "Some other generated narration text."


class RetryEventuallySucceedsLLMProvider(LLMProvider):
    """Test double that returns a duplicate on the FIRST attempt for one
    specific point, then genuinely distinct text on retry - verifies
    ScriptAgent actively regenerates instead of only ever dropping."""

    def __init__(self) -> None:
        self.point_attempts: dict[str, int] = {}

    def generate_text(self, prompt: str) -> str:
        if "Point to expand on: '" in prompt:
            point = prompt.split("Point to expand on: '", 1)[1].split("'.", 1)[0]
            self.point_attempts[point] = self.point_attempts.get(point, 0) + 1
            if point == "Point A":
                return "Exact shared text that will collide."
            if point == "Point B":
                if self.point_attempts[point] == 1:
                    return "Exact shared text that will collide."  # duplicate of Point A
                return "Genuinely distinct text for point B after retry."
            return f"Distinct unique content about {point}."
        if "video title" in prompt.lower():
            return "Title"
        if "HOOK" in prompt:
            return "Hook."
        if "INTRODUCTION" in prompt:
            return "Intro."
        if "CONCLUSION" in prompt:
            return "Conclusion."
        if "call-to-action" in prompt:
            return "CTA."
        return "Fallback."


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
    def llm_provider(self) -> LLMProvider:
        # VariedSectionLLMProvider (not MockLLMProvider) so section content
        # genuinely differs per point - see its docstring for why.
        return VariedSectionLLMProvider()

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
    async def test_generate_script_one_section_per_key_point(self) -> None:
        # Uses genuinely distinct per-point responses rather than
        # MockLLMProvider's generic fallback, which embeds only the first
        # ~50 chars of the prompt into ~300 chars of fixed boilerplate -
        # too similar across points to reliably stay under the near-
        # duplicate threshold (correctly so - see TestScriptAgentSectionDeduplication).
        provider = RepetitiveSectionLLMProvider(
            section_responses={"'Point A'": "Distinct text A.", "'Point B'": "Distinct text B."}
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B"])
        result = await agent.generate_script(research)
        assert len(result.sections) == 2

    @pytest.mark.asyncio
    async def test_generate_script_sections_capped_at_max_sections(self) -> None:
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'Point A'": "Distinct text A.",
                "'Point B'": "Distinct text B.",
                "'Point C'": "Distinct text C.",
                "'Point D'": "Distinct text D.",
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=2)
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


class TestScriptAgentSectionDeduplication:
    """Regression tests: ScriptAgent must never emit duplicate section narration.

    Root cause of the reported repeated-narration bug: ScriptAgent called the
    LLM once per key point with no check that the response actually differed
    from a previous section's narration - a provider (mock or real) that
    returns identical text for multiple prompts produced a ScriptResult with
    the same narration repeated across sections.
    """

    @pytest.mark.asyncio
    async def test_all_duplicate_section_responses_raises_validation_error(self) -> None:
        """When there's enough research (>= MIN_DISTINCT_SECTIONS points) but
        the provider degenerates into repeating itself for every section
        (even after retries), ScriptAgent must raise rather than silently
        ship a 1-section script - this is the floor validation catching the
        pathological case that triggered this whole investigation."""
        provider = RepetitiveSectionLLMProvider()  # every section prompt gets the same text
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(
            key_points=[
                "Dreams occur mainly during REM sleep",
                "Dreaming helps consolidate memories",
                "Most adults dream for about two hours a night",
                "Prefrontal cortex activity decreases during dreams",
                "Dreams may aid emotional processing",
            ]
        )

        with pytest.raises(ScriptAgentError, match="distinct section"):
            await agent.generate_script(research)

    @pytest.mark.asyncio
    async def test_mixed_duplicates_keep_first_occurrence_only(self) -> None:
        """Two duplicate pairs get collapsed, but a 5th unique point keeps
        the script above the MIN_DISTINCT_SECTIONS floor."""
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'Point A'": "Text X",
                "'Point B'": "Text X",  # duplicate of Point A -> dropped
                "'Point C'": "Text Y",
                "'Point D'": "Text Y",  # duplicate of Point C -> dropped
                "'Point E'": "Text Z",
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(
            key_points=["Point A", "Point B", "Point C", "Point D", "Point E"]
        )

        result = await agent.generate_script(research)

        assert len(result.sections) == 3
        assert [s.narration for s in result.sections] == ["Text X", "Text Y", "Text Z"]
        # First occurrence's heading/point is the one kept for each pair.
        assert result.sections[0].heading == "Point A"
        assert result.sections[1].heading == "Point C"
        assert result.sections[2].heading == "Point E"

    @pytest.mark.asyncio
    async def test_dedup_is_case_and_whitespace_insensitive(self) -> None:
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'Point A'": "Some Narration Text.",
                "'Point B'": "  some   narration text.  ",  # same content, different case/whitespace
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 1

    @pytest.mark.asyncio
    async def test_all_unique_sections_are_all_kept(self) -> None:
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'Point A'": "Unique text A.",
                "'Point B'": "Unique text B.",
                "'Point C'": "Unique text C.",
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B", "Point C"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 3
        assert [s.narration for s in result.sections] == [
            "Unique text A.",
            "Unique text B.",
            "Unique text C.",
        ]

    @pytest.mark.asyncio
    async def test_near_duplicate_shared_boilerplate_is_deduped(self) -> None:
        """Reproduces the actual reported bug: sections that are NOT
        byte-identical (each has a distinct lead-in) but share the same
        boilerplate body, so most of the text is repeated. This must be
        caught even though a plain exact-match check would miss it. Two
        genuinely unique points keep the script above the floor."""
        boilerplate = (
            ", here are the key findings: The topic involves multiple interconnected "
            "aspects that have been studied extensively. Current understanding "
            "suggests complex interactions between biological, psychological, and "
            "environmental factors."
        )
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'Point A'": "Based on research into Point A" + boilerplate,
                "'Point B'": "Based on research into Point B" + boilerplate,  # near-dup of A -> dropped
                "'Point C'": "REM sleep timing varies across the night, with cycles roughly every ninety minutes.",
                "'Point D'": "Memory consolidation appears to strengthen emotionally significant experiences during sleep.",
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B", "Point C", "Point D"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 3
        assert result.sections[0].narration == "Based on research into Point A" + boilerplate
        assert (
            result.sections[1].narration
            == "REM sleep timing varies across the night, with cycles roughly every ninety minutes."
        )
        assert (
            result.sections[2].narration
            == "Memory consolidation appears to strengthen emotionally significant experiences during sleep."
        )

    @pytest.mark.asyncio
    async def test_retry_recovers_from_a_single_duplicate_attempt(self) -> None:
        """When the first attempt for a point comes back duplicate,
        ScriptAgent must actively retry with a strengthened prompt and use
        the resulting distinct text, rather than only ever dropping."""
        provider = RetryEventuallySucceedsLLMProvider()
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B", "Point C"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 3
        assert result.sections[1].narration == "Genuinely distinct text for point B after retry."
        assert provider.point_attempts["Point B"] == 2  # confirms a retry actually happened
        assert provider.point_attempts["Point A"] == 1  # no retry needed - kept on first try

    @pytest.mark.asyncio
    async def test_floor_validation_not_triggered_below_min_key_points(self) -> None:
        """The MIN_DISTINCT_SECTIONS floor only applies when there was
        "enough research" to begin with - fewer than 3 available key points
        (e.g. because max_sections capped them) must not raise."""
        provider = RepetitiveSectionLLMProvider(
            section_responses={"'Point A'": "Text X", "'Point B'": "Text X"}  # exact dup
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["Point A", "Point B"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 1  # collapsed, but no error - only 2 points available

    @pytest.mark.asyncio
    async def test_legitimately_related_but_distinct_sections_are_kept(self) -> None:
        """Sections about a related topic, but each carrying genuinely
        different information, must NOT be deduped just for being related."""
        provider = RepetitiveSectionLLMProvider(
            section_responses={
                "'REM sleep timing'": (
                    "REM sleep is when the brain is highly active and most vivid "
                    "dreaming occurs."
                ),
                "'Memory consolidation'": (
                    "During sleep, the brain replays and strengthens memories from "
                    "the day, a process linked to dreaming."
                ),
            }
        )
        agent = ScriptAgent(llm_provider=provider, max_sections=5)
        research = _sample_research(key_points=["REM sleep timing", "Memory consolidation"])

        result = await agent.generate_script(research)

        assert len(result.sections) == 2


def _sample_script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why is the sky blue?",
        video_title="Why Is The Sky Blue?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(heading="Rayleigh Effect", narration="Blue light bends more than red light in the air."),
            ScriptSection(
                heading="Atmospheric Effects on the Moon",
                narration="The moon can appear to have a blue tinge from atmospheric scattering.",
            ),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


class RecordingLLMProvider(LLMProvider):
    """Test double returning a fixed canned response and recording prompts."""

    def __init__(self, response: str = "A corrected, grounded sentence.") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class TestReviseSection:
    """Tests for ScriptAgent.revise_section - Compliance Remediation's
    targeted, section-only script correction."""

    def test_only_targeted_section_narration_changes(self) -> None:
        provider = RecordingLLMProvider(response="Atmospheric scattering actually reddens the moon, not blues it.")
        agent = ScriptAgent(llm_provider=provider)
        script = _sample_script()
        research = _sample_research()

        revised = agent.revise_section(
            script, research, section_index=1, finding_description="Moon 'blue tinge' claim is scientifically wrong"
        )

        # Only the targeted section's narration/duration changed.
        assert revised.sections[1].narration == "Atmospheric scattering actually reddens the moon, not blues it."
        assert revised.sections[1].narration != script.sections[1].narration

        # Every other field is byte-identical to the input.
        assert revised.sections[0] == script.sections[0]
        assert revised.hook == script.hook
        assert revised.introduction == script.introduction
        assert revised.conclusion == script.conclusion
        assert revised.call_to_action == script.call_to_action
        assert revised.sources == script.sources
        assert revised.topic == script.topic
        assert revised.video_title == script.video_title

    def test_prompt_grounded_in_research_and_states_the_problem(self) -> None:
        provider = RecordingLLMProvider()
        agent = ScriptAgent(llm_provider=provider)
        script = _sample_script()
        research = _sample_research()

        agent.revise_section(script, research, section_index=1, finding_description="Moon 'blue tinge' claim is wrong")

        prompt = provider.calls[0]
        assert "Moon 'blue tinge' claim is wrong" in prompt
        assert "Atmospheric Effects on the Moon" in prompt
        assert research.summary in prompt

    def test_refreshed_fact_included_in_prompt_when_supplied(self) -> None:
        provider = RecordingLLMProvider()
        agent = ScriptAgent(llm_provider=provider)
        script = _sample_script()
        research = _sample_research()
        refreshed = ResearchFact(claim="Atmospheric scattering reddens the moon during a lunar eclipse.")

        agent.revise_section(
            script, research, section_index=1, finding_description="Moon color claim is wrong", refreshed_fact=refreshed
        )

        assert refreshed.claim in provider.calls[0]

    def test_estimated_duration_recomputed_for_total_script(self) -> None:
        provider = RecordingLLMProvider(response="A much, much longer corrected sentence with many more words in it.")
        agent = ScriptAgent(llm_provider=provider)
        script = _sample_script()
        research = _sample_research()

        revised = agent.revise_section(script, research, section_index=1, finding_description="wrong claim")

        assert revised.estimated_duration_seconds != script.estimated_duration_seconds
        assert revised.sections[1].estimated_duration_seconds > 0

    def test_invalid_section_index_raises(self) -> None:
        provider = RecordingLLMProvider()
        agent = ScriptAgent(llm_provider=provider)
        script = _sample_script()
        research = _sample_research()

        with pytest.raises(ScriptAgentError):
            agent.revise_section(script, research, section_index=99, finding_description="x")

    def test_missing_script_or_research_raises(self) -> None:
        provider = RecordingLLMProvider()
        agent = ScriptAgent(llm_provider=provider)

        with pytest.raises(ScriptAgentError):
            agent.revise_section(None, _sample_research(), section_index=0, finding_description="x")
        with pytest.raises(ScriptAgentError):
            agent.revise_section(_sample_script(), None, section_index=0, finding_description="x")

    def test_llm_failure_raises_script_agent_error(self) -> None:
        agent = ScriptAgent(llm_provider=ExplodingLLMProvider())
        script = _sample_script()
        research = _sample_research()

        with pytest.raises(ScriptAgentError):
            agent.revise_section(script, research, section_index=1, finding_description="wrong claim")

    def test_empty_correction_raises_never_fabricates(self) -> None:
        agent = ScriptAgent(llm_provider=RecordingLLMProvider(response="   "))
        script = _sample_script()
        research = _sample_research()

        with pytest.raises(ScriptAgentError):
            agent.revise_section(script, research, section_index=1, finding_description="wrong claim")
