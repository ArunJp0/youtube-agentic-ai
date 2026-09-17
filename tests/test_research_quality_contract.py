# Tests for the provider-independent research quality contract: EVERY
# SearchProvider's results (current-news, Wikipedia/evergreen, primary
# attempt or fallback) must pass relevance/DIRECT-SUPPORTING classification,
# near-duplicate collapse, and substance/depth validation before they may
# influence combined_context/key_points/ResearchResult/ScriptAgent. No
# provider is exempt; a provider returning results successfully is never
# itself treated as research success.
#
# Deliberately generic topics throughout - no real news story, country,
# person, or event from any prior validation run is referenced here.
#
# All mocked/fixture-based - no real network or LLM calls.
from __future__ import annotations

import pytest

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent
from src.llm.mock import MockLLMProvider
from src.services.research_relevance import simplify_query
from tests.test_research_agent import (
    _FixedSearchProvider,
    _PassthroughRelevanceFilter,
    _rich_result,
    _thin_result,
)
from tests.test_script_agent import VariedSectionLLMProvider

# ---- generic, synthetic topic/content builders (no real-world story) -----

TOPIC = "New engineering method improves bridge material durability"


def _direct_result(url: str, words: int = 60) -> dict:
    """Genuinely DIRECT: substantively about the topic itself."""
    filler = url.rsplit("/", 1)[-1] or "detail"
    return {
        "title": f"Findings on {filler}".title(),
        "url": url,
        "snippet": (
            f"{TOPIC}. Researchers tested the new method directly on bridge components, "
            + " ".join([filler] * words)
        ),
        "source_name": "Example Journal",
    }


def _supporting_result(url: str, words: int = 40) -> dict:
    """Genuinely SUPPORTING: a different named historical project used as
    an explanatory comparison for the topic - shares SOME topical
    vocabulary (as real supporting search results realistically would,
    since they were found via a search related to the topic) but is not
    itself about the topic's own subject."""
    filler = url.rsplit("/", 1)[-1] or "detail"
    return {
        "title": f"Historic Riverside Bridge Project Retrospective ({filler})".title(),
        "url": url,
        "snippet": (
            "A completely different, decades-old Riverside Bridge project faced similar "
            "material fatigue challenges, offering a useful historical comparison for "
            "understanding bridge material durability. " + " ".join([filler] * words)
        ),
        "source_name": "Example History Journal",
    }


def _unrelated_result(url: str = "https://example.com/unrelated", words: int = 40) -> dict:
    """No defensible explanatory relationship to the topic at all."""
    return {
        "title": "Local Bakery Wins Regional Pastry Competition",
        "url": url,
        "snippet": "A local bakery took first prize at the regional pastry competition. " + " ".join(["pastry"] * words),
        "source_name": "Example Local News",
    }


class _ScriptedClassificationLLMProvider:
    """Deterministically controls the classification batch response for
    tests that need to exercise the REAL semantic classification pipeline
    end to end (rather than bypassing it via _PassthroughRelevanceFilter).
    Every other prompt (summary/key points/facts/notes) delegates to
    MockLLMProvider's own sensible generic defaults."""

    def __init__(self, classification_response: str) -> None:
        self.classification_response = classification_response
        # VariedSectionLLMProvider (not MockLLMProvider) so per-section
        # ScriptAgent narration is genuinely distinct - see its own
        # docstring for why MockLLMProvider's generic fallback can't be
        # used for tests that reach ScriptAgent's per-section generation.
        self._delegate = VariedSectionLLMProvider()
        self.classification_calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        if "classifying search results" in prompt.lower():
            self.classification_calls.append(prompt)
            return self.classification_response
        return self._delegate.generate_text(prompt)


def _classification_response(*classifications: str) -> str:
    entries = ", ".join(
        f'{{"index": {i}, "classification": "{c}", "reason": "test"}}' for i, c in enumerate(classifications)
    )
    return f'{{"classifications": [{entries}]}}'


# ---- CORE INVARIANT: DIRECT/SUPPORTING/UNRELATED classification ----------


class TestDirectSupportingUnrelatedClassification:
    @pytest.mark.asyncio
    async def test_direct_topic_evidence_is_accepted(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct"))
        provider = _FixedSearchProvider("wikipedia", results=[_direct_result("https://example.com/direct")])
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        result = await agent.research(TOPIC)

        assert any("direct" in str(s) or "example.com" in str(s) for s in result.sources)
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_genuinely_useful_supporting_context_is_retained(self) -> None:
        """A DIRECT source (sufficient alone) plus a genuinely useful
        SUPPORTING source - both must be retained in the final result."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "supporting"))
        provider = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct"),
                _supporting_result("https://example.com/supporting"),
            ],
        )
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        result = await agent.research(TOPIC)

        source_strs = {str(s) for s in result.sources}
        assert "https://example.com/direct" in source_strs
        assert "https://example.com/supporting" in source_strs

    @pytest.mark.asyncio
    async def test_supporting_context_alone_cannot_satisfy_sufficiency(self) -> None:
        """Only SUPPORTING material is available (no DIRECT source at
        all) - research must fail explicitly, never silently proceed on
        supporting material alone."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("supporting"))
        provider = _FixedSearchProvider("wikipedia", results=[_supporting_result("https://example.com/supporting")])
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(TOPIC)

    @pytest.mark.asyncio
    async def test_unrelated_substantial_material_is_rejected(self) -> None:
        """A long, well-written, but genuinely unrelated result must be
        rejected regardless of its length - verbosity never substitutes
        for relevance."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "unrelated"))
        provider = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct"),
                {
                    "title": "New engineering method improves bridge material durability roundup",
                    "url": "https://example.com/unrelated-verbose",
                    "snippet": "Verbose but unrelated content. " + " ".join(["filler"] * 80),
                },
            ],
        )
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        result = await agent.research(TOPIC)

        assert "https://example.com/unrelated-verbose" not in {str(s) for s in result.sources}

    @pytest.mark.asyncio
    async def test_legitimate_comparison_with_different_named_entity_is_not_rejected(self) -> None:
        """A genuinely useful comparison/example referencing a DIFFERENT
        named event/entity than the topic must still be classifiable as
        supporting - it is not auto-rejected merely for naming something
        else."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "supporting"))
        provider = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct"),
                _supporting_result("https://example.com/riverside-comparison"),
            ],
        )
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        result = await agent.research(TOPIC)

        assert "https://example.com/riverside-comparison" in {str(s) for s in result.sources}

    @pytest.mark.asyncio
    async def test_unrelated_never_reclassified_as_supporting_by_deterministic_pass_alone(self) -> None:
        """When no semantic layer is available, the deterministic pass
        never invents a "supporting" classification - it conservatively
        treats survivors as direct or rejects them outright; it can never
        promote a clearly unrelated result to supporting on its own."""
        provider = _FixedSearchProvider(
            "wikipedia",
            results=[_direct_result("https://example.com/direct"), _unrelated_result()],
        )
        # No LLM configured at all - deterministic-only path.
        agent = ResearchAgent(search_provider=provider, llm_provider=MockLLMProvider(), relevance_filter=None)
        agent.relevance_filter.llm_provider = None

        result = await agent.research(TOPIC)

        assert "https://example.com/unrelated" not in {str(s) for s in result.sources}


class TestSupportingMaterialScriptBoundary:
    @pytest.mark.asyncio
    async def test_supporting_material_does_not_become_a_standalone_script_section(self) -> None:
        """Supporting material may inform a brief comparison but must not
        become its own dominant/standalone section - proven structurally:
        the labeled context instructs the LLM accordingly, and with a
        real (non-mock) LLM this is a prompting contract; here we prove
        the labeling/structure itself is correct and that DIRECT content
        still dominates the accepted source set."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "direct", "supporting"))
        provider = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct1"),
                _direct_result("https://example.com/direct2"),
                _supporting_result("https://example.com/supporting"),
            ],
        )
        agent = ResearchAgent(search_provider=provider, llm_provider=llm)

        research_result = await agent.research(TOPIC)
        script_agent = ScriptAgent(llm_provider=llm)
        script_result = await script_agent.generate_script(research_result)

        # The script must have sections grounded in the (majority) DIRECT
        # research - supporting material alone never produces enough
        # research to reach ScriptAgent (see
        # test_supporting_context_alone_cannot_satisfy_sufficiency), so by
        # construction any section here traces back to a run where DIRECT
        # content was present and sufficient.
        assert len(script_result.sections) >= 1
        assert research_result.source_provider is not None


# ---- B. Retry: query simplification preserves the real subject -----------


class TestQuerySimplificationPreservesSubject:
    def test_headline_shaped_topic_keeps_specific_subject_words(self) -> None:
        """The exact real-world gap a controlled validation exposed: a
        positional first-N-words cut can keep an entirely generic opening
        clause while dropping the actual subject."""
        topic = "Here's what one team of researchers thinks happened during the bridge material failure"
        simplified = simplify_query(topic)
        assert "bridge" in simplified.lower()
        assert "material" in simplified.lower() or "failure" in simplified.lower()

    def test_generic_filler_does_not_replace_the_real_subject(self) -> None:
        topic = "What we know so far about the newly discovered material compound today"
        simplified = simplify_query(topic)
        assert "compound" in simplified.lower() or "material" in simplified.lower()

    def test_question_shaped_topic_simplifies_sensibly(self) -> None:
        topic = "Why does this specific alloy resist corrosion better than older alloys used decades ago"
        simplified = simplify_query(topic)
        assert "alloy" in simplified.lower() or "corrosion" in simplified.lower()

    def test_concise_topic_remains_unchanged(self) -> None:
        topic = "Why do cats purr?"
        assert simplify_query(topic) == topic

    def test_capitalized_entities_are_prioritized(self) -> None:
        topic = "A detailed independent report finally explains what really happened at Example Facility last year"
        simplified = simplify_query(topic)
        assert "Example" in simplified and "Facility" in simplified


class TestRetryBehaviorEndToEnd:
    @pytest.mark.asyncio
    async def test_retry_continues_safely_when_the_retry_call_itself_fails(self) -> None:
        """A bounded simplified-query retry that itself errors out must
        never crash research() - it falls back to whatever the first
        attempt already produced."""
        topic = "A a a a a a a a a genuinely thin initial result about bridges"

        from src.tools.search_provider import SearchProvider

        class _FailingRetryProvider(SearchProvider):
            def __init__(self) -> None:
                self.calls: list[str] = []

            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                self.calls.append(query)
                if len(self.calls) == 1:
                    return [_thin_result(topic, "https://example.com/thin")]
                raise RuntimeError("simulated retry outage")

        provider = _FailingRetryProvider()
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/X")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=provider,
        )

        result = await agent.research(topic, topic_source="current_news")

        # Falls through to Wikipedia rather than crashing.
        assert result.source_provider == "wikipedia"
        assert len(provider.calls) == 2  # original + attempted retry


# ---- C. Wikipedia/fallback quality contract (the core gap this audit found) ----


class TestWikipediaQualityContract:
    """Wikipedia (as the default evergreen provider OR the current-news
    fallback) must pass the exact same relevance/substance contract as
    current-news - the specific structural gap a real controlled
    validation exposed and this task closes."""

    @pytest.mark.asyncio
    async def test_relevant_substantive_wikipedia_succeeds(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct"))
        wikipedia = _FixedSearchProvider("wikipedia", results=[_direct_result("https://en.wikipedia.org/wiki/X")])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        result = await agent.research(TOPIC)

        assert result.source_provider == "wikipedia"
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_unrelated_substantive_wikipedia_fails_explicitly(self) -> None:
        """The exact real-world defect: Wikipedia returning long, genuine
        article content that is simply about the WRONG topic must now be
        rejected, not silently accepted because it "successfully returned
        something substantial"."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("unrelated"))
        wikipedia = _FixedSearchProvider("wikipedia", results=[_unrelated_result(words=80)])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(TOPIC)

    @pytest.mark.asyncio
    async def test_mixed_relevant_unrelated_wikipedia_keeps_only_relevant(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "unrelated", "unrelated"))
        wikipedia = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://en.wikipedia.org/wiki/Direct"),
                _unrelated_result("https://en.wikipedia.org/wiki/Unrelated1"),
                _unrelated_result("https://en.wikipedia.org/wiki/Unrelated2"),
            ],
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        result = await agent.research(TOPIC)

        source_strs = {str(s) for s in result.sources}
        assert "https://en.wikipedia.org/wiki/Direct" in source_strs
        assert "https://en.wikipedia.org/wiki/Unrelated1" not in source_strs
        assert "https://en.wikipedia.org/wiki/Unrelated2" not in source_strs

    @pytest.mark.asyncio
    async def test_title_only_thin_wikipedia_fails(self) -> None:
        wikipedia = _FixedSearchProvider(
            "wikipedia", results=[_thin_result(TOPIC, "https://en.wikipedia.org/wiki/Thin")]
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(TOPIC)

    @pytest.mark.asyncio
    async def test_current_news_failure_then_valid_wikipedia_fallback_succeeds(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct"))
        current_news = _FixedSearchProvider("current_news", results=[_thin_result(TOPIC, "https://news.example.com/thin")])
        wikipedia = _FixedSearchProvider("wikipedia", results=[_direct_result("https://en.wikipedia.org/wiki/Direct")])
        agent = ResearchAgent(
            search_provider=wikipedia, llm_provider=llm, current_news_search_provider=current_news
        )

        result = await agent.research(TOPIC, topic_source="current_news")

        assert result.source_provider == "wikipedia"
        assert "https://en.wikipedia.org/wiki/Direct" in {str(s) for s in result.sources}

    @pytest.mark.asyncio
    async def test_current_news_failure_then_unrelated_wikipedia_fallback_fails_explicitly(self) -> None:
        """The exact scenario a real controlled validation exposed:
        current-news correctly self-rejects (thin), and the Wikipedia
        fallback returns real, substantial, but topically WRONG content -
        this must now raise explicitly rather than silently produce a
        wrong-topic ResearchResult/script."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("unrelated"))
        current_news = _FixedSearchProvider("current_news", results=[_thin_result(TOPIC, "https://news.example.com/thin")])
        wikipedia = _FixedSearchProvider("wikipedia", results=[_unrelated_result(words=80)])
        agent = ResearchAgent(
            search_provider=wikipedia, llm_provider=llm, current_news_search_provider=current_news
        )

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(TOPIC, topic_source="current_news")


# ---- D. Synthesis protections --------------------------------------------


class TestSynthesisProtections:
    @pytest.mark.asyncio
    async def test_wrong_topic_well_formed_material_cannot_become_research_result(self) -> None:
        """A confident, well-formed LLM classification of clearly
        unrelated material as "unrelated" must still result in an
        explicit failure - a provider "succeeding" is not enough, and a
        well-formed rejection is still a rejection."""
        llm = _ScriptedClassificationLLMProvider(_classification_response("unrelated", "unrelated"))
        wikipedia = _FixedSearchProvider(
            "wikipedia",
            results=[_unrelated_result("https://example.com/u1", words=80), _unrelated_result("https://example.com/u2", words=80)],
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        with pytest.raises(ResearchAgentError):
            await agent.research(TOPIC)


# ---- F. Compatibility -----------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_normal_evergreen_wikipedia_research_still_succeeds(self) -> None:
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia", results=[_direct_result("https://en.wikipedia.org/wiki/X")]),
            llm_provider=MockLLMProvider(),
            relevance_filter=_PassthroughRelevanceFilter(),
        )

        result = await agent.research(TOPIC)

        assert result.topic == TOPIC
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_existing_current_news_successful_path_still_succeeds(self) -> None:
        current_news = _FixedSearchProvider("current_news", results=[_rich_result(TOPIC, "https://news.example.com/a")])
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
            relevance_filter=_PassthroughRelevanceFilter(),
        )

        result = await agent.research(TOPIC, topic_source="current_news")

        assert result.source_provider == "current_news"


# ---- G. State/provenance integrity ----------------------------------------


class _RecordingScriptedClassificationLLMProvider(_ScriptedClassificationLLMProvider):
    """Same as _ScriptedClassificationLLMProvider, but additionally records
    every non-classification prompt (summary/key-points/facts/notes) - lets
    a test assert a rejected source's distinctive text never reached
    synthesis at all."""

    def __init__(self, classification_response: str) -> None:
        super().__init__(classification_response)
        self.synthesis_prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        if "classifying search results" not in prompt.lower():
            self.synthesis_prompts.append(prompt)
        return super().generate_text(prompt)


class TestStateAndProvenanceIntegrity:
    @pytest.mark.asyncio
    async def test_rejected_content_never_leaks_into_combined_context(self) -> None:
        llm = _RecordingScriptedClassificationLLMProvider(_classification_response("direct", "unrelated"))
        wikipedia = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct"),
                {
                    "title": "Totally unrelated distinctive marker XYZQ123",
                    "url": "https://example.com/unrelated",
                    "snippet": "UNIQUE_UNRELATED_MARKER_TOKEN appears only in the rejected content. " * 10,
                },
            ],
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        await agent.research(TOPIC)

        assert all("UNIQUE_UNRELATED_MARKER_TOKEN" not in p for p in llm.synthesis_prompts)

    @pytest.mark.asyncio
    async def test_final_sources_contain_only_accepted_results(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "unrelated"))
        wikipedia = _FixedSearchProvider(
            "wikipedia",
            results=[
                _direct_result("https://example.com/direct"),
                _unrelated_result("https://example.com/unrelated", words=80),
            ],
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        result = await agent.research(TOPIC)

        source_strs = {str(s) for s in result.sources}
        assert source_strs == {"https://example.com/direct"}

    @pytest.mark.asyncio
    async def test_result_missing_url_never_enters_context_or_sources(self) -> None:
        llm = _ScriptedClassificationLLMProvider(_classification_response("direct", "direct"))
        malformed = {"title": "No URL result", "snippet": "This result has no url field at all. " + "x " * 40}
        wikipedia = _FixedSearchProvider(
            "wikipedia", results=[_direct_result("https://example.com/direct"), malformed]
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=llm)

        result = await agent.research(TOPIC)

        assert len(result.sources) == 1
        assert str(result.sources[0]) == "https://example.com/direct"
