# Tests for Research Agent

from __future__ import annotations

import asyncio
import pytest

from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent
from src.llm.mock import MockLLMProvider
from src.services.research_relevance import RelevanceFilterResult, ResearchRelevanceFilter
from src.tools.search_provider import MockSearchProvider, SearchProvider
from src.models.research import ResearchResult, ResearchFact


class _PassthroughRelevanceFilter(ResearchRelevanceFilter):
    """Test double: every result is treated as DIRECT, unconditionally -
    for tests exercising general ResearchAgent parsing/structure/
    provenance mechanics rather than relevance-classification nuance
    itself, which has its own dedicated coverage elsewhere (see
    tests/test_research_relevance.py, tests/test_research_substance.py,
    and TestCurrentNewsRelevanceHardening/TestSubstanceDuplicateHardening
    below)."""

    def __init__(self) -> None:
        super().__init__(llm_provider=None)

    def filter(self, topic: str, results: list) -> RelevanceFilterResult:
        return RelevanceFilterResult(
            kept=list(results), direct=list(results), supporting=[], rejected_count=0,
            semantic_review_performed=False,
        )


class TestResearchAgent:
    """Tests for ResearchAgent functionality."""

    @pytest.fixture
    def mock_providers(self) -> tuple[MockSearchProvider, MockLLMProvider]:
        """Provide mock providers for testing."""
        return MockSearchProvider(), MockLLMProvider()

    @pytest.fixture
    def research_agent(self, mock_providers) -> ResearchAgent:
        """Provide a ResearchAgent with mock providers.

        Uses a permissive relevance_filter and a low min_context_words:
        these tests exercise general parsing/structure/provenance
        mechanics with MockSearchProvider's small canned fixture snippets
        (never written with a 40-word combined-substance threshold in
        mind), not relevance/substance-gate correctness itself, which has
        its own dedicated, thorough coverage elsewhere (see
        tests/test_research_relevance.py, tests/test_research_substance.py,
        TestCurrentNewsRelevanceHardening, TestSubstanceDuplicateHardening
        below)."""
        search_provider, llm_provider = mock_providers
        return ResearchAgent(
            search_provider=search_provider,
            llm_provider=llm_provider,
            max_sources=3,
            relevance_filter=_PassthroughRelevanceFilter(),
            min_context_words=10,
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
        result = await research_agent.research("Why do humans dream?")
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
        result = await research_agent.research("Why do humans dream?")
        # With mock LLM, should get several facts
        assert len(result.facts) >= 1
        # Check fact structure
        fact = result.facts[0]
        assert fact.claim
        assert 0.0 <= fact.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_research_key_points_parsing(self, research_agent) -> None:
        """Key points should be parsed from LLM response."""
        result = await research_agent.research("Why do humans dream?")
        # Mock LLM should return key points
        assert len(result.key_points) >= 1
        # Check they're strings
        for point in result.key_points:
            assert isinstance(point, str)
            assert len(point) > 0

    @pytest.mark.asyncio
    async def test_research_notes_generation(self, research_agent) -> None:
        """Research notes should be generated."""
        result = await research_agent.research("Why do humans dream?")
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
            relevance_filter=_PassthroughRelevanceFilter(),
            min_context_words=5,  # this test is about max_sources, not substance-gate tuning
        )
        result = await agent.research("Test topic")
        # Should have at most 2 sources from our mock
        assert len(result.sources) <= 2

    @pytest.mark.asyncio
    async def test_research_agent_repr(self, research_agent) -> None:
        """Agent should have a reasonable string representation."""
        repr_str = repr(research_agent)
        assert "ResearchAgent" in repr_str


class _EmptySearchProvider:
    """Test double: search always returns no results."""

    async def search(self, query: str, num_results: int = 5):
        return []


class _ExplodingSearchProvider:
    """Test double: search always raises, simulating a provider outage."""

    async def search(self, query: str, num_results: int = 5):
        raise RuntimeError("simulated search outage")


class _RecordingLLMProvider:
    """Test double returning a fixed canned response and recording prompts."""

    def __init__(self, response: str = "A corrected fact.") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _ExplodingLLMProvider:
    """Test double: generate_text always raises."""

    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


class TestResearchFocusedClaim:
    """Tests for ResearchAgent.research_focused_claim - Compliance
    Remediation's bounded, single-claim research refresh (used only when
    the original ResearchResult doesn't already ground a correction)."""

    @pytest.mark.asyncio
    async def test_returns_grounded_fact_on_success(self) -> None:
        llm = _RecordingLLMProvider(response="Atmospheric scattering reddens the moon, not blues it.")
        agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=llm)

        fact = await agent.research_focused_claim("Why is the sky blue?", "the moon has a blue tinge")

        assert isinstance(fact, ResearchFact)
        assert fact.claim == "Atmospheric scattering reddens the moon, not blues it."

    @pytest.mark.asyncio
    async def test_bounded_to_exactly_one_search_and_one_llm_call(self) -> None:
        search = MockSearchProvider()
        llm = _RecordingLLMProvider()
        agent = ResearchAgent(search_provider=search, llm_provider=llm)

        await agent.research_focused_claim("Why is the sky blue?", "the moon has a blue tinge")

        assert len(llm.calls) == 1
        assert "the moon has a blue tinge" in llm.calls[0]

    @pytest.mark.asyncio
    async def test_no_search_results_returns_none(self) -> None:
        agent = ResearchAgent(search_provider=_EmptySearchProvider(), llm_provider=_RecordingLLMProvider())

        fact = await agent.research_focused_claim("Why is the sky blue?", "an obscure claim")

        assert fact is None

    @pytest.mark.asyncio
    async def test_search_failure_returns_none_never_raises(self) -> None:
        agent = ResearchAgent(search_provider=_ExplodingSearchProvider(), llm_provider=_RecordingLLMProvider())

        fact = await agent.research_focused_claim("Why is the sky blue?", "a claim")

        assert fact is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none_never_raises(self) -> None:
        agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=_ExplodingLLMProvider())

        fact = await agent.research_focused_claim("Why is the sky blue?", "a claim")

        assert fact is None

    @pytest.mark.asyncio
    async def test_no_evidence_response_returns_none(self) -> None:
        llm = _RecordingLLMProvider(response="NO_EVIDENCE")
        agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=llm)

        fact = await agent.research_focused_claim("Why is the sky blue?", "an unrelated claim")

        assert fact is None

    @pytest.mark.asyncio
    async def test_empty_topic_or_claim_returns_none(self) -> None:
        agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=_RecordingLLMProvider())

        assert await agent.research_focused_claim("", "a claim") is None
        assert await agent.research_focused_claim("Why is the sky blue?", "") is None


class _FixedSearchProvider(SearchProvider):
    """Test double: returns a fixed, caller-supplied result list (or raises
    a caller-supplied exception), and records every query it was called
    with - for deterministic, network-free provider-routing/substance-gate
    tests."""

    def __init__(self, name: str, results=None, error: Exception | None = None) -> None:
        self._name = name
        self._results = results if results is not None else []
        self._error = error
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, num_results: int = 5):
        self.calls.append(query)
        if self._error:
            raise self._error
        return self._results[:num_results]


def _rich_result(topic: str, url: str, published_at: str | None = None, words: int = 60) -> dict:
    """A single search result whose snippet has >= ``words`` words -
    comfortably above DEFAULT_MIN_CONTEXT_WORDS on its own. Embeds
    ``topic``'s own words so it also clears ResearchRelevanceFilter's
    deterministic keyword-overlap pass (see TestResearchRelevanceFilter for
    dedicated relevance-filtering tests - these fixtures exist to exercise
    routing/substance/provenance behavior independently of relevance).
    Padding/title is derived from ``url`` (never a fixed repeated word,
    and the title never repeats ``topic`` verbatim like the snippet does)
    so two calls for the same topic but different URLs are genuinely
    DISTINCT content, not near-duplicate/syndicated text that
    ``dedupe_by_content`` would legitimately collapse - the snippet alone
    already carries the topical-overlap signal relevance filtering needs."""
    filler = url.rsplit("/", 1)[-1] or "detail"
    snippet = f"{topic}. " + " ".join([filler] * words)
    return {
        "title": filler.replace("-", " ").title(),
        "url": url,
        "snippet": snippet,
        "published_at": published_at,
        "source_name": "Example News",
    }


def _thin_result(topic: str, url: str, words: int = 3) -> dict:
    """A single search result whose snippet is far too short to clear
    DEFAULT_MIN_CONTEXT_WORDS on its own, but whose title still shares
    ``topic``'s own words so it clears relevance filtering independently of
    the (separate) substance check."""
    return {"title": f"{topic}", "url": url, "snippet": " ".join(["x"] * words)}


class TestTopicSourceAwareProviderRouting:
    """Covers STEP 2's core routing requirement: which SearchProvider(s)
    ResearchAgent tries is driven purely by the ``topic_source``
    classification hint, never by the topic's own text/content."""

    @pytest.mark.asyncio
    async def test_evergreen_topic_source_uses_default_provider_only(self) -> None:
        """topic_source=None (evergreen/no classification) must behave
        exactly as before: only the default search_provider is ever
        called, even when a current_news_search_provider is configured."""
        topic = "Why do cats purr?"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/X")])
        current_news = _FixedSearchProvider("current_news", results=[_rich_result(topic, "https://news.example.com/a")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source=None)

        assert result.source_provider == "wikipedia"
        assert wikipedia.calls == ["Why do cats purr?"]
        assert current_news.calls == []  # never invoked for a non-current-news topic

    @pytest.mark.asyncio
    async def test_current_news_topic_source_routes_to_current_news_provider_first(self) -> None:
        """topic_source='current_news' with a sufficient current-news result
        must be satisfied by current_news alone - Wikipedia is never even
        called once current_news already has enough substance."""
        topic = "Company announces new product today"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/X")])
        current_news = _FixedSearchProvider(
            "current_news",
            results=[_rich_result(topic, "https://news.example.com/a", published_at="2026-09-15T12:00:00+00:00")],
        )
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "current_news"
        assert current_news.calls == ["Company announces new product today"]
        assert wikipedia.calls == []  # thin/absent Wikipedia can't have "killed" a result it was never asked for

    @pytest.mark.asyncio
    async def test_current_news_topic_source_with_no_configured_provider_falls_back_to_default(self) -> None:
        """A current-news classified topic with no current_news_search_provider
        wired in (e.g. not configured for this environment) degrades
        gracefully to the exact original single-provider behavior."""
        topic = "Some current event"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/X")])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "wikipedia"

    @pytest.mark.asyncio
    async def test_mixed_topic_path_evergreen_selection_uses_default_provider(self) -> None:
        """The 'mixed' Topic Planner mode can select either an evergreen or
        a current-news candidate; when it selects evergreen (topic_source is
        the evergreen candidate's own source, not 'current_news'), research
        must route to the default provider exactly like a pure-evergreen
        topic would."""
        topic = "Why is the ocean salty?"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Y")])
        current_news = _FixedSearchProvider("current_news", results=[_rich_result(topic, "https://news.example.com/b")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="youtube")

        assert result.source_provider == "wikipedia"
        assert current_news.calls == []


class TestThinWikipediaDoesNotKillValidCurrentNewsResearch:
    @pytest.mark.asyncio
    async def test_exploding_wikipedia_does_not_prevent_sufficient_current_news_result(self) -> None:
        """Even if Wikipedia would outright error for a current-news query,
        that must never surface as a failure when current_news alone
        already cleared the substance bar - Wikipedia isn't consulted."""
        topic = "Breaking story today"
        wikipedia = _FixedSearchProvider("wikipedia", error=RuntimeError("simulated Wikipedia outage"))
        current_news = _FixedSearchProvider(
            "current_news",
            results=[_rich_result(topic, "https://news.example.com/c", published_at="2026-09-16T00:00:00+00:00")],
        )
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "current_news"
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_thin_current_news_is_supplemented_by_wikipedia_not_discarded(self) -> None:
        """A thin (but non-empty) current_news result should not be thrown
        away outright - the chain continues to try Wikipedia as a bounded
        second attempt, and if Wikipedia alone is sufficient, it is used."""
        topic = "Developing story"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Z")])
        current_news = _FixedSearchProvider("current_news", results=[_thin_result(topic, "https://news.example.com/d")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "wikipedia"
        assert current_news.calls == ["Developing story"]
        assert wikipedia.calls == ["Developing story"]


class TestInsufficientResearchExplicitFailure:
    @pytest.mark.asyncio
    async def test_insufficient_material_across_all_providers_raises_explicit_error(self) -> None:
        """When neither current_news nor Wikipedia can produce enough
        combined substance, ResearchAgent must fail loudly and
        deterministically rather than silently synthesizing a script from
        thin/tangential material."""
        topic = "Obscure breaking micro-story"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result(topic, "https://en.wikipedia.org/wiki/Thin")])
        current_news = _FixedSearchProvider("current_news", results=[_thin_result(topic, "https://news.example.com/e")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(topic, topic_source="current_news")

    @pytest.mark.asyncio
    async def test_default_single_provider_path_now_also_raises_for_thin_results(self) -> None:
        """Production-hardening behavior change (deliberate, per audit):
        the default (no topic_source) single-provider path no longer gets
        a free pass - Wikipedia/evergreen results must now satisfy the
        exact same relevance/substance contract as current-news. A
        provider returning results successfully is never itself treated
        as research success."""
        topic = "Why do cats purr?"
        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result(topic, "https://en.wikipedia.org/wiki/Thin")])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(topic)

    @pytest.mark.asyncio
    async def test_default_single_provider_path_succeeds_with_genuinely_sufficient_results(self) -> None:
        """The single-provider path still succeeds normally - it is only
        the previous unconditional tolerance for THIN content that
        changed, not evergreen research succeeding at all."""
        topic = "Why do cats purr?"
        wikipedia = _FixedSearchProvider(
            "wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Purring")]
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        result = await agent.research(topic)

        assert result.topic == "Why do cats purr?"
        assert result.source_provider == "wikipedia"


class TestProvenancePreserved:
    @pytest.mark.asyncio
    async def test_source_urls_and_published_at_preserved_and_aligned(self) -> None:
        topic = "Major event unfolds"
        current_news = _FixedSearchProvider(
            "current_news",
            results=[
                _rich_result(topic, "https://news.example.com/first-angle-analysis", published_at="2026-09-14T08:30:00+00:00"),
                _rich_result(topic, "https://news.example.com/second-angle-reaction", published_at=None),
            ],
        )
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "current_news"
        assert len(result.sources) == 2
        assert len(result.source_published_at) == len(result.sources)
        assert result.source_published_at[0] == "2026-09-14T08:30:00+00:00"
        assert result.source_published_at[1] is None
        assert str(result.sources[0]) == "https://news.example.com/first-angle-analysis"

    @pytest.mark.asyncio
    async def test_legacy_default_path_still_reports_source_provider(self) -> None:
        """source_provider is populated on every path, including the
        pre-existing default single-provider one - useful diagnostic
        metadata, not a behavior change for existing callers who simply
        ignore the new field."""
        agent = ResearchAgent(search_provider=MockSearchProvider(), llm_provider=MockLLMProvider())

        result = await agent.research("Why do humans dream?")

        assert result.source_provider is not None


class TestKeyPointPreambleParsingRegression:
    """Regression test for the real, reproducible Gemini habit of prefixing
    a key-points response with an unmarked, colon-terminated meta-commentary
    line (e.g. 'Based on the provided source material, here are 5 key
    points:') that was previously counted as a fabricated first point."""

    def test_leading_preamble_line_is_dropped(self) -> None:
        text = (
            "Based on the provided source material, here are 5 key points:\n"
            "1. Dreams occur mainly during REM sleep.\n"
            "2. Dreaming helps consolidate memories.\n"
            "3. Most adults dream for about two hours a night.\n"
        )

        points = ResearchAgent._parse_bullet_points(text)

        assert points == [
            "Dreams occur mainly during REM sleep.",
            "Dreaming helps consolidate memories.",
            "Most adults dream for about two hours a night.",
        ]

    def test_leading_blank_lines_before_preamble_still_stripped(self) -> None:
        """The fix tracks the first NON-BLANK line, not raw line index - a
        response with leading blank lines before the preamble must still
        drop the preamble correctly."""
        text = (
            "\n\n"
            "Here are the key points:\n"
            "- Point one.\n"
            "- Point two.\n"
        )

        points = ResearchAgent._parse_bullet_points(text)

        assert points == ["Point one.", "Point two."]

    def test_a_real_bullet_first_line_is_never_dropped(self) -> None:
        """A genuine first bullet point that happens to end with ':' (e.g.
        a point introducing a list of examples) must NOT be stripped -
        only an unmarked line is eligible."""
        text = "- Key finding one: temperatures rose significantly.\n- Key finding two.\n"

        points = ResearchAgent._parse_bullet_points(text)

        assert points == [
            "Key finding one: temperatures rose significantly.",
            "Key finding two.",
        ]

    def test_no_preamble_present_all_points_kept(self) -> None:
        text = "- Point one.\n- Point two.\n- Point three.\n"

        points = ResearchAgent._parse_bullet_points(text)

        assert points == ["Point one.", "Point two.", "Point three."]


def _on_topic_result(topic: str, url: str, words: int = 60) -> dict:
    # Pure filler repetition (no shared "-detail-N" template) so two
    # genuinely distinct results are never accidentally collapsed by
    # dedupe_by_content just for sharing the same numbered-placeholder
    # pattern, and the title never repeats topic verbatim - see
    # _rich_result's identical reasoning above.
    filler = url.rsplit("/", 1)[-1] or "detail"
    return {
        "title": filler.replace("-", " ").title(),
        "url": url,
        "snippet": f"{topic}. " + " ".join([filler] * words),
        "source_name": "Example News",
    }


def _off_topic_result(url: str = "https://example.com/off-topic") -> dict:
    return {
        "title": "Pop star clarifies unrelated tour rumor",
        "url": url,
        "snippet": " ".join(["unrelated entertainment gossip"] * 20),
        "source_name": "Entertainment Daily",
    }


class _PromptCapturingLLMProvider:
    """Test double recording every prompt it was asked to generate text
    for - lets a test assert a rejected source's distinctive text never
    reached the LLM at all (STEP 2: unrelated material cannot silently
    become a section)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "key points" in prompt.lower():
            return "- Genuine on-topic point one.\n- Genuine on-topic point two.\n- Genuine on-topic point three."
        return "A genuine on-topic response."


class TestCurrentNewsRelevanceHardening:
    """End-to-end (ResearchAgent.research) coverage for STEP 1's relevance
    hardening - complements the unit-level tests in
    tests/test_research_relevance.py."""

    @pytest.mark.asyncio
    async def test_unrelated_current_news_result_excluded_from_research_context(self) -> None:
        topic = "Poll shows rising trust in global institutions"
        current_news = _FixedSearchProvider(
            "current_news",
            results=[_on_topic_result(topic, "https://example.com/on-topic"), _off_topic_result()],
        )
        llm = _PromptCapturingLLMProvider()
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=llm,
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        source_strs = [str(s) for s in result.sources]
        assert "https://example.com/on-topic" in source_strs
        assert "https://example.com/off-topic" not in source_strs
        # The rejected source's distinctive wording never reached any prompt.
        assert all("unrelated entertainment gossip" not in p for p in llm.prompts)

    @pytest.mark.asyncio
    async def test_relevant_current_news_results_all_retained(self) -> None:
        """Two genuinely distinct on-topic results (not a repeated
        headline - see TestResearchSubstance/test_research_substance.py
        for dedicated near-duplicate-collapsing coverage) are both kept."""
        topic = "Poll shows rising trust in global institutions"
        current_news = _FixedSearchProvider(
            "current_news",
            results=[
                _on_topic_result(topic, "https://example.com/institutional-trust-survey-methodology"),
                _on_topic_result(topic, "https://example.com/public-opinion-leadership-reaction"),
            ],
        )
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert len(result.sources) == 2

    @pytest.mark.asyncio
    async def test_simplified_query_retry_triggered_when_filtered_context_is_insufficient(self) -> None:
        """When relevance filtering leaves too little context from the
        original query, ResearchAgent retries once with a generically
        simplified query before falling back to Wikipedia."""
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"
        simplified = "Poll shows rising trust in global institutions"

        class _TwoQueryProvider(SearchProvider):
            def __init__(self) -> None:
                self.calls: list[str] = []

            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                self.calls.append(query)
                if query == topic:
                    return [_off_topic_result()]
                return [_on_topic_result(simplified, "https://example.com/retry-hit")]

        provider = _TwoQueryProvider()
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=provider,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert provider.calls == [topic, simplified]
        assert result.source_provider == "current_news"
        assert any("retry-hit" in str(s) for s in result.sources)

    @pytest.mark.asyncio
    async def test_insufficient_relevant_context_after_retry_raises_explicitly(self) -> None:
        """Even after the bounded simplified-query retry, if nothing
        relevant/substantial survives from current_news OR Wikipedia,
        ResearchAgent must fail explicitly rather than proceed."""
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"
        current_news = _FixedSearchProvider("current_news", results=[_off_topic_result(), _off_topic_result("https://example.com/off2")])
        # Wikipedia isn't relevance-filtered (out of this hardening's scope -
        # see module docstring), so it must be made insufficient by SUBSTANCE
        # (too thin) rather than by topic, to prove the overall explicit-
        # failure gate still triggers when current_news has nothing usable.
        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result(topic, "https://example.com/off3")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        with pytest.raises(ResearchAgentError):
            await agent.research(topic, topic_source="current_news")


def _duplicate_headline_result(topic: str, url: str) -> dict:
    """Mimics the real gap this milestone fixed: several outlets
    syndicating the identical headline with a snippet that is effectively
    just the headline restated - title-only, contributes zero substance
    regardless of how many "copies" exist."""
    headline = f"{topic} developments continue"
    return {"title": headline, "url": url, "snippet": headline, "source_name": "Example News"}


def _syndicated_result(topic: str, url: str) -> dict:
    """Genuinely substantive content, but republished byte-identically at
    a different URL - the syndication pattern dedupe_by_content must
    collapse to a single distinct source."""
    body = f"{topic}. A shared syndicated wire report with the same real details repeated across outlets."
    return {
        "title": f"{topic} report",
        "url": url,
        "snippet": body + " " + " ".join(["shared"] * 50),
        "source_name": "Wire Service",
    }


class TestSubstanceDuplicateHardening:
    """End-to-end (ResearchAgent.research) coverage for the duplicated-
    headline/title-only substance gap a real controlled validation
    exposed - complements the unit-level tests in
    tests/test_research_substance.py."""

    @pytest.mark.asyncio
    async def test_duplicate_title_only_results_trigger_simplified_query_retry(self) -> None:
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"
        simplified = "Poll shows rising trust in global institutions"

        class _Provider(SearchProvider):
            def __init__(self) -> None:
                self.calls: list[str] = []

            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                self.calls.append(query)
                if query == topic:
                    # Four outlets, same headline, no real article body -
                    # relevant, but zero distinct substance.
                    return [_duplicate_headline_result(topic, f"https://outlet{i}.com/a") for i in range(4)]
                return [_on_topic_result(simplified, "https://example.com/retry-substantive")]

        provider = _Provider()
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=provider,
        )

        result = await agent.research(topic, topic_source="current_news")

        assert provider.calls == [topic, simplified]
        assert result.source_provider == "current_news"
        assert any("retry-substantive" in str(s) for s in result.sources)

    @pytest.mark.asyncio
    async def test_retry_results_are_content_deduplicated(self) -> None:
        """The simplified-query retry can itself return syndicated
        duplicates (a common real pattern) - the merged final result must
        still be content-deduplicated, not just URL-deduplicated."""
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"

        class _Provider(SearchProvider):
            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                if query == topic:
                    return [_duplicate_headline_result(topic, "https://outlet1.com/a")]
                return [
                    _syndicated_result(topic, "https://outlet2.com/b"),
                    _syndicated_result(topic, "https://outlet3.com/c"),
                ]

        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=_Provider(),
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "current_news"
        assert len(result.sources) == 1  # both syndicated retry copies collapsed to one

    @pytest.mark.asyncio
    async def test_wikipedia_fallback_used_when_retry_remains_thin(self) -> None:
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"

        class _Provider(SearchProvider):
            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                # Both the original and simplified-query attempts only
                # ever find title-only/duplicate results - never substantive.
                return [
                    _duplicate_headline_result(topic, "https://outlet1.com/a"),
                    _duplicate_headline_result(topic, "https://outlet2.com/b"),
                ]

        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Topic")])
        agent = ResearchAgent(
            search_provider=wikipedia, llm_provider=MockLLMProvider(), current_news_search_provider=_Provider()
        )

        result = await agent.research(topic, topic_source="current_news")

        assert result.source_provider == "wikipedia"

    @pytest.mark.asyncio
    async def test_explicit_failure_when_only_duplicate_title_only_content_available_everywhere(self) -> None:
        topic = "Poll shows rising trust in global institutions, less comfort with US as global leader"

        class _Provider(SearchProvider):
            @property
            def name(self) -> str:
                return "current_news"

            async def search(self, query: str, num_results: int = 5):
                return [
                    _duplicate_headline_result(topic, "https://outlet1.com/a"),
                    _duplicate_headline_result(topic, "https://outlet2.com/b"),
                ]

        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result(topic, "https://en.wikipedia.org/wiki/Topic")])
        agent = ResearchAgent(
            search_provider=wikipedia, llm_provider=MockLLMProvider(), current_news_search_provider=_Provider()
        )

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(topic, topic_source="current_news")

    @pytest.mark.asyncio
    async def test_evergreen_wikipedia_duplicate_title_only_content_now_also_raises(self) -> None:
        """Production-hardening behavior change (deliberate, per audit):
        Wikipedia's own results, on the default single-provider/evergreen
        path, now go through the exact same duplicate/title-only substance
        gate as current-news - a duplicated/title-only Wikipedia result is
        no longer silently treated as sufficient just because Wikipedia
        successfully returned something."""
        topic = "Why do cats purr?"
        wikipedia = _FixedSearchProvider(
            "wikipedia",
            results=[_duplicate_headline_result(topic, "https://en.wikipedia.org/wiki/X")],
        )
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        with pytest.raises(ResearchAgentError, match="Insufficient DIRECT research material"):
            await agent.research(topic)


class _KeyPointScriptedLLMProvider:
    """Test double returning a fixed, caller-scripted response for the
    key-points extraction prompt specifically, and a generic response for
    every other prompt (summary/facts/notes/title/etc.)."""

    def __init__(self, key_points_response: str, default_response: str = "A generic response.") -> None:
        self.key_points_response = key_points_response
        self.default_response = default_response

    def generate_text(self, prompt: str) -> str:
        if "key points" in prompt.lower():
            return self.key_points_response
        return self.default_response


class TestKeyPointRefusalValidation:
    """STEP 2 coverage: a model refusal/meta-response must never be
    accepted as a genuine research key point."""

    @pytest.mark.asyncio
    async def test_refusal_response_filtered_out_valid_points_kept(self) -> None:
        topic = "Why do cats purr?"
        response = (
            "- Cats purr through laryngeal muscle vibrations during both inhalation and exhalation.\n"
            "- If you can provide the full article text, I would be happy to extract more key points for you.\n"
            "- Purring may also serve a self-healing function via low-frequency vibration.\n"
        )
        llm = _KeyPointScriptedLLMProvider(response)
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Purring")]),
            llm_provider=llm,
            relevance_filter=_PassthroughRelevanceFilter(),
        )

        result = await agent.research(topic)

        assert len(result.key_points) == 2
        assert all("would be happy to" not in p for p in result.key_points)

    @pytest.mark.asyncio
    async def test_all_refusal_key_points_raises_explicitly(self) -> None:
        topic = "Why do cats purr?"
        response = (
            "- Based on the source material provided, there is no underlying text to extract findings from.\n"
            "- If you can provide the full article text, I would be happy to extract the key points for you.\n"
        )
        llm = _KeyPointScriptedLLMProvider(response)
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Purring")]),
            llm_provider=llm,
            relevance_filter=_PassthroughRelevanceFilter(),
        )

        with pytest.raises(ResearchAgentError, match="refusal/meta-commentary"):
            await agent.research(topic)

    @pytest.mark.asyncio
    async def test_valid_informational_key_points_all_pass(self) -> None:
        topic = "Why do cats purr?"
        response = (
            "- Cats purr through laryngeal muscle vibrations.\n"
            "- Purring occurs during both inhalation and exhalation.\n"
            "- Some evidence suggests purring may aid bone healing.\n"
        )
        llm = _KeyPointScriptedLLMProvider(response)
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Purring")]),
            llm_provider=llm,
            relevance_filter=_PassthroughRelevanceFilter(),
        )

        result = await agent.research(topic)

        assert len(result.key_points) == 3

    @pytest.mark.asyncio
    async def test_research_failure_prevents_script_agent_from_ever_being_invoked(self) -> None:
        """Structural proof that a ResearchAgentError (e.g. from
        all-refusal key points) stops the pipeline before ScriptAgent
        could ever turn refusal/meta-commentary into script sections."""
        topic = "Why do cats purr?"
        response = "- The source material provided lacks any real information to summarize.\n"
        llm = _KeyPointScriptedLLMProvider(response)
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia", results=[_rich_result(topic, "https://en.wikipedia.org/wiki/Purring")]),
            llm_provider=llm,
            relevance_filter=_PassthroughRelevanceFilter(),
        )
        script_agent = ScriptAgent(llm_provider=llm)

        with pytest.raises(ResearchAgentError):
            research_result = await agent.research(topic)
            await script_agent.generate_script(research_result)  # unreachable if research() raised correctly