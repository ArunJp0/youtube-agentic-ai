# Tests for Research Agent

from __future__ import annotations

import asyncio
import pytest

from src.agents.research import ResearchAgent, ResearchAgentError
from src.llm.mock import MockLLMProvider
from src.tools.search_provider import MockSearchProvider, SearchProvider
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


def _rich_result(url: str, published_at: str | None = None, words: int = 60) -> dict:
    """A single search result whose snippet has >= ``words`` words -
    comfortably above DEFAULT_MIN_CONTEXT_WORDS on its own."""
    snippet = " ".join(["substantive"] * words)
    return {
        "title": f"Article about {url}",
        "url": url,
        "snippet": snippet,
        "published_at": published_at,
        "source_name": "Example News",
    }


def _thin_result(url: str, words: int = 3) -> dict:
    """A single search result whose snippet is far too short to clear
    DEFAULT_MIN_CONTEXT_WORDS on its own."""
    return {"title": f"Thin result {url}", "url": url, "snippet": " ".join(["x"] * words)}


class TestTopicSourceAwareProviderRouting:
    """Covers STEP 2's core routing requirement: which SearchProvider(s)
    ResearchAgent tries is driven purely by the ``topic_source``
    classification hint, never by the topic's own text/content."""

    @pytest.mark.asyncio
    async def test_evergreen_topic_source_uses_default_provider_only(self) -> None:
        """topic_source=None (evergreen/no classification) must behave
        exactly as before: only the default search_provider is ever
        called, even when a current_news_search_provider is configured."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result("https://en.wikipedia.org/wiki/X")])
        current_news = _FixedSearchProvider("current_news", results=[_rich_result("https://news.example.com/a")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Why do cats purr?", topic_source=None)

        assert result.source_provider == "wikipedia"
        assert wikipedia.calls == ["Why do cats purr?"]
        assert current_news.calls == []  # never invoked for a non-current-news topic

    @pytest.mark.asyncio
    async def test_current_news_topic_source_routes_to_current_news_provider_first(self) -> None:
        """topic_source='current_news' with a sufficient current-news result
        must be satisfied by current_news alone - Wikipedia is never even
        called once current_news already has enough substance."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result("https://en.wikipedia.org/wiki/X")])
        current_news = _FixedSearchProvider(
            "current_news", results=[_rich_result("https://news.example.com/a", published_at="2026-09-15T12:00:00+00:00")]
        )
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Company announces new product today", topic_source="current_news")

        assert result.source_provider == "current_news"
        assert current_news.calls == ["Company announces new product today"]
        assert wikipedia.calls == []  # thin/absent Wikipedia can't have "killed" a result it was never asked for

    @pytest.mark.asyncio
    async def test_current_news_topic_source_with_no_configured_provider_falls_back_to_default(self) -> None:
        """A current-news classified topic with no current_news_search_provider
        wired in (e.g. not configured for this environment) degrades
        gracefully to the exact original single-provider behavior."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result("https://en.wikipedia.org/wiki/X")])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        result = await agent.research("Some current event", topic_source="current_news")

        assert result.source_provider == "wikipedia"

    @pytest.mark.asyncio
    async def test_mixed_topic_path_evergreen_selection_uses_default_provider(self) -> None:
        """The 'mixed' Topic Planner mode can select either an evergreen or
        a current-news candidate; when it selects evergreen (topic_source is
        the evergreen candidate's own source, not 'current_news'), research
        must route to the default provider exactly like a pure-evergreen
        topic would."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result("https://en.wikipedia.org/wiki/Y")])
        current_news = _FixedSearchProvider("current_news", results=[_rich_result("https://news.example.com/b")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Why is the ocean salty?", topic_source="youtube")

        assert result.source_provider == "wikipedia"
        assert current_news.calls == []


class TestThinWikipediaDoesNotKillValidCurrentNewsResearch:
    @pytest.mark.asyncio
    async def test_exploding_wikipedia_does_not_prevent_sufficient_current_news_result(self) -> None:
        """Even if Wikipedia would outright error for a current-news query,
        that must never surface as a failure when current_news alone
        already cleared the substance bar - Wikipedia isn't consulted."""
        wikipedia = _FixedSearchProvider("wikipedia", error=RuntimeError("simulated Wikipedia outage"))
        current_news = _FixedSearchProvider(
            "current_news", results=[_rich_result("https://news.example.com/c", published_at="2026-09-16T00:00:00+00:00")]
        )
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Breaking story today", topic_source="current_news")

        assert result.source_provider == "current_news"
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_thin_current_news_is_supplemented_by_wikipedia_not_discarded(self) -> None:
        """A thin (but non-empty) current_news result should not be thrown
        away outright - the chain continues to try Wikipedia as a bounded
        second attempt, and if Wikipedia alone is sufficient, it is used."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_rich_result("https://en.wikipedia.org/wiki/Z")])
        current_news = _FixedSearchProvider("current_news", results=[_thin_result("https://news.example.com/d")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Developing story", topic_source="current_news")

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
        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result("https://en.wikipedia.org/wiki/Thin")])
        current_news = _FixedSearchProvider("current_news", results=[_thin_result("https://news.example.com/e")])
        agent = ResearchAgent(
            search_provider=wikipedia,
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        with pytest.raises(ResearchAgentError, match="Insufficient research material"):
            await agent.research("Obscure breaking micro-story", topic_source="current_news")

    @pytest.mark.asyncio
    async def test_default_single_provider_path_never_raises_for_thin_results(self) -> None:
        """Regression guard: the pre-existing, default (no topic_source)
        single-provider path must keep its original tolerant behavior -
        thin Wikipedia results alone were never an explicit-failure
        condition before this milestone, and still aren't."""
        wikipedia = _FixedSearchProvider("wikipedia", results=[_thin_result("https://en.wikipedia.org/wiki/Thin")])
        agent = ResearchAgent(search_provider=wikipedia, llm_provider=MockLLMProvider())

        result = await agent.research("Why do cats purr?")

        assert result.topic == "Why do cats purr?"
        assert result.source_provider == "wikipedia"


class TestProvenancePreserved:
    @pytest.mark.asyncio
    async def test_source_urls_and_published_at_preserved_and_aligned(self) -> None:
        current_news = _FixedSearchProvider(
            "current_news",
            results=[
                _rich_result("https://news.example.com/f", published_at="2026-09-14T08:30:00+00:00"),
                _rich_result("https://news.example.com/g", published_at=None),
            ],
        )
        agent = ResearchAgent(
            search_provider=_FixedSearchProvider("wikipedia"),
            llm_provider=MockLLMProvider(),
            current_news_search_provider=current_news,
        )

        result = await agent.research("Major event unfolds", topic_source="current_news")

        assert result.source_provider == "current_news"
        assert len(result.sources) == 2
        assert len(result.source_published_at) == len(result.sources)
        assert result.source_published_at[0] == "2026-09-14T08:30:00+00:00"
        assert result.source_published_at[1] is None
        assert str(result.sources[0]) == "https://news.example.com/f"

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