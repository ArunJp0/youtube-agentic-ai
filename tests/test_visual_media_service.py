# Tests for VisualMediaService (query generation + asset selection/download).
# Uses MockMediaProvider only - no real network calls.
from __future__ import annotations

import os

import pytest

from src.models.script import ScriptResult, ScriptSection
from src.services.visual_media_service import (
    MAX_QUERY_TERMS,
    VisualMediaService,
    VisualMediaServiceError,
)
from src.tools.media_provider import MediaCandidate, MediaProvider, MockMediaProvider


def _sample_script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK_TEXT",
        introduction="INTRO_TEXT",
        sections=[
            ScriptSection(
                heading="REM Sleep Timing",
                narration="Most vivid dreaming happens during REM sleep cycles at night.",
            ),
            ScriptSection(
                heading="Memory Consolidation",
                narration="The brain strengthens memories and processes emotions while dreaming.",
            ),
        ],
        conclusion="CONCLUSION_TEXT",
        call_to_action="CTA_TEXT",
        sources=["https://en.wikipedia.org/wiki/Dream"],
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


class ExplodingMediaProvider(MediaProvider):
    """Test double whose search always raises, to simulate provider failure."""

    @property
    def name(self) -> str:
        return "exploding"

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        raise RuntimeError("simulated media provider outage")

    async def download(self, candidate, output_path: str) -> None:
        raise RuntimeError("should never be called")


class TestBuildSearchQueries:
    """Tests for deterministic, non-LLM search-query generation."""

    def test_query_derived_from_heading_and_narration(self) -> None:
        section = ScriptSection(
            heading="REM Sleep Timing",
            narration="Most vivid dreaming happens during REM sleep cycles at night.",
        )
        queries = VisualMediaService.build_search_queries(section)
        assert queries  # non-empty list
        assert "sleep" in queries[0].lower() or "dream" in queries[0].lower()

    def test_query_is_not_hardcoded_to_any_topic(self) -> None:
        """The generator must derive queries from the section itself, not
        contain any built-in topic-specific term like 'dreams'."""
        section = ScriptSection(
            heading="Photosynthesis Basics",
            narration="Plants convert sunlight into chemical energy using chlorophyll.",
        )
        query = VisualMediaService.build_search_queries(section)[0]
        assert "dream" not in query.lower()
        assert any(
            w in query.lower()
            for w in ["photosynthesis", "sunlight", "chemical", "chlorophyll", "plants", "energy"]
        )

    def test_query_excludes_stopwords(self) -> None:
        section = ScriptSection(
            heading="The Big Idea", narration="This is about the way we live and work."
        )
        query = VisualMediaService.build_search_queries(section)[0]
        for stopword in ("the", "is", "about", "way", "we"):
            assert stopword not in query.lower().split()

    def test_query_never_empty_even_for_short_heading(self) -> None:
        section = ScriptSection(heading="It", narration="We are.")
        queries = VisualMediaService.build_search_queries(section)
        assert queries
        assert queries[0].strip() != ""

    def test_specific_query_capped_at_max_terms(self) -> None:
        section = ScriptSection(
            heading="Elephants Giraffes Rhinoceros Crocodiles Alligators",
            narration="Extraordinary fascinating wonderful magnificent creatures roam savannas.",
        )
        query = VisualMediaService.build_search_queries(section)[0]
        assert len(query.split()) <= MAX_QUERY_TERMS

    def test_same_section_produces_same_queries(self) -> None:
        section = ScriptSection(heading="REM Sleep Timing", narration="Dreams occur during REM sleep.")
        assert VisualMediaService.build_search_queries(
            section
        ) == VisualMediaService.build_search_queries(section)

    def test_fallback_chain_includes_broader_topic_and_last_resort(self) -> None:
        section = ScriptSection(
            heading="REM Sleep Timing",
            narration="Most vivid dreaming happens during REM sleep cycles at night.",
        )
        queries = VisualMediaService.build_search_queries(section, topic="Why do humans dream?")
        assert len(queries) >= 2  # at least a specific query plus some fallback
        assert queries[-1] == "background footage"  # generic last resort always present

    def test_queries_are_deduplicated(self) -> None:
        """If the specific/broader/topic queries happen to coincide, the
        chain shouldn't contain the same string twice."""
        section = ScriptSection(heading="Dreams", narration="Dreams.")
        queries = VisualMediaService.build_search_queries(section, topic="Dreams")
        assert len(queries) == len(set(queries))


class TestConceptMappingRegression:
    """Regression tests for the reported semantically-weak queries.

    These use the literal reported section text as concrete examples, but
    every assertion is about generic properties (concreteness, absence of
    scientific jargon, boost-word presence) - not a hardcoded expectation
    that only holds for this one topic. TestBuildSearchQueries.
    test_query_is_not_hardcoded_to_any_topic already proves the logic
    generalizes to an unrelated topic (photosynthesis).
    """

    @pytest.mark.parametrize(
        "heading,narration,forbidden_terms",
        [
            (
                "Prefrontal cortex suppression creates dream illogic",
                "The part of the brain responsible for logic quiets down during dreams, "
                "which is why dream scenarios can feel so strange and illogical.",
                ["suppression", "creates", "illogic", "illogical", "mainly"],
            ),
            (
                "Most adults spend about 2 hours per night dreaming",
                "Each individual dream typically lasts around five to twenty minutes.",
                ["spend", "spends", "about"],
            ),
            (
                "Dreams occur mainly during REM sleep cycles",
                "Most vivid dreaming happens during REM sleep, a stage that repeats "
                "every ninety minutes.",
                ["mainly", "during", "occur"],
            ),
            (
                "Brain consolidates memories and processes emotions while dreaming",
                "While we dream, the brain sorts through the day's experiences and "
                "processes emotional moments.",
                ["consolidates", "processes", "while"],
            ),
            (
                "Dreams may serve evolutionary functions like threat simulation",
                "One theory suggests dreaming evolved as a safe rehearsal space for "
                "reacting to danger.",
                ["may", "serve", "functions", "simulation"],
            ),
        ],
    )
    def test_specific_query_drops_low_value_scientific_terms(
        self, heading, narration, forbidden_terms
    ) -> None:
        section = ScriptSection(heading=heading, narration=narration)
        query = VisualMediaService.build_search_queries(section, topic="Why do humans dream?")[0]
        query_words = query.lower().split()
        for term in forbidden_terms:
            assert term not in query_words, f"{term!r} should have been filtered from {query!r}"

    @pytest.mark.parametrize(
        "heading,narration,expected_any_of",
        [
            (
                "Prefrontal cortex suppression creates dream illogic",
                "The part of the brain responsible for logic quiets down during dreams.",
                ["brain", "neuroscience", "dream", "sleep"],
            ),
            (
                "Most adults spend about 2 hours per night dreaming",
                "Each individual dream typically lasts around five to twenty minutes.",
                ["person", "night", "sleep", "dream"],
            ),
            (
                "Dreams occur mainly during REM sleep cycles",
                "Most vivid dreaming happens during REM sleep, a stage that repeats.",
                ["dream", "sleep", "person", "sleeping"],
            ),
            (
                "Brain consolidates memories and processes emotions while dreaming",
                "While we dream, the brain sorts through experiences and emotions.",
                ["brain", "memory", "emotions", "feelings"],
            ),
            (
                "Dreams may serve evolutionary functions like threat simulation",
                "One theory suggests dreaming evolved as a rehearsal for danger.",
                ["dream", "sleep", "nature", "evolution", "danger", "survival"],
            ),
        ],
    )
    def test_specific_query_contains_concrete_visual_concept(
        self, heading, narration, expected_any_of
    ) -> None:
        section = ScriptSection(heading=heading, narration=narration)
        query = VisualMediaService.build_search_queries(section, topic="Why do humans dream?")[0]
        query_words = set(query.lower().split())
        assert query_words & set(expected_any_of), f"{query!r} has none of {expected_any_of}"


class TestVisualMediaServiceGenerateVisuals:
    """Tests for VisualMediaService.generate_visuals using MockMediaProvider."""

    @pytest.mark.asyncio
    async def test_generate_visuals_success(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script())

        assert result.success is True
        assert result.error is None
        assert result.provider == "mock"
        assert len(result.sections) == 2
        for mapping in result.sections:
            assert len(mapping.assets) == 1
            asset = mapping.assets[0]
            assert asset.success is True
            assert asset.local_file_path is not None
            assert os.path.exists(asset.local_file_path)

    @pytest.mark.asyncio
    async def test_section_index_and_heading_recorded(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=2)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script())

        assert result.sections[0].section_index == 0
        assert result.sections[0].section_heading == "REM Sleep Timing"
        assert result.sections[1].section_index == 1
        assert result.sections[1].section_heading == "Memory Consolidation"
        assert result.sections[0].assets[0].section_index == 0
        assert result.sections[1].assets[0].section_index == 1

    @pytest.mark.asyncio
    async def test_none_script_raises(self, tmp_path) -> None:
        service = VisualMediaService(
            media_provider=MockMediaProvider(), output_dir=str(tmp_path)
        )
        with pytest.raises(VisualMediaServiceError, match="ScriptResult is required"):
            await service.generate_visuals(None)

    @pytest.mark.asyncio
    async def test_no_sections_returns_unsuccessful_result(self, tmp_path) -> None:
        script = _sample_script(sections=[])
        service = VisualMediaService(
            media_provider=MockMediaProvider(), output_dir=str(tmp_path)
        )
        result = await service.generate_visuals(script)

        assert result.success is False
        assert result.sections == []
        assert "no sections" in result.error.lower()

    @pytest.mark.asyncio
    async def test_provider_failure_captured_per_section(self, tmp_path) -> None:
        service = VisualMediaService(
            media_provider=ExplodingMediaProvider(), output_dir=str(tmp_path)
        )
        result = await service.generate_visuals(_sample_script())

        assert result.success is False
        assert result.error is not None
        for mapping in result.sections:
            assert mapping.assets[0].success is False
            assert "simulated media provider outage" in mapping.assets[0].error

    @pytest.mark.asyncio
    async def test_empty_search_results_exhausts_fallback_chain_then_fails_cleanly(
        self, tmp_path
    ) -> None:
        """A section whose ENTIRE query chain (specific, broader, topic,
        last-resort) returns no candidates must get a clean, recorded
        failure (not a crash), while other sections still succeed."""
        section_a = ScriptSection(heading="Alpha Topic", narration="Content about alpha things entirely.")
        section_b = ScriptSection(heading="Beta Topic", narration="Content about beta things entirely.")
        script = _sample_script(sections=[section_a, section_b])

        all_queries_for_a = set(VisualMediaService.build_search_queries(section_a, script.topic))
        provider = MockMediaProvider(results_per_query=1, empty_for=all_queries_for_a)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script)

        assert result.success is False
        assert result.sections[0].assets[0].success is False
        assert result.sections[1].assets[0].success is True

    @pytest.mark.asyncio
    async def test_fallback_chain_recovers_when_specific_query_is_empty(self, tmp_path) -> None:
        """If the specific (most concrete) query returns nothing, the
        service must retry with the broader/topic/last-resort queries
        rather than failing the section outright."""
        section = ScriptSection(
            heading="Alpha Topic", narration="Content about alpha things entirely."
        )
        script = _sample_script(sections=[section])

        specific_query = VisualMediaService.build_search_queries(section, script.topic)[0]
        provider = MockMediaProvider(results_per_query=1, empty_for={specific_query})
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script)

        assert result.success is True
        assert result.sections[0].assets[0].success is True
        assert result.sections[0].assets[0].search_query != specific_query

    @pytest.mark.asyncio
    async def test_duplicate_assets_avoided_across_sections(self, tmp_path) -> None:
        """If two sections' queries would return the exact same candidate
        URLs, the second section must not reuse an already-used asset."""

        class SameCandidateProvider(MediaProvider):
            @property
            def name(self) -> str:
                return "same-candidate"

            async def search(self, query, prefer_video=True, max_results=5):
                # Every query returns the exact same single candidate.
                return [
                    MediaCandidate(
                        asset_type="video",
                        download_url="https://mock.media/same.mp4",
                        source_url="https://mock.media/page/same",
                        attribution="Someone",
                        width=1920,
                        height=1080,
                        duration_seconds=5.0,
                    )
                ]

            async def download(self, candidate, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"DATA")

        service = VisualMediaService(
            media_provider=SameCandidateProvider(), output_dir=str(tmp_path)
        )
        result = await service.generate_visuals(_sample_script())

        # First section gets the asset; second section can't reuse the same
        # URL and, since the provider only ever offers that one URL, fails
        # cleanly rather than duplicating it.
        assert result.sections[0].assets[0].success is True
        assert result.sections[1].assets[0].success is False
        assert "No suitable" in result.sections[1].assets[0].error

    @pytest.mark.asyncio
    async def test_duplicate_candidate_skipped_in_favor_of_next(self, tmp_path) -> None:
        """When a search returns multiple candidates and the first is
        already used, the service should pick the next non-duplicate one
        rather than failing."""

        class TwoCandidatesProvider(MediaProvider):
            @property
            def name(self) -> str:
                return "two-candidates"

            async def search(self, query, prefer_video=True, max_results=5):
                return [
                    MediaCandidate(
                        asset_type="video",
                        download_url="https://mock.media/shared.mp4",
                        source_url="https://mock.media/page/shared",
                    ),
                    MediaCandidate(
                        asset_type="video",
                        download_url="https://mock.media/unique.mp4",
                        source_url="https://mock.media/page/unique",
                    ),
                ]

            async def download(self, candidate, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"DATA")

        service = VisualMediaService(
            media_provider=TwoCandidatesProvider(), output_dir=str(tmp_path)
        )
        # Pre-seed generate_visuals by running it, then run again reusing
        # the SAME service instance is not how used_urls works (it's per
        # call) - instead simulate two sections both able to match the
        # shared URL first: first section takes "shared", second section
        # must skip "shared" and take "unique".
        result = await service.generate_visuals(_sample_script())

        assert result.sections[0].assets[0].source_url == "https://mock.media/page/shared"
        assert result.sections[1].assets[0].source_url == "https://mock.media/page/unique"
        assert result.sections[1].assets[0].success is True

    @pytest.mark.asyncio
    async def test_creates_output_dir(self, tmp_path) -> None:
        nested_dir = str(tmp_path / "nested" / "media")
        provider = MockMediaProvider(results_per_query=1)
        service = VisualMediaService(media_provider=provider, output_dir=nested_dir)

        result = await service.generate_visuals(_sample_script())

        assert os.path.isdir(nested_dir)
        assert result.sections[0].assets[0].local_file_path.startswith(nested_dir)

    @pytest.mark.asyncio
    async def test_prefer_video_false_passed_through_to_provider(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=1)
        service = VisualMediaService(
            media_provider=provider, output_dir=str(tmp_path), prefer_video=False
        )

        result = await service.generate_visuals(_sample_script())

        assert result.sections[0].assets[0].asset_type == "image"
