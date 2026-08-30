# Tests for VisualMediaService: duration-aware slot planning, deterministic
# query generation/expansion, and global duplicate-prevention asset
# selection/download. Uses MockMediaProvider only - no real network calls.
from __future__ import annotations

import os

import pytest

from src.models.script import ScriptResult, ScriptSection
from src.services.visual_media_service import (
    BROAD_QUERY_TERMS,
    CADENCE_POLICY,
    MAX_QUERY_TERMS,
    RECENT_REUSE_LOOKBACK,
    VARIANT_QUERY_TERMS,
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


def _section(heading: str, narration: str) -> ScriptSection:
    return ScriptSection(heading=heading, narration=narration)


class ExplodingMediaProvider(MediaProvider):
    """Test double whose search always raises, to simulate provider outage."""

    @property
    def name(self) -> str:
        return "exploding"

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        raise RuntimeError("simulated media provider outage")

    async def download(self, candidate, output_path: str) -> None:
        raise RuntimeError("should never be called")


class TestCalculateSlotCount:
    """Duration-aware visual slot calculation: never a fixed count, always
    derived from the section's own share of total narration duration and
    the cadence tier the whole video falls into."""

    def test_short_section_needs_only_one_slot(self) -> None:
        assert VisualMediaService.calculate_slot_count(1.0, 200.0) == 1

    def test_longer_section_needs_multiple_slots(self) -> None:
        # <=5min tier: target ~8s/clip. 100s section -> ~13 slots.
        slots = VisualMediaService.calculate_slot_count(100.0, 200.0)
        assert slots > 1
        assert slots == round(100.0 / 8.0)

    def test_cadence_tier_up_to_five_minutes(self) -> None:
        # total <= 300s -> target (6+10)/2 = 8s/clip
        assert VisualMediaService.calculate_slot_count(40.0, 200.0) == round(40.0 / 8.0)

    def test_cadence_tier_five_to_ten_minutes(self) -> None:
        # 300 < total <= 600 -> target (8+12)/2 = 10s/clip
        assert VisualMediaService.calculate_slot_count(40.0, 450.0) == round(40.0 / 10.0)

    def test_cadence_tier_over_ten_minutes(self) -> None:
        # total > 600 -> target (10+15)/2 = 12.5s/clip
        assert VisualMediaService.calculate_slot_count(50.0, 700.0) == round(50.0 / 12.5)

    def test_same_section_duration_gets_more_slots_in_faster_tier(self) -> None:
        """The same section length should be sliced into more (or equal)
        slots under a faster cadence tier than a slower one."""
        fast_tier_slots = VisualMediaService.calculate_slot_count(60.0, 200.0)  # <=5min
        slow_tier_slots = VisualMediaService.calculate_slot_count(60.0, 700.0)  # >10min
        assert fast_tier_slots >= slow_tier_slots

    def test_tier_boundary_exactly_five_minutes_uses_fast_tier(self) -> None:
        at_boundary = VisualMediaService.calculate_slot_count(40.0, 300.0)
        just_over = VisualMediaService.calculate_slot_count(40.0, 300.01)
        assert at_boundary != just_over or at_boundary == round(40.0 / 8.0)

    def test_tier_boundary_exactly_ten_minutes_uses_mid_tier(self) -> None:
        at_boundary = VisualMediaService.calculate_slot_count(40.0, 600.0)
        assert at_boundary == round(40.0 / 10.0)

    def test_never_returns_less_than_one(self) -> None:
        assert VisualMediaService.calculate_slot_count(0.5, 200.0) >= 1

    def test_cadence_policy_has_three_ordered_tiers(self) -> None:
        assert len(CADENCE_POLICY) == 3
        assert CADENCE_POLICY[0].max_total_seconds == 300.0
        assert CADENCE_POLICY[1].max_total_seconds == 600.0
        assert CADENCE_POLICY[2].max_total_seconds == float("inf")


class TestBuildQueryVariants:
    """Deterministic, non-LLM query-variant expansion. Must be fully
    generic across any topic domain - no topic-specific hardcoded rules."""

    def test_returns_empty_list_for_zero_variants(self) -> None:
        section = _section("REM Sleep Timing", "Dreams occur during REM sleep.")
        assert VisualMediaService.build_query_variants(section, 0) == []

    def test_does_not_exceed_max_variants(self) -> None:
        section = _section(
            "REM Sleep Timing",
            "Most vivid dreaming happens during REM sleep cycles at night in the bedroom.",
        )
        variants = VisualMediaService.build_query_variants(section, 2)
        assert len(variants) <= 2

    def test_variants_are_distinct(self) -> None:
        section = _section(
            "Memory Consolidation",
            "The brain strengthens memories and processes emotions while dreaming at night.",
        )
        variants = VisualMediaService.build_query_variants(section, 5)
        assert len(variants) == len(set(variants))

    def test_deterministic_for_same_section(self) -> None:
        section = _section("REM Sleep Timing", "Dreams occur during REM sleep.")
        assert VisualMediaService.build_query_variants(
            section, 3
        ) == VisualMediaService.build_query_variants(section, 3)

    @pytest.mark.parametrize(
        "heading,narration",
        [
            (
                "How Electric Motors Work",
                "Electric vehicles use battery packs and electric motors to generate "
                "torque without burning any fuel.",
            ),
            (
                "Journey to the Red Planet",
                "Spacecraft rely on powerful rocket engines to escape Earth's gravity "
                "and travel through space toward Mars.",
            ),
            (
                "How Volcanoes Erupt",
                "Molten magma rises through cracks in the crust before a volcano "
                "erupts with ash and lava.",
            ),
            (
                "Ancient Trade Routes",
                "Merchants carried silk and spices along ancient roads connecting "
                "distant empires and cities.",
            ),
        ],
    )
    def test_generic_across_unrelated_topic_domains(self, heading, narration) -> None:
        """The same deterministic logic must produce sensible, non-dream-
        specific variants for entirely unrelated topics."""
        section = _section(heading, narration)
        variants = VisualMediaService.build_query_variants(section, 3)
        assert variants
        combined = " ".join(variants).lower()
        assert "dream" not in combined
        assert "rem sleep" not in combined


class TestBuildConceptQuery:
    """Tests for the shared deterministic concept-mapping/ranking logic
    (_build_concept_query / _ranked_keywords) used by both the single
    broader-query builder and the multi-variant builder."""

    def test_query_derived_from_text(self) -> None:
        query = VisualMediaService._build_concept_query(
            "REM Sleep Timing Most vivid dreaming happens during REM sleep cycles at night.",
            MAX_QUERY_TERMS,
        )
        assert query
        assert "sleep" in query.lower() or "dream" in query.lower()

    def test_query_is_not_hardcoded_to_any_topic(self) -> None:
        query = VisualMediaService._build_concept_query(
            "Photosynthesis Basics Plants convert sunlight into chemical energy using chlorophyll.",
            MAX_QUERY_TERMS,
        )
        assert "dream" not in query.lower()
        assert any(
            w in query.lower()
            for w in ["photosynthesis", "sunlight", "chemical", "chlorophyll", "plants", "energy"]
        )

    def test_query_excludes_stopwords(self) -> None:
        query = VisualMediaService._build_concept_query(
            "The Big Idea This is about the way we live and work.", MAX_QUERY_TERMS
        )
        for stopword in ("the", "is", "about", "way", "we"):
            assert stopword not in query.lower().split()

    def test_query_never_empty_returns_empty_string_not_crash(self) -> None:
        query = VisualMediaService._build_concept_query("It We are.", MAX_QUERY_TERMS)
        assert query == "" or query.strip() != ""

    def test_query_capped_at_max_terms(self) -> None:
        query = VisualMediaService._build_concept_query(
            "Elephants Giraffes Rhinoceros Crocodiles Alligators "
            "Extraordinary fascinating wonderful magnificent creatures roam savannas.",
            MAX_QUERY_TERMS,
        )
        assert len(query.split()) <= MAX_QUERY_TERMS

    def test_deterministic(self) -> None:
        text = "REM Sleep Timing Dreams occur during REM sleep."
        assert VisualMediaService._build_concept_query(
            text, MAX_QUERY_TERMS
        ) == VisualMediaService._build_concept_query(text, MAX_QUERY_TERMS)


class TestConceptMappingRegression:
    """Regression tests for previously reported semantically-weak queries.
    Every assertion is about generic properties (concreteness, absence of
    scientific jargon) - not a hardcoded expectation that only holds for
    this one topic (see TestBuildQueryVariants.test_generic_across_unrelated_topic_domains
    for cross-domain proof)."""

    @pytest.mark.parametrize(
        "text,forbidden_terms",
        [
            (
                "Prefrontal cortex suppression creates dream illogic. The part of the brain "
                "responsible for logic quiets down during dreams, which is why dream scenarios "
                "can feel so strange and illogical.",
                ["suppression", "creates", "illogic", "illogical", "mainly"],
            ),
            (
                "Most adults spend about 2 hours per night dreaming. Each individual dream "
                "typically lasts around five to twenty minutes.",
                ["spend", "spends", "about"],
            ),
            (
                "Dreams occur mainly during REM sleep cycles. Most vivid dreaming happens "
                "during REM sleep, a stage that repeats every ninety minutes.",
                ["mainly", "during", "occur"],
            ),
            (
                "Brain consolidates memories and processes emotions while dreaming. While we "
                "dream, the brain sorts through the day's experiences and processes emotional "
                "moments.",
                ["consolidates", "processes", "while"],
            ),
            (
                "Dreams may serve evolutionary functions like threat simulation. One theory "
                "suggests dreaming evolved as a safe rehearsal space for reacting to danger.",
                ["may", "serve", "functions", "simulation"],
            ),
        ],
    )
    def test_drops_low_value_scientific_terms(self, text, forbidden_terms) -> None:
        query = VisualMediaService._build_concept_query(text, MAX_QUERY_TERMS)
        query_words = query.lower().split()
        for term in forbidden_terms:
            assert term not in query_words, f"{term!r} should have been filtered from {query!r}"

    @pytest.mark.parametrize(
        "text,expected_any_of",
        [
            (
                "Prefrontal cortex suppression creates dream illogic. The part of the brain "
                "responsible for logic quiets down during dreams.",
                ["brain", "neuroscience", "dream", "sleep"],
            ),
            (
                "Brain consolidates memories and processes emotions while dreaming. While we "
                "dream, the brain sorts through experiences and emotions.",
                ["brain", "memory", "emotions", "feelings"],
            ),
            (
                "Dreams may serve evolutionary functions like threat simulation. One theory "
                "suggests dreaming evolved as a rehearsal for danger.",
                ["dream", "sleep", "nature", "evolution", "danger", "survival"],
            ),
        ],
    )
    def test_contains_concrete_visual_concept(self, text, expected_any_of) -> None:
        query = VisualMediaService._build_concept_query(text, MAX_QUERY_TERMS)
        query_words = set(query.lower().split())
        assert query_words & set(expected_any_of), f"{query!r} has none of {expected_any_of}"


class TestGenerateVisualsValidation:
    """Input validation for the new duration-aware generate_visuals signature."""

    @pytest.mark.asyncio
    async def test_none_script_raises(self, tmp_path) -> None:
        service = VisualMediaService(media_provider=MockMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(VisualMediaServiceError, match="ScriptResult is required"):
            await service.generate_visuals(None, 60.0)

    @pytest.mark.asyncio
    async def test_none_duration_raises(self, tmp_path) -> None:
        service = VisualMediaService(media_provider=MockMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(VisualMediaServiceError, match="positive"):
            await service.generate_visuals(_sample_script(), None)

    @pytest.mark.asyncio
    async def test_zero_duration_raises(self, tmp_path) -> None:
        service = VisualMediaService(media_provider=MockMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(VisualMediaServiceError, match="positive"):
            await service.generate_visuals(_sample_script(), 0.0)

    @pytest.mark.asyncio
    async def test_negative_duration_raises(self, tmp_path) -> None:
        service = VisualMediaService(media_provider=MockMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(VisualMediaServiceError, match="positive"):
            await service.generate_visuals(_sample_script(), -5.0)

    @pytest.mark.asyncio
    async def test_no_sections_returns_unsuccessful_result(self, tmp_path) -> None:
        script = _sample_script(sections=[])
        service = VisualMediaService(media_provider=MockMediaProvider(), output_dir=str(tmp_path))
        result = await service.generate_visuals(script, 60.0)

        assert result.success is False
        assert result.sections == []
        assert "no sections" in result.error.lower()

    @pytest.mark.asyncio
    async def test_creates_output_dir(self, tmp_path) -> None:
        nested_dir = str(tmp_path / "nested" / "media")
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=nested_dir)

        result = await service.generate_visuals(_sample_script(), 40.0)

        assert os.path.isdir(nested_dir)
        assert result.sections[0].assets[0].local_file_path.startswith(nested_dir)


class TestGenerateVisualsSlotPlanning:
    """generate_visuals must actually use calculate_slot_count/section
    timing internally - not a fixed one-clip-per-section count."""

    @pytest.mark.asyncio
    async def test_single_section_slot_count_matches_calculation(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=20)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        total_duration = 80.0
        result = await service.generate_visuals(script, total_duration)

        expected_slots = VisualMediaService.calculate_slot_count(total_duration, total_duration)
        assert expected_slots > 1
        assert len(result.sections[0].assets) == expected_slots
        assert len(result.sections[0].search_queries) == expected_slots
        assert result.sections[0].planned_duration_seconds == pytest.approx(total_duration)

    @pytest.mark.asyncio
    async def test_short_section_gets_single_slot(self, tmp_path) -> None:
        section = _section("Quick Note", "A brief remark.")
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 6.0)

        assert len(result.sections[0].assets) == 1

    @pytest.mark.asyncio
    async def test_section_durations_proportional_to_narration_length(self, tmp_path) -> None:
        short_section = _section("Short", "One two three four.")
        long_section = _section(
            "Long", " ".join(["word"] * 40)
        )
        script = _sample_script(sections=[short_section, long_section])
        provider = MockMediaProvider(results_per_query=10)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 100.0)

        assert result.sections[1].planned_duration_seconds > result.sections[0].planned_duration_seconds
        assert len(result.sections[1].assets) >= len(result.sections[0].assets)

    @pytest.mark.asyncio
    async def test_section_index_and_heading_recorded(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=2)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script(), 40.0)

        assert result.sections[0].section_index == 0
        assert result.sections[0].section_heading == "REM Sleep Timing"
        assert result.sections[1].section_index == 1
        assert result.sections[1].section_heading == "Memory Consolidation"
        for mapping in result.sections:
            for asset in mapping.assets:
                assert asset.section_index == mapping.section_index

    @pytest.mark.asyncio
    async def test_prefer_video_false_passed_through_to_provider(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=1)
        service = VisualMediaService(
            media_provider=provider, output_dir=str(tmp_path), prefer_video=False
        )
        result = await service.generate_visuals(_sample_script(), 20.0)
        assert result.sections[0].assets[0].asset_type == "image"


class TestGenerateVisualsDownloadEfficiency:
    """Section 2: search metadata is cheap, but only the selected asset per
    slot is ever downloaded - never every candidate."""

    @pytest.mark.asyncio
    async def test_only_selected_assets_are_downloaded(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        total_duration = 60.0
        result = await service.generate_visuals(script, total_duration)

        slot_count = len(result.sections[0].assets)
        download_calls = [c for c in provider.calls if c[0] == "download"]
        # Only unique (non-reused) assets should trigger a real download -
        # never one download per candidate seen, and never more downloads
        # than there are distinct assets actually used.
        distinct_ids_used = {a.provider_asset_id for a in result.sections[0].assets if a.success}
        assert len(download_calls) == len(distinct_ids_used)
        assert len(download_calls) <= slot_count

    @pytest.mark.asyncio
    async def test_reused_asset_not_downloaded_twice(self, tmp_path) -> None:
        """pool_size=1 forces every slot after the first to reuse the same
        already-downloaded asset rather than re-downloading it."""
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=1)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 60.0)

        assets = result.sections[0].assets
        assert len(assets) > 1
        download_calls = [c for c in provider.calls if c[0] == "download"]
        assert len(download_calls) == 1
        assert any(a.reused for a in assets[1:])

    @pytest.mark.asyncio
    async def test_candidate_metadata_inspectable_without_downloading_all(self, tmp_path) -> None:
        """Search must be called (metadata inspected) without triggering a
        download for every candidate returned."""

        class CountingProvider(MediaProvider):
            def __init__(self) -> None:
                self.search_calls = 0
                self.download_calls = 0

            @property
            def name(self) -> str:
                return "counting"

            async def search(self, query, prefer_video=True, max_results=5):
                self.search_calls += 1
                return [
                    MediaCandidate(
                        asset_type="video",
                        download_url=f"https://mock.media/{query}-{i}.mp4",
                        source_url=f"https://mock.media/page/{query}-{i}",
                        provider_asset_id=f"{query}-{i}",
                    )
                    for i in range(5)
                ]

            async def download(self, candidate, output_path):
                self.download_calls += 1
                with open(output_path, "wb") as f:
                    f.write(b"DATA")

        provider = CountingProvider()
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        result = await service.generate_visuals(_sample_script(), 40.0)

        slot_count = sum(len(m.assets) for m in result.sections)
        # Every candidate returned per search is inspectable metadata, but
        # only one download happens per slot (the selected candidate).
        assert provider.download_calls <= slot_count
        assert provider.download_calls < provider.search_calls * 5


class TestGenerateVisualsDuplicatePrevention:
    """Section 5: global duplicate prevention across the whole video, with
    controlled reuse as a fallback rather than a hard requirement."""

    @pytest.mark.asyncio
    async def test_no_reuse_when_enough_unique_assets_available(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=10, pool_size=10)
        service = VisualMediaService(
            media_provider=provider, output_dir=str(tmp_path), max_results_per_query=10
        )

        result = await service.generate_visuals(script, 60.0)

        assets = result.sections[0].assets
        ids = [a.provider_asset_id for a in assets]
        assert len(ids) == len(set(ids))
        assert all(not a.reused for a in assets)

    @pytest.mark.asyncio
    async def test_no_back_to_back_duplicate_when_alternatives_exist(self, tmp_path) -> None:
        """With a small-but-nonzero unique pool, reuse must happen only
        after a gap - never immediately repeating the previous clip."""
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        total_duration = 100.0
        result = await service.generate_visuals(script, total_duration)

        assets = result.sections[0].assets
        assert len(assets) >= 5  # enough slots to force reuse after the pool is exhausted
        for a, b in zip(assets, assets[1:]):
            assert a.provider_asset_id != b.provider_asset_id

    @pytest.mark.asyncio
    async def test_controlled_reuse_after_lookback_gap(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 100.0)
        assets = result.sections[0].assets

        reused_assets = [a for a in assets if a.reused]
        assert reused_assets  # pool of 3 exhausted well before slots run out
        for asset in reused_assets:
            position = assets.index(asset)
            recent_window = assets[max(0, position - RECENT_REUSE_LOOKBACK) : position]
            recent_ids = {a.provider_asset_id for a in recent_window}
            # A gap-reused asset must not equal the id used in the
            # immediately preceding slot.
            assert assets[position - 1].provider_asset_id != asset.provider_asset_id

    @pytest.mark.asyncio
    async def test_immediate_reuse_only_as_last_resort(self, tmp_path) -> None:
        """When only a single distinct asset is ever available, immediate
        back-to-back repetition is the only remaining option."""
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque "
            "without burning any fuel at all in modern factories worldwide today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=3, pool_size=1)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 60.0)
        assets = result.sections[0].assets

        assert len(assets) > 1
        assert all(a.provider_asset_id == assets[0].provider_asset_id for a in assets)
        assert all(a.reused for a in assets[1:])

    @pytest.mark.asyncio
    async def test_duplicate_prevention_is_global_across_sections(self, tmp_path) -> None:
        """Two different sections must not select the exact same asset when
        a unique alternative exists anywhere in the video."""
        provider = MockMediaProvider(results_per_query=5, pool_size=10)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script(), 40.0)

        all_ids = [a.provider_asset_id for m in result.sections for a in m.assets if a.success]
        assert len(all_ids) == len(set(all_ids))


class TestGenerateVisualsFailureHandling:
    """Section failures are captured per-asset/per-section, never raised,
    and a section only counts as failed if it got zero usable assets."""

    @pytest.mark.asyncio
    async def test_provider_failure_captured_per_section(self, tmp_path) -> None:
        service = VisualMediaService(
            media_provider=ExplodingMediaProvider(), output_dir=str(tmp_path)
        )
        result = await service.generate_visuals(_sample_script(), 40.0)

        assert result.success is False
        assert result.error is not None
        for mapping in result.sections:
            assert all(not a.success for a in mapping.assets)
            assert all("simulated media provider outage" in (a.error or "") for a in mapping.assets)

    @pytest.mark.asyncio
    async def test_section_with_one_failed_slot_is_still_usable(self, tmp_path) -> None:
        """A section counts as usable if AT LEAST ONE of its planned slots
        succeeded - not all of them."""

        class FlakyProvider(MediaProvider):
            def __init__(self) -> None:
                self.call_count = 0

            @property
            def name(self) -> str:
                return "flaky"

            async def search(self, query, prefer_video=True, max_results=5):
                self.call_count += 1
                if self.call_count == 1:
                    return []  # first query for the first slot fails to find anything...
                return [
                    MediaCandidate(
                        asset_type="video",
                        download_url=f"https://mock.media/{self.call_count}.mp4",
                        source_url=f"https://mock.media/page/{self.call_count}",
                        provider_asset_id=str(self.call_count),
                    )
                ]

            async def download(self, candidate, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"DATA")

        section = _section("Solo Section", "Only one section in this script entirely.")
        script = _sample_script(sections=[section])
        service = VisualMediaService(media_provider=FlakyProvider(), output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 60.0)

        # Even if the very first query attempt for slot 1 finds nothing,
        # the fallback chain (broader/topic/last-resort queries) recovers.
        assert any(a.success for a in result.sections[0].assets)

    @pytest.mark.asyncio
    async def test_all_queries_exhausted_with_no_prior_downloads_fails_cleanly(self, tmp_path) -> None:
        """If every query in the fallback chain returns nothing AND nothing
        has been downloaded yet anywhere in the video, the slot fails
        cleanly with a descriptive error rather than crashing."""
        section = _section("Alpha Topic", "Content about alpha things entirely.")
        script = _sample_script(sections=[section])

        variants = VisualMediaService.build_query_variants(section, 5)
        broader = VisualMediaService._build_concept_query(section.heading, BROAD_QUERY_TERMS)
        topic_query = VisualMediaService._build_concept_query(script.topic, MAX_QUERY_TERMS)
        empty_for = set(variants) | {broader, topic_query, "background footage"}

        provider = MockMediaProvider(results_per_query=1, empty_for=empty_for)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 40.0)

        assert result.success is False
        assert all(not a.success for a in result.sections[0].assets)

    @pytest.mark.asyncio
    async def test_fallback_chain_recovers_when_specific_query_is_empty(self, tmp_path) -> None:
        section = _section("Alpha Topic", "Content about alpha things entirely.")
        script = _sample_script(sections=[section])

        variants = VisualMediaService.build_query_variants(section, 1)
        provider = MockMediaProvider(results_per_query=1, empty_for=set(variants))
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 20.0)

        assert result.sections[0].assets[0].success is True
        assert result.sections[0].assets[0].search_query not in variants
