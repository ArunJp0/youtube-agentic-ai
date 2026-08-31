# Tests for VisualMediaService: duration-aware slot planning, semantic-plan
# consumption (search_queries/avoid_concepts/neutral_fallback_queries),
# semantic filtering, and global duplicate-prevention asset selection/
# download. Uses MockMediaProvider/fake planners only - no real network or
# LLM calls.
from __future__ import annotations

import os

import pytest

from src.models.script import ScriptResult, ScriptSection
from src.models.visual_plan import SectionVisualPlan, VisualPlan
from src.services.query_generation import (
    BROAD_QUERY_TERMS,
    LAST_RESORT_QUERY,
    MAX_QUERY_TERMS,
    build_concept_query,
    build_query_variants,
)
from src.services.visual_media_service import (
    CADENCE_POLICY,
    RECENT_REUSE_LOOKBACK,
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


class FakeVisualPlanner:
    """Test double for VisualContextPlanner: returns a fixed VisualPlan and
    records how many times it was called (must be exactly one per
    generate_visuals() call, regardless of section/slot count)."""

    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.calls = 0

    def plan_visuals(self, script) -> VisualPlan:
        self.calls += 1
        return self.plan


class QueryAwareProvider(MediaProvider):
    """Test double returning different candidates (with controllable
    content_hint) per exact query string, so tests can simulate "this
    specific query only has a bad match, but the neutral query has a
    good one"."""

    def __init__(self, hints_by_query: dict) -> None:
        self.hints_by_query = hints_by_query
        self.searched_queries: list = []
        self.downloaded_ids: list = []

    @property
    def name(self) -> str:
        return "query-aware"

    async def search(self, query, prefer_video: bool = True, max_results: int = 5):
        self.searched_queries.append(query)
        hints = self.hints_by_query.get(query, [])
        return [
            MediaCandidate(
                asset_type="video",
                download_url=f"https://mock.media/{query}-{i}.mp4",
                source_url=f"https://mock.media/page/{query}-{i}",
                provider_asset_id=f"{query}-{i}",
                content_hint=hint,
            )
            for i, hint in enumerate(hints)
        ]

    async def download(self, candidate, output_path):
        self.downloaded_ids.append(candidate.provider_asset_id)
        with open(output_path, "wb") as f:
            f.write(b"DATA")


def _plan_for(*section_plans: SectionVisualPlan, topic: str = "Dreams") -> VisualPlan:
    return VisualPlan(topic=topic, sections=list(section_plans), used_semantic_planning=True)


class TestCalculateSlotCount:
    """Duration-aware visual slot calculation: never a fixed count, always
    derived from the section's own share of total narration duration and
    the cadence tier the whole video falls into."""

    def test_short_section_needs_only_one_slot(self) -> None:
        assert VisualMediaService.calculate_slot_count(1.0, 200.0) == 1

    def test_longer_section_needs_multiple_slots(self) -> None:
        slots = VisualMediaService.calculate_slot_count(100.0, 200.0)
        assert slots > 1
        assert slots == round(100.0 / 8.0)

    def test_cadence_tier_up_to_five_minutes(self) -> None:
        assert VisualMediaService.calculate_slot_count(40.0, 200.0) == round(40.0 / 8.0)

    def test_cadence_tier_five_to_ten_minutes(self) -> None:
        assert VisualMediaService.calculate_slot_count(40.0, 450.0) == round(40.0 / 10.0)

    def test_cadence_tier_over_ten_minutes(self) -> None:
        assert VisualMediaService.calculate_slot_count(50.0, 700.0) == round(50.0 / 12.5)

    def test_same_section_duration_gets_more_slots_in_faster_tier(self) -> None:
        fast_tier_slots = VisualMediaService.calculate_slot_count(60.0, 200.0)
        slow_tier_slots = VisualMediaService.calculate_slot_count(60.0, 700.0)
        assert fast_tier_slots >= slow_tier_slots

    def test_never_returns_less_than_one(self) -> None:
        assert VisualMediaService.calculate_slot_count(0.5, 200.0) >= 1

    def test_cadence_policy_has_three_ordered_tiers(self) -> None:
        assert len(CADENCE_POLICY) == 3
        assert CADENCE_POLICY[0].max_total_seconds == 300.0
        assert CADENCE_POLICY[1].max_total_seconds == 600.0
        assert CADENCE_POLICY[2].max_total_seconds == float("inf")


class TestGenerateVisualsValidation:
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


class TestGenerateVisualsWithoutPlanner:
    """Default behavior (visual_planner=None): identical shape to the
    duration-aware milestone before context-aware planning was added, now
    routed through query_generation.build_deterministic_visual_plan."""

    @pytest.mark.asyncio
    async def test_semantic_planning_used_is_false(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        result = await service.generate_visuals(_sample_script(), 40.0)
        assert result.semantic_planning_used is False
        assert result.semantic_planning_fallback_reason is None

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

    @pytest.mark.asyncio
    async def test_deterministic_plan_has_no_avoid_concepts(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        result = await service.generate_visuals(_sample_script(), 40.0)
        for mapping in result.sections:
            assert mapping.avoid_concepts == []

    @pytest.mark.asyncio
    async def test_all_queries_exhausted_with_no_prior_downloads_fails_cleanly(self, tmp_path) -> None:
        section = _section("Alpha Topic", "Content about alpha things entirely.")
        script = _sample_script(sections=[section])

        variants = build_query_variants(section, 5)
        broader = build_concept_query(section.heading, BROAD_QUERY_TERMS)
        topic_query = build_concept_query(script.topic, MAX_QUERY_TERMS)
        empty_for = set(variants) | {broader, topic_query, LAST_RESORT_QUERY}

        provider = MockMediaProvider(results_per_query=1, empty_for=empty_for)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 40.0)

        assert result.success is False
        assert all(not a.success for a in result.sections[0].assets)

    @pytest.mark.asyncio
    async def test_fallback_chain_recovers_when_specific_query_is_empty(self, tmp_path) -> None:
        section = _section("Alpha Topic", "Content about alpha things entirely.")
        script = _sample_script(sections=[section])

        variants = build_query_variants(section, 1)
        provider = MockMediaProvider(results_per_query=1, empty_for=set(variants))
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(script, 20.0)

        assert result.sections[0].assets[0].success is True
        assert result.sections[0].assets[0].search_query not in variants


class TestGenerateVisualsDownloadEfficiency:
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

        result = await service.generate_visuals(script, 60.0)

        slot_count = len(result.sections[0].assets)
        download_calls = [c for c in provider.calls if c[0] == "download"]
        distinct_ids_used = {a.provider_asset_id for a in result.sections[0].assets if a.success}
        assert len(download_calls) == len(distinct_ids_used)
        assert len(download_calls) <= slot_count

    @pytest.mark.asyncio
    async def test_reused_asset_not_downloaded_twice(self, tmp_path) -> None:
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
        assert provider.download_calls <= slot_count
        assert provider.download_calls < provider.search_calls * 5


class TestGenerateVisualsDuplicatePrevention:
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
        assert len(assets) >= 5
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
        assert reused_assets
        for asset in reused_assets:
            position = assets.index(asset)
            assert assets[position - 1].provider_asset_id != asset.provider_asset_id

    @pytest.mark.asyncio
    async def test_immediate_reuse_only_as_last_resort(self, tmp_path) -> None:
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
        assert all(a.relevance_tier == "reused" for a in assets[1:])

    @pytest.mark.asyncio
    async def test_duplicate_prevention_is_global_across_sections(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=5, pool_size=10)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script(), 40.0)

        all_ids = [a.provider_asset_id for m in result.sections for a in m.assets if a.success]
        assert len(all_ids) == len(set(all_ids))


class TestGenerateVisualsFailureHandling:
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
        class FlakyProvider(MediaProvider):
            def __init__(self) -> None:
                self.call_count = 0

            @property
            def name(self) -> str:
                return "flaky"

            async def search(self, query, prefer_video=True, max_results=5):
                self.call_count += 1
                if self.call_count == 1:
                    return []
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

        assert any(a.success for a in result.sections[0].assets)


class TestBuildPlanAndPreBuiltPlanReuse:
    """build_plan() is public so a caller (e.g. Visual QC) can build the
    plan once and hand it to generate_visuals(), avoiding a second,
    redundant LLM planning call."""

    @pytest.mark.asyncio
    async def test_build_plan_matches_internally_built_plan_shape(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        plan = service.build_plan(script)

        assert plan.used_semantic_planning is False
        assert len(plan.sections) == 1

    @pytest.mark.asyncio
    async def test_passing_prebuilt_plan_skips_second_planner_call(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(SectionVisualPlan(section_index=0, search_queries=["q"]))
        planner = FakeVisualPlanner(plan)
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, visual_planner=planner, output_dir=str(tmp_path))

        await service.generate_visuals(script, 20.0, visual_plan=plan)

        assert planner.calls == 0  # the pre-built plan was used, not re-derived

    @pytest.mark.asyncio
    async def test_no_plan_passed_still_builds_one_internally(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(SectionVisualPlan(section_index=0, search_queries=["q"]))
        planner = FakeVisualPlanner(plan)
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, visual_planner=planner, output_dir=str(tmp_path))

        await service.generate_visuals(script, 20.0)

        assert planner.calls == 1


class TestAcquireReplacementAsset:
    """Public entry point Visual QC uses for bounded replacement - reuses
    the exact same selection/reuse-fallback rules as normal slot filling,
    plus never selects/reuses an excluded id."""

    @pytest.mark.asyncio
    async def test_excluded_id_is_never_selected(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = service.build_plan(script)
        section_plan = plan.sections[0]

        downloaded_by_id: dict = {}
        used_ids_in_order: list = []
        first_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, set()
        )
        excluded = {first_asset.provider_asset_id}

        second_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, excluded
        )

        assert second_asset.provider_asset_id != first_asset.provider_asset_id

    @pytest.mark.asyncio
    async def test_mutates_shared_state_for_global_dedup(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=5, pool_size=5)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = service.build_plan(script)
        section_plan = plan.sections[0]

        downloaded_by_id: dict = {}
        used_ids_in_order: list = []
        asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, set()
        )

        # acquire_replacement_asset mutates downloaded_by_id itself (like
        # normal slot filling); used_ids_in_order tracking is the caller's
        # responsibility (VisualQCService appends it after evaluating the
        # replacement), the same division of labor generate_visuals uses.
        assert asset.provider_asset_id in downloaded_by_id
        assert downloaded_by_id[asset.provider_asset_id].provider_asset_id == asset.provider_asset_id

    @pytest.mark.asyncio
    async def test_falls_back_to_excluded_reuse_only_when_nothing_else_available(self, tmp_path) -> None:
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=3, pool_size=1)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = service.build_plan(script)
        section_plan = plan.sections[0]

        downloaded_by_id: dict = {}
        used_ids_in_order: list = []
        first_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, set()
        )
        # Only one distinct asset ever exists (pool_size=1) - excluding it
        # leaves nothing else, so it must still be returned as a last resort.
        result_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, {first_asset.provider_asset_id}
        )

        assert result_asset.provider_asset_id == first_asset.provider_asset_id
        assert result_asset.reused is True


class TestGenerateVisualsWithSemanticPlanner:
    """Semantic-planner integration: VisualMediaService uses the plan's
    search_queries/avoid_concepts/neutral_fallback_queries and applies the
    deterministic semantic filter against candidate metadata."""

    @pytest.mark.asyncio
    async def test_planner_called_exactly_once_per_generate_visuals_call(self, tmp_path) -> None:
        script = _sample_script()
        plan = _plan_for(
            SectionVisualPlan(section_index=0, search_queries=["q0"], neutral_fallback_queries=["n0"]),
            SectionVisualPlan(section_index=1, search_queries=["q1"], neutral_fallback_queries=["n1"]),
        )
        planner = FakeVisualPlanner(plan)
        provider = MockMediaProvider(results_per_query=5)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        # 5 sections' worth of slots would be requested from a naive
        # per-slot planner call; this must still be exactly one call.
        await service.generate_visuals(script, 200.0)

        assert planner.calls == 1

    @pytest.mark.asyncio
    async def test_semantic_summary_and_avoid_concepts_carried_into_mapping(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(
            SectionVisualPlan(
                section_index=0,
                semantic_summary="A mental process, not a physical one.",
                search_queries=["q0"],
                avoid_concepts=["construction site"],
                neutral_fallback_queries=["n0"],
            )
        )
        planner = FakeVisualPlanner(plan)
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)

        assert result.sections[0].semantic_summary == "A mental process, not a physical one."
        assert result.sections[0].avoid_concepts == ["construction site"]

    @pytest.mark.asyncio
    async def test_avoid_concept_candidate_is_rejected_in_favor_of_clean_candidate(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "The mind constructs a narrative.")])
        plan = _plan_for(
            SectionVisualPlan(
                section_index=0,
                search_queries=["mind narrative"],
                avoid_concepts=["construction site"],
                neutral_fallback_queries=["dream sleep"],
            )
        )
        planner = FakeVisualPlanner(plan)
        provider = QueryAwareProvider(
            {"mind narrative": ["workers on a construction site downtown"]}
        )
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)

        asset = result.sections[0].assets[0]
        assert asset.success is False  # "mind narrative"'s only candidate was rejected, "dream sleep" had none
        assert "mind narrative" in provider.searched_queries
        assert "dream sleep" in provider.searched_queries

    @pytest.mark.asyncio
    async def test_neutral_fallback_used_when_specific_query_has_only_bad_candidates(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "The mind constructs a narrative.")])
        plan = _plan_for(
            SectionVisualPlan(
                section_index=0,
                search_queries=["mind narrative"],
                avoid_concepts=["construction site"],
                neutral_fallback_queries=["dream sleep"],
            )
        )
        planner = FakeVisualPlanner(plan)
        provider = QueryAwareProvider(
            {
                "mind narrative": ["workers on a construction site downtown"],
                "dream sleep": ["person sleeping peacefully at night"],
            }
        )
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)

        asset = result.sections[0].assets[0]
        assert asset.success is True
        assert asset.search_query == "dream sleep"
        assert asset.relevance_tier == "neutral"
        assert provider.downloaded_ids == ["dream sleep-0"]

    @pytest.mark.asyncio
    async def test_relevance_tier_high_for_specific_query_match(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(
            SectionVisualPlan(section_index=0, search_queries=["good query"], neutral_fallback_queries=["n"])
        )
        planner = FakeVisualPlanner(plan)
        provider = QueryAwareProvider({"good query": [None]})
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)
        assert result.sections[0].assets[0].relevance_tier == "high"

    @pytest.mark.asyncio
    async def test_relevance_score_reflects_metadata_overlap(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(
            SectionVisualPlan(
                section_index=0,
                visual_intents=["person sleeping"],
                search_queries=["person sleeping"],
                neutral_fallback_queries=["n"],
            )
        )
        planner = FakeVisualPlanner(plan)
        provider = QueryAwareProvider({"person sleeping": ["person sleeping"]})
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)
        asset = result.sections[0].assets[0]
        assert asset.relevance_score == 1.0

    @pytest.mark.asyncio
    async def test_semantic_planning_used_true_when_planner_provides_plan(self, tmp_path) -> None:
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        plan = _plan_for(SectionVisualPlan(section_index=0, search_queries=["q"]))
        planner = FakeVisualPlanner(plan)
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)
        assert result.semantic_planning_used is True

    @pytest.mark.asyncio
    async def test_planner_reported_fallback_reason_propagated_to_result(self, tmp_path) -> None:
        """When the underlying planner itself already fell back (e.g. a real
        VisualContextPlanner whose LLM call failed), VisualMediaService must
        surface that on the VisualResult, not silently swallow it."""
        fallback_plan = VisualPlan(
            topic="Dreams",
            sections=[SectionVisualPlan(section_index=0, search_queries=["q"])],
            used_semantic_planning=False,
            fallback_reason="Semantic visual planning unavailable, used deterministic fallback: boom",
        )
        planner = FakeVisualPlanner(fallback_plan)
        script = _sample_script(sections=[_section("A", "Some narration text here.")])
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(script, 20.0)

        assert result.semantic_planning_used is False
        assert "boom" in result.semantic_planning_fallback_reason

    @pytest.mark.asyncio
    async def test_real_planner_failure_falls_back_and_pipeline_still_succeeds(self, tmp_path) -> None:
        """End-to-end with a real VisualContextPlanner wired in (not just a
        fake plan): an LLM outage must not break visual generation."""
        from src.agents.visual_context_planner import VisualContextPlanner
        from src.llm.provider import LLMProvider

        class ExplodingLLM(LLMProvider):
            def generate_text(self, prompt: str) -> str:
                raise RuntimeError("Gemini outage")

        planner = VisualContextPlanner(llm_provider=ExplodingLLM())
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path)
        )

        result = await service.generate_visuals(_sample_script(), 40.0)

        assert result.success is True
        assert result.semantic_planning_used is False
        assert "Gemini outage" in result.semantic_planning_fallback_reason
