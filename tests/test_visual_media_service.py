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
from src.tools.ai_video_provider import MockAIVideoProvider
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

        plan = await service.build_plan(script)

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

    @pytest.mark.asyncio
    async def test_normal_semantic_planning_call_succeeds_within_timeout(self, tmp_path) -> None:
        """A. A normal (fast) real VisualContextPlanner call must succeed
        exactly as before - the new outer timeout must never interfere
        with a genuinely successful call."""
        from src.agents.visual_context_planner import VisualContextPlanner
        from src.llm.provider import LLMProvider

        class FastLLM(LLMProvider):
            def generate_text(self, prompt: str) -> str:
                section = (
                    '{{"section_index": {i}, "semantic_summary": "s", '
                    '"visual_intents": ["i"], "search_queries": ["q"], '
                    '"avoid_concepts": [], "neutral_fallback_queries": ["n"]}}'
                )
                sections = ", ".join(section.format(i=i) for i in range(len(_sample_script().sections)))
                return f'{{"sections": [{sections}]}}'

        planner = VisualContextPlanner(llm_provider=FastLLM())
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path),
            visual_context_planner_timeout_seconds=5.0,
        )

        plan = await service.build_plan(_sample_script())

        assert plan.used_semantic_planning is True

    @pytest.mark.asyncio
    async def test_hanging_semantic_planning_call_times_out_and_falls_back(self, tmp_path) -> None:
        """B/C. A VisualContextPlanner call that never returns - the exact
        shape of the real production hang - must still be bounded by the
        outer timeout and degrade to the same deterministic fallback plan
        used for any other planning failure, never block indefinitely."""
        import time

        from src.agents.visual_context_planner import VisualContextPlanner
        from src.llm.provider import LLMProvider

        class HangingLLM(LLMProvider):
            def generate_text(self, prompt: str) -> str:
                # A background thread running time.sleep() cannot be
                # cancelled (unlike an awaited coroutine) - kept short
                # (not e.g. 3600s) so an orphaned worker thread never
                # blocks pytest's own process exit/thread-pool shutdown,
                # while still comfortably exceeding the tiny outer
                # timeout configured below.
                time.sleep(2)
                raise AssertionError("should never be reached - the outer timeout must fire first")

        planner = VisualContextPlanner(llm_provider=HangingLLM())
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path),
            visual_context_planner_timeout_seconds=0.05,
        )

        started = time.monotonic()
        plan = await service.build_plan(_sample_script())
        elapsed = time.monotonic() - started

        assert elapsed < 1.0  # bounded by the 0.05s outer timeout, not the LLM's own 2s "hang"
        assert plan.used_semantic_planning is False
        assert "timed out" in plan.fallback_reason.lower()
        assert "visual_context_planner" in plan.fallback_reason.lower()
        assert "timeout" in plan.fallback_reason.lower()  # category, per observability requirement

    @pytest.mark.asyncio
    async def test_hanging_semantic_planning_call_never_raises_to_caller(self, tmp_path) -> None:
        """A timeout must degrade exactly like any other planning failure -
        generate_visuals() must still succeed end to end, never propagate
        the timeout as an exception."""
        import time

        from src.agents.visual_context_planner import VisualContextPlanner
        from src.llm.provider import LLMProvider

        class HangingLLM(LLMProvider):
            def generate_text(self, prompt: str) -> str:
                time.sleep(2)  # see the sibling test above for why not 3600s
                raise AssertionError("should never be reached")

        planner = VisualContextPlanner(llm_provider=HangingLLM())
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(
            media_provider=provider, visual_planner=planner, output_dir=str(tmp_path),
            visual_context_planner_timeout_seconds=0.05,
        )

        result = await service.generate_visuals(_sample_script(), 20.0)

        assert result.success is True
        assert result.semantic_planning_used is False
        assert "timed out" in (result.semantic_planning_fallback_reason or "").lower()


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
        plan = await service.build_plan(script)
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
        plan = await service.build_plan(script)
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
    async def test_reports_failure_rather_than_reusing_an_excluded_asset(self, tmp_path) -> None:
        """A QC-driven replacement never falls back to reuse at all (see
        acquire_replacement_asset's docstring) - when the only candidate
        that exists is the one just excluded (e.g. an asset Visual QC just
        rejected), the call must report failure so VisualQCService can
        drop the slot and recompute section coverage, never silently hand
        back the excluded/rejected asset as if it were a genuine
        replacement."""
        section = _section(
            "How Electric Motors Work",
            "Electric vehicles use battery packs and electric motors to generate torque today.",
        )
        script = _sample_script(sections=[section])
        provider = MockMediaProvider(results_per_query=3, pool_size=1)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = await service.build_plan(script)
        section_plan = plan.sections[0]

        downloaded_by_id: dict = {}
        used_ids_in_order: list = []
        first_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, set()
        )
        # Only one distinct asset ever exists (pool_size=1) - excluding it
        # leaves nothing else, so the call must fail cleanly rather than
        # reuse the excluded asset.
        result_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, {first_asset.provider_asset_id}
        )

        assert result_asset.success is False


class TestBroadenQueryReplacement:
    """A later QC-driven replacement attempt (broaden_query=True) skips the
    slot's specific query entirely and searches only the neutral/broader
    tier - recovery generically, not by re-sampling the same possibly-
    unsuitable visual concept (see VisualQCService._resolve_slot)."""

    @pytest.mark.asyncio
    async def test_broaden_query_true_skips_specific_query(self, tmp_path) -> None:
        section_plan = SectionVisualPlan(
            section_index=0,
            search_queries=["specific narrow query"],
            neutral_fallback_queries=["broad safe query"],
        )
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        asset, used_query = await service.acquire_replacement_asset(
            section_plan, 0, 0, {}, [], set(), broaden_query=True
        )

        queried = [q for name, q in provider.calls if name == "search"]
        assert "specific narrow query" not in queried
        assert "broad safe query" in queried
        assert used_query == "broad safe query"
        assert asset.success is True

    @pytest.mark.asyncio
    async def test_broaden_query_false_tries_specific_query_first(self, tmp_path) -> None:
        section_plan = SectionVisualPlan(
            section_index=0,
            search_queries=["specific narrow query"],
            neutral_fallback_queries=["broad safe query"],
        )
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        asset, used_query = await service.acquire_replacement_asset(
            section_plan, 0, 0, {}, [], set(), broaden_query=False
        )

        assert used_query == "specific narrow query"
        assert asset.success is True

    @pytest.mark.asyncio
    async def test_broaden_query_still_avoids_excluded_ids(self, tmp_path) -> None:
        """Broadening the query tier never weakens exclusion - a
        previously-rejected id stays excluded regardless of which tier
        finds the replacement, and a QC-driven replacement reports failure
        rather than falling back to reusing it."""
        section_plan = SectionVisualPlan(
            section_index=0,
            search_queries=["specific narrow query"],
            neutral_fallback_queries=["broad safe query"],
        )
        provider = MockMediaProvider(results_per_query=1, pool_size=1)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        downloaded_by_id: dict = {}
        used_ids_in_order: list = []

        first_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, set(), broaden_query=False
        )
        excluded = {first_asset.provider_asset_id}

        result_asset, _ = await service.acquire_replacement_asset(
            section_plan, 0, 0, downloaded_by_id, used_ids_in_order, excluded, broaden_query=True
        )

        # Only one distinct asset exists in the pool either way - excluded
        # correctly reports failure rather than silently ignoring the
        # exclusion just because the query tier changed.
        assert result_asset.success is False


class TestRemediationScopedVisualReuse:
    """generate_visuals' prior_visual_result/changed_section_indices -
    avoids unnecessarily rebuilding sections a Compliance Remediation
    script revision didn't touch (see task: "avoid unnecessarily rebuilding
    unaffected sections/assets")."""

    @pytest.mark.asyncio
    async def test_unchanged_section_reused_without_new_search(self, tmp_path) -> None:
        script = _sample_script()
        provider = MockMediaProvider(results_per_query=5)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = await service.build_plan(script)

        first = await service.generate_visuals(script, 60.0, visual_plan=plan)
        assert first.success is True
        provider.calls.clear()

        second = await service.generate_visuals(
            script, 60.0, visual_plan=plan, prior_visual_result=first, changed_section_indices={1}
        )

        assert second.success is True
        # Section 0 (unchanged) must be byte-identical to the prior result -
        # no new search/download for it.
        assert second.sections[0] == first.sections[0]
        searched_queries = {q for name, q in provider.calls if name == "search"}
        assert searched_queries.isdisjoint(plan.sections[0].search_queries)
        # Section 1 (the actually-changed one) must still have been searched.
        assert searched_queries & set(plan.sections[1].search_queries)

    @pytest.mark.asyncio
    async def test_changed_section_is_freshly_reacquired(self, tmp_path) -> None:
        script = _sample_script()
        provider = MockMediaProvider(results_per_query=5)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = await service.build_plan(script)

        first = await service.generate_visuals(script, 60.0, visual_plan=plan)
        provider.calls.clear()

        second = await service.generate_visuals(
            script, 60.0, visual_plan=plan, prior_visual_result=first, changed_section_indices={1}
        )

        assert any(name == "search" for name, _ in provider.calls)
        # The changed section got at least one fresh search call.
        assert second.sections[1].section_index == 1

    @pytest.mark.asyncio
    async def test_changed_section_avoids_reselecting_its_own_prior_asset(self, tmp_path) -> None:
        """Global dedup state is seeded from EVERY prior asset (including
        the changed section's own) so a fresh acquire for that section
        naturally gets a different asset than it had before."""
        script = _sample_script(sections=[_section("Only Section", "Some narration text about this topic.")])
        # Ample pool headroom beyond however many slots this short section
        # needs, so "avoids reselecting" is meaningfully testable rather
        # than forced into reuse purely by pool exhaustion. max_results_per_query
        # must also be raised - it, not the provider's own results_per_query,
        # is what actually caps candidates considered per search call.
        provider = MockMediaProvider(results_per_query=20, pool_size=20)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path), max_results_per_query=20)
        plan = await service.build_plan(script)

        first = await service.generate_visuals(script, 30.0, visual_plan=plan)
        prior_ids = {a.provider_asset_id for m in first.sections for a in m.assets if a.success}

        second = await service.generate_visuals(
            script, 30.0, visual_plan=plan, prior_visual_result=first, changed_section_indices={0}
        )

        new_ids = {a.provider_asset_id for m in second.sections for a in m.assets if a.success}
        assert new_ids.isdisjoint(prior_ids)

    @pytest.mark.asyncio
    async def test_no_prior_result_behaves_exactly_as_before(self, tmp_path) -> None:
        """Omitting prior_visual_result/changed_section_indices (the
        default) must be byte-for-byte the original full-regeneration
        behavior - no accidental behavior change for every existing
        caller."""
        script = _sample_script()
        provider = MockMediaProvider(results_per_query=5)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = await service.build_plan(script)

        result = await service.generate_visuals(script, 60.0, visual_plan=plan)

        assert result.success is True
        assert len(result.sections) == len(script.sections)

    @pytest.mark.asyncio
    async def test_slot_count_mismatch_forces_fresh_reacquire_even_if_unchanged(self, tmp_path) -> None:
        """Safety guard: a section marked 'unchanged' is only reused when
        its recomputed plan still genuinely matches the prior attempt's -
        never reused blindly just because its index wasn't flagged."""
        script = _sample_script()
        provider = MockMediaProvider(results_per_query=5)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))
        plan = await service.build_plan(script)
        first = await service.generate_visuals(script, 60.0, visual_plan=plan)

        # Simulate a stale prior mapping whose slot count no longer matches
        # (e.g. a real duration shift) by hand-editing planned_duration.
        stale_mapping = first.sections[0].model_copy(update={"planned_duration_seconds": 0.5})
        stale_prior = first.model_copy(update={"sections": [stale_mapping] + list(first.sections[1:])})
        provider.calls.clear()

        second = await service.generate_visuals(
            script, 60.0, visual_plan=plan, prior_visual_result=stale_prior, changed_section_indices={1}
        )

        # Section 0 was "unchanged" but its prior plan no longer matches -
        # must be freshly reacquired, not reused verbatim from the stale mapping.
        assert second.sections[0] != stale_mapping
        assert any(name == "search" for name, _ in provider.calls)


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


class TestAIVideoIntegration:
    """Tests for the AI-video-generation integration seam
    (_acquire_visual_slot): AI-first when configured, Pexels-unchanged
    otherwise. No real PixVerse/Gemini/Pexels/network calls."""

    @pytest.mark.asyncio
    async def test_default_behavior_unchanged_when_ai_video_provider_not_set(self, tmp_path) -> None:
        provider = MockMediaProvider(results_per_query=3)
        service = VisualMediaService(media_provider=provider, output_dir=str(tmp_path))

        result = await service.generate_visuals(_sample_script(), 20.0)

        assert result.success is True
        assert all(a.provider == "mock" for m in result.sections for a in m.assets)
        assert all(a.relevance_tier != "ai_generated" for m in result.sections for a in m.assets)

    @pytest.mark.asyncio
    async def test_ai_video_provider_used_when_configured(self, tmp_path) -> None:
        media_provider = MockMediaProvider(results_per_query=3)
        ai_provider = MockAIVideoProvider()
        service = VisualMediaService(
            media_provider=media_provider, output_dir=str(tmp_path), ai_video_provider=ai_provider
        )

        result = await service.generate_visuals(_sample_script(), 20.0)

        assert result.success is True
        assets = [a for m in result.sections for a in m.assets]
        assert assets
        assert all(a.relevance_tier == "ai_generated" for a in assets)
        assert all(a.provider == "mock" for a in assets)  # MockAIVideoProvider's own name

    @pytest.mark.asyncio
    async def test_ai_generated_clip_materialized_into_output_dir(self, tmp_path) -> None:
        media_provider = MockMediaProvider(results_per_query=3)
        ai_provider = MockAIVideoProvider()
        service = VisualMediaService(
            media_provider=media_provider, output_dir=str(tmp_path), ai_video_provider=ai_provider
        )

        result = await service.generate_visuals(_sample_script(sections=[_section("A", "Some narration.")]), 20.0)

        asset = result.sections[0].assets[0]
        assert asset.local_file_path is not None
        assert os.path.dirname(asset.local_file_path) == str(tmp_path)
        assert os.path.exists(asset.local_file_path)

    @pytest.mark.asyncio
    async def test_ai_failure_falls_back_to_pexels_when_enabled(self, tmp_path) -> None:
        media_provider = MockMediaProvider(results_per_query=3)
        ai_provider = MockAIVideoProvider(fail=True)
        service = VisualMediaService(
            media_provider=media_provider,
            output_dir=str(tmp_path),
            ai_video_provider=ai_provider,
            ai_video_max_retries=0,
            stock_fallback_enabled=True,
        )

        result = await service.generate_visuals(_sample_script(), 20.0)

        assert result.success is True
        assets = [a for m in result.sections for a in m.assets]
        assert all(a.provider == "mock" for a in assets)
        assert all(a.relevance_tier != "ai_generated" for a in assets)

    @pytest.mark.asyncio
    async def test_ai_failure_fails_slot_cleanly_when_stock_fallback_disabled(self, tmp_path) -> None:
        """The exact configuration for a demo meant to show AI visuals
        only: no silent Pexels fetch when AI generation isn't available."""
        media_provider = MockMediaProvider(results_per_query=3)
        ai_provider = MockAIVideoProvider(fail=True)
        service = VisualMediaService(
            media_provider=media_provider,
            output_dir=str(tmp_path),
            ai_video_provider=ai_provider,
            ai_video_max_retries=0,
            stock_fallback_enabled=False,
        )

        result = await service.generate_visuals(_sample_script(sections=[_section("A", "Some narration.")]), 20.0)

        assert media_provider.calls == []  # Pexels never touched at all
        asset = result.sections[0].assets[0]
        assert asset.success is False
        assert "AI video generation failed" in asset.error

    @pytest.mark.asyncio
    async def test_ai_prompt_reuses_visual_plan_search_queries(self, tmp_path) -> None:
        ai_provider = MockAIVideoProvider()
        service = VisualMediaService(
            media_provider=MockMediaProvider(results_per_query=3),
            output_dir=str(tmp_path),
            ai_video_provider=ai_provider,
        )

        await service.generate_visuals(_sample_script(sections=[_section("REM Sleep", "Narration about REM sleep.")]), 20.0)

        assert ai_provider.calls
        assert "REM" in ai_provider.calls[0].prompt or "sleep" in ai_provider.calls[0].prompt.lower()

    @pytest.mark.asyncio
    async def test_ai_video_disabled_never_calls_ai_provider(self, tmp_path) -> None:
        """Redundant with the default-unchanged test above, but explicitly
        proves the AI provider is never even invoked when not configured -
        not merely that its result is discarded."""
        service = VisualMediaService(media_provider=MockMediaProvider(results_per_query=3), output_dir=str(tmp_path))
        result = await service.generate_visuals(_sample_script(), 20.0)
        assert result.success is True  # sanity: pipeline still works with no AI provider at all

    @pytest.mark.asyncio
    async def test_bounded_retries_respected_in_full_pipeline_context(self, tmp_path) -> None:
        ai_provider = MockAIVideoProvider(fail_times=1)  # fails once, succeeds on retry
        service = VisualMediaService(
            media_provider=MockMediaProvider(results_per_query=3),
            output_dir=str(tmp_path),
            ai_video_provider=ai_provider,
            ai_video_max_retries=2,
        )

        # A short section (well under MIN_SLOT_SECONDS' 3x floor) is
        # guaranteed exactly one visual slot, so the retry count below is
        # unambiguous - a longer section could need multiple slots, each
        # with its own attempt budget.
        result = await service.generate_visuals(_sample_script(sections=[_section("A", "Some narration.")]), 5.0)

        assert result.sections[0].assets[0].relevance_tier == "ai_generated"
        assert len(ai_provider.calls) == 2  # 1 failed + 1 retry that succeeded, never more than the bound
