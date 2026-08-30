# Tests for deterministic (non-LLM) stock-search query generation, the
# single source of truth used both directly by VisualMediaService (when no
# VisualContextPlanner is configured) and internally by
# VisualContextPlanner's own fallback path.
from __future__ import annotations

import pytest

from src.models.script import ScriptResult, ScriptSection
from src.services.query_generation import (
    BROAD_QUERY_TERMS,
    DEFAULT_QUERY_VARIANTS,
    LAST_RESORT_QUERY,
    MAX_QUERY_TERMS,
    build_concept_query,
    build_deterministic_visual_plan,
    build_query_variants,
    ordered_unique,
)


def _section(heading: str, narration: str) -> ScriptSection:
    return ScriptSection(heading=heading, narration=narration)


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK_TEXT",
        introduction="INTRO_TEXT",
        sections=[
            _section(
                "REM Sleep Timing",
                "Most vivid dreaming happens during REM sleep cycles at night.",
            ),
            _section(
                "Memory Consolidation",
                "The brain strengthens memories and processes emotions while dreaming.",
            ),
        ],
        conclusion="CONCLUSION_TEXT",
        call_to_action="CTA_TEXT",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


class TestOrderedUnique:
    def test_dedupes_preserving_order(self) -> None:
        assert ordered_unique(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_skips_falsy_values(self) -> None:
        assert ordered_unique(["a", "", None, "b"]) == ["a", "b"]

    def test_empty_input_returns_empty(self) -> None:
        assert ordered_unique([]) == []


class TestBuildQueryVariants:
    def test_returns_empty_list_for_zero_variants(self) -> None:
        section = _section("REM Sleep Timing", "Dreams occur during REM sleep.")
        assert build_query_variants(section, 0) == []

    def test_does_not_exceed_max_variants(self) -> None:
        section = _section(
            "REM Sleep Timing",
            "Most vivid dreaming happens during REM sleep cycles at night in the bedroom.",
        )
        variants = build_query_variants(section, 2)
        assert len(variants) <= 2

    def test_variants_are_distinct(self) -> None:
        section = _section(
            "Memory Consolidation",
            "The brain strengthens memories and processes emotions while dreaming at night.",
        )
        variants = build_query_variants(section, 5)
        assert len(variants) == len(set(variants))

    def test_deterministic_for_same_section(self) -> None:
        section = _section("REM Sleep Timing", "Dreams occur during REM sleep.")
        assert build_query_variants(section, 3) == build_query_variants(section, 3)

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
        section = _section(heading, narration)
        variants = build_query_variants(section, 3)
        assert variants
        combined = " ".join(variants).lower()
        assert "dream" not in combined
        assert "rem sleep" not in combined


class TestBuildConceptQuery:
    def test_query_derived_from_text(self) -> None:
        query = build_concept_query(
            "REM Sleep Timing Most vivid dreaming happens during REM sleep cycles at night.",
            MAX_QUERY_TERMS,
        )
        assert query
        assert "sleep" in query.lower() or "dream" in query.lower()

    def test_query_is_not_hardcoded_to_any_topic(self) -> None:
        query = build_concept_query(
            "Photosynthesis Basics Plants convert sunlight into chemical energy using chlorophyll.",
            MAX_QUERY_TERMS,
        )
        assert "dream" not in query.lower()
        assert any(
            w in query.lower()
            for w in ["photosynthesis", "sunlight", "chemical", "chlorophyll", "plants", "energy"]
        )

    def test_query_excludes_stopwords(self) -> None:
        query = build_concept_query(
            "The Big Idea This is about the way we live and work.", MAX_QUERY_TERMS
        )
        for stopword in ("the", "is", "about", "way", "we"):
            assert stopword not in query.lower().split()

    def test_query_capped_at_max_terms(self) -> None:
        query = build_concept_query(
            "Elephants Giraffes Rhinoceros Crocodiles Alligators "
            "Extraordinary fascinating wonderful magnificent creatures roam savannas.",
            MAX_QUERY_TERMS,
        )
        assert len(query.split()) <= MAX_QUERY_TERMS

    def test_deterministic(self) -> None:
        text = "REM Sleep Timing Dreams occur during REM sleep."
        assert build_concept_query(text, MAX_QUERY_TERMS) == build_concept_query(text, MAX_QUERY_TERMS)


class TestConceptMappingRegression:
    """Regression tests for previously reported semantically-weak queries.
    Every assertion is about generic properties (concreteness, absence of
    scientific jargon) - not a hardcoded expectation that only holds for
    this one topic (see test_generic_across_unrelated_topic_domains above)."""

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
                "Dreams may serve evolutionary functions like threat simulation. One theory "
                "suggests dreaming evolved as a safe rehearsal space for reacting to danger.",
                ["may", "serve", "functions", "simulation"],
            ),
        ],
    )
    def test_drops_low_value_scientific_terms(self, text, forbidden_terms) -> None:
        query = build_concept_query(text, MAX_QUERY_TERMS)
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
        ],
    )
    def test_contains_concrete_visual_concept(self, text, expected_any_of) -> None:
        query = build_concept_query(text, MAX_QUERY_TERMS)
        query_words = set(query.lower().split())
        assert query_words & set(expected_any_of), f"{query!r} has none of {expected_any_of}"


class TestBuildDeterministicVisualPlan:
    def test_returns_one_plan_per_section_in_order(self) -> None:
        script = _script()
        plan = build_deterministic_visual_plan(script)

        assert plan.used_semantic_planning is False
        assert plan.fallback_reason is None
        assert len(plan.sections) == len(script.sections)
        assert [s.section_index for s in plan.sections] == [0, 1]

    def test_every_section_has_nonempty_search_queries(self) -> None:
        plan = build_deterministic_visual_plan(_script())
        for section_plan in plan.sections:
            assert section_plan.search_queries

    def test_avoid_concepts_always_empty(self) -> None:
        """The deterministic path has no contextual understanding, so it
        never claims to know what to avoid - that's exactly the gap
        VisualContextPlanner exists to fill."""
        plan = build_deterministic_visual_plan(_script())
        for section_plan in plan.sections:
            assert section_plan.avoid_concepts == []

    def test_neutral_fallback_queries_include_last_resort(self) -> None:
        plan = build_deterministic_visual_plan(_script())
        for section_plan in plan.sections:
            assert LAST_RESORT_QUERY in section_plan.neutral_fallback_queries

    def test_search_queries_match_build_query_variants(self) -> None:
        script = _script()
        plan = build_deterministic_visual_plan(script)
        expected = build_query_variants(script.sections[0], DEFAULT_QUERY_VARIANTS)
        assert plan.sections[0].search_queries == expected

    def test_very_short_section_still_gets_a_search_query(self) -> None:
        script = _script(sections=[_section("It", "We are.")])
        plan = build_deterministic_visual_plan(script)
        assert plan.sections[0].search_queries

    def test_broader_query_uses_broad_term_count(self) -> None:
        script = _script()
        plan = build_deterministic_visual_plan(script)
        broader = build_concept_query(script.sections[0].heading, BROAD_QUERY_TERMS)
        assert broader in plan.sections[0].neutral_fallback_queries
