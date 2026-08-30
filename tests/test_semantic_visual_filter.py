# Tests for the deterministic semantic filtering layer that enforces a
# SectionVisualPlan against candidate metadata.
from __future__ import annotations

from src.models.visual_plan import SectionVisualPlan
from src.services.semantic_visual_filter import passes_avoid_filter, relevance_score
from src.tools.media_provider import MediaCandidate


def _candidate(content_hint=None) -> MediaCandidate:
    return MediaCandidate(
        asset_type="video",
        download_url="https://mock.media/x.mp4",
        source_url="https://mock.media/page/x",
        content_hint=content_hint,
    )


class TestPassesAvoidFilter:
    def test_no_content_hint_is_not_blocked(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=["construction site"])
        assert passes_avoid_filter(_candidate(content_hint=None), plan) is True

    def test_no_avoid_concepts_never_blocks(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=[])
        candidate = _candidate(content_hint="construction site building")
        assert passes_avoid_filter(candidate, plan) is True

    def test_matching_avoid_concept_is_rejected(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=["construction site"])
        candidate = _candidate(content_hint="workers on a construction site downtown")
        assert passes_avoid_filter(candidate, plan) is False

    def test_case_and_whitespace_insensitive(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=["  Building Construction  "])
        candidate = _candidate(content_hint="BUILDING CONSTRUCTION site aerial view")
        assert passes_avoid_filter(candidate, plan) is False

    def test_non_matching_hint_passes(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=["construction site"])
        candidate = _candidate(content_hint="person sleeping peacefully at night")
        assert passes_avoid_filter(candidate, plan) is True

    def test_blank_avoid_concept_entries_are_ignored(self) -> None:
        plan = SectionVisualPlan(section_index=0, avoid_concepts=["", "   "])
        candidate = _candidate(content_hint="anything at all")
        assert passes_avoid_filter(candidate, plan) is True

    def test_multiple_avoid_concepts_any_match_rejects(self) -> None:
        plan = SectionVisualPlan(
            section_index=0, avoid_concepts=["industrial machinery", "office building"]
        )
        candidate = _candidate(content_hint="tall office building at sunset")
        assert passes_avoid_filter(candidate, plan) is False


class TestRelevanceScore:
    def test_no_content_hint_scores_zero(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=["person sleeping"])
        assert relevance_score(_candidate(content_hint=None), plan) == 0.0

    def test_no_positive_terms_scores_zero(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=[], search_queries=[])
        candidate = _candidate(content_hint="person sleeping at night")
        assert relevance_score(candidate, plan) == 0.0

    def test_full_overlap_scores_high(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=["person sleeping"])
        candidate = _candidate(content_hint="person sleeping")
        assert relevance_score(candidate, plan) == 1.0

    def test_partial_overlap_scores_between_zero_and_one(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=["person sleeping at night"])
        candidate = _candidate(content_hint="person walking outdoors")
        score = relevance_score(candidate, plan)
        assert 0.0 < score < 1.0

    def test_no_overlap_scores_zero(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=["human brain neuroscience"])
        candidate = _candidate(content_hint="race car driving fast")
        assert relevance_score(candidate, plan) == 0.0

    def test_search_queries_also_contribute_to_score(self) -> None:
        plan = SectionVisualPlan(section_index=0, visual_intents=[], search_queries=["dream sleep"])
        candidate = _candidate(content_hint="dream sleep bedroom")
        assert relevance_score(candidate, plan) > 0.0
