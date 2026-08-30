# Tests for the SectionVisualPlan/VisualPlan models.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.visual_plan import SectionVisualPlan, VisualPlan


class TestSectionVisualPlan:
    def test_minimal_construction_uses_defaults(self) -> None:
        plan = SectionVisualPlan(section_index=0)
        assert plan.semantic_summary == ""
        assert plan.visual_intents == []
        assert plan.search_queries == []
        assert plan.avoid_concepts == []
        assert plan.neutral_fallback_queries == []
        assert plan.preferred_visual_types is None
        assert plan.confidence is None

    def test_full_construction(self) -> None:
        plan = SectionVisualPlan(
            section_index=2,
            semantic_summary="The brain combines fragmented memories into a dream narrative.",
            visual_intents=["person sleeping and dreaming", "human brain activity"],
            search_queries=["person dreaming sleep", "human brain neuroscience"],
            avoid_concepts=["construction site", "building construction"],
            neutral_fallback_queries=["person sleeping at night"],
            preferred_visual_types=["video"],
            confidence="high",
        )
        assert plan.section_index == 2
        assert "construction site" in plan.avoid_concepts
        assert plan.confidence == "high"

    def test_section_index_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            SectionVisualPlan(section_index=-1)

    def test_list_fields_are_independently_isolated(self) -> None:
        a = SectionVisualPlan(section_index=0)
        b = SectionVisualPlan(section_index=1)
        a.search_queries.append("x")
        assert b.search_queries == []


class TestVisualPlan:
    def test_construction(self) -> None:
        plan = VisualPlan(
            topic="Why do humans dream?",
            sections=[SectionVisualPlan(section_index=0)],
            used_semantic_planning=True,
        )
        assert plan.topic == "Why do humans dream?"
        assert len(plan.sections) == 1
        assert plan.fallback_reason is None

    def test_topic_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            VisualPlan(topic="", sections=[], used_semantic_planning=False)

    def test_fallback_reason_recorded(self) -> None:
        plan = VisualPlan(
            topic="Dreams",
            sections=[],
            used_semantic_planning=False,
            fallback_reason="LLM call failed",
        )
        assert plan.used_semantic_planning is False
        assert plan.fallback_reason == "LLM call failed"

    def test_serialization_roundtrip(self) -> None:
        plan = VisualPlan(
            topic="Dreams",
            sections=[
                SectionVisualPlan(
                    section_index=0,
                    semantic_summary="Summary",
                    visual_intents=["a"],
                    search_queries=["b"],
                    avoid_concepts=["c"],
                    neutral_fallback_queries=["d"],
                )
            ],
            used_semantic_planning=True,
        )
        restored = VisualPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan
