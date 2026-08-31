# Tests for the VisualRelevanceEvaluator abstraction and MockVisualRelevanceEvaluator.
from __future__ import annotations

import pytest

from src.models.visual_qc import RawAssetVerdict
from src.tools.visual_relevance_evaluator import (
    AssetFrames,
    MockVisualRelevanceEvaluator,
    SectionQCContext,
    VisualRelevanceEvaluator,
)


def _context(asset_ids) -> SectionQCContext:
    return SectionQCContext(
        topic="Dreams",
        section_index=0,
        section_heading="Intro",
        narration="Some narration text.",
        semantic_summary="A summary",
        visual_intents=["a"],
        avoid_concepts=[],
        assets=[AssetFrames(asset_id=aid, frame_paths=[f"{aid}.jpg"]) for aid in asset_ids],
    )


class TestMockVisualRelevanceEvaluator:
    def test_is_visual_relevance_evaluator(self) -> None:
        assert isinstance(MockVisualRelevanceEvaluator(), VisualRelevanceEvaluator)

    def test_name(self) -> None:
        assert MockVisualRelevanceEvaluator().name == "mock"

    @pytest.mark.asyncio
    async def test_default_verdict_used_for_unconfigured_asset(self) -> None:
        evaluator = MockVisualRelevanceEvaluator(default_score=0.42)
        verdicts = await evaluator.evaluate_section(_context(["a"]))
        assert verdicts[0].asset_id == "a"
        assert verdicts[0].relevance_score == 0.42

    @pytest.mark.asyncio
    async def test_explicit_verdict_used_when_configured(self) -> None:
        explicit = RawAssetVerdict(asset_id="a", relevance_score=0.1, misleading_or_conflicting=True)
        evaluator = MockVisualRelevanceEvaluator(verdicts_by_asset_id={"a": explicit})
        verdicts = await evaluator.evaluate_section(_context(["a"]))
        assert verdicts[0] == explicit

    @pytest.mark.asyncio
    async def test_one_verdict_per_asset_in_input_order_covered(self) -> None:
        evaluator = MockVisualRelevanceEvaluator()
        verdicts = await evaluator.evaluate_section(_context(["a", "b", "c"]))
        assert {v.asset_id for v in verdicts} == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_raises_configured_error(self) -> None:
        evaluator = MockVisualRelevanceEvaluator(raise_error=RuntimeError("vision outage"))
        with pytest.raises(RuntimeError, match="vision outage"):
            await evaluator.evaluate_section(_context(["a"]))

    @pytest.mark.asyncio
    async def test_calls_are_recorded(self) -> None:
        evaluator = MockVisualRelevanceEvaluator()
        context = _context(["a"])
        await evaluator.evaluate_section(context)
        assert evaluator.calls == [context]
