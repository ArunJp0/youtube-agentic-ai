# Tests for VisualQCService: policy thresholds, bounded replacement,
# metadata-fallback on evaluator failure, and repetition warnings. Uses
# fake evaluator/assembler/replacement-provider test doubles only - no
# real network, Gemini, or FFmpeg process.
from __future__ import annotations

import os

import pytest

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.models.visual_plan import SectionVisualPlan, VisualPlan
from src.models.visual_qc import RawAssetVerdict
from src.services.visual_qc_service import (
    APPROVE_SCORE_THRESHOLD,
    NEUTRAL_SCORE_THRESHOLD,
    VisualQCService,
    VisualQCServiceError,
)
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.visual_relevance_evaluator import MockVisualRelevanceEvaluator


class FakeAssembler(VideoAssembler):
    """Records extract_frames calls and writes tiny placeholder frame
    files instead of running real FFmpeg."""

    def __init__(self, duration: float = 5.0, fail_probe: bool = False, fail_extract: bool = False) -> None:
        self.duration = duration
        self.fail_probe = fail_probe
        self.fail_extract = fail_extract
        self.extract_calls: list[dict] = []

    def probe_duration_seconds(self, media_path: str) -> float:
        if self.fail_probe:
            raise VideoAssemblerError("simulated probe failure")
        return self.duration

    def build_section_clip(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Visual QC tests")

    def concatenate_and_mux_audio(self, *args, **kwargs) -> None:
        raise NotImplementedError("not exercised by Visual QC tests")

    def extract_frames(self, input_path, timestamps_seconds, output_dir, basename):
        if self.fail_extract:
            raise VideoAssemblerError("simulated extract failure")
        self.extract_calls.append(
            {"input_path": input_path, "timestamps_seconds": list(timestamps_seconds), "basename": basename}
        )
        paths = []
        for index, _ in enumerate(timestamps_seconds):
            path = os.path.join(output_dir, f"{basename}-{index + 1:02d}.jpg")
            with open(path, "wb") as f:
                f.write(b"FRAME")
            paths.append(path)
        return paths


class FakeReplacementProvider:
    """Duck-typed stand-in for VisualMediaService's acquire_replacement_asset:
    returns a pre-scripted queue of replacement assets per (section, slot),
    mutating downloaded_by_id/used_ids_in_order the same way the real
    method does, and records every call (including exclude_ids) for
    assertions."""

    def __init__(self, replacements_by_slot: dict) -> None:
        self.replacements_by_slot = {k: list(v) for k, v in replacements_by_slot.items()}
        self.calls: list[dict] = []

    async def acquire_replacement_asset(
        self, section_plan, slot_index, section_index, downloaded_by_id, used_ids_in_order, exclude_ids
    ):
        self.calls.append(
            {"section_index": section_index, "slot_index": slot_index, "exclude_ids": set(exclude_ids)}
        )
        queue = self.replacements_by_slot.get((section_index, slot_index), [])
        if not queue:
            return (
                MediaAsset(provider="mock", search_query="q", section_index=section_index, success=False, error="none left"),
                "q",
            )
        asset = queue.pop(0)
        key = _asset_key(asset)
        downloaded_by_id[key] = asset
        used_ids_in_order.append(key)
        return asset, "q"


def _asset_key(asset: MediaAsset) -> str:
    return asset.provider_asset_id or asset.source_url or asset.local_file_path or ""


def _script(sections) -> ScriptResult:
    return ScriptResult(
        topic="Why do humans dream?",
        video_title="Title",
        hook="Hook",
        introduction="Intro",
        sections=sections,
        conclusion="Conclusion",
        call_to_action="CTA",
    )


def _section_plan(index: int, **overrides) -> SectionVisualPlan:
    defaults = dict(
        section_index=index,
        semantic_summary="A mental process, not a physical one.",
        visual_intents=["person thinking"],
        search_queries=["person thinking"],
        avoid_concepts=["construction site"],
        neutral_fallback_queries=["neutral footage"],
    )
    defaults.update(overrides)
    return SectionVisualPlan(**defaults)


def _plan(*section_plans, topic: str = "Why do humans dream?") -> VisualPlan:
    return VisualPlan(topic=topic, sections=list(section_plans), used_semantic_planning=True)


def _asset(tmp_path, asset_id: str, section_index: int, asset_type: str = "image", success: bool = True, duration=None) -> MediaAsset:
    path = tmp_path / f"{asset_id}.jpg"
    if not path.exists():
        path.write_bytes(b"IMG")
    return MediaAsset(
        provider="mock",
        asset_type=asset_type if success else None,
        local_file_path=str(path) if success else None,
        provider_asset_id=asset_id,
        search_query="q",
        section_index=section_index,
        duration_seconds=duration,
        success=success,
    )


def _mapping(section_index: int, assets, heading: str = "Heading") -> SectionMediaMapping:
    return SectionMediaMapping(
        section_index=section_index,
        section_heading=heading,
        search_queries=["q"] * len(assets),
        assets=list(assets),
    )


def _visual_result(mappings, topic: str = "Why do humans dream?", success: bool = True) -> VisualResult:
    return VisualResult(topic=topic, provider="mock", sections=list(mappings), success=success)


class TestRunQcValidation:
    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self, tmp_path) -> None:
        service = VisualQCService(evaluator=MockVisualRelevanceEvaluator(), assembler=FakeAssembler())
        with pytest.raises(VisualQCServiceError):
            await service.run_qc("topic", None, _plan(), _visual_result([]))

    @pytest.mark.asyncio
    async def test_unsuccessful_visual_result_returns_clean_failure(self, tmp_path) -> None:
        service = VisualQCService(evaluator=MockVisualRelevanceEvaluator(), assembler=FakeAssembler())
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        visual_result = _visual_result([], success=False)

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.success is False
        assert qc_result.error is not None
        assert updated is visual_result


class TestRelevanceDecisions:
    @pytest.mark.asyncio
    async def test_highly_relevant_asset_approved(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"a1": RawAssetVerdict(asset_id="a1", relevance_score=0.9, reason="clear match")}
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "approved"
        assert result.approved is True
        assert result.evaluation_source == "vision"

    @pytest.mark.asyncio
    async def test_neutral_asset_accepted(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"a1": RawAssetVerdict(asset_id="a1", relevance_score=0.6, reason="acceptable b-roll")}
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "neutral"
        assert result.approved is True
        assert result.retry_recommended is False

    @pytest.mark.asyncio
    async def test_low_relevance_flags_replacement_recommended(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"a1": RawAssetVerdict(asset_id="a1", relevance_score=0.2, reason="weak")}
        )
        # No visual_media_service configured - can't actually replace, but
        # the verdict itself must still say a replacement was warranted.
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "weak"
        assert result.approved is False
        assert result.retry_recommended is True
        assert result.replaced is False

    @pytest.mark.asyncio
    async def test_misleading_asset_rejected_even_with_high_score(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="The mind constructs a narrative.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={
                "a1": RawAssetVerdict(
                    asset_id="a1", relevance_score=0.95, misleading_or_conflicting=True,
                    reason="construction site footage for a mental-narrative section",
                )
            }
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "rejected"
        assert result.approved is False
        assert result.misleading_or_conflicting is True

    @pytest.mark.asyncio
    async def test_threshold_boundaries_are_centralized_constants(self, tmp_path) -> None:
        assert APPROVE_SCORE_THRESHOLD > NEUTRAL_SCORE_THRESHOLD
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"a1": RawAssetVerdict(asset_id="a1", relevance_score=APPROVE_SCORE_THRESHOLD)}
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())
        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)
        assert qc_result.sections[0].assets[0].decision == "approved"


class TestBoundedReplacement:
    @pytest.mark.asyncio
    async def test_replacement_asset_approved_stops_retry_loop(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        original = _asset(tmp_path, "bad1", 0)
        replacement = _asset(tmp_path, "good1", 0)
        visual_result = _visual_result([_mapping(0, [original])])

        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={
                "bad1": RawAssetVerdict(asset_id="bad1", relevance_score=0.1, reason="weak"),
                "good1": RawAssetVerdict(asset_id="good1", relevance_score=0.95, reason="great match"),
            }
        )
        replacement_provider = FakeReplacementProvider({(0, 0): [replacement]})
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider)

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.asset_id == "good1"
        assert result.decision == "approved"
        assert result.replaced is True
        assert result.replacement_attempts == 1
        assert updated.sections[0].assets[0].provider_asset_id == "good1"

    @pytest.mark.asyncio
    async def test_bounded_by_max_replacement_attempts(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        original = _asset(tmp_path, "bad0", 0)
        replacements = [_asset(tmp_path, f"bad{i}", 0) for i in range(1, 5)]
        visual_result = _visual_result([_mapping(0, [original])])

        verdicts = {f"bad{i}": RawAssetVerdict(asset_id=f"bad{i}", relevance_score=0.1) for i in range(0, 5)}
        evaluator = MockVisualRelevanceEvaluator(verdicts_by_asset_id=verdicts)
        replacement_provider = FakeReplacementProvider({(0, 0): replacements})
        service = VisualQCService(
            evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider,
            max_replacement_attempts=2,
        )

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.replacement_attempts == 2  # bounded, not unbounded despite 4 replacements available
        assert len(replacement_provider.calls) == 2

    @pytest.mark.asyncio
    async def test_rejected_asset_ids_are_excluded_from_next_attempt(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        original = _asset(tmp_path, "bad0", 0)
        replacement = _asset(tmp_path, "bad1", 0)
        visual_result = _visual_result([_mapping(0, [original])])

        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={
                "bad0": RawAssetVerdict(asset_id="bad0", relevance_score=0.1),
                "bad1": RawAssetVerdict(asset_id="bad1", relevance_score=0.9),
            }
        )
        replacement_provider = FakeReplacementProvider({(0, 0): [replacement]})
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider)

        await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert replacement_provider.calls[0]["exclude_ids"] == {"bad0"}

    @pytest.mark.asyncio
    async def test_no_replacement_service_configured_keeps_best_available(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"a1": RawAssetVerdict(asset_id="a1", relevance_score=0.1)}
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=None)

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.sections[0].assets[0].replaced is False
        assert updated.sections[0].assets[0].provider_asset_id == "a1"

    @pytest.mark.asyncio
    async def test_replacement_provider_exhausted_keeps_last_attempted_asset(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        original = _asset(tmp_path, "bad0", 0)
        visual_result = _visual_result([_mapping(0, [original])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={"bad0": RawAssetVerdict(asset_id="bad0", relevance_score=0.1)}
        )
        # No replacements configured for this slot at all - acquire_replacement_asset returns success=False.
        replacement_provider = FakeReplacementProvider({})
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider)

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.asset_id == "bad0"
        assert result.replaced is False
        assert updated.sections[0].assets[0].provider_asset_id == "bad0"


class TestProviderFailureFallback:
    @pytest.mark.asyncio
    async def test_evaluator_failure_falls_back_to_metadata_approval(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(raise_error=RuntimeError("vision outage"))
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "metadata_fallback"
        assert result.approved is True
        assert result.evaluation_source == "metadata_fallback"
        assert qc_result.fallback_used is True
        assert "vision outage" in qc_result.fallback_reason
        # Nothing corrupted - the asset itself is unchanged.
        assert updated.sections[0].assets[0].provider_asset_id == "a1"

    @pytest.mark.asyncio
    async def test_vision_and_metadata_fallback_are_distinguishable(self, tmp_path) -> None:
        script = _script(
            [ScriptSection(heading="A", narration="Narration A."), ScriptSection(heading="B", narration="Narration B.")]
        )
        good_asset = _asset(tmp_path, "good", 0)
        other_asset = _asset(tmp_path, "other", 1)
        visual_result = _visual_result([_mapping(0, [good_asset]), _mapping(1, [other_asset])])

        class FlakyEvaluator(MockVisualRelevanceEvaluator):
            async def evaluate_section(self, context):
                if context.section_index == 1:
                    raise RuntimeError("section 2 vision outage")
                return await super().evaluate_section(context)

        evaluator = FlakyEvaluator(
            verdicts_by_asset_id={"good": RawAssetVerdict(asset_id="good", relevance_score=0.9)}
        )
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc(
            "topic", script, _plan(_section_plan(0), _section_plan(1)), visual_result
        )

        assert qc_result.sections[0].assets[0].evaluation_source == "vision"
        assert qc_result.sections[1].assets[0].evaluation_source == "metadata_fallback"

    @pytest.mark.asyncio
    async def test_missing_verdict_for_asset_produces_error_decision(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "unlisted", 0)
        visual_result = _visual_result([_mapping(0, [asset])])

        class OmittingEvaluator(MockVisualRelevanceEvaluator):
            async def evaluate_section(self, context):
                return []  # never returns a verdict for "unlisted"

        service = VisualQCService(evaluator=OmittingEvaluator(), assembler=FakeAssembler())

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.decision == "error"
        assert result.approved is True
        assert result.evaluation_source == "error"
        assert updated.sections[0].assets[0].provider_asset_id == "unlisted"


class TestVisualResultNotMutated:
    @pytest.mark.asyncio
    async def test_original_visual_result_unchanged_after_replacement(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        original = _asset(tmp_path, "bad0", 0)
        replacement = _asset(tmp_path, "good0", 0)
        visual_result = _visual_result([_mapping(0, [original])])

        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={
                "bad0": RawAssetVerdict(asset_id="bad0", relevance_score=0.1),
                "good0": RawAssetVerdict(asset_id="good0", relevance_score=0.9),
            }
        )
        replacement_provider = FakeReplacementProvider({(0, 0): [replacement]})
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider)

        _, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert visual_result.sections[0].assets[0].provider_asset_id == "bad0"  # original untouched
        assert updated.sections[0].assets[0].provider_asset_id == "good0"  # new object reflects the swap
        assert updated is not visual_result

    @pytest.mark.asyncio
    async def test_original_visual_result_unchanged_on_evaluator_failure(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Some narration.")])
        asset = _asset(tmp_path, "a1", 0)
        visual_result = _visual_result([_mapping(0, [asset])])
        evaluator = MockVisualRelevanceEvaluator(raise_error=RuntimeError("boom"))
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert visual_result.sections[0].assets[0].provider_asset_id == "a1"
        assert visual_result.sections[0].assets[0].success is True


class TestSectionAndAssetOrdering:
    @pytest.mark.asyncio
    async def test_ordering_preserved_across_multiple_sections_and_slots(self, tmp_path) -> None:
        script = _script(
            [
                ScriptSection(heading="A", narration="Narration A."),
                ScriptSection(heading="B", narration="Narration B."),
            ]
        )
        s0_assets = [_asset(tmp_path, f"s0-{i}", 0) for i in range(3)]
        s1_assets = [_asset(tmp_path, f"s1-{i}", 1) for i in range(2)]
        visual_result = _visual_result([_mapping(0, s0_assets), _mapping(1, s1_assets)])
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc(
            "topic", script, _plan(_section_plan(0), _section_plan(1)), visual_result
        )

        assert [s.section_index for s in qc_result.sections] == [0, 1]
        assert [a.slot_index for a in qc_result.sections[0].assets] == [0, 1, 2]
        assert [a.slot_index for a in qc_result.sections[1].assets] == [0, 1]
        assert [a.asset_id for a in qc_result.sections[0].assets] == ["s0-0", "s0-1", "s0-2"]

    @pytest.mark.asyncio
    async def test_section_with_no_usable_assets_produces_empty_result(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        failed_asset = MediaAsset(provider="mock", search_query="q", section_index=0, success=False, error="no asset")
        visual_result = _visual_result([_mapping(0, [failed_asset])])
        service = VisualQCService(evaluator=MockVisualRelevanceEvaluator(), assembler=FakeAssembler())

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.sections[0].assets == []
        assert updated.sections[0].assets[0].success is False


class TestRepetitionWarnings:
    @pytest.mark.asyncio
    async def test_back_to_back_duplicate_flagged(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        assets = [_asset(tmp_path, "same", 0), _asset(tmp_path, "same", 0)]
        visual_result = _visual_result([_mapping(0, assets)])
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert any("back-to-back" in w for w in qc_result.repetition_warnings)

    @pytest.mark.asyncio
    async def test_long_gap_reuse_not_flagged(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        assets = [_asset(tmp_path, f"id{i}", 0) for i in range(5)]
        assets.append(_asset(tmp_path, "id0", 0))  # reused after a long gap
        visual_result = _visual_result([_mapping(0, assets)])
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.repetition_warnings == []


class TestFrameSampling:
    @pytest.mark.asyncio
    async def test_video_asset_uses_assembler_extract_frames(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        asset = _asset(tmp_path, "v1", 0, asset_type="video", duration=4.0)
        visual_result = _visual_result([_mapping(0, [asset])])
        assembler = FakeAssembler(duration=4.0)
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=assembler)

        await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert len(assembler.extract_calls) == 1
        assert assembler.extract_calls[0]["timestamps_seconds"] == [2.0]  # midpoint of a 4s clip

    @pytest.mark.asyncio
    async def test_video_asset_without_known_duration_probes_it(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        asset = _asset(tmp_path, "v1", 0, asset_type="video", duration=None)
        visual_result = _visual_result([_mapping(0, [asset])])
        assembler = FakeAssembler(duration=10.0)
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=assembler)

        await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert assembler.extract_calls[0]["timestamps_seconds"] == [2.5, 7.5]  # 25%/75% of a probed 10s clip

    @pytest.mark.asyncio
    async def test_image_asset_skips_frame_extraction(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        asset = _asset(tmp_path, "img1", 0, asset_type="image")
        visual_result = _visual_result([_mapping(0, [asset])])
        assembler = FakeAssembler()
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=assembler)

        await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert assembler.extract_calls == []

    @pytest.mark.asyncio
    async def test_probe_failure_falls_back_to_assumed_duration_without_crashing(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        asset = _asset(tmp_path, "v1", 0, asset_type="video", duration=None)
        visual_result = _visual_result([_mapping(0, [asset])])
        assembler = FakeAssembler(fail_probe=True)
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=assembler)

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.sections[0].assets[0].evaluation_source == "vision"  # still completed, no crash

    @pytest.mark.asyncio
    async def test_frame_extraction_failure_falls_back_to_metadata(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="Narration A.")])
        asset = _asset(tmp_path, "v1", 0, asset_type="video", duration=4.0)
        visual_result = _visual_result([_mapping(0, [asset])])
        assembler = FakeAssembler(duration=4.0, fail_extract=True)
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=assembler)

        qc_result, updated = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        result = qc_result.sections[0].assets[0]
        assert result.evaluation_source == "metadata_fallback"
        assert result.approved is True
        assert qc_result.fallback_used is True
        assert updated.sections[0].assets[0].provider_asset_id == "v1"


class TestVisionCallCounting:
    @pytest.mark.asyncio
    async def test_one_call_per_section_without_replacement(self, tmp_path) -> None:
        script = _script(
            [ScriptSection(heading="A", narration="A."), ScriptSection(heading="B", narration="B.")]
        )
        assets0 = [_asset(tmp_path, f"s0-{i}", 0) for i in range(3)]
        assets1 = [_asset(tmp_path, f"s1-{i}", 1) for i in range(2)]
        visual_result = _visual_result([_mapping(0, assets0), _mapping(1, assets1)])
        evaluator = MockVisualRelevanceEvaluator(default_score=0.9)
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler())

        qc_result, _ = await service.run_qc(
            "topic", script, _plan(_section_plan(0), _section_plan(1)), visual_result
        )

        assert qc_result.vision_calls_made == 2  # one batched call per section, not per asset

    @pytest.mark.asyncio
    async def test_replacement_attempts_add_one_call_each(self, tmp_path) -> None:
        script = _script([ScriptSection(heading="A", narration="A.")])
        original = _asset(tmp_path, "bad0", 0)
        replacement = _asset(tmp_path, "good0", 0)
        visual_result = _visual_result([_mapping(0, [original])])
        evaluator = MockVisualRelevanceEvaluator(
            verdicts_by_asset_id={
                "bad0": RawAssetVerdict(asset_id="bad0", relevance_score=0.1),
                "good0": RawAssetVerdict(asset_id="good0", relevance_score=0.9),
            }
        )
        replacement_provider = FakeReplacementProvider({(0, 0): [replacement]})
        service = VisualQCService(evaluator=evaluator, assembler=FakeAssembler(), visual_media_service=replacement_provider)

        qc_result, _ = await service.run_qc("topic", script, _plan(_section_plan(0)), visual_result)

        assert qc_result.vision_calls_made == 2  # 1 initial section call + 1 replacement re-check
