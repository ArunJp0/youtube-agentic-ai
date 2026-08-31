# Visual QC Service: inspects actual representative frames from already-
# selected/downloaded media (not just search metadata) and decides whether
# each asset is semantically appropriate for its script section, replacing
# clearly weak/misleading ones through a bounded retry against
# VisualMediaService's existing selection logic.
#
# Deterministic orchestration only - no reasoning of its own. Semantic
# judgment comes entirely from the injected VisualRelevanceEvaluator (one
# call per section, batching that section's assets); this service owns
# policy (score thresholds -> decision), bounded replacement, repetition
# checking, and safe fallback when the evaluator is unavailable.
#
# Standalone component for this milestone: not wired into the main
# LangGraph pipeline yet (see src/visual_qc_demo.py).
from __future__ import annotations

import tempfile
from typing import Dict, List, Optional, Set, Tuple

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult
from src.models.visual_plan import SectionVisualPlan, VisualPlan
from src.models.visual_qc import AssetQCResult, RawAssetVerdict, SectionQCResult, VisualQCResult
from src.services.frame_sampling import calculate_sample_timestamps
from src.services.repetition_check import detect_repetition_warnings
from src.services.visual_media_service import RECENT_REUSE_LOOKBACK, VisualMediaService
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.visual_relevance_evaluator import (
    AssetFrames,
    SectionQCContext,
    VisualRelevanceEvaluator,
    VisualRelevanceEvaluatorError,
)

# >= this score: highly relevant, approved outright.
APPROVE_SCORE_THRESHOLD = 0.70

# >= this (but below approve): acceptable/neutral B-roll, still approved -
# below this: replacement recommended. Centralized here, not scattered
# magic numbers - see VisualQCService.__init__ to override per instance.
NEUTRAL_SCORE_THRESHOLD = 0.50

# Bounded replacement: a slot is retried at most this many times before
# the best-available asset is kept and explicitly flagged, rather than
# searching indefinitely.
DEFAULT_MAX_REPLACEMENT_ATTEMPTS = 2

# A best-effort assumed clip duration used only if a video asset's own
# duration is unknown AND ffprobe fails - keeps frame sampling from
# crashing QC over a single bad probe.
FALLBACK_ASSUMED_DURATION_SECONDS = 4.0


class VisualQCServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Evaluator failures are NOT raised - they are captured per-section as a
    metadata-fallback result so callers always get a structured
    VisualQCResult back.
    """


class VisualQCService:
    """Inspects actual downloaded media (representative frames), not just
    search metadata, and produces a structured VisualQCResult - replacing
    clearly weak/misleading assets through a bounded retry against the
    same VisualMediaService/VisualPlan that selected them originally.
    """

    def __init__(
        self,
        evaluator: VisualRelevanceEvaluator,
        assembler: VideoAssembler,
        visual_media_service: Optional[VisualMediaService] = None,
        max_replacement_attempts: int = DEFAULT_MAX_REPLACEMENT_ATTEMPTS,
        approve_threshold: float = APPROVE_SCORE_THRESHOLD,
        neutral_threshold: float = NEUTRAL_SCORE_THRESHOLD,
    ) -> None:
        """Initialize the Visual QC Service.

        Args:
            evaluator: Vision-capable relevance evaluator (real or mock)
            assembler: VideoAssembler used only for probing duration and
                extracting representative frames - no semantic decisions
                are ever delegated to it
            visual_media_service: Optional VisualMediaService to request
                replacement assets from when a slot is rejected. If None,
                rejected assets are kept as the best-available option
                (bounded replacement is skipped entirely, not attempted).
            max_replacement_attempts: Maximum replacement tries per slot
            approve_threshold: Minimum score for an outright "approved" decision
            neutral_threshold: Minimum score for a "neutral" (acceptable) decision
        """
        self.evaluator = evaluator
        self.assembler = assembler
        self.visual_media_service = visual_media_service
        self.max_replacement_attempts = max_replacement_attempts
        self.approve_threshold = approve_threshold
        self.neutral_threshold = neutral_threshold
        self._vision_calls = 0

    async def run_qc(
        self,
        topic: str,
        script: ScriptResult,
        visual_plan: VisualPlan,
        visual_result: VisualResult,
    ) -> Tuple[VisualQCResult, VisualResult]:
        """Run Visual QC over every section's selected assets.

        Never raises for evaluator failures - those are captured as a
        metadata-fallback result per affected section. Only raises
        VisualQCServiceError for configuration/programmer errors.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent
            visual_plan: The VisualPlan VisualMediaService used to select
                the media in ``visual_result`` (see
                VisualMediaService.build_plan)
            visual_result: Already-selected/downloaded media to QC

        Returns:
            (VisualQCResult, updated VisualResult with any replaced assets)
        """
        if script is None or visual_plan is None or visual_result is None:
            raise VisualQCServiceError("ScriptResult, VisualPlan, and VisualResult are all required")

        if not visual_result.success or not visual_result.sections:
            return (
                VisualQCResult(
                    topic=topic,
                    provider=self.evaluator.name,
                    success=False,
                    error="VisualResult was not successful; nothing to QC",
                ),
                visual_result,
            )

        self._vision_calls = 0
        downloaded_by_id, used_ids_in_order = self._reconstruct_global_state(visual_result)

        section_results: List[SectionQCResult] = []
        updated_mappings: List[SectionMediaMapping] = []
        fallback_reasons: List[str] = []

        with tempfile.TemporaryDirectory(prefix="visual_qc_") as tmp_dir:
            for mapping in visual_result.sections:
                section_plan = self._section_plan_for(visual_plan, mapping.section_index)
                narration = self._narration_for(script, mapping.section_index)

                section_result, final_assets, section_fallback_reason = await self._process_section(
                    topic, mapping, section_plan, narration, downloaded_by_id, used_ids_in_order, tmp_dir
                )
                section_results.append(section_result)
                updated_mappings.append(mapping.model_copy(update={"assets": final_assets}))
                if section_fallback_reason:
                    fallback_reasons.append(section_fallback_reason)

        all_ids_in_order = [
            self._asset_key(asset)
            for mapping in updated_mappings
            for asset in mapping.assets
            if asset.success
        ]
        repetition_warnings = detect_repetition_warnings(all_ids_in_order, min_gap=RECENT_REUSE_LOOKBACK)

        all_asset_results = [asset for section in section_results for asset in section.assets]
        updated_visual_result = visual_result.model_copy(update={"sections": updated_mappings})

        return (
            VisualQCResult(
                topic=topic,
                provider=self.evaluator.name,
                model=getattr(self.evaluator, "model", None),
                success=True,
                sections=section_results,
                total_assets_checked=len(all_asset_results),
                approved_count=sum(
                    1 for a in all_asset_results if a.decision in ("approved", "neutral", "metadata_fallback", "error")
                ),
                warning_count=sum(1 for a in all_asset_results if a.decision == "weak"),
                rejected_count=sum(1 for a in all_asset_results if a.decision == "rejected"),
                replaced_count=sum(1 for a in all_asset_results if a.replaced),
                vision_calls_made=self._vision_calls,
                fallback_used=bool(fallback_reasons),
                fallback_reason="; ".join(fallback_reasons) or None,
                repetition_warnings=repetition_warnings,
            ),
            updated_visual_result,
        )

    # ---- per-section processing ----------------------------------------------

    async def _process_section(
        self,
        topic: str,
        mapping: SectionMediaMapping,
        section_plan: Optional[SectionVisualPlan],
        narration: str,
        downloaded_by_id: Dict[str, MediaAsset],
        used_ids_in_order: List[str],
        tmp_dir: str,
    ) -> Tuple[SectionQCResult, List[MediaAsset], Optional[str]]:
        usable_indices = [i for i, a in enumerate(mapping.assets) if a.success and a.local_file_path]
        if not usable_indices:
            return SectionQCResult(section_index=mapping.section_index, assets=[]), list(mapping.assets), None

        verdicts_by_id, fallback_reason = await self._evaluate_section_assets(
            topic, mapping, section_plan, narration, usable_indices, tmp_dir
        )

        asset_results: List[AssetQCResult] = []
        final_assets: List[MediaAsset] = list(mapping.assets)

        for slot_index in usable_indices:
            asset = mapping.assets[slot_index]
            asset_id = self._asset_key(asset)

            if fallback_reason is not None:
                result = _metadata_fallback_result(asset_id, mapping.section_index, slot_index, fallback_reason)
                asset_results.append(result)
                continue

            result, final_asset = await self._resolve_slot(
                topic=topic,
                mapping=mapping,
                slot_index=slot_index,
                asset=asset,
                section_plan=section_plan,
                narration=narration,
                initial_verdict=verdicts_by_id.get(asset_id),
                downloaded_by_id=downloaded_by_id,
                used_ids_in_order=used_ids_in_order,
                tmp_dir=tmp_dir,
            )
            asset_results.append(result)
            final_assets[slot_index] = final_asset

        return (
            SectionQCResult(section_index=mapping.section_index, assets=asset_results),
            final_assets,
            fallback_reason,
        )

    async def _evaluate_section_assets(
        self,
        topic: str,
        mapping: SectionMediaMapping,
        section_plan: Optional[SectionVisualPlan],
        narration: str,
        usable_indices: List[int],
        tmp_dir: str,
    ) -> Tuple[Dict[str, RawAssetVerdict], Optional[str]]:
        """Batch-evaluate every usable asset in one section with a single
        evaluator call. Returns (verdicts_by_asset_id, fallback_reason) -
        fallback_reason is None on success."""
        try:
            frame_bundles = []
            for slot_index in usable_indices:
                asset = mapping.assets[slot_index]
                asset_id = self._asset_key(asset)
                basename = f"section-{mapping.section_index + 1:02d}-slot-{slot_index + 1:02d}"
                frame_paths = self._extract_frames_for_asset(asset, tmp_dir, basename)
                frame_bundles.append(AssetFrames(asset_id=asset_id, frame_paths=frame_paths))

            context = SectionQCContext(
                topic=topic,
                section_index=mapping.section_index,
                section_heading=mapping.section_heading,
                narration=narration,
                semantic_summary=section_plan.semantic_summary if section_plan else "",
                visual_intents=list(section_plan.visual_intents) if section_plan else [],
                avoid_concepts=list(section_plan.avoid_concepts) if section_plan else [],
                assets=frame_bundles,
            )
            verdicts = await self._call_evaluator(context)
            return {v.asset_id: v for v in verdicts}, None
        except VisualRelevanceEvaluatorError as e:
            return {}, str(e)
        except Exception as e:
            return {}, f"Unexpected Visual QC error: {e}"

    async def _resolve_slot(
        self,
        *,
        topic: str,
        mapping: SectionMediaMapping,
        slot_index: int,
        asset: MediaAsset,
        section_plan: Optional[SectionVisualPlan],
        narration: str,
        initial_verdict: Optional[RawAssetVerdict],
        downloaded_by_id: Dict[str, MediaAsset],
        used_ids_in_order: List[str],
        tmp_dir: str,
    ) -> Tuple[AssetQCResult, MediaAsset]:
        """Resolve one visual slot's final QC result, replacing the asset
        up to ``max_replacement_attempts`` times if the evaluator
        recommends it and a VisualMediaService is available to ask."""
        current_asset = asset
        asset_id = self._asset_key(asset)
        result = self._classify(asset_id, mapping.section_index, slot_index, initial_verdict, replaced=False, attempts=0)
        excluded: Set[str] = set()
        attempts = 0

        while (
            result.retry_recommended
            and attempts < self.max_replacement_attempts
            and self.visual_media_service is not None
            and section_plan is not None
        ):
            excluded.add(self._asset_key(current_asset))
            attempts += 1
            try:
                replacement_asset, _ = await self.visual_media_service.acquire_replacement_asset(
                    section_plan, slot_index, mapping.section_index, downloaded_by_id, used_ids_in_order, excluded
                )
            except Exception:
                break
            if not replacement_asset.success:
                break

            replacement_id = self._asset_key(replacement_asset)
            used_ids_in_order.append(replacement_id)
            verdict = await self._evaluate_single_asset(
                topic, mapping, slot_index, section_plan, narration, replacement_asset, replacement_id, attempts, tmp_dir
            )
            current_asset = replacement_asset
            result = self._classify(
                replacement_id, mapping.section_index, slot_index, verdict, replaced=True, attempts=attempts
            )

        return result, current_asset

    async def _evaluate_single_asset(
        self,
        topic: str,
        mapping: SectionMediaMapping,
        slot_index: int,
        section_plan: SectionVisualPlan,
        narration: str,
        asset: MediaAsset,
        asset_id: str,
        attempt_number: int,
        tmp_dir: str,
    ) -> Optional[RawAssetVerdict]:
        try:
            basename = f"replacement-{mapping.section_index + 1:02d}-{slot_index + 1:02d}-{attempt_number}"
            frame_paths = self._extract_frames_for_asset(asset, tmp_dir, basename)
            context = SectionQCContext(
                topic=topic,
                section_index=mapping.section_index,
                section_heading=mapping.section_heading,
                narration=narration,
                semantic_summary=section_plan.semantic_summary,
                visual_intents=list(section_plan.visual_intents),
                avoid_concepts=list(section_plan.avoid_concepts),
                assets=[AssetFrames(asset_id=asset_id, frame_paths=frame_paths)],
            )
            verdicts = await self._call_evaluator(context)
            return next((v for v in verdicts if v.asset_id == asset_id), None)
        except Exception:
            return None

    async def _call_evaluator(self, context: SectionQCContext) -> List[RawAssetVerdict]:
        verdicts = await self.evaluator.evaluate_section(context)
        self._vision_calls += 1
        return verdicts

    # ---- policy: verdict -> decision -----------------------------------------

    def _classify(
        self,
        asset_id: str,
        section_index: int,
        slot_index: int,
        verdict: Optional[RawAssetVerdict],
        replaced: bool,
        attempts: int,
    ) -> AssetQCResult:
        if verdict is None:
            return AssetQCResult(
                asset_id=asset_id,
                section_index=section_index,
                slot_index=slot_index,
                approved=True,
                decision="error",
                relevance_score=None,
                reason="No QC verdict returned for this asset; kept unvalidated.",
                misleading_or_conflicting=False,
                retry_recommended=False,
                evaluation_source="error",
                replaced=replaced,
                replacement_attempts=attempts,
            )

        if verdict.misleading_or_conflicting:
            decision, approved, retry_recommended = "rejected", False, True
        elif verdict.relevance_score >= self.approve_threshold:
            decision, approved, retry_recommended = "approved", True, False
        elif verdict.relevance_score >= self.neutral_threshold:
            decision, approved, retry_recommended = "neutral", True, False
        else:
            decision, approved, retry_recommended = "weak", False, True

        return AssetQCResult(
            asset_id=asset_id,
            section_index=section_index,
            slot_index=slot_index,
            approved=approved,
            decision=decision,
            relevance_score=verdict.relevance_score,
            reason=verdict.reason,
            detected_visual_summary=verdict.detected_visual_summary,
            misleading_or_conflicting=verdict.misleading_or_conflicting,
            retry_recommended=retry_recommended,
            evaluation_source="vision",
            replaced=replaced,
            replacement_attempts=attempts,
        )

    # ---- frame extraction -----------------------------------------------------

    def _extract_frames_for_asset(self, asset: MediaAsset, tmp_dir: str, basename: str) -> List[str]:
        if asset.asset_type != "video":
            return [asset.local_file_path]

        duration = asset.duration_seconds
        if not duration or duration <= 0:
            try:
                duration = self.assembler.probe_duration_seconds(asset.local_file_path)
            except VideoAssemblerError:
                duration = FALLBACK_ASSUMED_DURATION_SECONDS

        timestamps = calculate_sample_timestamps(duration)
        return self.assembler.extract_frames(asset.local_file_path, timestamps, tmp_dir, basename)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _reconstruct_global_state(visual_result: VisualResult) -> Tuple[Dict[str, MediaAsset], List[str]]:
        """Rebuild the global dedup state VisualMediaService would have had
        mid-generation, from an already-finished VisualResult - so a
        QC-driven replacement still respects global duplicate prevention."""
        downloaded_by_id: Dict[str, MediaAsset] = {}
        used_ids_in_order: List[str] = []
        for mapping in visual_result.sections:
            for asset in mapping.assets:
                if not asset.success:
                    continue
                key = VisualQCService._asset_key(asset)
                downloaded_by_id.setdefault(key, asset)
                used_ids_in_order.append(key)
        return downloaded_by_id, used_ids_in_order

    @staticmethod
    def _section_plan_for(visual_plan: VisualPlan, section_index: int) -> Optional[SectionVisualPlan]:
        return next((sp for sp in visual_plan.sections if sp.section_index == section_index), None)

    @staticmethod
    def _narration_for(script: ScriptResult, section_index: int) -> str:
        if 0 <= section_index < len(script.sections):
            return script.sections[section_index].narration
        return ""

    @staticmethod
    def _asset_key(asset: MediaAsset) -> str:
        return asset.provider_asset_id or asset.source_url or asset.local_file_path or ""


def _metadata_fallback_result(
    asset_id: str, section_index: int, slot_index: int, fallback_reason: str
) -> AssetQCResult:
    return AssetQCResult(
        asset_id=asset_id,
        section_index=section_index,
        slot_index=slot_index,
        approved=True,
        decision="metadata_fallback",
        relevance_score=None,
        reason=f"Vision QC unavailable, kept on the upstream metadata filter's prior approval: {fallback_reason}",
        misleading_or_conflicting=False,
        retry_recommended=False,
        evaluation_source="metadata_fallback",
        replaced=False,
        replacement_attempts=0,
    )
