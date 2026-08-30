# Visual Media Service: prepares stock image/video assets for a ScriptResult.
#
# This is a deterministic service, not an LLM-driven reasoning agent: it
# never calls an LLM itself. Semantic understanding of the script (what each
# section actually means, and which literal-but-wrong interpretations to
# avoid) is optionally supplied by a VisualContextPlanner - a single LLM
# call for the whole script, made once per generate_visuals() call - and
# this service is responsible only for turning that plan into actual
# candidate search/filter/select/download decisions. Asset selection itself
# is a fixed rule (prefer a never-used, plan-approved candidate; then a
# broader/neutral one; then controlled reuse; then a shared last resort).
#
# Visual quantity is duration-aware: how many distinct clips a section gets
# is derived from that section's own share of the real narration duration
# (see calculate_slot_count), never a fixed count per section or per video.
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.agents.visual_context_planner import VisualContextPlanner
from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult
from src.models.visual_plan import SectionVisualPlan, VisualPlan
from src.services.query_generation import LAST_RESORT_QUERY, build_deterministic_visual_plan, ordered_unique
from src.services.section_timing import calculate_section_durations
from src.services.semantic_visual_filter import passes_avoid_filter, relevance_score
from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError

DEFAULT_MEDIA_OUTPUT_DIR = os.path.join("output", "media")
DEFAULT_MAX_RESULTS_PER_QUERY = 5

# How many of the most recently *used* assets (across the whole video, not
# just this section) are off-limits for "controlled reuse" (selection
# priority 3 below) - avoids immediate back-to-back repeats and a couple of
# near-neighbors, without requiring full uniqueness across a long video.
RECENT_REUSE_LOOKBACK = 2

# A visual slot is never planned shorter than this, regardless of cadence
# math - prevents pathologically short sections from being sliced into
# many sub-clips that FFmpeg would barely show.
MIN_SLOT_SECONDS = 3.0


@dataclass(frozen=True)
class _CadenceTier:
    """One tier of the visual-cadence policy: for videos up to
    ``max_total_seconds`` long, aim to change the visual roughly every
    ``min_seconds_per_clip``-``max_seconds_per_clip`` seconds."""

    max_total_seconds: float
    min_seconds_per_clip: float
    max_seconds_per_clip: float


# Visual cadence policy, tiered by total narration duration: longer videos
# hold each clip slightly longer, so the number of clips needed does not
# grow linearly forever. Centralized here (not scattered magic numbers) so
# the whole policy can be tuned in one place.
CADENCE_POLICY: List[_CadenceTier] = [
    _CadenceTier(max_total_seconds=300.0, min_seconds_per_clip=6.0, max_seconds_per_clip=10.0),  # <= 5 min
    _CadenceTier(max_total_seconds=600.0, min_seconds_per_clip=8.0, max_seconds_per_clip=12.0),  # 5-10 min
    _CadenceTier(max_total_seconds=float("inf"), min_seconds_per_clip=10.0, max_seconds_per_clip=15.0),  # > 10 min
]


class VisualMediaServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Provider search/download failures are NOT raised - they are captured
    per-asset in the returned VisualResult (MediaAsset.success=False,
    error=...) so callers always get a structured result back.
    """


class VisualMediaService:
    """Deterministic service that prepares visual assets for a ScriptResult.

    For each section: derives how many distinct visual slots it needs from
    its own share of the real narration duration (duration-aware, never a
    fixed count), consults a per-section SectionVisualPlan (from an
    optional VisualContextPlanner, one LLM call for the whole script - or a
    deterministic equivalent) for which queries to search and which
    concepts to avoid, and fills each slot by searching candidate metadata
    first and downloading only the one candidate actually selected -
    preferring a never-used, plan-approved asset, then a broader/neutral
    one, then (only if no such candidate exists anywhere) reusing an
    already-downloaded asset that wasn't used immediately before, and only
    as a last resort reusing the most recent one.
    """

    def __init__(
        self,
        media_provider: MediaProvider,
        visual_planner: Optional[VisualContextPlanner] = None,
        output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
        max_results_per_query: int = DEFAULT_MAX_RESULTS_PER_QUERY,
        prefer_video: bool = True,
    ) -> None:
        """Initialize the Visual Media Service.

        Args:
            media_provider: Implementation of MediaProvider for search/download
            visual_planner: Optional VisualContextPlanner for context-aware
                query generation and semantic avoid-concepts. If None, a
                deterministic equivalent plan is built directly (see
                query_generation.build_deterministic_visual_plan) - no
                semantic filtering, but otherwise identical behavior.
            output_dir: Local directory to write downloaded assets into
                (expected to be excluded from version control)
            max_results_per_query: Candidates to request per query, giving
                room to find a unique (non-duplicate) match
            prefer_video: Ask the provider for video clips before images
        """
        self.media_provider = media_provider
        self.visual_planner = visual_planner
        self.output_dir = output_dir
        self.max_results_per_query = max_results_per_query
        self.prefer_video = prefer_video

    async def generate_visuals(
        self, script: ScriptResult, total_narration_duration_seconds: float
    ) -> VisualResult:
        """Prepare duration-aware, context-planned visual assets for every
        section of a ScriptResult.

        Never raises for search/download failures - those are captured
        per-asset in the returned VisualResult. Only raises
        VisualMediaServiceError for configuration/programmer errors (e.g. a
        missing ScriptResult or a non-positive duration).

        Args:
            script: Structured script produced by the Script Agent
            total_narration_duration_seconds: The real narration audio
                duration (VoiceResult.duration_seconds) - the upstream
                timing input this service's visual planning is based on.
                Section/slot durations are derived from this, not estimated
                independently.

        Returns:
            Structured VisualResult mapping each section to its ordered asset(s)
        """
        if script is None:
            raise VisualMediaServiceError("ScriptResult is required")
        if total_narration_duration_seconds is None or total_narration_duration_seconds <= 0:
            raise VisualMediaServiceError(
                "total_narration_duration_seconds must be a positive number "
                "(pass VoiceResult.duration_seconds)"
            )

        if not script.sections:
            return VisualResult(
                topic=script.topic,
                provider=self.media_provider.name,
                sections=[],
                success=False,
                error="ScriptResult has no sections to find visuals for",
            )

        os.makedirs(self.output_dir, exist_ok=True)

        plan = self._build_plan(script)
        section_durations = calculate_section_durations(
            script.sections, total_narration_duration_seconds
        )

        downloaded_by_id: Dict[str, MediaAsset] = {}
        used_ids_in_order: List[str] = []
        section_mappings: List[SectionMediaMapping] = []

        for index, (section, duration) in enumerate(zip(script.sections, section_durations)):
            section_plan = plan.sections[index]
            slot_count = self.calculate_slot_count(duration, total_narration_duration_seconds)

            slot_assets: List[MediaAsset] = []
            slot_queries: List[str] = []
            for slot_index in range(slot_count):
                asset, used_query = await self._acquire_slot_asset(
                    section_plan, slot_index, index, downloaded_by_id, used_ids_in_order
                )
                slot_assets.append(asset)
                slot_queries.append(used_query)
                if asset.success:
                    used_ids_in_order.append(self._asset_key(asset))

            section_mappings.append(
                SectionMediaMapping(
                    section_index=index,
                    section_heading=section.heading,
                    search_queries=slot_queries,
                    planned_duration_seconds=duration,
                    semantic_summary=section_plan.semantic_summary or None,
                    avoid_concepts=list(section_plan.avoid_concepts),
                    assets=slot_assets,
                )
            )

        failed_sections = sum(
            1 for mapping in section_mappings if not any(a.success for a in mapping.assets)
        )
        error = None
        if failed_sections:
            error = f"{failed_sections} of {len(section_mappings)} section(s) got no usable media asset"

        return VisualResult(
            topic=script.topic,
            provider=self.media_provider.name,
            sections=section_mappings,
            success=(failed_sections == 0),
            error=error,
            semantic_planning_used=plan.used_semantic_planning,
            semantic_planning_fallback_reason=plan.fallback_reason,
        )

    def _build_plan(self, script: ScriptResult) -> VisualPlan:
        """Get the per-section visual plan: from the configured planner (one
        LLM call for the whole script, with its own internal deterministic
        fallback) if one is set, otherwise directly from the deterministic
        query-generation logic."""
        if self.visual_planner is not None:
            return self.visual_planner.plan_visuals(script)
        return build_deterministic_visual_plan(script)

    # ---- duration-aware slot planning ---------------------------------------

    @staticmethod
    def calculate_slot_count(section_duration_seconds: float, total_video_duration_seconds: float) -> int:
        """Calculate the minimum reasonable number of visual slots a section needs.

        Duration-aware, not a fixed count: derived from the section's own
        duration and the cadence tier that applies to the *total* video
        duration (see CADENCE_POLICY). A section shorter than
        ``MIN_SLOT_SECONDS`` always gets exactly one slot.

        Args:
            section_duration_seconds: This section's own planned duration
            total_video_duration_seconds: The whole video's narration duration,
                used to pick which cadence tier applies

        Returns:
            Number of visual slots (>= 1) to plan for this section
        """
        tier = VisualMediaService._select_cadence_tier(total_video_duration_seconds)
        target_seconds_per_clip = (tier.min_seconds_per_clip + tier.max_seconds_per_clip) / 2

        desired = max(1, round(section_duration_seconds / target_seconds_per_clip))
        # Never slice a section into slots shorter than MIN_SLOT_SECONDS on
        # average - that would over-fetch for no visible benefit.
        max_by_floor = max(1, int(section_duration_seconds // MIN_SLOT_SECONDS))
        return max(1, min(desired, max_by_floor))

    @staticmethod
    def _select_cadence_tier(total_video_duration_seconds: float) -> _CadenceTier:
        for tier in CADENCE_POLICY:
            if total_video_duration_seconds <= tier.max_total_seconds:
                return tier
        return CADENCE_POLICY[-1]

    # ---- asset selection/download -------------------------------------------

    async def _acquire_slot_asset(
        self,
        section_plan: SectionVisualPlan,
        slot_index: int,
        section_index: int,
        downloaded_by_id: Dict[str, MediaAsset],
        used_ids_in_order: List[str],
    ) -> Tuple[MediaAsset, str]:
        """Fill one visual slot from a section's visual plan.

        Each slot gets its own specific query (search_queries, rotated by
        slot_index so different slots in the same section search different
        facets of its content - cheap, since it means at most one specific
        query per slot rather than re-trying every specific query on every
        slot), then falls through to the plan's neutral_fallback_queries,
        then a shared last resort. Every candidate is checked against the
        plan's avoid_concepts before it can be selected; only the one
        candidate actually chosen is downloaded.

        Selection priority:
            1. A never-used, plan-approved candidate for this slot's
               specific query.
            2. A never-used, plan-approved candidate for a neutral/broader
               query.
            3. An already-downloaded asset not used in the last
               RECENT_REUSE_LOOKBACK slots (no re-download - reuses the
               existing local file).
            4. Any already-downloaded asset, even the most recent one
               (immediate repetition - absolute last resort).

        Returns:
            (asset, query_used_to_find_it)
        """
        recent_ids = set(used_ids_in_order[-RECENT_REUSE_LOOKBACK:])
        specific_queries = section_plan.search_queries
        primary = specific_queries[slot_index % len(specific_queries)] if specific_queries else None

        chain_sources = ([primary] if primary else []) + list(section_plan.neutral_fallback_queries) + [
            LAST_RESORT_QUERY
        ]
        query_chain = ordered_unique(chain_sources)
        tier_by_query: Dict[str, str] = {}
        if primary:
            tier_by_query.setdefault(primary, "high")
        for query in list(section_plan.neutral_fallback_queries) + [LAST_RESORT_QUERY]:
            tier_by_query.setdefault(query, "neutral")

        last_error = "No search queries available"
        searched_any = False

        for query in query_chain:
            try:
                candidates = await self.media_provider.search(
                    query, prefer_video=self.prefer_video, max_results=self.max_results_per_query
                )
            except MediaProviderError as e:
                last_error = f"Media search failed: {e}"
                continue
            except Exception as e:
                last_error = f"Unexpected media search error: {e}"
                continue

            searched_any = True
            candidate = next(
                (
                    c
                    for c in candidates
                    if self._candidate_key(c) not in downloaded_by_id
                    and passes_avoid_filter(c, section_plan)
                ),
                None,
            )
            if candidate is not None:
                tier = tier_by_query.get(query, "neutral")
                score = relevance_score(candidate, section_plan)
                return await self._download_new_asset(
                    candidate, query, section_index, downloaded_by_id, tier, score
                )

        # No never-used, plan-approved candidate found anywhere in the
        # chain - fall back to reusing an already-downloaded asset rather
        # than failing the slot.
        fallback_query = query_chain[-1] if query_chain else ""

        for asset_id, asset in downloaded_by_id.items():
            if asset_id not in recent_ids:
                return self._reuse_asset(asset, fallback_query, section_index), fallback_query

        if downloaded_by_id:
            any_asset = next(iter(downloaded_by_id.values()))
            return self._reuse_asset(any_asset, fallback_query, section_index), fallback_query

        if not searched_any:
            last_error = last_error or "No search queries available"
        else:
            last_error = "No media asset available (search returned no usable candidates)"

        return (
            MediaAsset(
                provider=self.media_provider.name,
                search_query=fallback_query or "unknown",
                section_index=section_index,
                success=False,
                error=last_error,
            ),
            fallback_query,
        )

    async def _download_new_asset(
        self,
        candidate: MediaCandidate,
        query: str,
        section_index: int,
        downloaded_by_id: Dict[str, MediaAsset],
        relevance_tier: str,
        relevance_score_value: float,
    ) -> Tuple[MediaAsset, str]:
        output_path = os.path.join(self.output_dir, self._build_filename(section_index, candidate))
        try:
            await self.media_provider.download(candidate, output_path)
        except MediaProviderError as e:
            return (
                MediaAsset(
                    provider=self.media_provider.name,
                    search_query=query,
                    section_index=section_index,
                    success=False,
                    error=f"Media download failed: {e}",
                ),
                query,
            )
        except Exception as e:
            return (
                MediaAsset(
                    provider=self.media_provider.name,
                    search_query=query,
                    section_index=section_index,
                    success=False,
                    error=f"Unexpected download error: {e}",
                ),
                query,
            )

        asset = MediaAsset(
            provider=self.media_provider.name,
            asset_type=candidate.asset_type,
            local_file_path=output_path,
            source_url=candidate.source_url,
            provider_asset_id=candidate.provider_asset_id,
            attribution=candidate.attribution,
            search_query=query,
            section_index=section_index,
            duration_seconds=candidate.duration_seconds,
            width=candidate.width,
            height=candidate.height,
            reused=False,
            relevance_tier=relevance_tier,
            relevance_score=relevance_score_value,
            success=True,
        )
        downloaded_by_id[self._candidate_key(candidate)] = asset
        return asset, query

    @staticmethod
    def _reuse_asset(existing_asset: MediaAsset, query: str, section_index: int) -> MediaAsset:
        """Reference an already-downloaded asset for another slot - no new download."""
        return existing_asset.model_copy(
            update={
                "search_query": query,
                "section_index": section_index,
                "reused": True,
                "relevance_tier": "reused",
            }
        )

    @staticmethod
    def _candidate_key(candidate: MediaCandidate) -> str:
        return candidate.provider_asset_id or candidate.source_url or candidate.download_url

    @staticmethod
    def _asset_key(asset: MediaAsset) -> str:
        return asset.provider_asset_id or asset.source_url or asset.local_file_path or ""

    @staticmethod
    def _build_filename(section_index: int, candidate: MediaCandidate) -> str:
        default_ext = "mp4" if candidate.asset_type == "video" else "jpg"
        url_ext = os.path.splitext(urlparse(candidate.download_url).path)[1].lstrip(".")
        ext = url_ext if url_ext and len(url_ext) <= 4 else default_ext
        return f"section-{section_index + 1:02d}-{uuid.uuid4().hex[:8]}.{ext}"
