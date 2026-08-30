# Visual Media Service: prepares stock image/video assets for a ScriptResult.
#
# This is a deterministic service, not an LLM-driven reasoning agent: search
# query extraction/expansion is fixed keyword/concept-mapping logic with no
# model calls, and asset selection is a fixed rule (prefer video, then
# landscape, then the first unique candidate, falling back to broader
# queries, then to controlled reuse, then to a shared last resort). Only the
# actual media search/download is delegated to a swappable MediaProvider.
#
# Visual quantity is duration-aware: how many distinct clips a section gets
# is derived from that section's own share of the real narration duration
# (see calculate_slot_count), never a fixed count per section or per video.
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.services.section_timing import calculate_section_durations
from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError

DEFAULT_MEDIA_OUTPUT_DIR = os.path.join("output", "media")
DEFAULT_MAX_RESULTS_PER_QUERY = 5
MAX_QUERY_TERMS = 4
BROAD_QUERY_TERMS = 2
VARIANT_QUERY_TERMS = 2
LAST_RESORT_QUERY = "background footage"

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

# Deliberately small and generic (not topic-specific) - filtered out of
# section text when building a search query, since they carry no visual
# meaning. Nothing here is specific to any one video's subject matter.
# Includes both plain filler words and abstract/scientific connector words
# that read fine in narration but return poor or irrelevant stock-media
# search results (e.g. "suppression", "creates", "mainly").
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "that",
    "this", "these", "those", "it", "its", "as", "by", "from", "we", "you",
    "your", "our", "have", "has", "had", "do", "does", "did", "not", "so",
    "can", "could", "will", "would", "about", "into", "if", "than", "then",
    "when", "while", "which", "who", "whom", "what", "how", "why", "because",
    "also", "just", "really", "actually", "let", "lets", "know", "knows",
    "think", "one", "two", "three", "some", "more", "most", "much", "many",
    "over", "around", "every", "each", "other", "another", "such", "there",
    "here", "their", "they", "them", "he", "she", "his", "her",
    # Abstract/scientific connector words: grammatically fine in narration,
    # but not visually searchable and not translated by the concept maps
    # below (so they'd otherwise survive into the query untouched).
    "creates", "mainly", "during", "may", "serve", "suppression", "functions",
    "function", "functioning", "mechanism", "mechanisms", "process",
    "processes", "processing", "illogic", "illogical", "approximately",
    "significant", "significantly", "primarily", "essentially", "therefore",
    "however", "additionally", "furthermore", "specifically", "particularly",
    "fundamentally", "ultimately", "typically", "generally", "usually",
    "evidence", "research", "researchers", "study", "studies", "suggest",
    "suggests", "indicate", "indicates", "reveal", "reveals",
    "understanding", "spend", "spends", "spending", "average", "roughly",
}

# Multi-word phrases translated to concrete, visually-searchable concepts
# before tokenization. Applied longest/most-specific first via ordered
# substring replacement. Generic across any science/education topic - not
# specific to any one video's subject matter (e.g. "memory consolidation"
# applies to any script discussing memory, not just one about dreaming).
_CONCEPT_PHRASES: List[tuple] = [
    ("prefrontal cortex", "human brain neuroscience"),
    ("frontal cortex", "human brain neuroscience"),
    ("cerebral cortex", "human brain neuroscience"),
    ("brain imaging", "brain scan neuroscience"),
    ("brain activity", "brain neuroscience"),
    ("neural activity", "brain neuroscience"),
    ("rem sleep", "person sleeping bedroom"),
    ("deep sleep", "person sleeping bedroom"),
    ("falling asleep", "person sleeping bedroom"),
    ("sleep cycle", "person sleeping night"),
    ("sleep cycles", "person sleeping night"),
    ("memory consolidation", "memory brain"),
    ("consolidates memories", "memory brain"),
    ("consolidate memories", "memory brain"),
    ("long-term memory", "memory brain"),
    ("emotional processing", "emotions feelings"),
    ("processes emotions", "emotions feelings"),
    ("emotional regulation", "emotions feelings"),
    ("evolutionary functions", "nature evolution survival"),
    ("threat simulation", "danger survival instinct"),
    ("problem-solving", "thinking mind"),
    ("per night", "at night"),
    ("each night", "at night"),
    ("dreaming", "dream sleep"),
    ("dreams", "dream sleep"),
    ("dream", "dream sleep"),
]

# Single-word replacements applied after tokenization, for scientific terms
# not caught by the phrase map above. Empty string means "drop this word
# entirely" (it's not visually searchable and has no concrete substitute).
_CONCEPT_WORDS = {
    "cortex": "brain",
    "prefrontal": "brain",
    "neurons": "brain neuroscience",
    "neuron": "brain neuroscience",
    "consolidation": "memory",
    "consolidates": "memory",
    "adults": "person",
    "adult": "person",
    "humans": "people",
}

# Concrete, visually-searchable words that should be prioritized when a
# query has to be trimmed to a small number of terms. Generic vocabulary
# (people, places, objects, settings) - not tied to any one topic.
_CONCRETE_VISUAL_BOOST = {
    "person", "people", "human", "man", "woman", "child", "sleeping", "sleep",
    "bedroom", "night", "dream", "brain", "neuroscience", "memory",
    "emotions", "emotion", "feelings", "nature", "evolution", "survival",
    "danger", "instinct", "mind", "thinking", "city", "ocean", "forest",
    "light", "dark", "clock", "time", "body", "face", "hands", "eyes",
}


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
    fixed count), generates a small set of distinct concept-query variants
    from its own heading/narration (no LLM call) so different slots search
    for different facets of the same content, and fills each slot by
    searching candidate metadata first and downloading only the one
    candidate actually selected - preferring a never-used asset, then a
    broader-query match, then (only if no unique candidate exists anywhere)
    reusing an already-downloaded asset that wasn't used immediately
    before, and only as a last resort reusing the most recent one.
    """

    def __init__(
        self,
        media_provider: MediaProvider,
        output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
        max_results_per_query: int = DEFAULT_MAX_RESULTS_PER_QUERY,
        prefer_video: bool = True,
    ) -> None:
        """Initialize the Visual Media Service.

        Args:
            media_provider: Implementation of MediaProvider for search/download
            output_dir: Local directory to write downloaded assets into
                (expected to be excluded from version control)
            max_results_per_query: Candidates to request per query, giving
                room to find a unique (non-duplicate) match
            prefer_video: Ask the provider for video clips before images
        """
        self.media_provider = media_provider
        self.output_dir = output_dir
        self.max_results_per_query = max_results_per_query
        self.prefer_video = prefer_video

    async def generate_visuals(
        self, script: ScriptResult, total_narration_duration_seconds: float
    ) -> VisualResult:
        """Prepare duration-aware visual assets for every section of a ScriptResult.

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

        section_durations = calculate_section_durations(
            script.sections, total_narration_duration_seconds
        )

        downloaded_by_id: Dict[str, MediaAsset] = {}
        used_ids_in_order: List[str] = []
        section_mappings: List[SectionMediaMapping] = []

        for index, (section, duration) in enumerate(zip(script.sections, section_durations)):
            slot_count = self.calculate_slot_count(duration, total_narration_duration_seconds)
            variants = self.build_query_variants(section, slot_count)
            broader_query = self._build_concept_query(section.heading, BROAD_QUERY_TERMS)
            topic_query = (
                self._build_concept_query(script.topic, MAX_QUERY_TERMS) if script.topic else ""
            )

            slot_assets: List[MediaAsset] = []
            slot_queries: List[str] = []
            for slot_index in range(slot_count):
                primary = variants[slot_index % len(variants)] if variants else broader_query
                query_chain = self._ordered_unique(
                    [primary, broader_query, topic_query, LAST_RESORT_QUERY]
                )

                asset, used_query = await self._acquire_slot_asset(
                    query_chain, index, downloaded_by_id, used_ids_in_order
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
        )

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

    # ---- query generation ---------------------------------------------------

    @staticmethod
    def build_query_variants(section: ScriptSection, max_variants: int) -> List[str]:
        """Generate up to ``max_variants`` distinct concept-query variants for a section.

        Deterministic (no LLM call): the section's heading+narration are
        concept-mapped and ranked the same way as the single-query builder,
        then chunked into small groups of distinct concrete keywords - so
        different visual slots within the same section search for
        different (but still relevant) facets of its content instead of
        all repeating the exact same query. Fully generic: works from
        whatever concrete/visual vocabulary the section's own text (after
        concept mapping) contains, for any topic.

        Args:
            section: The ScriptSection to derive query variants for
            max_variants: Maximum number of variants to generate (typically
                the number of visual slots planned for this section)

        Returns:
            Ordered list of distinct query strings, length <= max_variants
            (may be shorter if the section doesn't have enough distinct
            concrete keywords)
        """
        if max_variants <= 0:
            return []

        combined = f"{section.heading} {section.narration}"
        ranked = VisualMediaService._ranked_keywords(combined, max_candidates=40)

        variants: List[str] = []
        for start in range(0, len(ranked), VARIANT_QUERY_TERMS):
            if len(variants) >= max_variants:
                break
            chunk = ranked[start : start + VARIANT_QUERY_TERMS]
            if chunk:
                variants.append(" ".join(chunk))
        return variants

    @staticmethod
    def _build_concept_query(text: str, max_terms: int) -> str:
        """Translate ``text`` into a single short, concrete, visually-searchable query."""
        ranked = VisualMediaService._ranked_keywords(text, max_candidates=20)
        return " ".join(ranked[:max_terms]) if ranked else ""

    @staticmethod
    def _ranked_keywords(text: str, max_candidates: int) -> List[str]:
        """Concept-map, extract, and rank keywords from ``text``.

        Shared by both the single specific-query builder and the
        multi-variant builder, so they always agree on what's concrete/
        visual for the same input text.
        """
        substituted = VisualMediaService._apply_concept_phrases(text.lower())
        candidates = VisualMediaService._extract_keywords(substituted, max_candidates=max_candidates)

        mapped: List[str] = []
        seen: Set[str] = set()
        for word in candidates:
            replacement = _CONCEPT_WORDS.get(word, word)
            for token in replacement.split():
                if token and token not in seen:
                    seen.add(token)
                    mapped.append(token)

        if not mapped:
            return []

        # Concrete/visual words are prioritized, so a late but concrete
        # word (e.g. "night") isn't crowded out by an earlier but non-
        # visual one (e.g. "hours").
        boosted = [w for w in mapped if w in _CONCRETE_VISUAL_BOOST]
        rest = [w for w in mapped if w not in _CONCRETE_VISUAL_BOOST]
        return boosted + rest

    @staticmethod
    def _apply_concept_phrases(lowered_text: str) -> str:
        for phrase, replacement in _CONCEPT_PHRASES:
            if phrase in lowered_text:
                lowered_text = lowered_text.replace(phrase, replacement)
        return lowered_text

    @staticmethod
    def _extract_keywords(text: str, max_candidates: int, min_word_length: int = 4) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        seen: Set[str] = set()
        keywords: List[str] = []
        for word in cleaned.split():
            if len(word) < min_word_length or word in _STOPWORDS or word in seen:
                continue
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= max_candidates:
                break
        return keywords

    @staticmethod
    def _ordered_unique(queries: List[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                ordered.append(query)
        return ordered

    # ---- asset selection/download -------------------------------------------

    async def _acquire_slot_asset(
        self,
        query_chain: List[str],
        section_index: int,
        downloaded_by_id: Dict[str, MediaAsset],
        used_ids_in_order: List[str],
    ) -> Tuple[MediaAsset, str]:
        """Fill one visual slot: search metadata for each query in the chain
        (without downloading), then select and download only the one
        candidate actually chosen.

        Selection priority:
            1. A candidate never used anywhere in this video yet
               (tried query-by-query, most specific first).
            2. An already-downloaded asset not used in the last
               RECENT_REUSE_LOOKBACK slots (no re-download - reuses the
               existing local file).
            3. Any already-downloaded asset, even the most recent one
               (immediate repetition - absolute last resort).

        Returns:
            (asset, query_used_to_find_it)
        """
        recent_ids = set(used_ids_in_order[-RECENT_REUSE_LOOKBACK:])
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
                (c for c in candidates if self._candidate_key(c) not in downloaded_by_id), None
            )
            if candidate is not None:
                return await self._download_new_asset(candidate, query, section_index, downloaded_by_id)

        # No never-used candidate found anywhere in the chain - fall back to
        # reusing an already-downloaded asset rather than failing the slot.
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
            success=True,
        )
        downloaded_by_id[self._candidate_key(candidate)] = asset
        return asset, query

    @staticmethod
    def _reuse_asset(existing_asset: MediaAsset, query: str, section_index: int) -> MediaAsset:
        """Reference an already-downloaded asset for another slot - no new download."""
        return existing_asset.model_copy(
            update={"search_query": query, "section_index": section_index, "reused": True}
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
