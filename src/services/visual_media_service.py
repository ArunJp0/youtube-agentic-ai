# Visual Media Service: prepares stock image/video assets for a ScriptResult.
#
# This is a deterministic service, not an LLM-driven reasoning agent: search
# query extraction is fixed keyword/concept-mapping logic with no model
# calls, and asset selection is a fixed rule (prefer video, then landscape,
# then first non-duplicate candidate, falling back to a broader query).
# Only the actual media search/download is delegated to a swappable
# MediaProvider.
from __future__ import annotations

import os
import re
import uuid
from typing import List, Set
from urllib.parse import urlparse

from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError

DEFAULT_MEDIA_OUTPUT_DIR = os.path.join("output", "media")
DEFAULT_MAX_RESULTS_PER_QUERY = 5
MAX_QUERY_TERMS = 4
BROAD_QUERY_TERMS = 2
LAST_RESORT_QUERY = "background footage"

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
    per-section in the returned VisualResult (MediaAsset.success=False,
    error=...) so callers always get a structured result back.
    """


class VisualMediaService:
    """Deterministic service that prepares visual assets for a ScriptResult.

    For each section: derives an ordered chain of search queries from its
    own heading/narration (no LLM call) - a specific concept-mapped query,
    a broader version of it, then the script's overall topic, then a last-
    resort generic query - and tries the configured MediaProvider with each
    in turn (video preferred, landscape orientation) until one yields a
    candidate not already used elsewhere in this script. Records the
    outcome as a MediaAsset even on total failure, so every section gets a
    mapping entry.
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
                room to skip duplicates already used by another section
            prefer_video: Ask the provider for video clips before images
        """
        self.media_provider = media_provider
        self.output_dir = output_dir
        self.max_results_per_query = max_results_per_query
        self.prefer_video = prefer_video

    async def generate_visuals(self, script: ScriptResult) -> VisualResult:
        """Prepare a visual asset for every section of a ScriptResult.

        Never raises for search/download failures - those are captured
        per-section in the returned VisualResult. Only raises
        VisualMediaServiceError for configuration/programmer errors (e.g. a
        missing ScriptResult).

        Args:
            script: Structured script produced by the Script Agent

        Returns:
            Structured VisualResult mapping each section to its asset(s)
        """
        if script is None:
            raise VisualMediaServiceError("ScriptResult is required")

        if not script.sections:
            return VisualResult(
                topic=script.topic,
                provider=self.media_provider.name,
                sections=[],
                success=False,
                error="ScriptResult has no sections to find visuals for",
            )

        os.makedirs(self.output_dir, exist_ok=True)

        used_urls: Set[str] = set()
        section_mappings: List[SectionMediaMapping] = []

        for index, section in enumerate(script.sections):
            queries = self.build_search_queries(section, script.topic)
            asset = await self._find_and_download_asset(queries, index, used_urls)
            if asset.success:
                if asset.source_url:
                    used_urls.add(asset.source_url)
                used_urls.add(asset.local_file_path or "")

            section_mappings.append(
                SectionMediaMapping(
                    section_index=index,
                    section_heading=section.heading,
                    search_query=asset.search_query,
                    assets=[asset],
                )
            )

        failed_count = sum(1 for m in section_mappings if not m.assets[0].success)
        error = None
        if failed_count:
            error = f"{failed_count} of {len(section_mappings)} section(s) could not get a media asset"

        return VisualResult(
            topic=script.topic,
            provider=self.media_provider.name,
            sections=section_mappings,
            success=(failed_count == 0),
            error=error,
        )

    # ---- query generation ---------------------------------------------------

    @staticmethod
    def build_search_queries(section: ScriptSection, topic: str = "") -> List[str]:
        """Derive an ordered chain of search-query candidates for a section.

        Deterministic concept mapping + keyword extraction (no LLM call):
        1. A specific query from the section's own heading + narration,
           with scientific/abstract terms translated to concrete visual
           concepts (e.g. "prefrontal cortex" -> "human brain neuroscience").
        2. A broader version using just the heading, fewer terms.
        3. A query derived from the script's overall topic.
        4. A generic last-resort query.

        Callers try each in order until one yields a usable asset.

        Args:
            section: The ScriptSection to derive queries for
            topic: The script's overall topic, used for the broader fallback

        Returns:
            Ordered, deduplicated, non-empty list of query strings
        """
        specific = VisualMediaService._build_concept_query(
            f"{section.heading} {section.narration}", MAX_QUERY_TERMS
        )
        broader = VisualMediaService._build_concept_query(section.heading, BROAD_QUERY_TERMS)
        topic_query = (
            VisualMediaService._build_concept_query(topic, MAX_QUERY_TERMS) if topic else ""
        )

        queries: List[str] = []
        for query in (specific, broader, topic_query, LAST_RESORT_QUERY):
            if query and query not in queries:
                queries.append(query)
        return queries

    @staticmethod
    def _build_concept_query(text: str, max_terms: int) -> str:
        """Translate ``text`` into a short, concrete, visually-searchable query."""
        substituted = VisualMediaService._apply_concept_phrases(text.lower())
        candidates = VisualMediaService._extract_keywords(substituted, max_candidates=20)

        mapped: List[str] = []
        seen: Set[str] = set()
        for word in candidates:
            replacement = _CONCEPT_WORDS.get(word, word)
            for token in replacement.split():
                if token and token not in seen:
                    seen.add(token)
                    mapped.append(token)

        if not mapped:
            return ""

        # Concrete/visual words are prioritized when trimming to max_terms,
        # so a late but concrete word (e.g. "night") isn't crowded out by
        # earlier but non-visual ones (e.g. "hours").
        boosted = [w for w in mapped if w in _CONCRETE_VISUAL_BOOST]
        rest = [w for w in mapped if w not in _CONCRETE_VISUAL_BOOST]
        ordered = boosted + rest
        return " ".join(ordered[:max_terms])

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

    # ---- asset selection/download -------------------------------------------

    async def _find_and_download_asset(
        self, queries: List[str], section_index: int, used_urls: Set[str]
    ) -> MediaAsset:
        provider_name = self.media_provider.name
        last_query = queries[-1] if queries else ""
        last_error = "No search queries available"

        for query in queries:
            last_query = query
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

            candidate = next(
                (
                    c
                    for c in candidates
                    if c.source_url not in used_urls and c.download_url not in used_urls
                ),
                None,
            )
            if candidate is None:
                last_error = "No suitable (non-duplicate) media asset found"
                continue

            output_path = os.path.join(
                self.output_dir, self._build_filename(section_index, candidate)
            )
            try:
                await self.media_provider.download(candidate, output_path)
            except MediaProviderError as e:
                last_error = f"Media download failed: {e}"
                continue
            except Exception as e:
                last_error = f"Unexpected download error: {e}"
                continue

            return MediaAsset(
                provider=provider_name,
                asset_type=candidate.asset_type,
                local_file_path=output_path,
                source_url=candidate.source_url,
                attribution=candidate.attribution,
                search_query=query,
                section_index=section_index,
                duration_seconds=candidate.duration_seconds,
                width=candidate.width,
                height=candidate.height,
                success=True,
            )

        return MediaAsset(
            provider=provider_name,
            search_query=last_query,
            section_index=section_index,
            success=False,
            error=last_error,
        )

    @staticmethod
    def _build_filename(section_index: int, candidate: MediaCandidate) -> str:
        default_ext = "mp4" if candidate.asset_type == "video" else "jpg"
        url_ext = os.path.splitext(urlparse(candidate.download_url).path)[1].lstrip(".")
        ext = url_ext if url_ext and len(url_ext) <= 4 else default_ext
        return f"section-{section_index + 1:02d}-{uuid.uuid4().hex[:8]}.{ext}"
