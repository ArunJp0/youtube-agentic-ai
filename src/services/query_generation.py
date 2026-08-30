# Deterministic (non-LLM) stock-search query generation.
#
# This is the single source of truth for turning a ScriptSection's own
# heading/narration into concrete, visually-searchable query text: concept
# mapping, keyword extraction/ranking, and multi-variant expansion. It is
# used two ways:
#   1. Directly, as VisualMediaService's fallback when no VisualContextPlanner
#      is configured (or one fails) - see build_deterministic_visual_plan.
#   2. Indirectly, inside VisualContextPlanner's own fallback path, so a real
#      LLM outage degrades to the exact same deterministic behavior rather
#      than a second, divergent implementation.
#
# Fully generic across topics - nothing here is specific to any one video's
# subject matter (see the module docstring notes on individual maps below).
from __future__ import annotations

import re
from typing import List, Set

from src.models.script import ScriptResult, ScriptSection
from src.models.visual_plan import SectionVisualPlan, VisualPlan

MAX_QUERY_TERMS = 4
BROAD_QUERY_TERMS = 2
VARIANT_QUERY_TERMS = 2
DEFAULT_QUERY_VARIANTS = 5
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


def _apply_concept_phrases(lowered_text: str) -> str:
    for phrase, replacement in _CONCEPT_PHRASES:
        if phrase in lowered_text:
            lowered_text = lowered_text.replace(phrase, replacement)
    return lowered_text


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


def ranked_keywords(text: str, max_candidates: int) -> List[str]:
    """Concept-map, extract, and rank keywords from ``text``.

    Shared by both the single broader-query builder and the multi-variant
    builder, so they always agree on what's concrete/visual for the same
    input text.
    """
    substituted = _apply_concept_phrases(text.lower())
    candidates = _extract_keywords(substituted, max_candidates=max_candidates)

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

    # Concrete/visual words are prioritized, so a late but concrete word
    # (e.g. "night") isn't crowded out by an earlier but non-visual one
    # (e.g. "hours").
    boosted = [w for w in mapped if w in _CONCRETE_VISUAL_BOOST]
    rest = [w for w in mapped if w not in _CONCRETE_VISUAL_BOOST]
    return boosted + rest


def build_concept_query(text: str, max_terms: int) -> str:
    """Translate ``text`` into a single short, concrete, visually-searchable query."""
    ranked = ranked_keywords(text, max_candidates=20)
    return " ".join(ranked[:max_terms]) if ranked else ""


def build_query_variants(section: ScriptSection, max_variants: int) -> List[str]:
    """Generate up to ``max_variants`` distinct concept-query variants for a section.

    Deterministic (no LLM call): the section's heading+narration are
    concept-mapped and ranked the same way as the single-query builder, then
    chunked into small groups of distinct concrete keywords - so different
    visual slots within the same section search for different (but still
    relevant) facets of its content instead of all repeating the exact same
    query. Fully generic: works from whatever concrete/visual vocabulary the
    section's own text (after concept mapping) contains, for any topic.

    Args:
        section: The ScriptSection to derive query variants for
        max_variants: Maximum number of variants to generate

    Returns:
        Ordered list of distinct query strings, length <= max_variants (may
        be shorter if the section doesn't have enough distinct concrete
        keywords)
    """
    if max_variants <= 0:
        return []

    combined = f"{section.heading} {section.narration}"
    ranked = ranked_keywords(combined, max_candidates=40)

    variants: List[str] = []
    for start in range(0, len(ranked), VARIANT_QUERY_TERMS):
        if len(variants) >= max_variants:
            break
        chunk = ranked[start : start + VARIANT_QUERY_TERMS]
        if chunk:
            variants.append(" ".join(chunk))
    return variants


def ordered_unique(queries: List[str]) -> List[str]:
    """Dedupe a list of query strings, preserving order and skipping falsy values."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for query in queries:
        if query and query not in seen:
            seen.add(query)
            ordered.append(query)
    return ordered


def build_deterministic_visual_plan(script: ScriptResult) -> VisualPlan:
    """Build a VisualPlan using only deterministic keyword/concept-mapping
    logic - no LLM call.

    Used directly when VisualMediaService has no VisualContextPlanner
    configured, and internally by VisualContextPlanner itself when a real
    semantic-planning call fails, so both cases degrade to the exact same
    well-tested behavior rather than two divergent fallbacks. The resulting
    plan has no semantic understanding of context (its ``avoid_concepts``
    are always empty - it cannot resolve lexical ambiguity), which is
    precisely the limitation VisualContextPlanner exists to improve on.
    """
    topic_query = build_concept_query(script.topic, MAX_QUERY_TERMS) if script.topic else ""

    sections: List[SectionVisualPlan] = []
    for index, section in enumerate(script.sections):
        variants = build_query_variants(section, DEFAULT_QUERY_VARIANTS)
        broader_query = build_concept_query(section.heading, BROAD_QUERY_TERMS)
        neutral_queries = ordered_unique([broader_query, topic_query, LAST_RESORT_QUERY])

        if not variants:
            single = build_concept_query(f"{section.heading} {section.narration}", MAX_QUERY_TERMS)
            variants = [single] if single else []

        sections.append(
            SectionVisualPlan(
                section_index=index,
                semantic_summary="",
                visual_intents=list(variants),
                search_queries=variants or neutral_queries[:1],
                avoid_concepts=[],
                neutral_fallback_queries=neutral_queries,
            )
        )

    return VisualPlan(topic=script.topic, sections=sections, used_semantic_planning=False)
