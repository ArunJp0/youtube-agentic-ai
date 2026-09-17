# Research relevance/classification: validates that search results (from
# ANY SearchProvider - current-news, Wikipedia, or any future provider) are
# genuinely useful for the selected topic before they are admitted into
# ResearchResult/ScriptAgent context, and classifies accepted results as:
#
#   DIRECT:      substantively about the selected topic itself.
#   SUPPORTING:  not directly about the topic, but has a clear, defensible
#                explanatory relationship to it - a historical comparison,
#                an analogous event, background/context, an illustrative
#                example - the kind of material a narrator would legitimately
#                use as a brief comparison or backdrop, never as the main
#                subject. May concern a different named event/entity/place
#                and still be genuinely supporting.
#   UNRELATED:   no defensible explanatory relationship to the topic at all.
#
# Two-stage, cheapest-first policy:
#   1. Deterministic keyword-overlap filter (free, instant) rejects results
#      that share NO significant term with the topic at all - catches
#      clearly unrelated stories (e.g. an unrelated entertainment/product
#      item that happened to appear in the same search results) without
#      ever needing an LLM call. This is a coarse pre-filter only - it never
#      itself decides DIRECT vs SUPPORTING, which requires genuine semantic
#      judgment.
#   2. For the remaining, merely-plausible survivors, at most ONE batched
#      LLM classification call judges all of them together (never one call
#      per source) - word overlap alone can't reliably tell "genuinely on
#      topic", "useful supporting context", or "coincidentally shares a
#      word", so this catches what the deterministic pass can't. An
#      unrelated source is never reclassified as supporting just because
#      the LLM can invent a loose connection - the prompt explicitly
#      requires a defensible, explainable relationship.
#
# Purely topical - never inspects age/freshness (see
# docs/DECISIONS.md: "Current-news freshness is a window, not a same-day-
# only requirement") and never rewards length (a verbose irrelevant result
# is still rejected; a short genuinely-relevant one is still kept).
#
# Generic across every topic/story - nothing here is specific to any one
# video's subject matter, person, or outlet.
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.llm.provider import LLMProvider
from src.services.llm_json import JsonExtractionError, extract_json_object

# Small, generic English stopword list for topical-overlap scoring only -
# deliberately separate from query_generation.py's own stopword list, which
# is tuned for VISUAL searchability (it drops meaningful topical words like
# "research"/"study"/"evidence" that matter here but carry no visual
# meaning there). Duplicating a handful of short, closed-class words across
# two purpose-specific lists is simpler and safer than sharing one list
# whose meaning would have to serve two different jobs.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "that",
    "this", "these", "those", "it", "its", "as", "by", "from", "we", "you",
    "your", "our", "have", "has", "had", "do", "does", "did", "not", "so",
    "can", "could", "will", "would", "about", "into", "if", "than", "then",
    "when", "while", "which", "who", "whom", "what", "how", "why", "because",
    "also", "just", "more", "most", "much", "many", "over", "around",
    "every", "each", "other", "another", "such", "there", "here", "their",
    "they", "them", "he", "she", "his", "her", "new", "says", "says",
}

_WORD_RE = re.compile(r"[a-z0-9]+")

# Query simplification: never invents wording, purely mechanical.
_CLAUSE_SPLIT_RE = re.compile(r"\s*[,;:]\s*|\s+-\s+|\s+—\s+")
MIN_CLAUSE_WORDS = 3
MAX_SIMPLIFIED_WORDS = 8

# Conservative morphological normalization for token-overlap comparison
# only (see _normalize_term) - common English endings that end in "s" but
# are NOT a simple plural, so they must never be stripped. Deliberately
# does NOT include "-is": names/demonyms ending in "i" (e.g. "Houthi",
# "Israeli", "Iraqi", "Somali", "Pakistani") pluralize to "-is" and are
# common in exactly the current-news domain this exists for - protecting
# a handful of Latin-derived "-is" singular nouns (crisis, analysis) from
# being over-stripped is not worth silently reintroducing the real
# singular/plural mismatch this normalization was built to fix. An
# over-stripped word (e.g. "crisis" -> "crisi") is harmless here - it is
# never displayed, only compared for overlap - unless it coincidentally
# collides with another real word's stripped form, which is rare.
_NON_PLURAL_S_ENDINGS = ("ss", "us", "os")
_MIN_LENGTH_FOR_PLURAL_STRIP = 4


def _normalize_term(word: str) -> str:
    """Strip a trailing possessive/simple plural "s" (e.g. "Houthis" <->
    "Houthi") so a trivial singular/plural mismatch never causes a false
    zero-overlap rejection in the deterministic relevance pass.

    Deliberately NOT a stemmer - a single, small, generic English
    morphology rule with explicit exceptions for common non-plural
    "-us"/"-ss"/"-os" endings (e.g. "focus", "class", "chaos") to avoid
    creating unrelated matches (see _NON_PLURAL_S_ENDINGS for why "-is" is
    deliberately NOT in this exception list). Never references any
    specific topic/person/vocabulary; ambiguous cases stay the semantic
    batch check's job, not this deterministic pass's.
    """
    if word.endswith("'s"):
        return word[:-2]
    if (
        word.endswith("s")
        and len(word) > _MIN_LENGTH_FOR_PLURAL_STRIP
        and not word.endswith(_NON_PLURAL_S_ENDINGS)
    ):
        return word[:-1]
    return word


def significant_terms(text: str) -> Set[str]:
    """Lowercased, stopword-filtered, length>=3, morphologically-
    normalized word tokens - the unit both the deterministic filter and
    the query simplifier reason about."""
    return {
        _normalize_term(w)
        for w in _WORD_RE.findall((text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _bare_word(token: str) -> str:
    """Strip surrounding punctuation and a trailing possessive from one
    whitespace-split token, for stopword/capitalization testing only - the
    ORIGINAL token (not this bare form) is what ends up in a simplified
    query, so casing/punctuation a real search engine cares about survives."""
    bare = token.strip(string.punctuation)
    if bare.endswith("'s"):
        bare = bare[:-2]
    return bare


def _is_significant_token(token: str) -> bool:
    bare = _bare_word(token)
    return len(bare) >= 3 and bare.lower() not in _STOPWORDS


def _is_capitalized_token(token: str) -> bool:
    """Cheap, generic proxy for a proper noun/named entity - never a
    lookup of specific words. A capitalized stopword (e.g. a sentence-
    initial "The") is not treated as a distinguishing term."""
    bare = _bare_word(token)
    return bool(bare) and bare[0].isupper() and bare.lower() not in _STOPWORDS


def simplify_query(topic: str) -> str:
    """Generic, topic-agnostic simplification for a bounded retry search.

    Two strategies, in order:
      1. Prefer the topic's own first clause (split on common clause
         separators) when it is substantial enough on its own.
      2. Otherwise, keep the topic's own most DISTINGUISHING words rather
         than blindly truncating to the first N words. A positional
         first-N-words cut can preserve an entirely generic opening clause
         while silently dropping the actual subject - e.g. "Here's what
         one team of scientists thinks happened in [the real subject]"
         truncated to 8 words keeps none of "[the real subject]" at all.
         Capitalized tokens (a cheap, generic proxy for named
         entities/proper nouns - never a lookup of specific words) are
         kept first; other significant (non-stopword) terms fill any
         remaining budget; generic filler is dropped last. Selected words
         are always re-emitted in their ORIGINAL sentence order, never
         scrambled by importance.

    Purely mechanical - never hardcodes or invents any wording, so it
    works identically for any topic string.
    """
    topic = (topic or "").strip()
    if not topic:
        return topic

    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(topic) if c.strip()]
    if clauses and len(clauses[0].split()) >= MIN_CLAUSE_WORDS and clauses[0] != topic:
        return clauses[0]

    tokens = topic.split()
    if len(tokens) <= MAX_SIMPLIFIED_WORDS:
        return topic

    def _rank(indexed) -> int:
        _, tok = indexed
        if _is_capitalized_token(tok):
            return 0
        if _is_significant_token(tok):
            return 1
        return 2

    indexed_tokens = list(enumerate(tokens))
    # sorted() is stable - within the same rank tier, original relative
    # order is preserved, so ties never depend on anything but position.
    ranked = sorted(indexed_tokens, key=_rank)
    keep_indices = {i for i, _ in ranked[:MAX_SIMPLIFIED_WORDS]}
    selected = [tok for i, tok in indexed_tokens if i in keep_indices]

    if len(selected) >= MIN_CLAUSE_WORDS:
        return " ".join(selected)
    return " ".join(tokens[:MAX_SIMPLIFIED_WORDS])


@dataclass
class RelevanceFilterResult:
    """Outcome of one ``ResearchRelevanceFilter.filter()`` call.

    ``kept`` is always exactly ``direct + supporting`` (in that order) -
    preserved as a convenience for callers that only need "was this
    accepted at all", while ``direct``/``supporting`` let a caller enforce
    the "supporting material alone can never satisfy sufficiency" rule.
    """

    kept: List[Dict[str, Any]] = field(default_factory=list)
    direct: List[Dict[str, Any]] = field(default_factory=list)
    supporting: List[Dict[str, Any]] = field(default_factory=list)
    rejected_count: int = 0
    semantic_review_performed: bool = False


class ResearchRelevanceFilter:
    """Filters raw search results down to the ones genuinely useful for the
    selected topic, classifying each survivor as DIRECT or SUPPORTING,
    before they are ever admitted into ResearchResult/ScriptAgent context.

    Provider-agnostic: used identically for every SearchProvider's results
    (current-news, Wikipedia, any future provider) by ResearchAgent - never
    special-cased per provider.

    Never mutates a surviving result dict - source URL, source name,
    publication timestamp, and every other field stay exactly as the
    provider returned them (only whole-result inclusion/classification is
    decided here), so provenance is always preserved for kept sources.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    def filter(self, topic: str, results: List[Dict[str, Any]]) -> RelevanceFilterResult:
        if not results:
            return RelevanceFilterResult()

        topic_terms = significant_terms(topic)
        survivors = [r for r in results if self._deterministically_plausible(topic_terms, r)]

        if not survivors or self.llm_provider is None:
            # No semantic judgment is possible - every deterministic
            # survivor is conservatively treated as DIRECT (the original,
            # backward-compatible behavior before DIRECT/SUPPORTING
            # existed): we have no basis to demote it to merely
            # "supporting", and never inventing a supporting
            # classification without genuine semantic review.
            return RelevanceFilterResult(
                kept=list(survivors),
                direct=list(survivors),
                supporting=[],
                rejected_count=len(results) - len(survivors),
                semantic_review_performed=False,
            )

        try:
            classifications = self._semantic_classification_batch(topic, survivors)
        except Exception:
            # LLM/parsing failure on the semantic pass never means "treat
            # everything as irrelevant" - the deterministic pass already
            # established a real topical signal for these survivors, so it
            # alone decides (as DIRECT, for the same reason as above) when
            # the semantic layer is unavailable.
            return RelevanceFilterResult(
                kept=list(survivors),
                direct=list(survivors),
                supporting=[],
                rejected_count=len(results) - len(survivors),
                semantic_review_performed=False,
            )

        direct = [r for i, r in enumerate(survivors) if classifications.get(i) == "direct"]
        supporting = [r for i, r in enumerate(survivors) if classifications.get(i) == "supporting"]
        kept = direct + supporting
        return RelevanceFilterResult(
            kept=kept,
            direct=direct,
            supporting=supporting,
            rejected_count=len(results) - len(kept),
            semantic_review_performed=True,
        )

    # ---- deterministic pass ---------------------------------------------

    @staticmethod
    def _deterministically_plausible(topic_terms: Set[str], result: Dict[str, Any]) -> bool:
        """Reject only when there is NO significant term shared with the
        topic at all - a conservative, cheap first pass that catches
        clearly unrelated results (see module docstring) without ever
        risking a false rejection of genuinely relevant-but-differently-
        worded content (that nuance is left to the semantic pass below,
        which is also where DIRECT vs SUPPORTING is actually decided)."""
        if not topic_terms:
            return True
        candidate_text = f"{result.get('title') or ''} {result.get('snippet') or ''}"
        return bool(significant_terms(candidate_text) & topic_terms)

    # ---- semantic (batch LLM) pass ---------------------------------------

    def _semantic_classification_batch(self, topic: str, candidates: List[Dict[str, Any]]) -> Dict[int, str]:
        prompt = self._build_prompt(topic, candidates)
        raw_response = self.llm_provider.generate_text(prompt)
        return self._parse_classification_response(raw_response, len(candidates))

    @staticmethod
    def _build_prompt(topic: str, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for i, c in enumerate(candidates):
            title = (c.get("title") or "").strip()
            snippet = (c.get("snippet") or "").strip()
            lines.append(f'{i}. Title: "{title}"\n   Snippet: "{snippet}"')
        candidates_block = "\n".join(lines)
        return (
            "You are classifying search results for their usefulness researching this specific video "
            f"topic: \"{topic}\"\n\n"
            "For EACH numbered source below, classify it as exactly one of:\n"
            '- "direct": substantively about the topic itself - it directly reports on it or provides '
            "real information about it.\n"
            '- "supporting": NOT directly about the topic, but has a clear, DEFENSIBLE explanatory '
            "relationship to it - e.g. a historical comparison, an analogous event, relevant "
            "background/context, or an example that genuinely helps explain, compare, or illustrate the "
            "topic. It may concern a completely different named event, place, or person and still "
            "legitimately be supporting - do not require the same names/entities as the topic.\n"
            '- "unrelated": no defensible explanatory relationship to the topic at all.\n\n'
            "Be conservative with \"supporting\": only use it when you could clearly explain IN ONE "
            "SENTENCE why this source helps explain, compare to, or contextualize the topic. Never "
            "invent a loose or tenuous connection just to avoid \"unrelated\" - an unrelated story (e.g. "
            "celebrity news, an unrelated product story, sports, unrelated politics/entertainment) must "
            "be marked unrelated even if it is well-written or detailed. A genuinely relevant older "
            "background/context article should be marked direct or supporting regardless of how long "
            "ago it was published - age is never the deciding factor.\n\n"
            f"Sources:\n{candidates_block}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with exactly this "
            "shape:\n"
            "{\n"
            '  "classifications": [\n'
            '    {"index": 0, "classification": "direct", "reason": "one short phrase"}\n'
            "  ]\n"
            "}\n"
            "Include exactly one entry per numbered source above."
        )

    @staticmethod
    def _parse_classification_response(raw_text: str, count: int) -> Dict[int, str]:
        payload = extract_json_object(raw_text)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise JsonExtractionError("Classification response was not a JSON object")

        judged: Dict[int, str] = {}
        for item in data.get("classifications") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            classification = str(item.get("classification") or "").strip().lower()
            if classification in ("direct", "supporting", "unrelated"):
                judged[idx] = classification

        # A source missing from a partial/malformed response defaults to
        # "direct", not dropped - it already passed the deterministic
        # pass, and one missing judgment should never silently drop a real
        # candidate (mirrors TopicRankingPlanner's own per-item fallback).
        # Defaulting to "direct" (the more useful tier) rather than
        # "supporting" avoids silently weakening sufficiency just because
        # one entry in the response was malformed.
        return {i: judged.get(i, "direct") for i in range(count)}


def dedupe_by_url(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop later entries that repeat an earlier one's URL - used when
    merging an initial search with its simplified-query retry, so the
    same story never counts twice."""
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        url = r.get("url")
        if url:
            if url in seen:
                continue
            seen.add(url)
        deduped.append(r)
    return deduped
