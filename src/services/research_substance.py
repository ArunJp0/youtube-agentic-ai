# Research substance/depth validation: distinguishes genuinely substantive
# research context from duplicated headlines, near-duplicate syndicated
# snippets, and title-only RSS descriptions - the exact real-world gap a
# controlled validation exposed (several outlets syndicating the identical
# headline with no distinct article body, each snippet effectively just
# repeating its own title). Fully deterministic and cheap - no LLM call.
#
# Complements src.services.research_relevance: topical relevance and
# informational depth are separate concerns. A result can be perfectly
# on-topic and still carry zero information beyond its own headline.
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, List

# Near-duplicate CONTENT collapse threshold - mirrors ScriptAgent's own
# SECTION_SIMILARITY_THRESHOLD (0.82) for the same reason: genuinely
# distinct content scores far lower, while syndicated copies of the same
# story score very high.
CONTENT_SIMILARITY_THRESHOLD = 0.82
MIN_LENGTH_FOR_FUZZY_CONTENT_MATCH = 20

# A snippet within this similarity of its own title carries no real
# information beyond the headline - "title-only" (see is_title_only_snippet).
TITLE_ONLY_SIMILARITY_THRESHOLD = 0.85
MIN_LENGTH_FOR_FUZZY_TITLE_MATCH = 15


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _is_near_duplicate_text(a: str, b: str, threshold: float, min_length: int) -> bool:
    """Exact match always caught; fuzzy matching only for long-enough text
    (short strings can score a deceptively high ratio from sharing most of
    a short common prefix) - mirrors ScriptAgent._is_near_duplicate."""
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < min_length or len(b) < min_length:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def is_title_only_snippet(title: str, snippet: str) -> bool:
    """True when ``snippet`` carries no meaningful information beyond
    ``title`` itself - an empty snippet, an exact repeat, or a near-
    identical variant (e.g. the same headline with a trailing outlet name
    or minor punctuation difference, which real Google News RSS
    descriptions sometimes reduce to)."""
    norm_title = _normalize(title)
    norm_snippet = _normalize(snippet)
    if not norm_snippet:
        return True
    return _is_near_duplicate_text(
        norm_title, norm_snippet, TITLE_ONLY_SIMILARITY_THRESHOLD, MIN_LENGTH_FOR_FUZZY_TITLE_MATCH
    )


def _content_key(result: Dict[str, Any]) -> str:
    return _normalize(f"{result.get('title') or ''} {result.get('snippet') or ''}")


def _same_story(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two results are "the same story" (for grouping) if EITHER their
    titles alone are near-duplicate, or their combined title+snippet text
    is. Title-based matching is checked first and is deliberately robust
    to snippet-length differences: real syndication often has one outlet's
    snippet far richer than another's for the identical headline, and a
    length-dominated similarity ratio over the combined text alone would
    wrongly treat that as "different stories". The combined-text check
    remains as a secondary signal for near-identical bodies published
    under a differently-worded title."""
    title_a, title_b = _normalize(a.get("title") or ""), _normalize(b.get("title") or "")
    if _is_near_duplicate_text(title_a, title_b, CONTENT_SIMILARITY_THRESHOLD, MIN_LENGTH_FOR_FUZZY_TITLE_MATCH):
        return True
    return _is_near_duplicate_text(
        _content_key(a), _content_key(b), CONTENT_SIMILARITY_THRESHOLD, MIN_LENGTH_FOR_FUZZY_CONTENT_MATCH
    )


def group_near_duplicate_content(results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Greedy near-duplicate grouping over each result's own title/content -
    syndicated re-postings of the same story collapse into one group
    regardless of differing URLs/outlets/publish times/snippet depth. A
    new result joins the first existing group any of whose current members
    it matches (not just that group's original member), so a short chain
    of gradually-varying near-duplicates still collapses correctly."""
    groups: List[List[Dict[str, Any]]] = []
    for r in results:
        matched = next((g for g in groups if any(_same_story(r, member) for member in g)), None)
        if matched is not None:
            matched.append(r)
        else:
            groups.append([r])
    return groups


def _most_informative(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A group's best representative: the one with the longest snippet
    (the most real content to offer); ties keep first-seen order."""
    return max(group, key=lambda r: len(r.get("snippet") or ""))


def dedupe_by_content(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse near-duplicate/syndicated results down to one
    representative (the most informative) per distinct story, so the same
    headline repeated across outlets never inflates the final source list,
    LLM context, or key-point extraction input. Preserves every kept
    result's original fields (URL, source name, published_at) unchanged -
    only whole-result inclusion is decided here."""
    return [_most_informative(group) for group in group_near_duplicate_content(results)]


@dataclass
class SubstanceAssessment:
    """Deterministic assessment of how much genuinely distinct, usable
    information a batch of search results actually carries - as opposed to
    their raw aggregate word count, which duplicated/title-only snippets
    can inflate without adding any real information."""

    distinct_word_count: int
    distinct_source_count: int
    title_only_count: int
    duplicate_count: int
    sufficient: bool


def assess_substance(results: List[Dict[str, Any]], min_words: int) -> SubstanceAssessment:
    """Assess ``results`` for genuine informational substance.

    Two things a naive aggregate word count misses, both handled here:
    1. Multiple sources syndicating the identical headline/snippet -
       collapsed to one distinct content group; only that group's most
       informative representative's words count, once.
    2. A snippet that is effectively just its own title restated - never
       counted as substantive content, even when it is the only source
       (see ``is_title_only_snippet``).

    Never inspects age/freshness and never rewards verbosity beyond what
    is genuinely distinct - a single long, non-duplicated, non-title-only
    snippet is scored the same whether or not other thin sources exist
    alongside it.
    """
    if not results:
        return SubstanceAssessment(0, 0, 0, 0, False)

    title_only_count = sum(
        1 for r in results if is_title_only_snippet(r.get("title") or "", r.get("snippet") or "")
    )
    groups = group_near_duplicate_content(results)

    distinct_word_count = 0
    for group in groups:
        best = _most_informative(group)
        if is_title_only_snippet(best.get("title") or "", best.get("snippet") or ""):
            continue  # no information beyond the headline - contributes nothing
        distinct_word_count += len((best.get("snippet") or "").split())

    return SubstanceAssessment(
        distinct_word_count=distinct_word_count,
        distinct_source_count=len(groups),
        title_only_count=title_only_count,
        duplicate_count=len(results) - len(groups),
        sufficient=distinct_word_count >= min_words,
    )


# ---- key-point output validation ---------------------------------------

# Generic LLM-refusal/meta-response detector: a "key point" is treated as
# a non-answer only when it BOTH (a) talks ABOUT the source material/data
# itself - meta-discourse a genuine factual key point about the topic
# essentially never uses - AND (b) either states insufficiency or asks for
# more input. Requiring both signals together (rather than either alone)
# keeps this from false-flagging a genuine news fact that happens to
# mention "data" or contain the word "no". Nothing here references any
# specific topic, person, or story - it is a structural pattern, not a
# lookup table of exact phrases from any one real response.
_META_DISCOURSE_PHRASES = (
    "source material",
    "source text",
    "article text",
    "underlying text",
    "the source",
    "the data provided",
    "research data",
    "the information provided",
    "the content provided",
    "specific findings",
    "key points for you",
    "the provided titles",
    "the provided headlines",
    "the provided text",
)
_INSUFFICIENCY_WORD_RE = re.compile(
    r"\b(no|not|cannot|can't|unable|lack|lacks|lacking|insufficient|nothing)\b"
)
_REQUEST_PHRASES = (
    "if you can provide",
    "if you could provide",
    "please provide",
    "would be happy to",
    "let me know",
)


def is_meta_refusal_response(text: str) -> bool:
    """True when ``text`` reads as the LLM declining to answer (talking
    about the absence of source material/data itself) rather than stating
    an actual fact about the topic."""
    lower = (text or "").lower()
    if not any(phrase in lower for phrase in _META_DISCOURSE_PHRASES):
        return False
    if _INSUFFICIENCY_WORD_RE.search(lower):
        return True
    return any(phrase in lower for phrase in _REQUEST_PHRASES)
