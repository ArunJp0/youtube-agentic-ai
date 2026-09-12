# Deterministic sensitive-category flagging for Topic Planner candidates -
# a keyword heuristic, NOT a fact-checker or a second Compliance Agent. It
# only sets a typed flag (TopicCandidate.is_sensitive/sensitivity_reasons)
# for downstream Research/Compliance to apply stricter treatment - it never
# suppresses, rejects, rewrites, or auto-fabricates anything about a
# candidate. See STEP 9.
#
# Known MVP limitation (documented, not hidden): this is an English-
# language keyword heuristic, matching the default TOPIC_LANGUAGE="en".
# A future non-English deployment would need an equivalent keyword set for
# its own configured language - this module does not attempt automatic
# translation/multilingual detection.
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Generic category keywords - not region/outlet-specific, not tied to any
# single country's politics/geography. Each category name is itself a
# stable, typed "reason" a downstream consumer can branch on.
SENSITIVE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "active_conflict": ["war", "airstrike", "invasion", "military strike", "ceasefire", "insurgency"],
    "elections_politics": ["election", "referendum", "parliament", "impeachment", "coup"],
    "crime_allegations": ["arrested", "indicted", "charged with", "allegations of", "accused of", "lawsuit filed"],
    "medical_emergency": ["outbreak", "pandemic", "epidemic", "public health emergency"],
    "casualties": ["killed", "death toll", "casualties", "dead after", "fatalities"],
}

_WORD_BOUNDARY_CACHE: Dict[str, re.Pattern] = {}


def _pattern_for(keyword: str) -> re.Pattern:
    if keyword not in _WORD_BOUNDARY_CACHE:
        _WORD_BOUNDARY_CACHE[keyword] = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
    return _WORD_BOUNDARY_CACHE[keyword]


def detect_sensitivity(text: str) -> Tuple[bool, List[str]]:
    """Return (is_sensitive, matched_category_names) for ``text`` (typically
    a candidate's raw_title, optionally with a short snippet appended).

    Deterministic substring/word-boundary matching only - no LLM call, no
    opinion, no suppression decision. An empty/None text is never sensitive.
    """
    if not text:
        return False, []

    matched: List[str] = []
    for category, keywords in SENSITIVE_CATEGORY_KEYWORDS.items():
        if any(_pattern_for(keyword).search(text) for keyword in keywords):
            matched.append(category)

    return bool(matched), matched
