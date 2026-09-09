# Finding Localizer: maps a Compliance Agent's semantic findings to the
# specific ScriptSection(s) they concern, so remediation can target a
# correction rather than blindly regenerating the whole script.
#
# Deterministic only - no LLM call here. Primary signal is the
# SemanticReviewFinding.related_section_heading the reviewer itself already
# supplies (it has full section content when writing the finding, so it is
# the most reliable source); a conservative keyword-overlap fallback covers
# older/malformed responses. A finding with no confident match is reported
# as not actionable - never guessed, never defaulted to "the whole script".
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from src.models.compliance import SemanticReviewFinding
from src.models.remediation import LocalizedFinding
from src.models.research import ResearchResult
from src.models.script import ScriptResult

_WORD_RE = re.compile(r"[a-z]{4,}")

# Generic, non-topic-specific stopwords that would otherwise inflate
# overlap scores without indicating any real subject-matter match.
_STOPWORDS = frozenset(
    {
        "this", "that", "with", "from", "have", "claim", "claims", "script",
        "content", "video", "about", "which", "there", "their", "would",
        "could", "should", "does", "doesn", "into", "than", "then", "also",
        "such", "some", "when", "what", "these", "those", "being", "been",
    }
)

# Minimum overlapping significant words required before a fallback match is
# trusted - one shared word is too weak to act on; two is a real signal.
MIN_OVERLAP_WORDS = 2


def _significant_words(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _localize_one(finding: SemanticReviewFinding, script: ScriptResult) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    if not script.sections:
        return None, None, None

    if finding.related_section_heading:
        target = finding.related_section_heading.strip().lower()
        for index, section in enumerate(script.sections):
            if section.heading.strip().lower() == target:
                return index, section.heading, "llm"

    finding_words = _significant_words(finding.description)
    if len(finding_words) < MIN_OVERLAP_WORDS:
        return None, None, None

    matches = []
    for index, section in enumerate(script.sections):
        section_words = _significant_words(f"{section.heading} {section.narration}")
        overlap = finding_words & section_words
        if len(overlap) >= MIN_OVERLAP_WORDS:
            matches.append(index)

    if len(matches) != 1:
        # Zero confident matches, or an ambiguous tie across sections -
        # never guess which one was actually meant.
        return None, None, None

    index = matches[0]
    return index, script.sections[index].heading, "fallback_match"


def localize_findings(findings: List[SemanticReviewFinding], script: Optional[ScriptResult]) -> List[LocalizedFinding]:
    """Map each finding to a ScriptSection, or mark it non-actionable.

    Never raises, never guesses: a finding with no confident section match
    (LLM-supplied reference or a conservative keyword-overlap fallback)
    comes back with ``actionable=False`` and no section assigned.
    """
    if script is None:
        return [
            LocalizedFinding(
                finding_category=f.category, finding_description=f.description, actionable=False
            )
            for f in findings
        ]

    localized: List[LocalizedFinding] = []
    for finding in findings:
        section_index, section_heading, source = _localize_one(finding, script)
        localized.append(
            LocalizedFinding(
                finding_category=finding.category,
                finding_description=finding.description,
                section_index=section_index,
                section_heading=section_heading,
                localization_source=source,
                actionable=section_index is not None,
            )
        )
    return localized


def has_grounding_evidence(research: Optional[ResearchResult], finding_description: str) -> bool:
    """Whether the existing ResearchResult already contains evidence
    plausibly relevant to correcting ``finding_description`` - a
    conservative keyword-overlap heuristic (same primitive as
    localization), used only to decide whether a bounded research refresh
    is needed before a targeted script correction.
    """
    if research is None:
        return False

    finding_words = _significant_words(finding_description)
    if len(finding_words) < MIN_OVERLAP_WORDS:
        return False

    research_text = " ".join(
        [research.summary or ""] + list(research.key_points) + [fact.claim for fact in research.facts]
    )
    research_words = _significant_words(research_text)
    return len(finding_words & research_words) >= MIN_OVERLAP_WORDS
