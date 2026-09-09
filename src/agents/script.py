# Script Agent implementation
from __future__ import annotations

import difflib
from typing import List, Optional

from src.llm.provider import LLMProvider
from src.models.research import ResearchFact, ResearchResult
from src.models.script import ScriptResult, ScriptSection

DEFAULT_WORDS_PER_MINUTE = 150.0
DEFAULT_MAX_SECTIONS = 5

# Similarity ratio (difflib.SequenceMatcher, on normalized text) at/above
# which two sections are treated as near-duplicates rather than distinct
# content. Calibrated against real examples: genuinely distinct sections
# score ~0.02-0.4, while sections sharing the same boilerplate body with only
# a different lead-in clause (the observed bug) score ~0.93. 0.82 leaves a
# wide margin on both sides.
SECTION_SIMILARITY_THRESHOLD = 0.82

# Minimum normalized length (characters) before the similarity ratio is
# trusted for fuzzy matching. Short strings that differ by only one word
# (e.g. "Point A" vs "Point B") can score a deceptively high ratio purely
# because most of the string is shared - fuzzy matching is only meaningful
# for paragraph-length narration. Exact duplicates are always caught
# regardless of length.
MIN_LENGTH_FOR_FUZZY_MATCH = 60

# If a section's narration comes back as a duplicate of an already-kept
# section, retry with a strengthened prompt this many times (in addition to
# the first attempt) before giving up and dropping the section. Actively
# trying to regenerate distinct content is the primary defense; dropping is
# the last resort.
MAX_SECTION_GENERATION_ATTEMPTS = 3

# When at least this many research key points are available, the resulting
# script must end up with at least this many distinct sections. Below this
# many available points, there isn't "enough research" for the floor to be
# meaningful (see generate_script's Raises docstring).
MIN_DISTINCT_SECTIONS = 3


class ScriptAgentError(Exception):
    """Custom exception for Script Agent errors."""
    pass


class ScriptAgent:
    """Script Agent that converts a ResearchResult into a structured YouTube script.

    This agent:
    1. Accepts a completed ResearchResult (produced by the Research Agent)
    2. Uses an LLM to write natural, spoken-style narration grounded strictly
       in the supplied research (title, hook, intro, one section per key
       point, conclusion, call to action)
    3. Returns a structured ScriptResult ready for a future Voice/Visual Agent

    Every prompt is built from the same research context and explicitly
    instructed not to introduce claims beyond what was researched, and every
    section carries the research's source URLs for traceability.

    Each section's prompt is told which other points are covered elsewhere
    (so it doesn't repeat them) and, if the LLM still returns a duplicate,
    is retried with a strengthened prompt before the section is dropped -
    dropping/deduplicating is a last-resort safety net, not the primary way
    duplicate content is avoided.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_sections: int = DEFAULT_MAX_SECTIONS,
        words_per_minute: float = DEFAULT_WORDS_PER_MINUTE,
    ) -> None:
        """Initialize the Script Agent.

        Args:
            llm_provider: Implementation of LLMProvider for narration writing
            max_sections: Maximum number of body sections to generate
            words_per_minute: Speaking rate used to estimate narration duration
        """
        self.llm_provider = llm_provider
        self.max_sections = max_sections
        self.words_per_minute = words_per_minute

    async def generate_script(self, research: ResearchResult) -> ScriptResult:
        """Convert a ResearchResult into a structured ScriptResult.

        Args:
            research: Structured research produced by the Research Agent

        Returns:
            Structured ScriptResult with narration ready for narration/voice

        Raises:
            ScriptAgentError: If research is missing/invalid, script
                generation fails, or (when at least MIN_DISTINCT_SECTIONS
                research points are available) fewer than
                MIN_DISTINCT_SECTIONS genuinely distinct sections could be
                produced after deduplication/retries.
        """
        if research is None:
            raise ScriptAgentError("ResearchResult is required")
        if not research.summary or not research.summary.strip():
            raise ScriptAgentError("ResearchResult must have a non-empty summary")

        context = self._build_context(research)

        try:
            video_title = self._clean_line(self._generate_title(context))
            hook = self._clean_line(self._generate_hook(context), keep_multiline=True)
            introduction = self._clean_line(self._generate_introduction(context), keep_multiline=True)
            sections = self._generate_sections(research, context)
            conclusion = self._clean_line(self._generate_conclusion(context), keep_multiline=True)
            call_to_action = self._clean_line(
                self._generate_call_to_action(research.topic), keep_multiline=True
            )
        except ScriptAgentError:
            raise
        except Exception as e:
            raise ScriptAgentError(f"LLM processing failed: {e}")

        total_duration = (
            self._estimate_duration(hook)
            + self._estimate_duration(introduction)
            + sum(section.estimated_duration_seconds for section in sections)
            + self._estimate_duration(conclusion)
            + self._estimate_duration(call_to_action)
        )

        return ScriptResult(
            topic=research.topic,
            video_title=video_title or research.topic,
            hook=hook,
            introduction=introduction,
            sections=sections,
            conclusion=conclusion,
            call_to_action=call_to_action,
            estimated_duration_seconds=round(total_duration, 1),
            sources=research.sources,
            script_notes=research.research_notes,
        )

    # ---- context building --------------------------------------------------

    @staticmethod
    def _build_context(research: ResearchResult) -> str:
        """Build the compact, grounding context shared by every prompt.

        Using the same context for every generation step (rather than
        re-deriving it) keeps every piece of narration grounded in the same
        facts, which is what keeps the script from inventing claims.
        """
        parts = [f"Topic: {research.topic}", f"Summary: {research.summary}"]
        if research.key_points:
            # Deliberately not labeled "Key points:" - that phrase collides
            # with MockLLMProvider's built-in "key point" canned response,
            # which would otherwise make every prompt (not just per-section
            # ones) get the same generic mock text whenever this context
            # block is included, masking real narration differences.
            parts.append("Notable points:\n" + "\n".join(f"- {p}" for p in research.key_points))
        if research.facts:
            fact_lines = []
            for fact in research.facts:
                source = f" (Source: {fact.source})" if fact.source else ""
                fact_lines.append(f"- {fact.claim}{source}")
            parts.append("Facts:\n" + "\n".join(fact_lines))
        return "\n\n".join(parts)

    # ---- generation steps ---------------------------------------------------

    def _generate_title(self, context: str) -> str:
        prompt = (
            "You are writing a single, catchy YouTube video title for the "
            "researched topic below. Return ONLY the title text, nothing else, "
            f"and do not state anything not supported by the research.\n\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    def _generate_hook(self, context: str) -> str:
        prompt = (
            "Write a 1-2 sentence spoken YouTube video HOOK for the topic below. "
            "It must be natural spoken narration (not written like an article), "
            "grab attention immediately, and must not state anything not "
            f"supported by the research below.\n\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    def _generate_introduction(self, context: str) -> str:
        prompt = (
            "Write a short spoken YouTube video INTRODUCTION (2-4 sentences) for "
            "the topic below. It should be natural narration a narrator would say "
            "out loud, setting up what the video will cover, grounded only in the "
            f"research provided.\n\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    def _generate_sections(self, research: ResearchResult, context: str) -> List[ScriptSection]:
        key_points = research.key_points[: self.max_sections] if research.key_points else []
        source_refs = [str(s) for s in research.sources]

        if not key_points:
            # No key points to structure sections around: fall back to a
            # single overview section so the script is never empty.
            narration = self._clean_line(
                self.llm_provider.generate_text(
                    "Write one short spoken YouTube narration paragraph covering the "
                    f"following research, grounded only in what is provided.\n\n{context}"
                ),
                keep_multiline=True,
            )
            return [
                ScriptSection(
                    heading="Overview",
                    narration=narration,
                    source_refs=source_refs,
                    estimated_duration_seconds=self._estimate_duration(narration),
                )
            ]

        sections: List[ScriptSection] = []
        kept_narrations: List[str] = []
        for point in key_points:
            other_points = [p for p in key_points if p != point]
            narration = ""

            for attempt in range(MAX_SECTION_GENERATION_ATTEMPTS):
                prompt = self._build_section_prompt(
                    point, other_points, context, is_retry=attempt > 0
                )
                candidate = self._clean_line(
                    self.llm_provider.generate_text(prompt), keep_multiline=True
                )
                narration = candidate
                if not any(self._is_near_duplicate(candidate, prior) for prior in kept_narrations):
                    break
                # Duplicate of an already-kept section: actively try again
                # with a strengthened prompt (primary defense) before
                # falling back to dropping this section (last resort).

            if any(self._is_near_duplicate(narration, prior) for prior in kept_narrations):
                # Every attempt, including retries, came back as a
                # duplicate/near-duplicate of an already-kept section (e.g.
                # a degenerate/repetitive LLM response sharing the same
                # boilerplate body with only a different lead-in) - skip it
                # rather than repeating the same content twice. Genuinely
                # distinct content, even on a related topic, stays well
                # under the similarity threshold and is kept. This should
                # be rare; MIN_DISTINCT_SECTIONS below catches it happening
                # too often.
                continue
            kept_narrations.append(narration)

            sections.append(
                ScriptSection(
                    heading=point[:60],
                    narration=narration,
                    source_refs=source_refs,
                    estimated_duration_seconds=self._estimate_duration(narration),
                )
            )

        if len(key_points) >= MIN_DISTINCT_SECTIONS and len(sections) < MIN_DISTINCT_SECTIONS:
            raise ScriptAgentError(
                f"Script generation produced only {len(sections)} distinct section(s) "
                f"from {len(key_points)} available research points (after "
                f"deduplication/retries); expected at least {MIN_DISTINCT_SECTIONS}. "
                "The LLM provider may be returning repetitive/boilerplate content."
            )

        return sections

    @staticmethod
    def _build_section_prompt(
        point: str, other_points: List[str], context: str, is_retry: bool
    ) -> str:
        """Build the prompt for one section, listing other points to avoid
        repeating and, on retry, explicitly asking for more distinct content.

        The point being expanded on is deliberately the very first thing in
        the prompt (rather than embedded mid-sentence), so each section's
        prompt is distinguishable from the very start.
        """
        other_points_note = ""
        if other_points:
            other_points_note = (
                "Other points already covered elsewhere in this video (do NOT "
                "repeat these; focus only on the point below):\n"
                + "\n".join(f"- {p}" for p in other_points)
                + "\n\n"
            )

        retry_note = ""
        if is_retry:
            retry_note = (
                "Your previous attempt was too similar to another section. Write "
                "something meaningfully different, focused specifically and only "
                "on the point below, and avoid generic filler phrasing.\n\n"
            )

        return (
            f"Point to expand on: '{point}'.\n"
            f"{retry_note}{other_points_note}"
            "Write one short spoken YouTube narration paragraph (2-3 sentences) "
            "about ONLY this specific point. Use natural spoken narration, not "
            "article style, and do not add any claim that isn't supported by "
            f"the research below.\n\n{context}"
        )

    def _generate_conclusion(self, context: str) -> str:
        prompt = (
            "Write a short spoken YouTube video CONCLUSION (2-3 sentences) that "
            "wraps up the topic below in natural narration, grounded only in the "
            f"research provided.\n\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    def _generate_call_to_action(self, topic: str) -> str:
        prompt = (
            "Write a short, natural spoken YouTube call-to-action (1-2 sentences) "
            f"for a video about '{topic}', asking viewers to like, subscribe, and "
            "comment. Do not state any research facts, only the call to action."
        )
        return self.llm_provider.generate_text(prompt)

    # ---- targeted remediation revision ---------------------------------------

    def revise_section(
        self,
        script: ScriptResult,
        research: ResearchResult,
        section_index: int,
        finding_description: str,
        refreshed_fact: Optional[ResearchFact] = None,
    ) -> ScriptResult:
        """Return a corrected ScriptResult for Compliance Remediation, with
        ONLY ``sections[section_index]``'s narration (and its own/the
        script's total ``estimated_duration_seconds``) changed - every
        other field (hook, introduction, other sections, conclusion,
        call_to_action, sources) is byte-identical to ``script``.

        Grounded strictly in ``research`` (plus ``refreshed_fact``, when a
        bounded research refresh was needed) - never a blind rewrite of the
        whole script, and never a correction unsupported by the evidence
        it's given.

        Args:
            script: The ScriptResult a Compliance REVIEW flagged
            research: The original ResearchResult this script was grounded in
            section_index: Index into ``script.sections`` to revise
            finding_description: The specific compliance finding to address
            refreshed_fact: Optional additional evidence from a bounded
                research refresh (see ResearchAgent.research_focused_claim),
                when the original research didn't already cover this claim

        Returns:
            A new ScriptResult with only the targeted section revised

        Raises:
            ScriptAgentError: If inputs are invalid, the LLM call fails, or
                the correction comes back empty - never a fabricated fix.
        """
        if script is None or research is None:
            raise ScriptAgentError("ScriptResult and ResearchResult are both required")
        if not (0 <= section_index < len(script.sections)):
            raise ScriptAgentError(f"section_index {section_index} is out of range for this script")

        context = self._build_context(research)
        section = script.sections[section_index]
        prompt = self._build_correction_prompt(section, finding_description, context, refreshed_fact)

        try:
            candidate = self._clean_line(self.llm_provider.generate_text(prompt), keep_multiline=True)
        except Exception as e:
            raise ScriptAgentError(f"Section revision LLM processing failed: {e}")

        if not candidate:
            raise ScriptAgentError("Section revision produced empty narration")

        revised_section = section.model_copy(
            update={"narration": candidate, "estimated_duration_seconds": self._estimate_duration(candidate)}
        )
        new_sections = list(script.sections)
        new_sections[section_index] = revised_section

        total_duration = (
            self._estimate_duration(script.hook)
            + self._estimate_duration(script.introduction)
            + sum(s.estimated_duration_seconds for s in new_sections)
            + self._estimate_duration(script.conclusion)
            + self._estimate_duration(script.call_to_action)
        )
        return script.model_copy(
            update={"sections": new_sections, "estimated_duration_seconds": round(total_duration, 1)}
        )

    @staticmethod
    def _build_correction_prompt(
        section: ScriptSection,
        finding_description: str,
        context: str,
        refreshed_fact: Optional[ResearchFact],
    ) -> str:
        refreshed_note = (
            f"\n\nAdditional verified evidence for this correction: {refreshed_fact.claim}"
            if refreshed_fact is not None
            else ""
        )
        return (
            "The following section of a YouTube video script has a specific factual/consistency problem "
            "that a compliance review identified and that MUST be corrected:\n\n"
            f"Section heading: '{section.heading}'\n"
            f"Current narration: {section.narration}\n\n"
            f"Problem identified by compliance review: {finding_description}"
            f"{refreshed_note}\n\n"
            "Rewrite ONLY this section's narration (2-3 sentences, natural spoken YouTube narration, not "
            "article style) so the problem above is fixed. Stay strictly grounded in the research below - "
            "do not introduce any new claim the research doesn't support, and do not change what topic "
            f"this section otherwise covers.\n\n{context}\n\n"
            "Return ONLY the corrected narration text, nothing else."
        )

    # ---- helpers -------------------------------------------------------------

    def _estimate_duration(self, text: str) -> float:
        """Estimate narration duration from word count at a fixed speaking rate."""
        word_count = len(text.split())
        if not word_count:
            return 0.0
        return round((word_count / self.words_per_minute) * 60.0, 1)

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        """Collapse whitespace and case so near-identical narration is caught."""
        return " ".join(text.split()).strip().lower()

    @staticmethod
    def _is_near_duplicate(a: str, b: str, threshold: float = SECTION_SIMILARITY_THRESHOLD) -> bool:
        """True if two narration texts are exact or near-duplicates.

        Exact duplicates are always caught, regardless of length. Fuzzy
        (strongly similar but not identical) matching only kicks in for
        paragraph-length text, since short strings can score a deceptively
        high similarity ratio from sharing most of a short common prefix.
        """
        norm_a = ScriptAgent._normalize_for_dedup(a)
        norm_b = ScriptAgent._normalize_for_dedup(b)
        if not norm_a or not norm_b:
            return False
        if norm_a == norm_b:
            return True
        if len(norm_a) < MIN_LENGTH_FOR_FUZZY_MATCH or len(norm_b) < MIN_LENGTH_FOR_FUZZY_MATCH:
            return False
        return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold

    @staticmethod
    def _clean_line(text: str, keep_multiline: bool = False) -> str:
        """Strip whitespace/quote noise from an LLM text response.

        When ``keep_multiline`` is False, only the first non-empty line is
        kept (used for single-line outputs like the video title).
        """
        cleaned = text.strip().strip('"').strip("'").strip()
        if not keep_multiline:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            cleaned = lines[0] if lines else cleaned
        return cleaned
