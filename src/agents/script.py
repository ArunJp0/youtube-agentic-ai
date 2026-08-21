# Script Agent implementation
from __future__ import annotations

from typing import List

from src.llm.provider import LLMProvider
from src.models.research import ResearchResult
from src.models.script import ScriptResult, ScriptSection

DEFAULT_WORDS_PER_MINUTE = 150.0
DEFAULT_MAX_SECTIONS = 5


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
            ScriptAgentError: If research is missing/invalid, or script
                generation fails
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
            parts.append("Key points:\n" + "\n".join(f"- {p}" for p in research.key_points))
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
        for point in key_points:
            prompt = (
                "Write one short spoken YouTube narration paragraph (2-3 sentences) "
                f"expanding on this single point: '{point}'. Use natural spoken "
                "narration, not article style, and do not add any claim that isn't "
                f"supported by the research below.\n\n{context}"
            )
            narration = self._clean_line(self.llm_provider.generate_text(prompt), keep_multiline=True)
            sections.append(
                ScriptSection(
                    heading=point[:60],
                    narration=narration,
                    source_refs=source_refs,
                    estimated_duration_seconds=self._estimate_duration(narration),
                )
            )
        return sections

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

    # ---- helpers -------------------------------------------------------------

    def _estimate_duration(self, text: str) -> float:
        """Estimate narration duration from word count at a fixed speaking rate."""
        word_count = len(text.split())
        if not word_count:
            return 0.0
        return round((word_count / self.words_per_minute) * 60.0, 1)

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
