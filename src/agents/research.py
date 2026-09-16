# Research Agent implementation
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import HttpUrl

from src.models.research import ResearchResult, ResearchFact
from src.tools.search_provider import SearchProvider
from src.llm.provider import LLMProvider

# Deterministic minimum-substance gate for the multi-provider (topic-source-
# aware) search chain only - never applied to the default single-provider
# path, which keeps its exact prior tolerant behavior. Combined snippet
# word count below this is treated as "insufficient reliable research",
# not silently handed to the LLM to make the best of scattered/tangential
# material (see research()'s explicit-failure branch).
DEFAULT_MIN_CONTEXT_WORDS = 40


def _has_sufficient_substance(results: List[Dict[str, Any]], min_words: int) -> bool:
    """Deterministic, provider-agnostic substance check: total combined
    snippet word count across ``results`` at/above ``min_words``. Never
    inspects content/meaning - purely a length gate, so it can never
    become a topic-specific heuristic."""
    total_words = sum(len((r.get("snippet") or "").split()) for r in results)
    return total_words >= min_words


class ResearchAgentError(Exception):
    """Custom exception for Research Agent errors."""
    pass


class ResearchAgent:
    """Research Agent that performs topic research using search and LLM providers.

    This agent:
    1. Accepts a research topic
    2. Queries search provider for relevant sources
    3. Uses LLM to organize/summarize findings
    4. Returns structured ResearchResult

    Source selection is topic-CHARACTERISTIC-aware, not topic-text-aware:
    ``research(topic, topic_source=...)``'s optional ``topic_source`` hint
    (e.g. ``"current_news"``, mirroring ``TopicCandidate.source``/
    ``TopicSelectionResult.selected_topic_source`` - the Topic Planner's own
    real classification of where the topic came from) decides which
    provider(s) to try, never a keyword/content heuristic over the topic
    string itself. Wikipedia (the default ``search_provider``) remains the
    sole source for the default/evergreen path (topic_source is anything
    other than "current_news", or no ``current_news_search_provider`` is
    configured) - existing behavior for every current call site is
    byte-for-byte unchanged.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        llm_provider: LLMProvider,
        max_sources: int = 5,
        timeout_seconds: float = 30.0,
        current_news_search_provider: Optional[SearchProvider] = None,
        min_context_words: int = DEFAULT_MIN_CONTEXT_WORDS,
    ) -> None:
        """Initialize the Research Agent.

        Args:
            search_provider: Implementation of SearchProvider for retrieving
                sources - the default/evergreen path (e.g. Wikipedia)
            llm_provider: Implementation of LLMProvider for synthesis
            max_sources: Maximum number of sources to retrieve
            timeout_seconds: Timeout for search operations
            current_news_search_provider: Optional SearchProvider used only
                when ``research(topic, topic_source="current_news")`` is
                called - e.g. ``CurrentNewsSearchProvider``, reusing the
                same real Google News RSS infrastructure the Topic Planner
                already trusts. ``None`` (default) means current-news topics
                fall back to the same single default provider as before -
                never a hard requirement, always a graceful degrade.
            min_context_words: Deterministic minimum combined-snippet word
                count required before proceeding to LLM synthesis, when
                more than one provider is in play (see class docstring) -
                never applied to the default single-provider path.
        """
        self.search_provider = search_provider
        self.llm_provider = llm_provider
        self.max_sources = max_sources
        self.timeout_seconds = timeout_seconds
        self.current_news_search_provider = current_news_search_provider
        self.min_context_words = min_context_words

    async def research(self, topic: str, topic_source: Optional[str] = None) -> ResearchResult:
        """Execute full research workflow for a topic.

        Args:
            topic: Research topic/query
            topic_source: Optional classification of where this topic came
                from (mirrors ``TopicCandidate.source``, e.g.
                ``"current_news"``) - selects which search provider(s) to
                try; ``None`` (default) preserves the exact original
                single-provider behavior.

        Returns:
            Structured ResearchResult with findings

        Raises:
            ResearchAgentError: If research fails, or - only on the
                multi-provider current-news-aware path - if every provider
                tried produced insufficient reliable material (explicit,
                deterministic failure rather than silently proceeding on
                thin/tangential content).
        """
        if not topic or not topic.strip():
            raise ResearchAgentError("Topic cannot be empty")

        providers = self._select_search_providers(topic_source)

        # Step 1: Search for sources - try each candidate provider in
        # order, stopping at the first with sufficient substance. A
        # provider failing outright (timeout/exception) is isolated and
        # skipped, never crashing the whole call, mirroring
        # CompositeTopicSourceProvider's own per-source failure isolation.
        raw_results: List[Dict[str, Any]] = []
        used_provider_label: Optional[str] = None
        providers_tried: List[str] = []
        last_error: Optional[str] = None

        for provider in providers:
            label = getattr(provider, "name", type(provider).__name__)
            providers_tried.append(label)
            try:
                results = await asyncio.wait_for(
                    provider.search(topic, self.max_sources),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = f"{label} search timed out after {self.timeout_seconds}s"
                continue
            except Exception as e:
                last_error = f"{label} search failed: {e}"
                continue

            if not raw_results and results:
                # Keep the first non-empty result as a best-effort
                # candidate even if it turns out too thin - never discard
                # real results just because a later provider might do
                # better.
                raw_results, used_provider_label = results, label
            if _has_sufficient_substance(results, self.min_context_words):
                raw_results, used_provider_label = results, label
                break

        if not raw_results:
            # Existing, unchanged explicit-empty-result path - a genuine
            # zero-results case across every provider tried.
            tried = ", ".join(providers_tried) or "none"
            notes = f"Search returned no results. Providers tried: {tried}."
            if last_error:
                notes += f" Last error: {last_error}"
            return ResearchResult(
                topic=topic,
                summary="No sources found for this topic.",
                key_points=[],
                facts=[],
                sources=[],
                research_notes=notes,
                source_provider=used_provider_label,
            )

        if len(providers) > 1 and not _has_sufficient_substance(raw_results, self.min_context_words):
            # Multi-provider (topic-source-aware) chain only - explicit,
            # deterministic failure rather than silently synthesizing from
            # thin/tangential material (STEP 2's explicit requirement).
            # The default single-provider path never reaches this branch,
            # preserving its original tolerant behavior exactly.
            word_count = sum(len((r.get("snippet") or "").split()) for r in raw_results)
            raise ResearchAgentError(
                f"Insufficient research material for '{topic}' after trying {', '.join(providers_tried)} "
                f"({word_count} context words, below the {self.min_context_words}-word minimum)."
            )

        # Step 2: Extract sources and snippets
        sources: List[HttpUrl] = []
        source_published_at: List[Optional[str]] = []
        snippets: List[str] = []

        for result in raw_results:
            if "url" in result:
                try:
                    sources.append(HttpUrl(result["url"]))
                except Exception:
                    continue  # Skip invalid URLs - and their paired published_at, so lists stay aligned
                source_published_at.append(result.get("published_at"))
            if "snippet" in result:
                snippets.append(result["snippet"])

        combined_context = "\n\n".join(snippets)

        # Step 3: Use LLM to generate structured output
        try:
            summary = await self._generate_summary(topic, combined_context)
            key_points = await self._extract_key_points(topic, combined_context)
            facts = await self._extract_facts(topic, combined_context)
            research_notes = await self._generate_notes(topic, combined_context)
        except Exception as e:
            raise ResearchAgentError(f"LLM processing failed: {e}")

        return ResearchResult(
            topic=topic,
            summary=summary,
            key_points=key_points,
            facts=facts,
            sources=sources,
            research_notes=research_notes,
            source_provider=used_provider_label,
            source_published_at=source_published_at,
        )

    def _select_search_providers(self, topic_source: Optional[str]) -> List[SearchProvider]:
        """Choose which provider(s) to try, based purely on the topic's own
        SOURCE CLASSIFICATION (never its text/content) - see class
        docstring. Returns exactly one provider (the existing default)
        unless a current-news topic AND a configured news provider both
        apply, in which case current-news is tried first, Wikipedia second
        as a bounded background/fallback attempt."""
        if topic_source == "current_news" and self.current_news_search_provider is not None:
            return [self.current_news_search_provider, self.search_provider]
        return [self.search_provider]

    async def research_focused_claim(self, topic: str, claim: str) -> Optional[ResearchFact]:
        """Bounded, single-claim research refresh for Compliance Remediation:
        at most one search call and one LLM call, scoped to ``claim``
        rather than re-researching the whole topic.

        Never raises - returns ``None`` whenever the refresh can't produce
        usable, source-grounded evidence (search failure/timeout, no
        results, LLM failure, or the source material simply doesn't cover
        this claim), so a caller can treat that as "cannot safely correct"
        rather than inventing a fix.

        Args:
            topic: Overall video topic (for search/prompt context)
            claim: The specific disputed claim to verify/correct

        Returns:
            A single source-grounded ResearchFact, or None
        """
        if not topic or not claim:
            return None

        try:
            raw_results = await asyncio.wait_for(
                self.search_provider.search(f"{topic}: {claim}", self.max_sources),
                timeout=self.timeout_seconds,
            )
        except Exception:
            return None

        if not raw_results:
            return None

        snippets = [r["snippet"] for r in raw_results if "snippet" in r]
        context = "\n\n".join(snippets)
        if not context:
            return None

        prompt = (
            f"You are fact-checking ONE specific claim about '{topic}' using only the source material below.\n\n"
            f"Claim to verify/correct: {claim}\n\n"
            f"Source material:\n{context}\n\n"
            "Respond with ONE corrected, source-grounded factual statement about this specific point "
            "(a single sentence). If the source material does not address this claim at all, respond "
            "with exactly: NO_EVIDENCE"
        )
        try:
            response = self.llm_provider.generate_text(prompt)
        except Exception:
            return None

        cleaned = (response or "").strip()
        if not cleaned or cleaned.upper().startswith("NO_EVIDENCE"):
            return None

        return ResearchFact(claim=cleaned, source=None, confidence=0.6)

    async def _generate_summary(self, topic: str, context: str) -> str:
        """Generate a high-level summary using LLM."""
        prompt = (
            f"Summarize the following research on '{topic}' in 2-3 sentences. "
            f"Focus on the most important findings.\n\n"
            f"Source material:\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    async def _extract_key_points(self, topic: str, context: str) -> List[str]:
        """Extract key bullet points using LLM."""
        prompt = (
            f"Extract 5-7 key points as a bullet list from this research on '{topic}'. "
            f"Each point should be one clear sentence.\n\n"
            f"Source material:\n{context}"
        )
        response = self.llm_provider.generate_text(prompt)
        # Parse bullet points from response
        return self._parse_bullet_points(response)

    async def _extract_facts(self, topic: str, context: str) -> List[ResearchFact]:
        """Extract structured facts with confidence using LLM."""
        prompt = (
            f"Extract 8-10 verifiable facts from this research on '{topic}'. "
            f"Format each as: 'Fact: [claim] | Source: [source name] | Confidence: [0.0-1.0]'\n\n"
            f"Source material:\n{context}"
        )
        response = self.llm_provider.generate_text(prompt)
        return self._parse_facts(response)

    async def _generate_notes(self, topic: str, context: str) -> Optional[str]:
        """Generate additional research notes."""
        prompt = (
            f"Provide brief research notes on '{topic}' highlighting "
            f"gaps, controversies, or areas needing further investigation.\n\n"
            f"Source material:\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    @staticmethod
    def _parse_bullet_points(text: str) -> List[str]:
        """Parse bullet points from LLM response.

        A leading line that isn't itself a bullet/numbered item (e.g. "Here
        are 5 key points extracted from the provided text:") is a real,
        reproducible LLM habit - observed directly with real Gemini
        responses - and was previously kept as if it were a genuine point.
        Only the very FIRST line is ever eligible to be dropped this way
        (a real point appearing later that happens to end in ":" is never
        touched), and only when it has no bullet/number marker of its own -
        a narrow, targeted fix, not a rewrite of the parser.
        """
        points: List[str] = []
        seen_first_nonblank_line = False
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            is_first_nonblank_line = not seen_first_nonblank_line
            seen_first_nonblank_line = True

            has_marker = False
            # Remove common bullet prefixes
            for prefix in ("•", "-", "*", "–", "—"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    has_marker = True
                    break
            # Remove numbering like "1."
            if not has_marker and line and line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
                has_marker = True

            if is_first_nonblank_line and not has_marker and line.endswith(":"):
                # Unmarked, colon-terminated opening line - introductory
                # meta-commentary, not a real point (e.g. "Based on the
                # provided source material, here are 5 key points:").
                continue

            if line:
                points.append(line)
        return points[:7]  # Cap at 7 points

    @staticmethod
    def _parse_facts(text: str) -> List[ResearchFact]:
        """Parse structured facts from LLM response.

        Supports multiple formats:
        - "Fact: claim | Source: source | Confidence: 0.X"
        - "1. claim" (numbered list)
        - "• claim" (bullet list)
        """
        facts = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            claim = ""
            source = None
            confidence = 0.5

            # Try to parse "Fact: claim | Source: source | Confidence: 0.X"
            if "fact:" in line.lower() or "claim:" in line.lower():
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.lower().startswith("fact:"):
                        claim = part[5:].strip()
                    elif part.lower().startswith("claim:"):
                        claim = part[6:].strip()
                    elif part.lower().startswith("source:"):
                        source = part[7:].strip()
                    elif part.lower().startswith("confidence:"):
                        conf_str = part[11:].strip()
                        try:
                            confidence = float(conf_str)
                            confidence = max(0.0, min(1.0, confidence))
                        except ValueError:
                            pass

                if claim:
                    facts.append(ResearchFact(
                        claim=claim,
                        source=source,
                        confidence=confidence,
                    ))
            else:
                # Try to parse as numbered list or bullet
                for prefix in ("•", "-", "*", "–", "—"):
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break

                # Remove numbering like "1." or "1)"
                if line and line[0].isdigit():
                    # Find the separator
                    for sep in (".", ")", " "):
                        if sep in line[:3]:
                            line = line.split(sep, 1)[1].strip()
                            break

                if line:
                    facts.append(ResearchFact(
                        claim=line,
                        source=None,
                        confidence=0.8,
                    ))

        return facts[:10]  # Cap at 10 facts