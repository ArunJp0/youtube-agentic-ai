# Research Agent implementation
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import HttpUrl

from src.models.research import ResearchResult, ResearchFact
from src.services.research_relevance import ResearchRelevanceFilter, dedupe_by_url, simplify_query
from src.services.research_substance import assess_substance, dedupe_by_content, is_meta_refusal_response
from src.tools.search_provider import SearchProvider
from src.llm.provider import LLMProvider

# Deterministic minimum-substance gate, applied UNIFORMLY to every provider
# (current-news and Wikipedia/evergreen alike, primary attempt or fallback -
# no provider is exempt; a provider returning results successfully is never
# itself treated as research success). Measured as DISTINCT DIRECT-only
# informational word count (see src.services.research_substance), not a raw
# aggregate - duplicated/syndicated snippets, title-only descriptions, and
# SUPPORTING-only content never satisfy this threshold on their own (see
# research()'s explicit-failure branch).
DEFAULT_MIN_CONTEXT_WORDS = 40

# Shared instruction fragment for every synthesis prompt (summary/key
# points/facts) - reused rather than duplicated, so "how supporting
# material must be weighted" is defined in exactly one place. Only takes
# effect when _build_labeled_context actually included a SUPPORTING
# section; harmless (a no-op instruction) when it didn't.
_SUPPORTING_MATERIAL_GUIDANCE = (
    " If a SUPPORTING background/context section is present below, use it only briefly - for a single "
    "comparison, example, or piece of context - never let it dominate, replace, or become a separate "
    "point on its own; the topic itself must remain the clear main subject throughout."
)


def _has_sufficient_substance(results: List[Dict[str, Any]], min_words: int) -> bool:
    """Deterministic, provider-agnostic substance check, delegating to
    ``assess_substance`` (distinct-content-aware, not a naive aggregate
    word count). Never inspects age or topical meaning - purely a
    length/duplication gate, so it can never become a topic-specific
    heuristic."""
    return assess_substance(results, min_words).sufficient


class ResearchAgentError(Exception):
    """Custom exception for Research Agent errors."""
    pass


@dataclass
class _ProviderSearchOutcome:
    """One provider's fully-validated results (after the research quality
    contract - relevance/classification, near-duplicate collapse, and any
    bounded retry): never raw, never provider-specific. ``direct`` and
    ``supporting`` mirror ``RelevanceFilterResult`` - see its docstring for
    the DIRECT/SUPPORTING distinction."""

    raw_count: int = 0
    direct: List[Dict[str, Any]] = field(default_factory=list)
    supporting: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> List[Dict[str, Any]]:
        return self.direct + self.supporting


class ResearchAgent:
    """Research Agent that performs topic research using search and LLM providers.

    This agent:
    1. Accepts a research topic
    2. Queries search provider(s) for candidate sources
    3. Validates EVERY candidate against one provider-agnostic research
       quality contract before it may influence synthesis (see
       ``_search_and_validate``/``_classify_and_dedupe``) - topical
       relevance/DIRECT-vs-SUPPORTING classification, near-duplicate
       collapse, and substance/depth, applied identically to every
       ``SearchProvider`` (current-news, Wikipedia, any future provider,
       primary attempt or fallback). A provider returning results
       successfully is never itself treated as research success.
    4. Uses an LLM to organize/summarize only the ACCEPTED set
    5. Returns structured ResearchResult, or raises ``ResearchAgentError``
       explicitly when no provider/retry combination produced sufficient
       DIRECT research - never a wrong-topic or unusably-thin result

    Source selection is topic-CHARACTERISTIC-aware, not topic-text-aware:
    ``research(topic, topic_source=...)``'s optional ``topic_source`` hint
    (e.g. ``"current_news"``, mirroring ``TopicCandidate.source``/
    ``TopicSelectionResult.selected_topic_source`` - the Topic Planner's own
    real classification of where the topic came from) decides which
    provider(s) to try, never a keyword/content heuristic over the topic
    string itself. Wikipedia (the default ``search_provider``) remains the
    sole source for the default/evergreen path (topic_source is anything
    other than "current_news", or no ``current_news_search_provider`` is
    configured) - it now goes through the exact same quality contract as
    current-news, rather than being exempt from it.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        llm_provider: LLMProvider,
        max_sources: int = 5,
        timeout_seconds: float = 30.0,
        current_news_search_provider: Optional[SearchProvider] = None,
        min_context_words: int = DEFAULT_MIN_CONTEXT_WORDS,
        relevance_filter: Optional[ResearchRelevanceFilter] = None,
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
            min_context_words: Deterministic minimum DISTINCT-DIRECT-content
                word count required before proceeding to LLM synthesis (see
                ``_has_sufficient_substance``) - applied to every provider's
                DIRECT-classified results, current-news and Wikipedia/
                evergreen alike. SUPPORTING content can never satisfy this
                threshold on its own.
            relevance_filter: Classifies every provider's search results as
                DIRECT/SUPPORTING/UNRELATED before they may enter context
                (see src.services.research_relevance) - applied identically
                to EVERY SearchProvider's results, current-news and
                Wikipedia alike; no provider is exempt. Defaults to a
                filter built from the same ``llm_provider`` above (no
                second LLM dependency to wire through callers).
        """
        self.search_provider = search_provider
        self.llm_provider = llm_provider
        self.max_sources = max_sources
        self.timeout_seconds = timeout_seconds
        self.current_news_search_provider = current_news_search_provider
        self.min_context_words = min_context_words
        self.relevance_filter = relevance_filter or ResearchRelevanceFilter(llm_provider)

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
            Structured ResearchResult with findings - guaranteed to contain
            only content that passed the research quality contract
            (DIRECT/SUPPORTING classification, near-duplicate collapse,
            substance/depth) for whichever provider ultimately supplied it.

        Raises:
            ResearchAgentError: If every provider (and bounded retry) tried
                failed to produce sufficient DIRECT research - explicit,
                deterministic failure rather than silently proceeding on
                thin, unrelated, or tangential content. This now applies
                uniformly, including the single-provider (evergreen)
                path - a provider returning results successfully is never
                itself treated as research success.
        """
        if not topic or not topic.strip():
            raise ResearchAgentError("Topic cannot be empty")

        providers = self._select_search_providers(topic_source)

        # Step 1: Search for sources - try each candidate provider in
        # order, stopping at the first whose DIRECT content alone is
        # sufficient. A provider failing outright (timeout/exception) is
        # isolated and skipped, never crashing the whole call, mirroring
        # CompositeTopicSourceProvider's own per-source failure isolation.
        # Every provider's results - current-news, Wikipedia, any future
        # provider, primary attempt or fallback - go through the identical
        # quality contract in _search_and_validate; none is exempt.
        accepted_direct: List[Dict[str, Any]] = []
        accepted_supporting: List[Dict[str, Any]] = []
        used_provider_label: Optional[str] = None
        providers_tried: List[str] = []
        last_error: Optional[str] = None

        for provider in providers:
            label = getattr(provider, "name", type(provider).__name__)
            providers_tried.append(label)
            try:
                outcome = await self._search_and_validate(provider, topic)
            except asyncio.TimeoutError:
                last_error = f"{label} search timed out after {self.timeout_seconds}s"
                continue
            except Exception as e:
                last_error = f"{label} search failed: {e}"
                continue

            if not accepted_direct and not accepted_supporting and outcome.accepted:
                # Keep the first non-empty accepted set as a best-effort
                # candidate even if it turns out insufficient - never
                # discard real accepted content just because a later
                # provider might do better.
                accepted_direct, accepted_supporting, used_provider_label = (
                    outcome.direct,
                    outcome.supporting,
                    label,
                )
            if _has_sufficient_substance(outcome.direct, self.min_context_words):
                accepted_direct, accepted_supporting, used_provider_label = (
                    outcome.direct,
                    outcome.supporting,
                    label,
                )
                break

        if not _has_sufficient_substance(accepted_direct, self.min_context_words):
            # Covers BOTH a genuine zero-results case AND a "results
            # existed but nothing was DIRECT/substantive enough" case
            # uniformly - assess_substance([], ...) is deterministically
            # insufficient, so no separate "no results" branch is needed.
            # SUPPORTING material is reported for diagnostics but can
            # never satisfy this on its own (the explicit invariant this
            # milestone enforces) - a real research failure now always
            # raises rather than ever silently reaching ScriptAgent with
            # an unusable/wrong-topic result.
            assessment = assess_substance(accepted_direct, self.min_context_words)
            tried = ", ".join(providers_tried) or "none"
            error_detail = f" Last error: {last_error}." if last_error else ""
            raise ResearchAgentError(
                f"Insufficient DIRECT research material for '{topic}' after trying {tried} "
                f"({assessment.distinct_word_count} distinct DIRECT context words across "
                f"{assessment.distinct_source_count} distinct DIRECT source(s) - "
                f"{assessment.duplicate_count} duplicate/syndicated and {assessment.title_only_count} "
                f"title-only result(s) excluded; {len(accepted_supporting)} supporting source(s) "
                f"available but supporting material alone can never satisfy sufficiency - below the "
                f"{self.min_context_words}-word minimum).{error_detail}"
            )

        # Step 2: Extract sources/snippets from the FINAL accepted set
        # ONLY (DIRECT + SUPPORTING) - never from rejected/raw/retry
        # content that didn't survive validation. A result missing a
        # usable URL is skipped entirely (never lets unattributable
        # content influence context while staying untracked in sources -
        # a provenance-integrity gap the previous implementation had).
        sources: List[HttpUrl] = []
        source_published_at: List[Optional[str]] = []
        direct_snippets: List[str] = []
        supporting_snippets: List[str] = []

        for tier_results, tier_snippets in (
            (accepted_direct, direct_snippets),
            (accepted_supporting, supporting_snippets),
        ):
            for result in tier_results:
                snippet = result.get("snippet")
                url = result.get("url")
                if not url or not snippet:
                    continue
                try:
                    parsed_url = HttpUrl(url)
                except Exception:
                    continue
                sources.append(parsed_url)
                source_published_at.append(result.get("published_at"))
                tier_snippets.append(snippet)

        combined_context = self._build_labeled_context(direct_snippets, supporting_snippets)

        # Step 3: Use LLM to generate structured output - only ever from
        # the validated, labeled context above; the LLM never sees
        # rejected/unvalidated material.
        try:
            summary = await self._generate_summary(topic, combined_context)
            key_points = await self._extract_key_points(topic, combined_context)
            facts = await self._extract_facts(topic, combined_context)
            research_notes = await self._generate_notes(topic, combined_context)
        except Exception as e:
            raise ResearchAgentError(f"LLM processing failed: {e}")

        # A "key point" that is actually the LLM declining to answer (e.g.
        # "the source material only contains a headline...") is never a
        # genuine finding - filtered out unconditionally (never allowed to
        # reach ScriptAgent), and an explicit failure if NOTHING genuine
        # survives (as opposed to silently returning zero key points).
        valid_key_points = [p for p in key_points if not is_meta_refusal_response(p)]
        if key_points and not valid_key_points:
            raise ResearchAgentError(
                f"Key-point extraction for '{topic}' produced only refusal/meta-commentary "
                "responses (no genuine informational content) - the source material was "
                "likely too thin for the LLM to work with despite passing the substance gate."
            )
        key_points = valid_key_points

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

    async def _search_and_validate(self, provider: SearchProvider, topic: str) -> _ProviderSearchOutcome:
        """Execute one provider's search and run its results through the
        SAME research quality contract regardless of which provider it is
        (current-news, Wikipedia, or any future SearchProvider; a primary
        attempt or a fallback) - no provider-specific special-casing:

          1. classify + validate (relevance/DIRECT-SUPPORTING/near-
             duplicate collapse - see _classify_and_dedupe),
          2. if the DIRECT content alone still isn't substantive enough,
             one bounded retry against a generically simplified query is
             attempted, its results are classified/validated too, and
             merged in (deduplicated by URL AND by content),
          3. still-insufficient DIRECT substance is returned as-is (the
             caller falls through to the next provider, or ultimately
             raises ResearchAgentError - it is never treated as success).
        """
        results = await asyncio.wait_for(
            provider.search(topic, self.max_sources), timeout=self.timeout_seconds
        )
        outcome = self._classify_and_dedupe(topic, results)
        if _has_sufficient_substance(outcome.direct, self.min_context_words):
            return outcome

        simplified_query = simplify_query(topic)
        if not simplified_query or simplified_query.strip().lower() == topic.strip().lower():
            return outcome

        try:
            retry_results = await asyncio.wait_for(
                provider.search(simplified_query, self.max_sources), timeout=self.timeout_seconds
            )
        except Exception:
            # The bounded simplified-query retry itself failed (timeout/
            # provider error) - keep whatever the first attempt already
            # produced rather than discarding it.
            return outcome

        seen_urls = {r.get("url") for r in results if r.get("url")}
        new_candidates = [r for r in retry_results if r.get("url") not in seen_urls]
        retry_outcome = self._classify_and_dedupe(topic, new_candidates)

        return _ProviderSearchOutcome(
            raw_count=outcome.raw_count + retry_outcome.raw_count,
            direct=dedupe_by_content(dedupe_by_url(outcome.direct + retry_outcome.direct)),
            supporting=dedupe_by_content(dedupe_by_url(outcome.supporting + retry_outcome.supporting)),
        )

    def _classify_and_dedupe(self, topic: str, results: List[Dict[str, Any]]) -> _ProviderSearchOutcome:
        """The single, reusable acceptance pipeline every provider's raw
        results flow through: normalize (drop malformed/empty results) ->
        relevance + DIRECT/SUPPORTING/UNRELATED classification -> near-
        duplicate content collapse. Nothing downstream of this method ever
        sees a result that didn't pass through it."""
        normalized = [r for r in results if (r.get("title") or "").strip() or (r.get("snippet") or "").strip()]
        classified = self.relevance_filter.filter(topic, normalized)
        return _ProviderSearchOutcome(
            raw_count=len(results),
            direct=dedupe_by_content(classified.direct),
            supporting=dedupe_by_content(classified.supporting),
        )

    @staticmethod
    def _build_labeled_context(direct_snippets: List[str], supporting_snippets: List[str]) -> str:
        """Combine accepted snippets into one prompt-ready context string,
        clearly labeling supporting material as secondary so downstream
        synthesis (_generate_summary/_extract_key_points/_extract_facts/
        _generate_notes) can weight it correctly - it may inform a brief
        comparison/example, but must never dominate or replace the direct
        research (see class docstring)."""
        context = "\n\n".join(direct_snippets)
        if supporting_snippets:
            context += (
                "\n\n--- SUPPORTING background/context (use only briefly - for comparison, "
                "illustration, or context - never as a main point on its own, and never let it "
                "dominate or replace the direct research above) ---\n\n"
                + "\n\n".join(supporting_snippets)
            )
        return context

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
            f"Focus on the most important findings about the topic itself.{_SUPPORTING_MATERIAL_GUIDANCE}\n\n"
            f"Source material:\n{context}"
        )
        return self.llm_provider.generate_text(prompt)

    async def _extract_key_points(self, topic: str, context: str) -> List[str]:
        """Extract key bullet points using LLM."""
        prompt = (
            f"Extract 5-7 key points as a bullet list from this research on '{topic}'. "
            f"Each point should be one clear sentence, primarily about the topic itself."
            f"{_SUPPORTING_MATERIAL_GUIDANCE}\n\n"
            f"Source material:\n{context}"
        )
        response = self.llm_provider.generate_text(prompt)
        # Parse bullet points from response
        return self._parse_bullet_points(response)

    async def _extract_facts(self, topic: str, context: str) -> List[ResearchFact]:
        """Extract structured facts with confidence using LLM."""
        prompt = (
            f"Extract 8-10 verifiable facts from this research on '{topic}'. "
            f"Format each as: 'Fact: [claim] | Source: [source name] | Confidence: [0.0-1.0]'."
            f"{_SUPPORTING_MATERIAL_GUIDANCE}\n\n"
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