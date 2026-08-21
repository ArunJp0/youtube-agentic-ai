# Research Agent implementation
from __future__ import annotations

import asyncio
from typing import List, Optional

from pydantic import HttpUrl

from src.models.research import ResearchResult, ResearchFact
from src.tools.search_provider import SearchProvider
from src.llm.provider import LLMProvider


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
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        llm_provider: LLMProvider,
        max_sources: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the Research Agent.

        Args:
            search_provider: Implementation of SearchProvider for retrieving sources
            llm_provider: Implementation of LLMProvider for synthesis
            max_sources: Maximum number of sources to retrieve
            timeout_seconds: Timeout for search operations
        """
        self.search_provider = search_provider
        self.llm_provider = llm_provider
        self.max_sources = max_sources
        self.timeout_seconds = timeout_seconds

    async def research(self, topic: str) -> ResearchResult:
        """Execute full research workflow for a topic.

        Args:
            topic: Research topic/query

        Returns:
            Structured ResearchResult with findings

        Raises:
            ResearchAgentError: If research fails
        """
        if not topic or not topic.strip():
            raise ResearchAgentError("Topic cannot be empty")

        # Step 1: Search for sources
        try:
            raw_results = await asyncio.wait_for(
                self.search_provider.search(topic, self.max_sources),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise ResearchAgentError(f"Search timed out after {self.timeout_seconds}s")
        except Exception as e:
            raise ResearchAgentError(f"Search failed: {e}")

        if not raw_results:
            # Return empty but valid result
            return ResearchResult(
                topic=topic,
                summary="No sources found for this topic.",
                key_points=[],
                facts=[],
                sources=[],
                research_notes="Search returned no results.",
            )

        # Step 2: Extract sources and snippets
        sources: List[HttpUrl] = []
        snippets: List[str] = []

        for result in raw_results:
            if "url" in result:
                try:
                    sources.append(HttpUrl(result["url"]))
                except Exception:
                    pass  # Skip invalid URLs
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
        )

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
        """Parse bullet points from LLM response."""
        points = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove common bullet prefixes
            for prefix in ("•", "-", "*", "–", "—"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            # Remove numbering like "1."
            if line and line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
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