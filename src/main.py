# Demo/test runner for Research workflow
from __future__ import annotations

import asyncio
import sys
from typing import Optional

from src.tools.search_provider import MockSearchProvider, get_mock_search_provider
from src.llm.mock import MockLLMProvider
from src.workflows.research_graph import run_research_workflow
from src.models.research import ResearchResult


async def run_research_demo(topic: str) -> ResearchResult:
    """Run the research workflow with mock providers.

    Args:
        topic: Research topic to investigate

    Returns:
        ResearchResult with structured findings
    """
    search_provider = get_mock_search_provider()
    llm_provider = MockLLMProvider()

    print(f"🔍 Researching: {topic}")
    print("=" * 60)

    result = await run_research_workflow(topic, search_provider, llm_provider)
    return result


def print_research_result(result: ResearchResult) -> None:
    """Pretty print a ResearchResult."""
    print(f"\n📋 TOPIC: {result.topic}")
    print("-" * 60)

    print(f"\n📝 SUMMARY:")
    print(f"   {result.summary}")

    print(f"\n🔑 KEY POINTS ({len(result.key_points)}):")
    for i, point in enumerate(result.key_points, 1):
        print(f"   {i}. {point}")

    print(f"\n✅ FACTS ({len(result.facts)}):")
    for i, fact in enumerate(result.facts, 1):
        conf = int(fact.confidence * 100)
        source = f" (Source: {fact.source})" if fact.source else ""
        print(f"   {i}. [{conf}%] {fact.claim}{source}")

    print(f"\n📚 SOURCES ({len(result.sources)}):")
    for i, source in enumerate(result.sources, 1):
        print(f"   {i}. {source}")

    if result.research_notes:
        print(f"\n📓 RESEARCH NOTES:")
        print(f"   {result.research_notes}")

    print("\n" + "=" * 60)
    print("✅ Research complete!")


async def main() -> None:
    """Main entry point for demo."""
    # Allow topic to be passed as command-line argument
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "Why do humans dream?"

    try:
        result = await run_research_demo(topic)
        print_research_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())