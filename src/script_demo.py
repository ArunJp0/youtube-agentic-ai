# Demo/test runner for the Research Agent -> Script Agent pipeline.
#
# Uses mock providers only (no real Gemini/network calls), so it can be run
# anywhere without an API key. The response_map below customizes the mock
# LLM's canned answers for script-writing prompts specifically, purely for
# a readable demo transcript - it does not modify MockLLMProvider itself.
from __future__ import annotations

import asyncio
import sys

from src.llm.mock import MockLLMProvider
from src.models.script import ScriptResult
from src.tools.search_provider import get_mock_search_provider
from src.workflows.research_graph import run_research_workflow
from src.workflows.script_graph import run_script_workflow

DEFAULT_TOPIC = "Why do humans dream?"

_SCRIPT_DEMO_RESPONSES = {
    "video title": "Why Do We Dream? The Science Behind Your Nightly Adventures",
    "hook": "Ever woken up from a dream so vivid it felt real? Tonight we're finding out why your brain does that.",
    "introduction": (
        "In this video we're exploring the science of dreaming - what your brain "
        "is actually doing while you sleep, and why dreams might matter more than you think."
    ),
    "expanding on this single point": (
        "Researchers point to this as one of the clearer explanations for what's "
        "happening in your brain while you dream."
    ),
    "conclusion": "So next time you wake up remembering a strange dream, you'll know there's real science behind it.",
    "call-to-action": (
        "If you found this interesting, hit like, subscribe for more science explainers, "
        "and tell us your weirdest dream in the comments."
    ),
}


async def run_script_demo(topic: str) -> ScriptResult:
    """Run Research Agent -> Script Agent end to end with mock providers.

    Args:
        topic: Research topic to investigate and script

    Returns:
        ScriptResult with structured narration
    """
    search_provider = get_mock_search_provider()
    research_llm_provider = MockLLMProvider()
    script_llm_provider = MockLLMProvider(response_map=_SCRIPT_DEMO_RESPONSES)

    print(f"[1/2] Researching: {topic}")
    research_result = await run_research_workflow(topic, search_provider, research_llm_provider)

    print("[2/2] Writing script from research...")
    script_result = await run_script_workflow(research_result, script_llm_provider)

    return script_result


def print_script_result(result: ScriptResult) -> None:
    """Pretty print a ScriptResult."""
    print(f"\nVIDEO TITLE: {result.video_title}")
    print("-" * 60)

    print("\nHOOK:")
    print(f"   {result.hook}")

    print("\nINTRODUCTION:")
    print(f"   {result.introduction}")

    print(f"\nSECTIONS ({len(result.sections)}):")
    for i, section in enumerate(result.sections, 1):
        print(f"\n   {i}. {section.heading}  (~{section.estimated_duration_seconds:.1f}s)")
        print(f"      {section.narration}")
        if section.visual_notes:
            print(f"      [visual: {section.visual_notes}]")

    print("\nCONCLUSION:")
    print(f"   {result.conclusion}")

    print("\nCALL TO ACTION:")
    print(f"   {result.call_to_action}")

    minutes = result.estimated_duration_seconds / 60.0
    print(f"\nESTIMATED TOTAL DURATION: {result.estimated_duration_seconds:.1f}s (~{minutes:.1f} min)")

    print(f"\nSOURCES ({len(result.sources)}):")
    for i, source in enumerate(result.sources, 1):
        print(f"   {i}. {source}")

    if result.script_notes:
        print("\nSCRIPT NOTES (inherited from research):")
        print(f"   {result.script_notes}")

    print("\n" + "=" * 60)
    print("Script complete!")


async def main() -> None:
    """Main entry point for the demo."""
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = DEFAULT_TOPIC

    try:
        result = await run_script_demo(topic)
        print_script_result(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
