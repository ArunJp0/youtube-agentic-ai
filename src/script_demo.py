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
    # One genuinely distinct canned narration per known MockLLMProvider
    # key point (see src/llm/mock.py's "key point" response), keyed by the
    # exact quoted point text ScriptAgent embeds as "Point to expand on:
    # '<point>'." A single shared response here would make every section
    # near-identical boilerplate and get collapsed by ScriptAgent's
    # duplicate detection - see docs/DECISIONS.md.
    "'Dreams occur mainly during REM sleep cycles'": (
        "Most of our vivid dreaming happens during REM sleep, a stage that repeats "
        "every ninety minutes or so through the night, when brain activity ramps up "
        "to levels similar to being awake."
    ),
    "'Brain consolidates memories and processes emotions while dreaming'": (
        "While we dream, the brain is busy sorting through the day's experiences, "
        "strengthening important memories and working through emotional moments so "
        "we wake up feeling a little more settled."
    ),
    "'Most adults spend about 2 hours per night dreaming'": (
        "On average, adults spend around two hours every night dreaming, spread "
        "across several REM cycles - dreaming is a much bigger part of sleep than "
        "most people realize."
    ),
    "'Prefrontal cortex suppression creates dream illogic'": (
        "The part of the brain responsible for logic and self-control quiets down "
        "during dreams, which is exactly why dream scenarios can feel so strange, "
        "illogical, or even impossible."
    ),
    "'Dreams may serve evolutionary functions like threat simulation'": (
        "One leading theory suggests dreaming evolved as a kind of safe rehearsal "
        "space, letting our ancestors practice reacting to threats without any "
        "real risk."
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
