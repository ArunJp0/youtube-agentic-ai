# Live demo runner for the Visual Media Service.
#
# Produces a ScriptResult via the existing mock Research -> Script demo
# pipeline (fast, no LLM/network calls needed to build the script), then
# fetches REAL stock media for each section using the configured real media
# provider (Pexels by default), so this demo downloads actual image/video
# files. Since this demo doesn't run VoiceService, it uses the script's own
# deterministic duration estimate as the timing input for visual planning
# (the real pipeline uses VoiceResult.duration_seconds instead - see
# src/pipeline_demo.py).
from __future__ import annotations

import asyncio
import sys

from src.agents.visual_context_planner import VisualContextPlanner
from src.config.providers import ProviderConfigError, get_llm_provider, get_media_provider
from src.config.settings import Settings
from src.models.media import VisualResult
from src.script_demo import run_script_demo
from src.services.visual_media_service import VisualMediaService

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_media_demo(topic: str) -> VisualResult:
    """Build a script (mock pipeline) then fetch real, duration-aware visual assets for it.

    Args:
        topic: Research topic to build a script for and find visuals for

    Returns:
        VisualResult mapping each section to its downloaded, ordered asset(s)
    """
    script_result = await run_script_demo(topic)

    settings = Settings(media_provider="pexels")
    media_provider = get_media_provider(settings)
    # Uses whatever LLM_PROVIDER is configured (falls back to the
    # deterministic query generator automatically if it's mock, or if a
    # real call fails) - see VisualContextPlanner.plan_visuals.
    visual_planner = VisualContextPlanner(llm_provider=get_llm_provider(settings))
    visual_service = VisualMediaService(media_provider=media_provider, visual_planner=visual_planner)

    print(f"\n[3/3] Finding visuals for {len(script_result.sections)} section(s) via Pexels...")
    return await visual_service.generate_visuals(
        script_result, script_result.estimated_duration_seconds
    )


def print_visual_result(result: VisualResult) -> None:
    """Pretty print a VisualResult as a section -> slots -> asset mapping."""
    print("\n" + "=" * 60)
    print(f"VISUAL RESULT (provider: {result.provider})")
    print("=" * 60)
    planner_status = "LLM semantic plan" if result.semantic_planning_used else "deterministic fallback"
    print(f"Visual planning: {planner_status}")
    if result.semantic_planning_fallback_reason:
        print(f"   Fallback reason: {result.semantic_planning_fallback_reason}")

    total_slots = 0
    unique_ids = set()
    reused_count = 0

    for mapping in result.sections:
        print(
            f"\nSection {mapping.section_index + 1} ({mapping.section_heading}) "
            f"- {len(mapping.assets)} slot(s), {mapping.planned_duration_seconds:.1f}s planned"
        )
        if mapping.semantic_summary:
            print(f"   Meaning: {mapping.semantic_summary}")
        if mapping.avoid_concepts:
            print(f"   Avoiding: {', '.join(mapping.avoid_concepts)}")
        for slot_index, asset in enumerate(mapping.assets):
            query = mapping.search_queries[slot_index] if slot_index < len(mapping.search_queries) else ""
            total_slots += 1
            if not asset.success:
                print(f"   Slot {slot_index + 1}: FAILED ({query!r}) - {asset.error}")
                continue

            asset_id = asset.provider_asset_id or asset.source_url or asset.local_file_path
            unique_ids.add(asset_id)
            reused_tag = " [REUSED]" if asset.reused else ""
            tier_tag = f" [{asset.relevance_tier}]" if asset.relevance_tier else ""
            if asset.reused:
                reused_count += 1
            print(f"   Slot {slot_index + 1}: {query!r}{reused_tag}{tier_tag}")
            print(f"      Asset: {asset.local_file_path}")
            print(f"      Type:  {asset.asset_type}  ({asset.width}x{asset.height})")
            if asset.attribution:
                print(f"      Credit: {asset.attribution}")
            if asset.source_url:
                print(f"      Source: {asset.source_url}")

    print(f"\nTotal visual slots: {total_slots}")
    print(f"Unique assets used: {len(unique_ids)}")
    print(f"Reused (no re-download): {reused_count}")
    print(f"\nOverall success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = await run_media_demo(topic)
    except ProviderConfigError as e:
        print(f"Provider configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_visual_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
