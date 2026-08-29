# Live demo runner for the Visual Media Service.
#
# Produces a ScriptResult via the existing mock Research -> Script demo
# pipeline (fast, no LLM/network calls needed to build the script), then
# fetches REAL stock media for each section using the configured real media
# provider (Pexels by default), so this demo downloads actual image/video
# files.
from __future__ import annotations

import asyncio
import sys

from src.config.providers import ProviderConfigError, get_media_provider
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
    """Build a script (mock pipeline) then fetch real visual assets for it.

    Args:
        topic: Research topic to build a script for and find visuals for

    Returns:
        VisualResult mapping each section to its downloaded asset(s)
    """
    script_result = await run_script_demo(topic)

    settings = Settings(media_provider="pexels")
    media_provider = get_media_provider(settings)
    visual_service = VisualMediaService(media_provider=media_provider)

    print(f"\n[3/3] Finding visuals for {len(script_result.sections)} section(s) via Pexels...")
    return await visual_service.generate_visuals(script_result)


def print_visual_result(result: VisualResult) -> None:
    """Pretty print a VisualResult as a section -> query -> asset mapping."""
    print("\n" + "=" * 60)
    print(f"VISUAL RESULT (provider: {result.provider})")
    print("=" * 60)

    for mapping in result.sections:
        asset = mapping.assets[0] if mapping.assets else None
        print(f"\nSection {mapping.section_index + 1} ({mapping.section_heading})")
        print(f"   Query: {mapping.search_query}")
        if asset and asset.success:
            print(f"   Asset: {asset.local_file_path}")
            print(f"   Type:  {asset.asset_type}  ({asset.width}x{asset.height})")
            if asset.attribution:
                print(f"   Credit: {asset.attribution}")
            if asset.source_url:
                print(f"   Source: {asset.source_url}")
        else:
            error = asset.error if asset else "no asset produced"
            print(f"   FAILED: {error}")

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
