# Standalone live demo for the Thumbnail Agent.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope.
#
# Reuses the most recently produced final video (preferring the BGM-mixed
# MP4, then captioned, then plain) and its matching .srt transcript to
# reconstruct narration context - exactly like src/bgm_demo.py and
# src/metadata_demo.py, via the same shared
# src/services/script_context_reconstruction.py module (no duplicated
# logic). Also reuses the most recently generated Metadata JSON artifact
# (if any) for extra title/SEO context - a free, no-LLM-call enrichment.
#
# Does NOT re-run Research/Script/Voice/Visual Media/Visual QC/Video
# Assembly/Captions/BGM/Metadata - only thumbnail planning, image
# retrieval, rendering, and validation run here.
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Optional, Tuple

from src.agents.thumbnail_agent import ThumbnailAgent, ThumbnailAgentError
from src.config.providers import ProviderConfigError, get_llm_provider, get_media_provider
from src.config.settings import Settings
from src.models.thumbnail import ThumbnailResult
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR
from src.services.script_context_reconstruction import (
    build_context_from_srt,
    build_topic_only_context,
    find_matching_srt,
    original_base_name,
)
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR

DEFAULT_TOPIC = "Why do humans dream?"
DEFAULT_METADATA_OUTPUT_DIR = os.path.join("output", "metadata")

_VIDEO_SUFFIXES_BY_PREFERENCE = ("-captioned-bgm.mp4", "-captioned.mp4", ".mp4")


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _find_latest_final_video(directory: str) -> Optional[str]:
    """Prefer the final BGM-mixed MP4; fall back to the captioned MP4, then
    any plain assembled MP4."""
    if not os.path.isdir(directory):
        return None
    for suffix in _VIDEO_SUFFIXES_BY_PREFERENCE:
        if suffix == ".mp4":
            candidates = [
                path
                for path in glob.glob(os.path.join(directory, "*.mp4"))
                if not path.endswith("-captioned.mp4") and not path.endswith("-captioned-bgm.mp4")
            ]
        else:
            candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def _find_latest_metadata_json(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = glob.glob(os.path.join(directory, "*.json"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _load_metadata_context(metadata_json_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Returns (title, seo_summary) from an existing metadata JSON artifact,
    or (None, None) if unavailable/unreadable - a free enrichment, never a
    hard requirement."""
    if not metadata_json_path:
        return None, None
    try:
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("title"), data.get("seo_summary")
    except (OSError, json.JSONDecodeError):
        return None, None


async def run_thumbnail_demo(topic: str, video_path: Optional[str] = None) -> ThumbnailResult:
    """Generate a thumbnail for the most recently generated final video (or
    an explicit ``video_path``), reconstructing narration context from that
    video's own existing .srt transcript when available, and enriching the
    prompt with the most recently generated Metadata JSON's title/SEO
    summary when available - never by re-running Research/Script/Voice/
    Visual Media/Visual QC/Video Assembly/Captions/BGM/Metadata.

    Args:
        topic: Overall video topic (used for planning and, when no
            matching .srt exists, as the only context available)
        video_path: Final video to base the thumbnail's slug/context on;
            auto-discovered under output/video/ if omitted

    Returns:
        ThumbnailResult describing the outcome

    Raises:
        ThumbnailAgentError: If no source video is available (configuration-
            level, not a processing failure)
    """
    print("[1/3] Locating the latest final video and metadata context...")
    video_path = video_path or _find_latest_final_video(DEFAULT_VIDEO_OUTPUT_DIR)
    if not video_path:
        raise ThumbnailAgentError(
            f"No existing assembled/captioned/mixed MP4 found under {DEFAULT_VIDEO_OUTPUT_DIR}/. "
            "Run the pipeline demo (python -m src.pipeline_demo) at least once first, "
            "or pass an explicit video path."
        )
    print(f"      Video: {video_path}")

    metadata_json_path = _find_latest_metadata_json(DEFAULT_METADATA_OUTPUT_DIR)
    metadata_title, seo_summary = _load_metadata_context(metadata_json_path)
    if metadata_json_path:
        print(f"      Reusing existing metadata context from: {metadata_json_path}")
    else:
        print("      No existing metadata JSON found - continuing without that extra context")

    print("\n[2/3] Reconstructing narration context...")
    srt_path = find_matching_srt(video_path, DEFAULT_SUBTITLE_OUTPUT_DIR)
    if srt_path:
        print(f"      Reusing existing narration context from: {srt_path}")
        script_result = build_context_from_srt(topic, srt_path, video_path)
    else:
        print("      No matching .srt transcript found - using topic-only context (Research/Script are NOT re-run)")
        script_result = build_topic_only_context(topic)

    print("\n[3/3] Planning, selecting a source image, rendering, and validating...")
    settings = Settings()
    llm_provider = get_llm_provider(settings)
    media_provider = get_media_provider(settings)
    agent = ThumbnailAgent(media_provider=media_provider, llm_provider=llm_provider)
    return await agent.generate_thumbnail(
        topic,
        script_result,
        metadata_title=metadata_title,
        seo_summary=seo_summary,
        video_slug=original_base_name(video_path),
    )


def print_thumbnail_result(result: ThumbnailResult) -> None:
    """Pretty print a ThumbnailResult."""
    print("\n" + "=" * 60)
    print("THUMBNAIL RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")

    if result.plan:
        plan = result.plan
        print(f"\nThumbnail plan (semantic planning used: {plan.used_semantic_planning}):")
        print(f"   Hook text:      {plan.hook_text}")
        print(f"   Visual concept: {plan.visual_concept}")
        print(f"   Search query:   {plan.search_query}")
        print(f"   Mood:           {plan.mood}")
        print(f"   Subject:        {plan.subject}")
        print(f"   Composition:    {plan.composition} (text position: {plan.text_position})")
        print(f"   Avoid concepts: {plan.avoid_concepts}")
        if plan.fallback_reason:
            print(f"   Fallback reason (deterministic fallback used): {plan.fallback_reason}")

    if result.selected_asset:
        asset = result.selected_asset
        print("\nSelected source image:")
        print(f"   Provider: {asset.provider} (id: {asset.provider_asset_id})")
        print(f"   Source:   {asset.source_url}")
        print(f"   Attribution: {asset.attribution}")
        print(f"   Dimensions:  {asset.width}x{asset.height}")

    if not result.success:
        print(f"\nError: {result.error}")
        return

    print(f"\nLLM provider: {result.llm_provider} | model: {result.llm_model} | fallback used: {result.used_fallback_model}")
    print(f"Final dimensions: {result.width}x{result.height}")
    print(f"Thumbnail path: {result.output_path}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = await run_thumbnail_demo(topic)
    except (ProviderConfigError, ThumbnailAgentError) as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_thumbnail_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
