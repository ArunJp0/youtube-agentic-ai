# Standalone live demo for the Metadata Agent.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope.
#
# Reuses the most recently produced final video (preferring the BGM-mixed
# MP4, then the captioned MP4, then the plain assembled MP4) and its
# matching .srt transcript rather than regenerating anything. No
# ScriptResult is persisted to disk anywhere in this project, so - exactly
# like src/bgm_demo.py - this reconstructs one from the existing .srt
# transcript via the shared src.services.script_context_reconstruction
# module, instead of duplicating that logic or re-running Research/Script.
from __future__ import annotations

import glob
import os
import sys
from typing import Optional

from src.agents.metadata_agent import DEFAULT_METADATA_OUTPUT_DIR, MetadataAgent, MetadataAgentError
from src.config.providers import ProviderConfigError, get_llm_provider
from src.config.settings import Settings
from src.models.metadata import MetadataResult
from src.services.script_context_reconstruction import (
    build_context_from_srt,
    build_topic_only_context,
    find_matching_srt,
    original_base_name,
)
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError

DEFAULT_TOPIC = "Why do humans dream?"

# Preference order: the final user-facing output first, falling back
# toward earlier pipeline stages' outputs if later ones don't exist yet.
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


def run_metadata_demo(topic: str, video_path: Optional[str] = None) -> MetadataResult:
    """Generate YouTube metadata for the most recently generated final
    video (or an explicit ``video_path``), reconstructing narration context
    from that video's own existing .srt transcript when available - never
    by re-running Research/Script/Voice/Visual Media/Visual QC/Video
    Assembly/Captions/BGM.

    Synchronous end to end: MetadataAgent only calls the synchronous
    LLMProvider.generate_text plus local file I/O, and duration probing is
    a plain FFmpeg subprocess call - there is no async work in this demo.

    Args:
        topic: Overall video topic (used for metadata generation and, when
            no matching .srt exists, as the only context available)
        video_path: Final video to generate metadata for; auto-discovered
            under output/video/ if omitted

    Returns:
        MetadataResult describing the outcome

    Raises:
        MetadataAgentError: If no source video is available (configuration-
            level, not a processing failure)
        VideoAssemblerError: If FFmpeg is unavailable/misconfigured
    """
    print("[1/3] Checking FFmpeg availability and locating the latest final video...")
    assembler = FFmpegVideoAssembler()

    video_path = video_path or _find_latest_final_video(DEFAULT_VIDEO_OUTPUT_DIR)
    if not video_path:
        raise MetadataAgentError(
            f"No existing assembled/captioned/mixed MP4 found under {DEFAULT_VIDEO_OUTPUT_DIR}/. "
            "Run the pipeline demo (python -m src.pipeline_demo) at least once first, "
            "or pass an explicit video path."
        )
    print(f"      Video: {video_path}")
    video_duration = assembler.probe_duration_seconds(video_path)
    print(f"      Duration: {video_duration:.1f}s")

    print("\n[2/3] Reconstructing narration context...")
    srt_path = find_matching_srt(video_path, DEFAULT_SUBTITLE_OUTPUT_DIR)
    if srt_path:
        print(f"      Reusing existing narration context from: {srt_path}")
        script_result = build_context_from_srt(topic, srt_path, video_path)
    else:
        print("      No matching .srt transcript found - using topic-only context (Research/Script are NOT re-run)")
        script_result = build_topic_only_context(topic)

    print("\n[3/3] Generating metadata (one semantic LLM call)...")
    settings = Settings()
    llm_provider = get_llm_provider(settings)
    agent = MetadataAgent(llm_provider=llm_provider)
    return agent.generate_metadata(
        topic, script_result, duration_seconds=video_duration, video_slug=original_base_name(video_path)
    )


def print_metadata_result(result: MetadataResult) -> None:
    """Pretty print a MetadataResult."""
    print("\n" + "=" * 60)
    print("METADATA RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")

    if not result.success:
        print(f"Error: {result.error}")
        return

    print(f"\nTitle ({len(result.title)} chars):\n   {result.title}")
    print(f"\nSEO summary:\n   {result.seo_summary}")
    print(f"\nDescription ({len(result.description)} chars):\n{result.description}")
    print(f"\nTags ({len(result.tags)}): {result.tags}")
    print(f"\nHashtags ({len(result.hashtags)}): {result.hashtags}")

    print(f"\nChapters available: {result.chapters_available}")
    if result.chapters_available:
        for chapter in result.chapters:
            print(f"   {chapter.timestamp_text:>8s}  {chapter.title}")
    elif result.chapters_omitted_reason:
        print(f"   Chapters omitted: {result.chapters_omitted_reason}")

    print(f"\nLLM provider: {result.llm_provider} | model: {result.llm_model} | fallback used: {result.used_fallback_model}")
    print(f"Metadata JSON: {result.output_path}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = run_metadata_demo(topic)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except (ProviderConfigError, MetadataAgentError) as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_metadata_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
