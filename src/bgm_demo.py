# Standalone live demo for the Background Music / Audio Mixing Service.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope.
#
# Reuses the EXISTING captioned MP4 most recently produced by a prior
# pipeline_demo/caption_demo run (output/video/*-captioned.mp4, falling
# back to any other assembled MP4) rather than regenerating it.
#
# No ScriptResult is persisted to disk anywhere in this project, so this
# demo deliberately does NOT re-run Research/Script/Voice/Visual Media/
# Visual QC/Video Assembly just to reconstruct one - that would spend real
# API quota (and, per real runs, be slow/unreliable under Gemini rate
# limiting) purely to re-derive text this project already generated once.
# Instead, it reconstructs a minimal ScriptResult directly from the
# existing .srt transcript (output/subtitles/<name>.srt) that CaptionService
# already produced for the located video - the real, timestamped narration
# text as actually spoken, which is a more faithful source for music-mood
# planning than a freshly re-generated script would be anyway. If no
# matching .srt exists, a topic-only context is used instead (a weaker
# mood signal, but still exactly one MusicContextPlanner LLM call - no
# Research/Script fallback).
from __future__ import annotations

import asyncio
import glob
import os
import re
import sys
from typing import Optional

from src.config.providers import ProviderConfigError, get_llm_provider
from src.config.settings import Settings
from src.models.music import AudioMixResult
from src.models.script import ScriptResult, ScriptSection
from src.models.video import VideoAssemblyResult
from src.services.audio_mixing_service import AudioMixingService, AudioMixingServiceError
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError
from src.tools.music_catalog_provider import (
    DEFAULT_CATALOG_PATH,
    LocalMusicCatalogProvider,
    MusicCatalogProviderError,
)

DEFAULT_TOPIC = "Why do humans dream?"
CAPTIONED_SUFFIX = "-captioned.mp4"
BGM_SUFFIX = "-bgm.mp4"

# VideoAssemblyService names output files "<slug>-<8-char-hex>.mp4" (see
# DEFAULT_VIDEO_OUTPUT_DIR's producer) - stripped to recover a readable
# title when no real ScriptResult.video_title is available.
_ID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")

_RECONSTRUCTED_PLACEHOLDER = "(reconstructed for standalone BGM validation - not part of the real script)"

CATALOG_SETUP_INSTRUCTIONS = f"""
No approved BGM tracks were found in the local catalog.

Required setup:
  1. Obtain instrumental background tracks from an approved source
     (YouTube Audio Library is preferred - use tracks marked
     "Attribution not required"). Do not use random "No Copyright Music"
     channel downloads.
  2. Save each audio file under: assets/bgm/tracks/
  3. Add one entry per track to: {DEFAULT_CATALOG_PATH}
     (a JSON array - see assets/bgm/README.md for the exact field format
     and an example entry).

Once at least one track is in the catalog, re-run this demo.
""".strip()


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _find_latest_video(directory: str) -> Optional[str]:
    """Prefer the most recent captioned MP4; fall back to any assembled
    MP4 that isn't itself a previous BGM output."""
    if not os.path.isdir(directory):
        return None
    captioned = glob.glob(os.path.join(directory, f"*{CAPTIONED_SUFFIX}"))
    if captioned:
        return max(captioned, key=os.path.getmtime)
    plain = [
        path
        for path in glob.glob(os.path.join(directory, "*.mp4"))
        if not path.endswith(CAPTIONED_SUFFIX) and not path.endswith(BGM_SUFFIX)
    ]
    if not plain:
        return None
    return max(plain, key=os.path.getmtime)


def _original_base_name(video_path: str) -> str:
    """The base filename CaptionService derived its .srt/-captioned.mp4
    names from, regardless of whether ``video_path`` is itself the plain
    or the captioned variant."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    if base.endswith("-captioned"):
        base = base[: -len("-captioned")]
    return base


def _find_matching_srt(video_path: str, subtitle_dir: str) -> Optional[str]:
    """Locate the .srt transcript CaptionService produced for this same
    video, if any (see CaptionService._build_srt_filename)."""
    srt_path = os.path.join(subtitle_dir, f"{_original_base_name(video_path)}.srt")
    return srt_path if os.path.exists(srt_path) else None


def _extract_narration_from_srt(srt_path: str) -> str:
    """Concatenate every caption's text into one narration string, in
    chronological order - ignores block indices/timestamps entirely."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    narration_parts = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        if text_lines:
            narration_parts.append(" ".join(text_lines))
    return " ".join(narration_parts).strip()


def _derive_title_from_base_name(base_name: str) -> str:
    slug = _ID_SUFFIX_RE.sub("", base_name)
    words = [w for w in slug.replace("_", "-").split("-") if w]
    return " ".join(w.capitalize() for w in words) if words else base_name


def _build_context_from_srt(topic: str, srt_path: str, video_path: str) -> ScriptResult:
    narration = _extract_narration_from_srt(srt_path) or topic
    title = _derive_title_from_base_name(_original_base_name(video_path))
    hook = narration[:200].rsplit(" ", 1)[0] if len(narration) > 200 else narration
    return ScriptResult(
        topic=topic,
        video_title=title,
        hook=hook or topic,
        introduction=_RECONSTRUCTED_PLACEHOLDER,
        sections=[ScriptSection(heading="Full narration (from existing captions)", narration=narration)],
        conclusion=_RECONSTRUCTED_PLACEHOLDER,
        call_to_action=_RECONSTRUCTED_PLACEHOLDER,
    )


def _build_topic_only_context(topic: str) -> ScriptResult:
    return ScriptResult(
        topic=topic,
        video_title=topic,
        hook=topic,
        introduction=_RECONSTRUCTED_PLACEHOLDER,
        sections=[ScriptSection(heading="Topic only (no existing narration context found)", narration=topic)],
        conclusion=_RECONSTRUCTED_PLACEHOLDER,
        call_to_action=_RECONSTRUCTED_PLACEHOLDER,
    )


async def run_bgm_demo(topic: str, video_path: Optional[str] = None) -> AudioMixResult:
    """Mix background music under the most recently generated captioned MP4
    (or an explicit ``video_path``), planning mood/context from that
    video's own existing .srt transcript when available - never by
    re-running Research/Script/Voice/Visual Media/Visual QC/Video Assembly.

    Args:
        topic: Overall video topic (used for the music plan and, when no
            matching .srt exists, as the only context available)
        video_path: Assembled/captioned MP4 to mix BGM onto; auto-discovered
            under output/video/ if omitted

    Returns:
        AudioMixResult describing the outcome

    Raises:
        AudioMixingServiceError: If no source video or no BGM catalog track
            is available (configuration-level, not a processing failure)
        VideoAssemblerError: If FFmpeg is unavailable/misconfigured
    """
    print("[1/3] Checking FFmpeg availability and locating an existing video...")
    assembler = FFmpegVideoAssembler()

    video_path = video_path or _find_latest_video(DEFAULT_VIDEO_OUTPUT_DIR)
    if not video_path:
        raise AudioMixingServiceError(
            f"No existing assembled/captioned MP4 found under {DEFAULT_VIDEO_OUTPUT_DIR}/. "
            "Run the pipeline demo (python -m src.pipeline_demo) at least once first, "
            "or pass an explicit video path."
        )
    print(f"      Video: {video_path}")
    video_duration = assembler.probe_duration_seconds(video_path)
    video_result = VideoAssemblyResult(
        success=True, output_path=video_path, duration_seconds=video_duration, format="mp4"
    )

    print("\n[2/3] Checking the approved local BGM catalog...")
    catalog_provider = LocalMusicCatalogProvider()
    try:
        catalog = catalog_provider.list_tracks()
    except MusicCatalogProviderError as e:
        raise AudioMixingServiceError(str(e)) from e
    if not catalog:
        raise AudioMixingServiceError(CATALOG_SETUP_INSTRUCTIONS)
    print(f"      {len(catalog)} approved track(s) found")

    print("\n[3/3] Reconstructing narration context, planning mood, selecting a track, and mixing...")
    srt_path = _find_matching_srt(video_path, DEFAULT_SUBTITLE_OUTPUT_DIR)
    if srt_path:
        print(f"      Reusing existing narration context from: {srt_path}")
        script_result = _build_context_from_srt(topic, srt_path, video_path)
    else:
        print("      No matching .srt transcript found - using topic-only context (Research/Script are NOT re-run)")
        script_result = _build_topic_only_context(topic)

    settings = Settings()
    llm_provider = get_llm_provider(settings)  # only consumer: MusicContextPlanner's single mood-planning call
    service = AudioMixingService(catalog_provider=catalog_provider, assembler=assembler, llm_provider=llm_provider)
    return await service.generate_mix(topic, script_result, video_result)


def print_mix_result(result: AudioMixResult) -> None:
    """Pretty print an AudioMixResult."""
    print("\n" + "=" * 60)
    print("AUDIO MIX RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")

    if result.music_plan:
        plan = result.music_plan
        print(f"\nMusic plan (semantic planning used: {plan.used_semantic_planning}):")
        print(f"   Primary mood:    {plan.primary_mood}")
        print(f"   Secondary mood:  {plan.secondary_mood}")
        print(f"   Energy level:    {plan.energy_level}")
        print(f"   Genres:          {plan.preferred_genres}")
        print(f"   Instrumentation: {plan.preferred_instrumentation}")
        print(f"   Avoid styles:    {plan.avoid_styles}")
        print(f"   Neutral/subtle required: {plan.requires_neutral_subtle}")
        print(f"   Reasoning: {plan.reasoning_summary}")
        if plan.fallback_reason:
            print(f"   Fallback reason (deterministic fallback used): {plan.fallback_reason}")

    if result.selected_track:
        track = result.selected_track
        print("\nSelected track:")
        print(f"   {track.title} (id: {track.track_id})")
        print(f"   Source: {track.source} | License: {track.license_type}")
        print(f"   Attribution required: {track.attribution_required} ({track.attribution_text})")
        print(f"   Mood tags: {track.mood_tags} | Genre: {track.genre} | Energy: {track.energy_level}")

    if not result.success:
        print(f"\nError: {result.error}")
        return

    print(f"\nBGM gain: {result.bgm_gain_db} dB | Ducking used: {result.ducking_used} | Looped: {result.looped}")
    print(f"Source video duration: {result.source_duration_seconds}")
    print(f"Output duration:       {result.output_duration_seconds}")
    print(f"Output path: {result.output_path}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = await run_bgm_demo(topic)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except (ProviderConfigError, AudioMixingServiceError) as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_mix_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
