# Live demo runner for the Video Assembly Service.
#
# Produces a ScriptResult via the existing mock Research -> Script demo
# pipeline (fast, no LLM/network calls needed to build the script), then
# generates REAL narration audio (Edge TTS) and REAL section media
# (Pexels) for it - both free - so the Video Assembly Service has genuine
# media to work with, then assembles a REAL MP4 with FFmpeg.
#
# Standalone only: this does not touch src/workflows/pipeline_graph.py.
from __future__ import annotations

import asyncio
import sys

from src.config.providers import ProviderConfigError, get_media_provider, get_voice_provider
from src.config.settings import Settings
from src.models.video import VideoAssemblyResult
from src.script_demo import run_script_demo
from src.services.video_assembly_service import VideoAssemblyService, VideoAssemblyServiceError
from src.services.visual_media_service import VisualMediaService
from src.services.voice_service import VoiceService
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_video_demo(topic: str) -> VideoAssemblyResult:
    """Build a script (mock), real narration audio (Edge TTS), real section
    media (Pexels), then assemble a real MP4 (FFmpeg).

    Args:
        topic: Research topic to build a script, narrate, illustrate, and
            assemble a video for

    Returns:
        VideoAssemblyResult describing the final video (or the failure)
    """
    # Check FFmpeg availability FIRST, before spending any real Edge
    # TTS/Pexels calls on a demo that couldn't finish anyway.
    print("[1/3] Checking FFmpeg availability...")
    assembler = FFmpegVideoAssembler()
    video_service = VideoAssemblyService(assembler=assembler)
    print(f"      ffmpeg:  {assembler.ffmpeg_path}")
    print(f"      ffprobe: {assembler.ffprobe_path}")

    script_result = await run_script_demo(topic)

    voice_settings = Settings(voice_provider="edge")
    voice_service = VoiceService(
        voice_provider=get_voice_provider(voice_settings), voice_name=voice_settings.voice_name
    )
    print("\n[2/3] Synthesizing narration audio (Edge TTS, real synthesis)...")
    voice_result = await voice_service.generate_voice(script_result)
    print(f"      Audio: {voice_result.audio_file_path}  (success={voice_result.success})")
    if not voice_result.success:
        print(f"      Error: {voice_result.error}")

    media_settings = Settings(media_provider="pexels")
    visual_service = VisualMediaService(media_provider=get_media_provider(media_settings))
    print(f"\n[2/3] Finding visuals for {len(script_result.sections)} section(s) via Pexels...")
    visual_result = await visual_service.generate_visuals(script_result)
    found = sum(1 for m in visual_result.sections if m.assets and m.assets[0].success)
    print(f"      Media: {found}/{len(visual_result.sections)} section(s) (success={visual_result.success})")

    print("\n[3/3] Assembling final MP4 (FFmpeg: scale/crop/trim/loop per section, concat, mux audio)...")
    return await video_service.assemble_video(script_result, voice_result, visual_result)


def print_video_result(result: VideoAssemblyResult) -> None:
    """Pretty print a VideoAssemblyResult."""
    print("\n" + "=" * 60)
    print("VIDEO ASSEMBLY RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Output:     {result.output_path}")
        print(f"Duration:   {result.duration_seconds:.1f}s")
        print(f"Resolution: {result.width}x{result.height} @ {result.fps}fps")
        print(f"Format:     {result.format} ({result.video_codec} video / {result.audio_codec} audio)")
        print(f"Sections:   {result.section_count}")
        print(
            "Section durations (s): "
            + ", ".join(f"{d:.1f}" for d in result.section_durations_seconds)
        )
    else:
        print(f"Error: {result.error}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = await run_video_demo(topic)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except (ProviderConfigError, VideoAssemblyServiceError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_video_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
