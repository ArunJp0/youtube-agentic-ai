# Standalone live demo for the Caption Service.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - this is a deliberately standalone runner, per this milestone's scope.
#
# Reuses the EXISTING narration audio (output/audio/*.mp3) and assembled
# MP4 (output/video/*.mp4, excluding already-captioned copies) most
# recently produced by a prior pipeline_demo/video_demo run, rather than
# regenerating them - captioning doesn't need Research/Gemini/Pexels/Visual
# QC to run again. Explicit paths can be passed instead if given.
from __future__ import annotations

import asyncio
import glob
import os
import sys
from typing import Optional

from src.config.providers import ProviderConfigError, get_transcription_provider
from src.config.settings import Settings
from src.models.captions import CaptionResult
from src.models.video import VideoAssemblyResult
from src.models.voice import VoiceResult
from src.services.caption_service import CaptionService, CaptionServiceError
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR
from src.services.voice_service import DEFAULT_OUTPUT_DIR
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError

CAPTIONED_SUFFIX = "-captioned.mp4"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _find_latest_file(directory: str, extension: str, exclude_suffix: Optional[str] = None) -> Optional[str]:
    """Return the most recently modified file matching ``extension`` under
    ``directory``, skipping any file ending in ``exclude_suffix``."""
    if not os.path.isdir(directory):
        return None
    candidates = [
        path
        for path in glob.glob(os.path.join(directory, f"*{extension}"))
        if not (exclude_suffix and path.endswith(exclude_suffix))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


async def run_caption_demo(
    audio_path: Optional[str] = None, video_path: Optional[str] = None
) -> CaptionResult:
    """Caption the most recently generated narration audio + assembled MP4
    (or explicit paths, if given) using the configured TranscriptionProvider.

    Args:
        audio_path: Narration MP3 to transcribe; auto-discovered under
            output/audio/ if omitted
        video_path: Assembled MP4 to caption; auto-discovered under
            output/video/ (excluding already-captioned copies) if omitted

    Returns:
        CaptionResult describing the outcome
    """
    print("[1/3] Checking FFmpeg availability and locating existing artifacts...")
    assembler = FFmpegVideoAssembler()

    audio_path = audio_path or _find_latest_file(DEFAULT_OUTPUT_DIR, ".mp3")
    video_path = video_path or _find_latest_file(DEFAULT_VIDEO_OUTPUT_DIR, ".mp4", exclude_suffix=CAPTIONED_SUFFIX)
    if not audio_path or not video_path:
        raise CaptionServiceError(
            "No existing narration audio / assembled video found under "
            f"{DEFAULT_OUTPUT_DIR}/ and {DEFAULT_VIDEO_OUTPUT_DIR}/. Run "
            "the pipeline demo (python -m src.pipeline_demo) at least once first, "
            "or pass explicit paths."
        )
    print(f"      Audio: {audio_path}")
    print(f"      Video: {video_path}")

    audio_duration = assembler.probe_duration_seconds(audio_path)
    video_duration = assembler.probe_duration_seconds(video_path)
    voice_result = VoiceResult(
        audio_file_path=audio_path, provider="existing", voice_name="existing", format="mp3",
        success=True, duration_seconds=audio_duration,
    )
    video_result = VideoAssemblyResult(
        success=True, output_path=video_path, duration_seconds=video_duration, format="mp4",
    )

    settings = Settings()
    transcription_provider = get_transcription_provider(settings)
    print(f"\n[2/3] Transcribing narration audio via '{transcription_provider.name}'...")
    caption_service = CaptionService(transcription_provider=transcription_provider, assembler=assembler)

    print("\n[3/3] Building readable captions, writing .srt, and burning subtitles...")
    return await caption_service.generate_captions(voice_result, video_result)


def print_caption_result(result: CaptionResult) -> None:
    """Pretty print a CaptionResult."""
    print("\n" + "=" * 60)
    print("CAPTION RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error}")
        return

    print(f"Transcription provider: {result.transcription_provider} (model: {result.transcription_model})")
    print(f"Caption segments: {len(result.segments)}")
    print(f"SRT file:        {result.srt_path}")
    print(f"Captioned video: {result.captioned_video_path}")
    print(f"Narration duration: {result.narration_duration_seconds}")
    print(f"Original video duration:  {result.video_duration_seconds}")
    print(f"Captioned video duration: {result.captioned_duration_seconds}")

    print("\nFirst captions:")
    for segment in result.segments[:5]:
        text_preview = segment.text.replace("\n", " / ")
        print(f"   [{segment.start_seconds:.2f}s - {segment.end_seconds:.2f}s] {text_preview}")
    if len(result.segments) > 5:
        print(f"   ... and {len(result.segments) - 5} more")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    audio_arg = sys.argv[1] if len(sys.argv) > 1 else None
    video_arg = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = await run_caption_demo(audio_arg, video_arg)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except (ProviderConfigError, CaptionServiceError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_caption_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
