# Live demo runner for the complete Research -> Script -> Voice -> Visual
# Media -> Visual QC -> Video Assembly -> Subtitle/Caption pipeline.
#
# Uses the existing provider factory (src.config.providers / Settings), so
# with the current .env configuration (LLM_PROVIDER=gemini,
# SEARCH_PROVIDER=wikipedia, VOICE_PROVIDER=edge, MEDIA_PROVIDER=pexels,
# TRANSCRIPTION_PROVIDER=whisper) this calls real Wikipedia search, real
# Gemini LLM processing (including its existing retry/backoff/fallback-
# model behavior, untouched by this module), real Edge TTS narration
# synthesis, real Pexels stock media retrieval, real Gemini vision QC, real
# FFmpeg video assembly, and real local Whisper transcription/captioning.
from __future__ import annotations

import asyncio
import dataclasses
import sys

from src.caption_demo import print_caption_result
from src.config.providers import (
    ProviderConfigError,
    get_llm_provider,
    get_media_provider,
    get_search_provider,
    get_transcription_provider,
    get_visual_relevance_evaluator,
    get_voice_provider,
)
from src.config.settings import Settings
from src.main import print_research_result
from src.media_demo import print_visual_result
from src.script_demo import print_script_result
from src.visual_qc_demo import print_qc_result
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR
from src.services.video_assembly_service import DEFAULT_VIDEO_OUTPUT_DIR
from src.services.visual_media_service import DEFAULT_MEDIA_OUTPUT_DIR
from src.services.voice_service import DEFAULT_OUTPUT_DIR
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError
from src.workflows.pipeline_graph import PipelineState, build_pipeline_graph

DEFAULT_TOPIC = "Why do humans dream?"

# Visual Context Planner runs inside the "media" stage (VisualMediaService
# builds the plan and consumes it in the same step) rather than as its own
# top-level progress stage - it isn't a separately observable pipeline step
# from the outside, just an internal part of how Visual Media selects assets.
_STAGE_LABELS = {
    "research": "[1/7] Research",
    "script": "[2/7] Script",
    "voice": "[3/7] Voice",
    "media": "[4/7] Visual Media",
    "visual_qc": "[5/7] Visual QC",
    "video_assembly": "[6/7] Video Assembly",
    "captions": "[7/7] Subtitles / Captions",
}


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_pipeline_demo(topic: str) -> PipelineState:
    """Run the full 7-stage pipeline, printing progress as each stage completes.

    Provider selection comes entirely from Settings/.env via the existing
    provider factory (src.config.providers) - this function does not
    hardcode Gemini, Wikipedia, Edge TTS, or Pexels. FFmpeg availability is
    checked eagerly (constructing FFmpegVideoAssembler) before any real
    Research/Script/Voice/Media calls are made, so a missing FFmpeg install
    is reported immediately rather than after burning API quota.

    Streams the compiled graph (rather than using the simpler
    ``run_pipeline`` convenience function) purely so this demo can print
    per-stage progress as each stage finishes; the underlying graph and
    node logic (including VideoAssemblyService, reused unchanged) are
    identical either way.

    Args:
        topic: Research topic to investigate, script, narrate, illustrate,
            and assemble into a final video

    Returns:
        Final PipelineState (see PipelineState.status/.error for outcome)
    """
    assembler = FFmpegVideoAssembler()

    settings = Settings()
    llm_provider = get_llm_provider(settings)
    search_provider = get_search_provider(settings)
    voice_provider = get_voice_provider(settings)
    media_provider = get_media_provider(settings)
    visual_relevance_evaluator = get_visual_relevance_evaluator(settings)
    transcription_provider = get_transcription_provider(settings)

    print(f"Pipeline: {topic}")
    print(
        f"   LLM: {settings.llm_provider} | Search: {settings.search_provider} "
        f"| Voice: {settings.voice_provider} ({settings.voice_name}) "
        f"| Media: {settings.media_provider} | Visual QC: {visual_relevance_evaluator.name} "
        f"| Captions: {transcription_provider.name} | Video: ffmpeg"
    )
    print("=" * 60)

    graph = build_pipeline_graph(
        search_provider,
        llm_provider,
        voice_provider,
        settings.voice_name,
        media_provider,
        assembler,
        visual_relevance_evaluator,
        transcription_provider,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_MEDIA_OUTPUT_DIR,
        DEFAULT_VIDEO_OUTPUT_DIR,
        DEFAULT_SUBTITLE_OUTPUT_DIR,
    ).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    accumulated: dict = dataclasses.asdict(initial_state)
    async for update in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            accumulated.update(node_output)
            label = _STAGE_LABELS.get(node_name, node_name)
            if accumulated.get("error") and node_output.get("error"):
                print(f"{label}: FAILED - {node_output['error']}")
            else:
                print(f"{label}: done")

    return PipelineState(
        topic=accumulated.get("topic", topic),
        research_result=accumulated.get("research_result"),
        script_result=accumulated.get("script_result"),
        voice_result=accumulated.get("voice_result"),
        visual_plan=accumulated.get("visual_plan"),
        visual_result=accumulated.get("visual_result"),
        visual_qc_result=accumulated.get("visual_qc_result"),
        qc_approved_visual_result=accumulated.get("qc_approved_visual_result"),
        video_assembly_result=accumulated.get("video_assembly_result"),
        caption_result=accumulated.get("caption_result"),
        status=accumulated.get("status", "unknown"),
        error=accumulated.get("error"),
    )


def _print_final_summary(state: PipelineState) -> None:
    """Concise wrap-up: status + one line per stage - no full object dumps."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Pipeline status: {state.status}")
    print(f"Research success: {state.research_result is not None}")

    if state.script_result:
        print(f"Script success: True ({len(state.script_result.sections)} sections)")
    else:
        print("Script success: False")

    if state.voice_result:
        print(f"Voice success: {state.voice_result.success} (audio: {state.voice_result.audio_file_path})")
    else:
        print("Voice success: False")

    if state.visual_result:
        found = sum(1 for m in state.visual_result.sections if any(a.success for a in m.assets))
        slot_count = sum(len(m.assets) for m in state.visual_result.sections)
        planner_status = (
            "LLM semantic plan" if state.visual_result.semantic_planning_used else "deterministic fallback"
        )
        print(
            f"Visual Media success: {state.visual_result.success} "
            f"({found}/{len(state.visual_result.sections)} sections, {slot_count} visual slots, "
            f"visual planning: {planner_status})"
        )
    else:
        print("Visual Media success: False")

    if state.visual_qc_result:
        qc = state.visual_qc_result
        print(
            f"Visual QC success: {qc.success} ({qc.approved_count} approved incl. fallback, "
            f"{qc.warning_count} warning, {qc.rejected_count} rejected, {qc.replaced_count} replaced, "
            f"{qc.vision_calls_made} vision call(s), fallback used: {qc.fallback_used})"
        )
    else:
        print("Visual QC success: False")

    if state.video_assembly_result:
        result = state.video_assembly_result
        print(f"Video Assembly success: {result.success}")
        if result.success:
            print(f"Assembled MP4 (non-captioned): {result.output_path}")
            print(f"Duration:   {result.duration_seconds:.1f}s")
            print(f"Resolution: {result.width}x{result.height} @ {result.fps}fps")
        else:
            print(f"Video Assembly error: {result.error}")
    else:
        print("Video Assembly success: False")

    if state.caption_result:
        caption = state.caption_result
        print(f"Captions success: {caption.success}")
        if caption.success:
            print(f"Caption segments: {len(caption.segments)}")
            print(f"SRT file:        {caption.srt_path}")
            print(f"Captioned MP4:   {caption.captioned_video_path}")
            print(
                f"Duration (original vs captioned): "
                f"{caption.video_duration_seconds} vs {caption.captioned_duration_seconds}"
            )
        else:
            print(f"Captions error: {caption.error}")
    else:
        print("Captions success: False")


async def main() -> None:
    """Main entry point for the pipeline demo."""
    _ensure_utf8_stdout()
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = DEFAULT_TOPIC

    try:
        state = await run_pipeline_demo(topic)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except ProviderConfigError as e:
        print(f"Provider configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if state.research_result:
        print("\n" + "#" * 60)
        print("# RESEARCH RESULT")
        print("#" * 60)
        print_research_result(state.research_result)

    if state.script_result:
        print("\n" + "#" * 60)
        print("# SCRIPT RESULT")
        print("#" * 60)
        print_script_result(state.script_result)

    if state.voice_result:
        print("\n" + "#" * 60)
        print("# VOICE RESULT")
        print("#" * 60)
        print(f"Success:  {state.voice_result.success}")
        print(f"Provider: {state.voice_result.provider}")
        print(f"Voice:    {state.voice_result.voice_name}")
        print(f"Format:   {state.voice_result.format}")
        if state.voice_result.duration_seconds is not None:
            print(f"Duration: {state.voice_result.duration_seconds:.1f}s")
        if state.voice_result.audio_file_path:
            print(f"Audio file: {state.voice_result.audio_file_path}")
        if state.voice_result.error:
            print(f"Error: {state.voice_result.error}")

    if state.visual_result:
        print("\n" + "#" * 60)
        print("# VISUAL MEDIA RESULT")
        print("#" * 60)
        print_visual_result(state.visual_result)

    if state.visual_qc_result:
        print("\n" + "#" * 60)
        print("# VISUAL QC RESULT")
        print("#" * 60)
        print_qc_result(state.visual_qc_result, state.visual_result)

    if state.video_assembly_result:
        print("\n" + "#" * 60)
        print("# VIDEO ASSEMBLY RESULT")
        print("#" * 60)
        result = state.video_assembly_result
        print(f"Success: {result.success}")
        if result.success:
            print(f"Output:     {result.output_path}")
            print(f"Duration:   {result.duration_seconds:.1f}s")
            print(f"Resolution: {result.width}x{result.height} @ {result.fps}fps")
            print(f"Format:     {result.format} ({result.video_codec} video / {result.audio_codec} audio)")
            print(f"Sections:   {result.section_count}")
        else:
            print(f"Error: {result.error}")

    if state.caption_result:
        print("\n" + "#" * 60)
        print("# CAPTION RESULT")
        print("#" * 60)
        print_caption_result(state.caption_result)

    _print_final_summary(state)

    if state.status != "completed":
        print(f"\nPipeline error: {state.error}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
