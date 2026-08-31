# Standalone live demo for the Visual QC Service.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - this is a deliberately standalone runner so Visual QC can be
# validated against real downloaded media and a real vision model on its
# own, per this milestone's scope.
#
# Produces a ScriptResult via the existing mock Research -> Script demo
# pipeline (fast, no LLM/network calls needed to build the script), then
# real visual context planning + real Pexels media (same as media_demo.py),
# then runs real Visual QC (Gemini vision calls) against the actual
# downloaded assets.
from __future__ import annotations

import asyncio
import sys

from src.agents.visual_context_planner import VisualContextPlanner
from src.config.providers import (
    ProviderConfigError,
    get_llm_provider,
    get_media_provider,
    get_visual_relevance_evaluator,
)
from src.config.settings import Settings
from src.models.media import VisualResult
from src.models.visual_qc import VisualQCResult
from src.script_demo import run_script_demo
from src.services.visual_media_service import VisualMediaService
from src.services.visual_qc_service import VisualQCService, VisualQCServiceError
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_visual_qc_demo(topic: str) -> tuple[VisualQCResult, VisualResult, VisualResult]:
    """Build a script (mock), real visual plan + real Pexels media, then run
    real standalone Visual QC against the actual downloaded assets.

    Args:
        topic: Research topic to build a script, plan visuals, and QC for

    Returns:
        (VisualQCResult, original VisualResult, QC-updated VisualResult)
    """
    print("[1/4] Checking FFmpeg availability...")
    assembler = FFmpegVideoAssembler()

    script_result = await run_script_demo(topic)

    settings = Settings(media_provider="pexels")
    print("\n[2/4] Building visual context plan (one Gemini call for the whole script)...")
    visual_planner = VisualContextPlanner(llm_provider=get_llm_provider(settings))
    visual_service = VisualMediaService(media_provider=get_media_provider(settings), visual_planner=visual_planner)
    plan = visual_service.build_plan(script_result)
    print(f"      Visual planning: {'LLM semantic plan' if plan.used_semantic_planning else 'deterministic fallback'}")

    print(f"\n[3/4] Selecting/downloading visuals for {len(script_result.sections)} section(s) via Pexels...")
    visual_result = await visual_service.generate_visuals(
        script_result, script_result.estimated_duration_seconds, visual_plan=plan
    )
    slot_count = sum(len(m.assets) for m in visual_result.sections)
    print(f"      Selected {slot_count} visual slot(s) (success={visual_result.success})")

    print("\n[4/4] Running Visual QC (real vision calls against downloaded frames)...")
    evaluator = get_visual_relevance_evaluator(settings)
    qc_service = VisualQCService(evaluator=evaluator, assembler=assembler, visual_media_service=visual_service)
    qc_result, updated_visual_result = await qc_service.run_qc(topic, script_result, plan, visual_result)

    return qc_result, visual_result, updated_visual_result


def print_qc_result(qc_result: VisualQCResult, visual_result: VisualResult) -> None:
    """Pretty print a VisualQCResult."""
    print("\n" + "=" * 60)
    print(f"VISUAL QC RESULT (provider: {qc_result.provider}, model: {qc_result.model})")
    print("=" * 60)
    print(f"QC completed: {qc_result.success}")
    if qc_result.error:
        print(f"Error: {qc_result.error}")
        return

    section_headings = {m.section_index: m.section_heading for m in visual_result.sections}

    for section in qc_result.sections:
        heading = section_headings.get(section.section_index, "")
        print(f"\nSection {section.section_index + 1} ({heading}):")
        for asset in section.assets:
            replaced_tag = f" [REPLACED x{asset.replacement_attempts}]" if asset.replaced else ""
            score = f"{asset.relevance_score:.2f}" if asset.relevance_score is not None else "n/a"
            print(
                f"   Slot {asset.slot_index + 1}: {asset.decision.upper()} "
                f"(score={score}, source={asset.evaluation_source}){replaced_tag}"
            )
            if asset.detected_visual_summary:
                print(f"      Detected: {asset.detected_visual_summary}")
            print(f"      Reason: {asset.reason}")
            if asset.misleading_or_conflicting:
                print("      ⚠ Flagged misleading/conflicting")

    if qc_result.repetition_warnings:
        print("\nRepetition warnings:")
        for warning in qc_result.repetition_warnings:
            print(f"   - {warning}")

    print("\n" + "-" * 60)
    print(f"Total assets checked: {qc_result.total_assets_checked}")
    print(f"Approved (incl. neutral/fallback): {qc_result.approved_count}")
    print(f"Warnings (weak, kept): {qc_result.warning_count}")
    print(f"Rejected (kept only as last resort): {qc_result.rejected_count}")
    print(f"Replaced: {qc_result.replaced_count}")
    print(f"Vision calls made: {qc_result.vision_calls_made}")
    print(f"Fallback used: {qc_result.fallback_used}")
    if qc_result.fallback_reason:
        print(f"Fallback reason: {qc_result.fallback_reason}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        qc_result, visual_result, _updated_visual_result = await run_visual_qc_demo(topic)
    except VideoAssemblerError as e:
        print(f"FFmpeg is unavailable or misconfigured:\n{e}")
        sys.exit(1)
    except (ProviderConfigError, VisualQCServiceError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print_qc_result(qc_result, visual_result)
    if not qc_result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
