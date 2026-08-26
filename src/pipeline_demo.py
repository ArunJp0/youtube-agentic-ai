# Live demo runner for the complete Research -> Script -> Voice pipeline.
#
# Uses the existing provider factory (src.config.providers / Settings), so
# with the current .env configuration (LLM_PROVIDER=gemini,
# SEARCH_PROVIDER=wikipedia, VOICE_PROVIDER=edge) this calls real Wikipedia
# search, real Gemini LLM processing (including its existing retry/backoff/
# fallback-model behavior, untouched by this module), and real Edge TTS
# narration synthesis.
from __future__ import annotations

import asyncio
import dataclasses
import sys

from src.config.providers import ProviderConfigError, get_llm_provider, get_search_provider, get_voice_provider
from src.config.settings import Settings
from src.main import print_research_result
from src.script_demo import print_script_result
from src.services.voice_service import DEFAULT_OUTPUT_DIR
from src.workflows.pipeline_graph import PipelineState, build_pipeline_graph

DEFAULT_TOPIC = "Why do humans dream?"

_STAGE_LABELS = {
    "research": "[1/3] Research",
    "script": "[2/3] Script",
    "voice": "[3/3] Voice",
}


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_pipeline_demo(topic: str) -> PipelineState:
    """Run the full Research -> Script -> Voice pipeline, printing progress
    as each stage completes.

    Provider selection comes entirely from Settings/.env via the existing
    provider factory (src.config.providers) - this function does not
    hardcode Gemini, Wikipedia, or Edge TTS.

    Streams the compiled graph (rather than using the simpler
    ``run_pipeline`` convenience function) purely so this demo can print
    "[1/3] Research" / "[2/3] Script" / "[3/3] Voice" as each stage
    finishes; the underlying graph and node logic are unchanged.

    Args:
        topic: Research topic to investigate, script, and narrate

    Returns:
        Final PipelineState (see PipelineState.status/.error for outcome)
    """
    settings = Settings()
    llm_provider = get_llm_provider(settings)
    search_provider = get_search_provider(settings)
    voice_provider = get_voice_provider(settings)

    print(f"Pipeline: {topic}")
    print(
        f"   LLM provider: {settings.llm_provider} | Search provider: {settings.search_provider} "
        f"| Voice provider: {settings.voice_provider} ({settings.voice_name})"
    )
    print("=" * 60)

    graph = build_pipeline_graph(
        search_provider, llm_provider, voice_provider, settings.voice_name, DEFAULT_OUTPUT_DIR
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
        status=accumulated.get("status", "unknown"),
        error=accumulated.get("error"),
    )


async def main() -> None:
    """Main entry point for the pipeline demo."""
    _ensure_utf8_stdout()
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = DEFAULT_TOPIC

    try:
        state = await run_pipeline_demo(topic)
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

    print(f"\nPipeline status: {state.status}")
    if state.status != "completed":
        print(f"Pipeline error: {state.error}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
