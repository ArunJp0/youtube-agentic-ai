# Live demo runner for the complete Research Agent -> Script Agent pipeline.
#
# Uses the existing provider factory (src.config.providers / Settings), so
# with the current .env configuration (LLM_PROVIDER=gemini,
# SEARCH_PROVIDER=wikipedia) this calls real Wikipedia search and real
# Gemini LLM processing - including Gemini's existing retry/backoff/
# fallback-model behavior, which is untouched by this module.
from __future__ import annotations

import asyncio
import sys

from src.config.providers import ProviderConfigError, get_llm_provider, get_search_provider
from src.config.settings import Settings
from src.main import print_research_result
from src.script_demo import print_script_result
from src.workflows.pipeline_graph import PipelineState, run_pipeline

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_pipeline_demo(topic: str) -> PipelineState:
    """Run the full Research -> Script pipeline using the configured providers.

    Provider selection comes entirely from Settings/.env via the existing
    provider factory (src.config.providers) - this function does not
    hardcode Gemini or Wikipedia.

    Args:
        topic: Research topic to investigate and script

    Returns:
        Final PipelineState (see PipelineState.status/.error for outcome)
    """
    settings = Settings()
    llm_provider = get_llm_provider(settings)
    search_provider = get_search_provider(settings)

    print(f"Pipeline: {topic}")
    print(f"   LLM provider: {settings.llm_provider} | Search provider: {settings.search_provider}")
    print("=" * 60)

    return await run_pipeline(topic, search_provider, llm_provider)


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

    print(f"\nPipeline status: {state.status}")
    if state.status != "completed":
        print(f"Pipeline error: {state.error}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
