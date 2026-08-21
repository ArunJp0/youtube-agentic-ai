# Real-provider demo/test runner for the Research workflow.
#
# Unlike src/main.py (which always uses mock providers), this runs the
# Research Agent against real Gemini (LLM) and real Wikipedia (search).
# Requires GEMINI_API_KEY to be set in the environment or a .env file.
from __future__ import annotations

import asyncio
import sys

from src.config.settings import Settings
from src.config.providers import ProviderConfigError, get_llm_provider, get_search_provider
from src.main import print_research_result
from src.models.research import ResearchResult
from src.workflows.research_graph import run_research_workflow

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for emoji output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_real_research_demo(topic: str) -> ResearchResult:
    """Run the research workflow with real Gemini + Wikipedia providers.

    Args:
        topic: Research topic to investigate

    Returns:
        ResearchResult with structured findings
    """
    settings = Settings(llm_provider="gemini", search_provider="wikipedia")
    llm_provider = get_llm_provider(settings)
    search_provider = get_search_provider(settings)

    print(f"🔍 Researching (REAL providers): {topic}")
    print("   LLM: Gemini | Search: Wikipedia")
    print("=" * 60)

    return await run_research_workflow(topic, search_provider, llm_provider)


async def main() -> None:
    """Main entry point for the real research demo."""
    _ensure_utf8_stdout()
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = DEFAULT_TOPIC

    try:
        result = await run_real_research_demo(topic)
        print_research_result(result)
    except ProviderConfigError as e:
        print(f"❌ Provider configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
