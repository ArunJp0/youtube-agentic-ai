# Live demo runner for the Voice Generation Service.
#
# Produces a ScriptResult via the existing mock Research -> Script demo
# pipeline (fast, no LLM/network calls needed to build the script), then
# synthesizes REAL narration audio for it using the free Edge TTS provider,
# so this demo produces an actual playable audio file.
from __future__ import annotations

import asyncio
import sys

from src.config.providers import ProviderConfigError, get_voice_provider
from src.config.settings import Settings
from src.models.voice import VoiceResult
from src.script_demo import run_script_demo
from src.services.voice_service import VoiceService

DEFAULT_TOPIC = "Why do humans dream?"


def _ensure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError for non-ASCII output on legacy Windows consoles (cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


async def run_voice_demo(topic: str) -> VoiceResult:
    """Build a script (mock pipeline) then synthesize real narration audio for it.

    Args:
        topic: Research topic to build a script for and narrate

    Returns:
        VoiceResult describing the generated audio (or the failure)
    """
    script_result = await run_script_demo(topic)

    settings = Settings(voice_provider="edge")
    voice_provider = get_voice_provider(settings)
    voice_service = VoiceService(voice_provider=voice_provider, voice_name=settings.voice_name)

    print("\n" + "=" * 60)
    print(f"SECTION TITLES AND NARRATION ({len(script_result.sections)} section(s) kept)")
    print("=" * 60)
    for i, section in enumerate(script_result.sections, 1):
        print(f"\n{i}. {section.heading}")
        print(f"   {section.narration}")

    final_narration = VoiceService.extract_narration(script_result)
    print("\n" + "=" * 60)
    print("FINAL NARRATION TEXT (post validation/dedup, sent to TTS)")
    print("=" * 60)
    print(final_narration)

    print("\n[3/3] Synthesizing narration audio (Edge TTS, real synthesis)...")
    return await voice_service.generate_voice(script_result)


def print_voice_result(result: VoiceResult) -> None:
    """Pretty print a VoiceResult."""
    print("\n" + "=" * 60)
    print("VOICE RESULT")
    print("=" * 60)
    print(f"Success:  {result.success}")
    print(f"Provider: {result.provider}")
    print(f"Voice:    {result.voice_name}")
    print(f"Format:   {result.format}")
    if result.duration_seconds is not None:
        print(f"Duration: {result.duration_seconds:.1f}s")
    if result.audio_file_path:
        print(f"Audio file: {result.audio_file_path}")
    if result.error:
        print(f"Error: {result.error}")


async def main() -> None:
    """Main entry point for the demo."""
    _ensure_utf8_stdout()
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    try:
        result = await run_voice_demo(topic)
    except ProviderConfigError as e:
        print(f"Provider configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_voice_result(result)
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
