# Standalone Gemini model health-check tool.
#
# Tests model AVAILABILITY/RELIABILITY only - one tiny, low-token prompt
# per candidate model, no Research/Script/Voice/Visual Media/Visual QC/
# Captions/BGM/Pexels/Whisper/FFmpeg involved. Uses the existing configured
# GEMINI_API_KEY (never printed/logged).
#
# Two data sources for candidates, in priority order:
#   1. The real ListModels endpoint (GET .../v1beta/models) - authoritative,
#      not guessed. Only models the key's project can actually see and that
#      support generateContent are considered.
#   2. If listing itself fails/is unavailable, falls back to only the
#      already-configured GEMINI_MODEL/GEMINI_FALLBACK_MODEL - never an
#      invented model name.
#
# Each probe is a single request with NO retry/backoff of its own - this
# tool measures raw availability/latency; GeminiLLMProvider's own bounded
# retry policy is a separate, already-tested concern (see test_llm_gemini.py).
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from src.config.settings import Settings

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
PROBE_TIMEOUT_SECONDS = 15.0
PROBE_PROMPT = "Reply with exactly one word: OK"
MAX_OUTPUT_TOKENS = 5

RETRYABLE_SERVER_STATUS_CODES = {500, 502, 503, 504}


@dataclass
class ModelHealthResult:
    model: str
    success: bool
    latency_seconds: float
    category: str  # "ok", "429", "503", "timeout", "malformed", "error"
    detail: str


def list_available_models(api_key: str) -> List[str]:
    """Return real model names the API key can access that support
    generateContent - the authoritative source, never a guessed list.

    Returns an empty list (not an exception) if listing itself fails, so
    callers can fall back to explicitly-configured candidates instead.
    """
    try:
        response = httpx.get(f"{API_BASE}/models", params={"key": api_key}, timeout=PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError:
        return []

    models = []
    for entry in data.get("models", []):
        name = entry.get("name", "")  # e.g. "models/gemini-3.6-flash"
        methods = entry.get("supportedGenerationMethods", [])
        if name.startswith("models/") and "generateContent" in methods:
            models.append(name.split("models/", 1)[1])
    return models


def probe_model(api_key: str, model: str) -> ModelHealthResult:
    """Send exactly one minimal generateContent request - no retries."""
    url = f"{API_BASE}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": PROBE_PROMPT}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS},
    }

    start = time.monotonic()
    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.TimeoutException:
        return ModelHealthResult(model, False, time.monotonic() - start, "timeout", "request timed out")
    except httpx.HTTPError as e:
        return ModelHealthResult(model, False, time.monotonic() - start, "error", str(e))

    latency = time.monotonic() - start

    if response.status_code == 429:
        return ModelHealthResult(model, False, latency, "429", "rate limited")
    if response.status_code in RETRYABLE_SERVER_STATUS_CODES:
        return ModelHealthResult(model, False, latency, "503", f"HTTP {response.status_code}")
    if response.status_code >= 400:
        return ModelHealthResult(model, False, latency, "error", f"HTTP {response.status_code}: {response.text[:200]}")

    try:
        data = response.json()
        candidates = data.get("candidates")
        if not candidates:
            return ModelHealthResult(model, False, latency, "malformed", f"no candidates: {str(data)[:200]}")
        parts = candidates[0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
    except Exception as e:
        return ModelHealthResult(model, False, latency, "malformed", str(e))

    if not text.strip():
        return ModelHealthResult(model, False, latency, "malformed", "empty response text")

    return ModelHealthResult(model, True, latency, "ok", text.strip()[:50])


def run_health_check(candidates: Optional[List[str]] = None) -> List[ModelHealthResult]:
    """Probe each candidate model once. If ``candidates`` is None, tries
    ListModels first, falling back to only the currently-configured
    GEMINI_MODEL/GEMINI_FALLBACK_MODEL if listing is unavailable."""
    settings = Settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set - cannot run a Gemini health check.")

    if candidates is None:
        candidates = list_available_models(settings.gemini_api_key)
        if not candidates:
            candidates = [m for m in (settings.gemini_model, settings.gemini_fallback_model) if m]

    results = []
    for model in candidates:
        result = probe_model(settings.gemini_api_key, model)
        results.append(result)
        print(
            f"{model:30s} success={str(result.success):5s} category={result.category:10s} "
            f"latency={result.latency_seconds:5.2f}s  {result.detail}"
        )
    return results


def main() -> None:
    candidate_args = sys.argv[1:] or None
    try:
        results = run_health_check(candidate_args)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    healthy = [r for r in results if r.success]
    print(f"\n{len(healthy)}/{len(results)} model(s) healthy.")
    if not healthy:
        sys.exit(1)


if __name__ == "__main__":
    main()
