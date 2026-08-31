# Small shared helper for extracting a single JSON object from LLM text
# output that may be wrapped in a markdown code fence or surrounded by
# commentary. Used by both VisualContextPlanner and
# GeminiVisualRelevanceEvaluator so there is one implementation, not two.
from __future__ import annotations

import re

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class JsonExtractionError(Exception):
    """Raised when no JSON object could be found in an LLM text response."""


def extract_json_object(raw_text: str) -> str:
    """Best-effort extraction of a single JSON object substring from
    ``raw_text``, stripping a surrounding markdown code fence if present
    and ignoring any leading/trailing commentary.

    Raises:
        JsonExtractionError: If no ``{...}`` object could be located.
    """
    text = raw_text.strip()
    fence_match = _JSON_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JsonExtractionError("No JSON object found in response")
    return text[start : end + 1]
