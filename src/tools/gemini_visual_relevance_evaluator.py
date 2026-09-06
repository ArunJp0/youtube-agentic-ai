# Real VisualRelevanceEvaluator backed by the Gemini API's multimodal
# generateContent endpoint (image + text input in one request).
#
# Reuses the same REST endpoint GeminiLLMProvider already calls
# (generateContent) - Gemini's flash models are natively multimodal, so
# this only adds image parts to the same request shape already proven to
# work in src/llm/gemini.py, rather than inventing a new API surface. Kept
# as its own small class (not a change to GeminiLLMProvider/LLMProvider):
# LLMProvider's generate_text(prompt) -> str contract has no way to carry
# images, and Research/Script's use of it is intentionally left untouched.
from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import httpx

from src.llm.gemini import DEFAULT_GEMINI_MODEL, GEMINI_API_BASE
from src.models.visual_qc import RawAssetVerdict
from src.services.llm_json import JsonExtractionError, extract_json_object
from src.tools.visual_relevance_evaluator import (
    SectionQCContext,
    VisualRelevanceEvaluator,
    VisualRelevanceEvaluatorError,
)

# Reuses GeminiLLMProvider's own default model constant (this module
# already depends on src.llm.gemini for GEMINI_API_BASE) rather than a
# second hardcoded literal - one source of truth for "the default Gemini
# text/vision model", chosen via a real health check (see
# src/gemini_health_check.py and src/llm/gemini.py's DEFAULT_GEMINI_MODEL).
DEFAULT_VISION_MODEL = DEFAULT_GEMINI_MODEL


class GeminiVisualRelevanceEvaluator(VisualRelevanceEvaluator):
    """Vision-capable relevance evaluator backed by the Gemini API.

    One request per ``evaluate_section`` call, with every candidate
    asset's representative frame(s) attached as image parts alongside a
    single concise text prompt - never one request per frame or per asset.
    """

    def __init__(
        self, api_key: Optional[str], model: str = DEFAULT_VISION_MODEL, timeout_seconds: float = 30.0
    ) -> None:
        if not api_key:
            raise VisualRelevanceEvaluatorError(
                "GEMINI_API_KEY is not set. Set it in the environment or .env file "
                "to use the Gemini vision evaluator."
            )
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "gemini"

    async def evaluate_section(self, context: SectionQCContext) -> List[RawAssetVerdict]:
        payload = {"contents": [{"parts": self._build_parts(context)}]}
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, params={"key": self.api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise VisualRelevanceEvaluatorError(
                f"Gemini vision API returned an error: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            raise VisualRelevanceEvaluatorError(f"Gemini vision request failed: {e}") from e

        raw_text = self._extract_text(data)
        return self._parse_verdicts(raw_text)

    # ---- request building --------------------------------------------------

    def _build_parts(self, context: SectionQCContext) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"text": self._build_prompt(context)}]
        for asset in context.assets:
            parts.append({"text": f"[asset_id={asset.asset_id}]"})
            for frame_path in asset.frame_paths:
                parts.append(
                    {"inline_data": {"mime_type": _mime_type_for(frame_path), "data": _encode_image(frame_path)}}
                )
        return parts

    @staticmethod
    def _build_prompt(context: SectionQCContext) -> str:
        intents = ", ".join(context.visual_intents) or "(none specified)"
        avoid = ", ".join(context.avoid_concepts) or "(none specified)"
        asset_ids = ", ".join(asset.asset_id for asset in context.assets)

        return (
            "You are quality-checking stock video/photo clips already selected for one "
            "section of a YouTube video, by looking at representative frames from each "
            "clip. Prefer a neutral, relevant visual over a literal-but-misleading one, "
            "and do not require every frame to be a literal depiction of every sentence - "
            "acceptable neutral B-roll should not be rejected just because it isn't a "
            "literal visualization of the narration.\n\n"
            f"Video topic: {context.topic}\n"
            f"Section meaning: {context.semantic_summary or context.section_heading}\n"
            f"Section narration: {context.narration}\n"
            f"Intended visual concepts: {intents}\n"
            f"Concepts to avoid (literal-but-wrong interpretations for this section): {avoid}\n\n"
            f"Below are representative frames for {len(context.assets)} candidate asset(s), "
            "each preceded by a text label giving its asset_id. Asset IDs, in order: "
            f"{asset_ids}\n\n"
            "Return ONLY a single JSON object (no markdown fences, no commentary) with "
            "exactly this shape:\n"
            "{\n"
            '  "assets": [\n'
            "    {\n"
            '      "asset_id": "...",\n'
            '      "relevance_score": 0.0,\n'
            '      "misleading_or_conflicting": false,\n'
            '      "detected_visual_summary": "one short phrase describing what the frame(s) actually show",\n'
            '      "reason": "one short sentence explaining the score"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Include exactly one object per asset_id listed above, in any order."
        )

    # ---- response parsing ---------------------------------------------------

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        candidates = data.get("candidates")
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise VisualRelevanceEvaluatorError(f"Gemini blocked the vision request: {block_reason}")
            raise VisualRelevanceEvaluatorError(f"Malformed Gemini vision response (no candidates): {data}")

        try:
            parts = candidates[0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise VisualRelevanceEvaluatorError(f"Malformed Gemini vision response: {data}") from e

        if not text.strip():
            raise VisualRelevanceEvaluatorError("Gemini vision call returned an empty response")
        return text

    @staticmethod
    def _parse_verdicts(raw_text: str) -> List[RawAssetVerdict]:
        try:
            payload = extract_json_object(raw_text)
            data = json.loads(payload)
            items = data["assets"]
            if not isinstance(items, list) or not items:
                raise VisualRelevanceEvaluatorError("Vision response had no assets")
            return [RawAssetVerdict.model_validate(item) for item in items]
        except VisualRelevanceEvaluatorError:
            raise
        except (JsonExtractionError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise VisualRelevanceEvaluatorError(f"Could not parse Gemini vision response: {e}") from e


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _mime_type_for(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"
