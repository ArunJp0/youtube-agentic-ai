# Tests for GeminiVisualRelevanceEvaluator. All httpx calls are mocked in
# every test in this module, so no real network/Gemini call is ever made.
from __future__ import annotations

import base64
import json

import httpx
import pytest

from src.tools.gemini_visual_relevance_evaluator import GeminiVisualRelevanceEvaluator
from src.tools.visual_relevance_evaluator import AssetFrames, SectionQCContext, VisualRelevanceEvaluatorError


class FakeResponse:
    def __init__(self, json_data=None, status_code: int = 200) -> None:
        self._json_data = json_data or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com/fake")
            raise httpx.HTTPStatusError(
                f"HTTP error {self.status_code}", request=request, response=self  # type: ignore[arg-type]
            )

    def json(self):
        return self._json_data


class FakeAsyncClient:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.last_json = None
        self.last_url = None

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, params=None, json=None):
        self.last_url = url
        self.last_json = json
        if self.exc is not None:
            raise self.exc
        return self.response


def _gemini_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _context(asset_ids, frame_path: str) -> SectionQCContext:
    return SectionQCContext(
        topic="Why do humans dream?",
        section_index=0,
        section_heading="Intro",
        narration="The mind constructs a narrative from memories.",
        semantic_summary="The mind forms a dream narrative.",
        visual_intents=["person dreaming"],
        avoid_concepts=["construction site"],
        assets=[AssetFrames(asset_id=aid, frame_paths=[frame_path]) for aid in asset_ids],
    )


def _verdicts_json(*asset_ids) -> str:
    return json.dumps(
        {
            "assets": [
                {
                    "asset_id": aid,
                    "relevance_score": 0.8,
                    "misleading_or_conflicting": False,
                    "detected_visual_summary": "a person sleeping",
                    "reason": "matches the section",
                }
                for aid in asset_ids
            ]
        }
    )


class TestGeminiVisualRelevanceEvaluatorInit:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(VisualRelevanceEvaluatorError, match="GEMINI_API_KEY"):
            GeminiVisualRelevanceEvaluator(api_key=None)

    def test_name(self) -> None:
        assert GeminiVisualRelevanceEvaluator(api_key="k").name == "gemini"


class TestGeminiVisualRelevanceEvaluatorEvaluate:
    @pytest.mark.asyncio
    async def test_returns_verdicts_for_all_assets(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"FAKEJPEG")

        payload = _gemini_payload(_verdicts_json("a", "b"))
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        verdicts = await evaluator.evaluate_section(_context(["a", "b"], str(frame_path)))

        assert {v.asset_id for v in verdicts} == {"a", "b"}
        assert verdicts[0].relevance_score == 0.8

    @pytest.mark.asyncio
    async def test_image_is_base64_encoded_and_sent_as_inline_data(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"FAKEJPEG-BYTES")

        payload = _gemini_payload(_verdicts_json("a"))
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        await evaluator.evaluate_section(_context(["a"], str(frame_path)))

        parts = fake_client.last_json["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inline_data" in p]
        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "image/jpeg"
        decoded = base64.b64decode(inline_parts[0]["inline_data"]["data"])
        assert decoded == b"FAKEJPEG-BYTES"

    @pytest.mark.asyncio
    async def test_prompt_includes_topic_summary_and_avoid_concepts(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        payload = _gemini_payload(_verdicts_json("a"))
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        context = _context(["a"], str(frame_path))
        await evaluator.evaluate_section(context)

        prompt_text = fake_client.last_json["contents"][0]["parts"][0]["text"]
        assert context.topic in prompt_text
        assert context.semantic_summary in prompt_text
        assert "construction site" in prompt_text

    @pytest.mark.asyncio
    async def test_single_request_regardless_of_asset_count(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        payload = _gemini_payload(_verdicts_json("a", "b", "c"))
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        await evaluator.evaluate_section(_context(["a", "b", "c"], str(frame_path)))

        assert fake_client.last_url is not None  # exactly one post() call was made (no retry loop)

    @pytest.mark.asyncio
    async def test_http_error_raises_evaluator_error(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        fake_client = FakeAsyncClient(response=FakeResponse({}, status_code=500))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        with pytest.raises(VisualRelevanceEvaluatorError):
            await evaluator.evaluate_section(_context(["a"], str(frame_path)))

    @pytest.mark.asyncio
    async def test_network_error_raises_evaluator_error(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        fake_client = FakeAsyncClient(exc=httpx.ConnectError("connection refused"))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        with pytest.raises(VisualRelevanceEvaluatorError):
            await evaluator.evaluate_section(_context(["a"], str(frame_path)))

    @pytest.mark.asyncio
    async def test_malformed_response_raises_evaluator_error(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        fake_client = FakeAsyncClient(response=FakeResponse({"candidates": []}))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        with pytest.raises(VisualRelevanceEvaluatorError):
            await evaluator.evaluate_section(_context(["a"], str(frame_path)))

    @pytest.mark.asyncio
    async def test_unparseable_json_raises_evaluator_error(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        payload = _gemini_payload("this is not json")
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        with pytest.raises(VisualRelevanceEvaluatorError):
            await evaluator.evaluate_section(_context(["a"], str(frame_path)))

    @pytest.mark.asyncio
    async def test_empty_assets_list_in_response_raises(self, monkeypatch, tmp_path) -> None:
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"X")

        payload = _gemini_payload(json.dumps({"assets": []}))
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        evaluator = GeminiVisualRelevanceEvaluator(api_key="test-key")
        with pytest.raises(VisualRelevanceEvaluatorError):
            await evaluator.evaluate_section(_context(["a"], str(frame_path)))
