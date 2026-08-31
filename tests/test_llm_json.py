# Tests for the shared JSON-extraction helper used by VisualContextPlanner
# and GeminiVisualRelevanceEvaluator.
from __future__ import annotations

import pytest

from src.services.llm_json import JsonExtractionError, extract_json_object


class TestExtractJsonObject:
    def test_plain_json_object(self) -> None:
        assert extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_strips_markdown_fence(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert extract_json_object(text) == '{"a": 1}'

    def test_strips_bare_fence_without_language_tag(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert extract_json_object(text) == '{"a": 1}'

    def test_strips_leading_and_trailing_commentary(self) -> None:
        text = 'Sure, here you go:\n{"a": 1}\nHope that helps!'
        assert extract_json_object(text) == '{"a": 1}'

    def test_no_object_raises(self) -> None:
        with pytest.raises(JsonExtractionError):
            extract_json_object("not json at all")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(JsonExtractionError):
            extract_json_object("")
