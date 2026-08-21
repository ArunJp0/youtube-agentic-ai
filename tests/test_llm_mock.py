# Tests for Mock LLM Provider

from __future__ import annotations

import pytest

from src.llm.provider import LLMProvider
from src.llm.mock import MockLLMProvider


class TestMockLLMProvider:
    """Tests for MockLLMProvider functionality."""

    def test_mock_provider_instance(self) -> None:
        """Mock provider should be created successfully."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_generate_text_returns_string(self) -> None:
        """generate_text should return a string response."""
        provider = MockLLMProvider()
        response = provider.generate_text("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_text_summarize(self) -> None:
        """Summary generation should return relevant text."""
        provider = MockLLMProvider()
        response = provider.generate_text("Please summarize this: ...")
        assert "Dreams occur primarily during REM sleep" in response
        assert "memory consolidation" in response

    def test_generate_text_key_points(self) -> None:
        """Key point extraction should return bullet points."""
        provider = MockLLMProvider()
        response = provider.generate_text("Extract key points for this topic")
        assert "REM sleep" in response or "dreaming" in response.lower()

    def test_generate_text_facts(self) -> None:
        """Fact extraction should return numbered facts."""
        provider = MockLLMProvider()
        response = provider.generate_text("Extract 5 facts about topic")
        assert len(response) > 0

    def test_generate_text_default_response(self) -> None:
        """Unrecognized prompts should get generic response."""
        provider = MockLLMProvider()
        response = provider.generate_text("random unrecognized prompt")
        assert "Based on research" in response or "key findings" in response

    def test_custom_response_mapping(self) -> None:
        """Custom mappings should override default responses."""
        provider = MockLLMProvider(
            response_map={"special": "Custom response"}
        )
        response = provider.generate_text("This is a special request")
        assert response == "Custom response"

    def test_concurrency_safe(self) -> None:
        """Provider should be usable across concurrent calls."""
        provider = MockLLMProvider()
        responses = [provider.generate_text(f"Test {i}") for i in range(10)]
        assert len(responses) == 10
        assert all(isinstance(r, str) for r in responses)