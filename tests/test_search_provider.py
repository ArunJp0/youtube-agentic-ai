# Tests for SearchProvider abstraction

from __future__ import annotations

import asyncio

import pytest

from src.tools.search_provider import SearchProvider, MockSearchProvider, search_raw


class TestMockSearchProvider:
    """Tests for MockSearchProvider functionality."""

    def test_mock_provider_instance(self):
        """Mock provider should be created successfully."""
        provider = MockSearchProvider()
        assert isinstance(provider, SearchProvider)

    def test_mock_search_returns_results(self):
        """Mock provider should return valid search results."""
        results = asyncio.run(search_raw("How does photosynthesis work?"))
        assert isinstance(results, list)
        assert len(results) >= 2  # Mock data has 2 results for photosynthesis
        assert isinstance(results[0], dict)
        assert "title" in results[0]


class TestSearchRawFunction:
    """Tests for the convenience search_raw function."""

    def test_search_raw_returns_list(self):
        """search_raw should return a list of results."""
        results = asyncio.run(search_raw("What causes earthquakes?"))
        assert isinstance(results, list)

    def test_search_raw_unknown_query(self):
        """search_raw should return mock results for unknown queries."""
        results = asyncio.run(search_raw("unknown_query_xyz123"))
        assert len(results) >= 2
        assert results[0]["title"].startswith("Mock result")
