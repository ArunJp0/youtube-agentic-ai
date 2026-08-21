# Tests for ResearchResult and ResearchFact models
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.research import ResearchResult, ResearchFact


class TestResearchFact:
    """Tests for the ResearchFact model."""

    def test_fact_creation_with_all_fields(self) -> None:
        """Fact can be created with all fields."""
        fact = ResearchFact(
            claim="Dreams occur during REM sleep",
            source="Harvard Medical School",
            confidence=0.9,
        )
        assert fact.claim == "Dreams occur during REM sleep"
        assert fact.source == "Harvard Medical School"
        assert fact.confidence == 0.9

    def test_fact_creation_with_minimal_fields(self) -> None:
        """Fact can be created with only the claim field."""
        fact = ResearchFact(claim="Test claim")
        assert fact.claim == "Test claim"
        assert fact.source is None
        assert fact.confidence == 0.5  # default

    def test_fact_confidence_must_be_between_0_and_1(self) -> None:
        """Confidence values outside 0.0-1.0 should be rejected."""
        with pytest.raises(ValidationError):
            ResearchFact(claim="Test", confidence=1.5)
        with pytest.raises(ValidationError):
            ResearchFact(claim="Test", confidence=-0.1)

    def test_fact_confidence_boundaries(self) -> None:
        """Confidence values at exact boundaries should be accepted."""
        low = ResearchFact(claim="Test", confidence=0.0)
        high = ResearchFact(claim="Test", confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_fact_serialization(self) -> None:
        """Fact can be serialized to dict."""
        fact = ResearchFact(claim="Test", confidence=0.7)
        data = fact.model_dump()
        assert data["claim"] == "Test"
        assert data["confidence"] == 0.7


class TestResearchResult:
    """Tests for the ResearchResult model."""

    def test_result_creation_with_minimal_fields(self) -> None:
        """Result can be created with only required fields."""
        result = ResearchResult(
            topic="Test Topic",
            summary="Test summary",
        )
        assert result.topic == "Test Topic"
        assert result.summary == "Test summary"
        assert result.key_points == []
        assert result.facts == []
        assert result.sources == []
        assert result.research_notes is None

    def test_result_creation_with_all_fields(self) -> None:
        """Result can be created with all fields populated."""
        result = ResearchResult(
            topic="Dreaming",
            summary="Research on dreams",
            key_points=["Point 1", "Point 2"],
            facts=[ResearchFact(claim="Fact 1")],
            sources=["https://example.com"],
            research_notes="Some notes",
        )
        assert len(result.key_points) == 2
        assert len(result.facts) == 1
        assert len(result.sources) == 1
        assert result.research_notes == "Some notes"

    def test_result_topic_cannot_be_empty(self) -> None:
        """Empty topic should be rejected."""
        with pytest.raises(ValidationError):
            ResearchResult(topic="", summary="Test")

    def test_result_summary_cannot_be_empty(self) -> None:
        """Empty summary should be rejected."""
        with pytest.raises(ValidationError):
            ResearchResult(topic="Test", summary="")

    def test_result_key_points_default_factory(self) -> None:
        """Default key_points should be a fresh list per instance."""
        r1 = ResearchResult(topic="A", summary="A")
        r2 = ResearchResult(topic="B", summary="B")
        r1.key_points.append("shared")
        assert r2.key_points == []  # No mutation across instances

    def test_result_fact_default_factory(self) -> None:
        """Default facts should be a fresh list per instance."""
        r1 = ResearchResult(topic="A", summary="A")
        r2 = ResearchResult(topic="B", summary="B")
        r1.facts.append(ResearchFact(claim="Test"))
        assert r2.facts == []

    def test_result_serialization(self) -> None:
        """Result can be serialized to JSON."""
        result = ResearchResult(
            topic="Test",
            summary="Summary",
            facts=[ResearchFact(claim="Fact", confidence=0.8)],
        )
        data = result.model_dump()
        assert data["topic"] == "Test"
        assert data["facts"][0]["confidence"] == 0.8

    def test_result_json_roundtrip(self) -> None:
        """Result can be serialized to JSON and deserialized."""
        result = ResearchResult(
            topic="Test",
            summary="Summary",
            facts=[ResearchFact(claim="Fact", confidence=0.8)],
        )
        json_str = result.model_dump_json()
        restored = ResearchResult.model_validate_json(json_str)
        assert restored.topic == result.topic
        assert restored.facts[0].confidence == 0.8