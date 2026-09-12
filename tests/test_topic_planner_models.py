# Tests for Topic Planner typed models (src/models/topic_planner.py).
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.topic_planner import (
    SemanticTopicJudgment,
    TopicCandidate,
    TopicRankingOutcome,
    TopicScore,
    TopicSelectionResult,
)


class TestTopicCandidate:
    def test_valid_construction(self) -> None:
        c = TopicCandidate(raw_title="Why do cats purr?", normalized_title="why do cats purr", source="mock")
        assert c.source_id is None
        assert c.popularity_signal is None

    def test_popularity_signal_bounded(self) -> None:
        with pytest.raises(ValidationError):
            TopicCandidate(raw_title="x", normalized_title="x", source="mock", popularity_signal=1.5)

    def test_empty_raw_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopicCandidate(raw_title="", normalized_title="x", source="mock")


class TestTopicScore:
    def test_valid_construction(self) -> None:
        candidate = TopicCandidate(raw_title="x", normalized_title="x", source="mock")
        score = TopicScore(
            candidate=candidate,
            relevance_score=0.8,
            novelty_score=1.0,
            evergreen_score=0.7,
            suitability_score=0.9,
            popularity_score=0.5,
            total_score=0.8,
        )
        assert score.rationale == ""

    def test_scores_bounded_zero_to_one(self) -> None:
        candidate = TopicCandidate(raw_title="x", normalized_title="x", source="mock")
        with pytest.raises(ValidationError):
            TopicScore(
                candidate=candidate,
                relevance_score=1.5,
                novelty_score=1.0,
                evergreen_score=0.7,
                suitability_score=0.9,
                popularity_score=0.5,
                total_score=0.8,
            )


class TestTopicSelectionResult:
    def test_failure_result_has_no_selected_topic(self) -> None:
        result = TopicSelectionResult(success=False, status="no_candidates")
        assert result.selected_topic is None

    def test_success_result_round_trips(self) -> None:
        result = TopicSelectionResult(
            success=True, status="selected", selected_topic="Why do cats purr?", score=0.8
        )
        dumped = result.model_dump(mode="json")
        restored = TopicSelectionResult.model_validate(dumped)
        assert restored == result

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopicSelectionResult(success=False, status="not_a_real_status")


class TestTopicRankingOutcome:
    def test_unperformed_outcome_still_has_judgments(self) -> None:
        judgment = SemanticTopicJudgment(
            normalized_title="x", relevance_score=0.5, evergreen_score=0.5, suitability_score=0.5
        )
        outcome = TopicRankingOutcome(performed=False, judgments=[judgment], fallback_reason="LLM outage")
        assert outcome.performed is False
        assert len(outcome.judgments) == 1
