# Tests for TopicRankingPlanner: one bounded semantic LLM call, structured
# JSON response parsing, and a conservative neutral-fallback outcome on any
# failure. Uses fake LLMProvider test doubles only - no real Gemini calls.
from __future__ import annotations

import json

import pytest

from src.agents.topic_ranking_planner import TopicRankingPlanner, TopicRankingPlannerError
from src.llm.provider import LLMProvider
from src.models.topic_planner import TopicCandidate


def _candidate(title: str, category: str = "education") -> TopicCandidate:
    from src.services.topic_normalization import normalize_topic

    return TopicCandidate(raw_title=title, normalized_title=normalize_topic(title), source="mock", category=category)


def _valid_response(candidates, **overrides) -> str:
    judgments = [
        {
            "normalized_title": c.normalized_title,
            "relevance_score": 0.8,
            "evergreen_score": 0.9,
            "suitability_score": 0.7,
            "rationale": "Fits the niche well.",
        }
        for c in candidates
    ]
    payload = {"judgments": judgments}
    payload.update(overrides)
    return json.dumps(payload)


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: str = "", raise_error: Exception | None = None) -> None:
        self.response = response
        self.raise_error = raise_error
        self.calls: list[str] = []
        self.name = "fake-llm"
        self.last_model_used = "fake-model"
        self.last_used_fallback = False

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_error:
            raise self.raise_error
        return self.response


class TestSuccessfulRanking:
    def test_valid_response_parsed(self) -> None:
        candidates = [_candidate("Why do cats purr?"), _candidate("How do vaccines work?")]
        llm = FakeLLMProvider(response=_valid_response(candidates))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert outcome.performed is True
        assert len(outcome.judgments) == 2
        assert outcome.judgments[0].relevance_score == 0.8

    def test_exactly_one_llm_call_regardless_of_candidate_count(self) -> None:
        candidates = [_candidate(f"Topic {i}") for i in range(8)]
        llm = FakeLLMProvider(response=_valid_response(candidates))
        planner = TopicRankingPlanner(llm)

        planner.rank(candidates)

        assert len(llm.calls) == 1

    def test_prompt_mentions_niche_when_configured(self) -> None:
        candidates = [_candidate("Why do cats purr?")]
        llm = FakeLLMProvider(response=_valid_response(candidates))
        planner = TopicRankingPlanner(llm)

        planner.rank(candidates, niche="science")

        assert "science" in llm.calls[0]

    def test_one_judgment_per_candidate_always(self) -> None:
        candidates = [_candidate("A"), _candidate("B"), _candidate("C")]
        llm = FakeLLMProvider(response=_valid_response(candidates))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert {j.normalized_title for j in outcome.judgments} == {c.normalized_title for c in candidates}

    def test_partial_response_missing_one_candidate_gets_neutral_fallback(self) -> None:
        candidates = [_candidate("Topic A"), _candidate("Topic B")]
        # Only judge the first candidate - the response is otherwise valid.
        payload = {
            "judgments": [
                {
                    "normalized_title": candidates[0].normalized_title,
                    "relevance_score": 0.9,
                    "evergreen_score": 0.9,
                    "suitability_score": 0.9,
                    "rationale": "great",
                }
            ]
        }
        llm = FakeLLMProvider(response=json.dumps(payload))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert len(outcome.judgments) == 2
        second = next(j for j in outcome.judgments if j.normalized_title == candidates[1].normalized_title)
        assert second.relevance_score == 0.5


class TestRankingFailureFallback:
    def test_raised_exception_returns_unperformed_with_neutral_judgments(self) -> None:
        candidates = [_candidate("Topic A"), _candidate("Topic B")]
        llm = FakeLLMProvider(raise_error=RuntimeError("simulated 503"))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert outcome.performed is False
        assert "simulated 503" in outcome.fallback_reason
        assert len(outcome.judgments) == 2
        assert all(j.relevance_score == 0.5 for j in outcome.judgments)

    def test_malformed_json_falls_back(self) -> None:
        candidates = [_candidate("Topic A")]
        llm = FakeLLMProvider(response="not json at all")
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert outcome.performed is False
        assert len(outcome.judgments) == 1

    def test_non_object_json_falls_back(self) -> None:
        candidates = [_candidate("Topic A")]
        llm = FakeLLMProvider(response=json.dumps(["not", "an", "object"]))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert outcome.performed is False

    def test_never_raises_for_llm_failure(self) -> None:
        candidates = [_candidate("Topic A")]
        llm = FakeLLMProvider(raise_error=RuntimeError("simulated outage"))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)  # should not raise
        assert isinstance(outcome.performed, bool)

    def test_out_of_range_score_clamped(self) -> None:
        candidates = [_candidate("Topic A")]
        payload = {
            "judgments": [
                {
                    "normalized_title": candidates[0].normalized_title,
                    "relevance_score": 5.0,
                    "evergreen_score": -1.0,
                    "suitability_score": 0.5,
                    "rationale": "x",
                }
            ]
        }
        llm = FakeLLMProvider(response=json.dumps(payload))
        planner = TopicRankingPlanner(llm)

        outcome = planner.rank(candidates)

        assert outcome.judgments[0].relevance_score == 1.0
        assert outcome.judgments[0].evergreen_score == 0.0


class TestConfigurationErrors:
    def test_empty_candidates_raises(self) -> None:
        llm = FakeLLMProvider(response="{}")
        planner = TopicRankingPlanner(llm)
        with pytest.raises(TopicRankingPlannerError):
            planner.rank([])
