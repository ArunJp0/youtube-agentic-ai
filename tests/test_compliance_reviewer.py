# Tests for ComplianceReviewer: one bounded semantic LLM call, structured
# JSON response parsing, and a conservative unperformed-review result on any
# failure. Uses fake LLMProvider test doubles only - no real Gemini calls.
from __future__ import annotations

import json

import pytest

from src.agents.compliance_reviewer import ComplianceReviewer, ComplianceReviewerError
from src.llm.provider import LLMProvider
from src.models.metadata import MetadataResult
from src.models.script import ScriptResult, ScriptSection
from src.models.thumbnail import ThumbnailPlan, ThumbnailResult


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[ScriptSection(heading="REM Sleep", narration="Dreams occur mainly during REM sleep.")],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _metadata(**overrides) -> MetadataResult:
    defaults = dict(success=True, topic="Why do humans dream?", title="Why Do We Dream?", description="A grounded look.")
    defaults.update(overrides)
    return MetadataResult(**defaults)


def _thumbnail(**overrides) -> ThumbnailResult:
    plan = ThumbnailPlan(
        hook_text="WHY DO WE DREAM",
        search_query="person sleeping",
        used_semantic_planning=True,
    )
    defaults = dict(success=True, topic="Why do humans dream?", plan=plan)
    defaults.update(overrides)
    return ThumbnailResult(**defaults)


def _valid_response(findings=None, summary="No issues found") -> str:
    return json.dumps({"findings": findings or [], "summary": summary})


class FakeLLMProvider(LLMProvider):
    """Test double returning a fixed canned response and counting calls."""

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


class TestComplianceReviewerSuccess:
    def test_valid_empty_findings_response(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.performed is True
        assert result.findings == []
        assert result.summary == "No issues found"
        assert result.fallback_reason is None

    def test_valid_findings_parsed(self) -> None:
        findings = [{"category": "title_content_mismatch", "description": "Title overstates the content", "severity": "high"}]
        llm = FakeLLMProvider(response=_valid_response(findings=findings))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.performed is True
        assert len(result.findings) == 1
        assert result.findings[0].category == "title_content_mismatch"
        assert result.findings[0].severity == "high"

    def test_exactly_one_llm_call(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())
        assert len(llm.calls) == 1

    def test_prompt_contains_topic_and_content(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        prompt = llm.calls[0]
        assert "Why do humans dream?" in prompt
        assert "SEMANTIC COMPLIANCE REVIEW" in prompt
        assert "Why Do We Dream?" in prompt

    def test_prompt_never_asks_if_copyright_safe(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        prompt = llm.calls[0].lower()
        assert "is this copyright safe" not in prompt
        assert "legal" in prompt  # explicitly told it is NOT a legal authority

    def test_missing_context_handled_gracefully(self) -> None:
        """No script/metadata/thumbnail available - reviewer still builds a
        valid prompt from the topic alone."""
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", None, None, None)

        assert result.performed is True
        assert "Why do humans dream?" in llm.calls[0]

    def test_severity_defaults_to_medium_when_missing(self) -> None:
        findings = [{"category": "unsupported_claim", "description": "Claim not supported by script"}]
        llm = FakeLLMProvider(response=_valid_response(findings=findings))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.findings[0].severity == "medium"

    def test_invalid_severity_normalized_to_medium(self) -> None:
        findings = [{"category": "unsupported_claim", "description": "Claim not supported", "severity": "extreme"}]
        llm = FakeLLMProvider(response=_valid_response(findings=findings))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.findings[0].severity == "medium"

    def test_malformed_finding_entries_skipped_not_fatal(self) -> None:
        findings = [
            {"category": "x", "description": "valid one"},
            {"description": "missing category"},
            "not even a dict",
        ]
        llm = FakeLLMProvider(response=_valid_response(findings=findings))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.performed is True
        assert len(result.findings) == 1
        assert result.findings[0].category == "x"


class TestComplianceReviewerFailure:
    def test_raised_exception_returns_unperformed_result(self) -> None:
        llm = FakeLLMProvider(raise_error=RuntimeError("simulated 503"))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.performed is False
        assert "simulated 503" in result.fallback_reason
        assert result.findings == []

    def test_timeout_style_error_falls_back(self) -> None:
        llm = FakeLLMProvider(raise_error=TimeoutError("timed out"))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())
        assert result.performed is False

    def test_malformed_json_falls_back(self) -> None:
        llm = FakeLLMProvider(response="not json at all")
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())

        assert result.performed is False
        assert result.findings == []

    def test_non_object_json_falls_back(self) -> None:
        llm = FakeLLMProvider(response=json.dumps(["not", "an", "object"]))
        reviewer = ComplianceReviewer(llm)
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())
        assert result.performed is False

    def test_never_raises_for_llm_failure(self) -> None:
        llm = FakeLLMProvider(raise_error=RuntimeError("simulated outage"))
        reviewer = ComplianceReviewer(llm)
        # Should not raise
        result = reviewer.review("Why do humans dream?", _script(), _metadata(), _thumbnail())
        assert isinstance(result.performed, bool)

    def test_missing_topic_raises_configuration_error(self) -> None:
        llm = FakeLLMProvider(response=_valid_response())
        reviewer = ComplianceReviewer(llm)
        with pytest.raises(ComplianceReviewerError):
            reviewer.review("", _script(), _metadata(), _thumbnail())
