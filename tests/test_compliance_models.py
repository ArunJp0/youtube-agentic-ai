# Tests for Copyright/Compliance typed models (src/models/compliance.py).
from __future__ import annotations

from src.models.compliance import (
    DISCLAIMER,
    ComplianceResult,
    ProvenanceCheckResult,
    RequiredAttribution,
    SemanticReviewFinding,
    SemanticReviewResult,
)


class TestDisclaimer:
    def test_disclaimer_does_not_claim_legal_or_copyright_safety(self) -> None:
        lowered = DISCLAIMER.lower()
        assert "legally safe" not in lowered
        assert "copyright safe" not in lowered
        assert "not a legal" in lowered or "not legal" in lowered

    def test_compliance_result_carries_disclaimer_by_default(self) -> None:
        result = ComplianceResult(success=True, topic="t")
        assert result.disclaimer == DISCLAIMER


class TestProvenanceCheckResult:
    def test_construction(self) -> None:
        check = ProvenanceCheckResult(check_name="final_video_present", status="ok", detail="present")
        assert check.status == "ok"


class TestRequiredAttribution:
    def test_construction(self) -> None:
        attribution = RequiredAttribution(
            asset_type="bgm", asset_id="calm-music", source="YouTube Audio Library", attribution_text="Credit here"
        )
        assert attribution.asset_type == "bgm"


class TestSemanticReviewResult:
    def test_defaults(self) -> None:
        result = SemanticReviewResult(performed=False, fallback_reason="outage")
        assert result.findings == []
        assert result.summary == ""

    def test_with_findings(self) -> None:
        finding = SemanticReviewFinding(category="unsupported_claim", description="x")
        result = SemanticReviewResult(performed=True, findings=[finding])
        assert result.findings[0].severity == "medium"  # default


class TestComplianceResult:
    def test_defaults(self) -> None:
        result = ComplianceResult(success=True, topic="t")
        assert result.publish_decision is None
        assert result.checks == []
        assert result.warnings == []
        assert result.blockers == []
        assert result.required_attributions == []
        assert result.attribution_required is False
        assert result.llm_used is False

    def test_full_construction(self) -> None:
        result = ComplianceResult(
            success=True,
            topic="Why do humans dream?",
            publish_decision="PASS",
            risk_level="low",
            checks=[ProvenanceCheckResult(check_name="x", status="ok", detail="d")],
            warnings=[],
            blockers=[],
            attribution_required=False,
            required_attributions=[],
            provenance_summary="x: ok",
            semantic_review=SemanticReviewResult(performed=True, findings=[], summary="clean"),
            llm_used=True,
            artifacts_inspected=["final video: a.mp4"],
        )
        assert result.publish_decision == "PASS"
