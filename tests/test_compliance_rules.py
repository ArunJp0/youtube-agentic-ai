# Tests for centralized publish-decision rules
# (src/services/compliance_rules.py). Pure logic, no LLM/network involved.
from __future__ import annotations

from src.models.compliance import SemanticReviewFinding, SemanticReviewResult
from src.services.compliance_rules import decide_publish_status

_CLEAN_REVIEW = SemanticReviewResult(performed=True, findings=[], summary="No issues found")
_UNPERFORMED_REVIEW = SemanticReviewResult(performed=False, fallback_reason="simulated outage")
_FINDING_REVIEW = SemanticReviewResult(
    performed=True,
    findings=[SemanticReviewFinding(category="title_content_mismatch", description="Title overstates content")],
    summary="One finding",
)


class TestDecidePublishStatus:
    def test_no_blockers_no_warnings_clean_review_passes(self) -> None:
        decision, risk = decide_publish_status([], [], _CLEAN_REVIEW)
        assert decision == "PASS"
        assert risk == "low"

    def test_blocker_forces_block_regardless_of_review(self) -> None:
        decision, risk = decide_publish_status(["missing catalog track"], [], _CLEAN_REVIEW)
        assert decision == "BLOCK"
        assert risk == "high"

    def test_blocker_overrides_clean_positive_llm_result(self) -> None:
        """A deterministic blocker must win even when the semantic review
        found nothing wrong - blockers are never overridden by the LLM."""
        decision, _ = decide_publish_status(["selected music is not in the approved catalog"], [], _CLEAN_REVIEW)
        assert decision == "BLOCK"

    def test_unperformed_review_never_passes(self) -> None:
        decision, risk = decide_publish_status([], [], _UNPERFORMED_REVIEW)
        assert decision == "REVIEW"
        assert risk == "medium"

    def test_none_review_never_passes(self) -> None:
        decision, _ = decide_publish_status([], [], None)
        assert decision == "REVIEW"

    def test_semantic_finding_alone_is_review_not_block(self) -> None:
        decision, risk = decide_publish_status([], [], _FINDING_REVIEW)
        assert decision == "REVIEW"
        assert risk == "medium"

    def test_warning_alone_forces_review(self) -> None:
        decision, risk = decide_publish_status([], ["visual provenance unavailable"], _CLEAN_REVIEW)
        assert decision == "REVIEW"
        assert risk == "medium"

    def test_multiple_warnings_still_review_not_block(self) -> None:
        decision, _ = decide_publish_status(
            [], ["visual provenance unavailable", "bgm provenance unavailable", "topic mismatch"], _CLEAN_REVIEW
        )
        assert decision == "REVIEW"

    def test_blockers_take_priority_over_unperformed_review(self) -> None:
        decision, _ = decide_publish_status(["missing thumbnail"], [], _UNPERFORMED_REVIEW)
        assert decision == "BLOCK"
