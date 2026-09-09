# Tests for the Finding Localizer (src/services/finding_localizer.py):
# deterministic mapping of Compliance semantic findings to the specific
# ScriptSection they concern, for Compliance Remediation's targeted
# correction. No LLM/network calls - pure functions only.
from __future__ import annotations

from src.models.compliance import SemanticReviewFinding
from src.models.research import ResearchFact, ResearchResult
from src.models.script import ScriptResult, ScriptSection
from src.services.finding_localizer import has_grounding_evidence, localize_findings


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why is the sky blue?",
        video_title="Why Is The Sky Blue?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(heading="Rayleigh Effect", narration="Blue light bends more than red light in the air."),
            ScriptSection(
                heading="Atmospheric Effects on the Moon",
                narration="The moon can appear to have a blue tinge from atmospheric scattering.",
            ),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _finding(**overrides) -> SemanticReviewFinding:
    defaults = dict(category="factual_consistency_error", description="A generic finding.")
    defaults.update(overrides)
    return SemanticReviewFinding(**defaults)


class TestLlmSuppliedLocalization:
    def test_exact_heading_match_is_localized(self) -> None:
        finding = _finding(
            description="The moon 'blue tinge' claim is wrong.",
            related_section_heading="Atmospheric Effects on the Moon",
        )
        result = localize_findings([finding], _script())[0]

        assert result.actionable is True
        assert result.section_index == 1
        assert result.section_heading == "Atmospheric Effects on the Moon"
        assert result.localization_source == "llm"

    def test_case_insensitive_heading_match(self) -> None:
        finding = _finding(related_section_heading="atmospheric effects on the moon")
        result = localize_findings([finding], _script())[0]

        assert result.actionable is True
        assert result.section_index == 1

    def test_llm_supplied_heading_not_found_falls_through_to_fallback(self) -> None:
        """A related_section_heading that doesn't match any real section
        (e.g. the LLM paraphrased it) must not be trusted blindly - it
        falls through to the deterministic keyword fallback instead."""
        finding = _finding(
            description="atmospheric scattering moon tinge claim",
            related_section_heading="Some Heading That Does Not Exist",
        )
        result = localize_findings([finding], _script())[0]

        # Falls back to keyword overlap, which does match section 1.
        assert result.localization_source == "fallback_match"
        assert result.section_index == 1


class TestDeterministicFallbackLocalization:
    def test_keyword_overlap_localizes_without_llm_reference(self) -> None:
        finding = _finding(
            category="factual_consistency_error",
            description="The script claims atmospheric scattering gives the moon a blue tinge, which is wrong.",
        )
        result = localize_findings([finding], _script())[0]

        assert result.actionable is True
        assert result.section_index == 1
        assert result.localization_source == "fallback_match"

    def test_single_shared_word_is_not_enough(self) -> None:
        """One overlapping word is too weak a signal to act on, even
        though it only overlaps a single section."""
        finding = _finding(description="Something about scattering is slightly off.")
        result = localize_findings([finding], _script())[0]

        assert result.actionable is False


class TestNoConfidentLocalization:
    def test_unrelated_finding_is_not_actionable(self) -> None:
        finding = _finding(description="Claim not fully supported by the script")
        result = localize_findings([finding], _script())[0]

        assert result.actionable is False
        assert result.section_index is None
        assert result.section_heading is None

    def test_ambiguous_match_across_multiple_sections_is_not_actionable(self) -> None:
        script = _script(
            sections=[
                ScriptSection(heading="A", narration="atmospheric scattering blue light physics explained"),
                ScriptSection(heading="B", narration="atmospheric scattering blue light physics discussed"),
            ]
        )
        finding = _finding(description="atmospheric scattering blue light physics claim is wrong")
        result = localize_findings([finding], script)[0]

        assert result.actionable is False

    def test_no_script_available_is_not_actionable(self) -> None:
        finding = _finding(description="atmospheric scattering moon claim")
        result = localize_findings([finding], None)[0]

        assert result.actionable is False

    def test_no_sections_is_not_actionable(self) -> None:
        script = _script(sections=[])
        finding = _finding(description="atmospheric scattering moon claim")
        result = localize_findings([finding], script)[0]

        assert result.actionable is False

    def test_multiple_findings_localized_independently(self) -> None:
        findings = [
            _finding(description="atmospheric scattering moon blue tinge claim is wrong"),
            _finding(description="Claim not fully supported by the script"),
        ]
        results = localize_findings(findings, _script())

        assert results[0].actionable is True
        assert results[1].actionable is False


class TestHasGroundingEvidence:
    def _research(self, **overrides) -> ResearchResult:
        defaults = dict(
            topic="Why is the sky blue?",
            summary="Rayleigh scattering explains the sky's blue color.",
            key_points=["Blue light scatters more than red light"],
            facts=[ResearchFact(claim="Atmospheric scattering reddens the moon during eclipses", confidence=0.8)],
        )
        defaults.update(overrides)
        return ResearchResult(**defaults)

    def test_covered_claim_has_grounding_evidence(self) -> None:
        research = self._research()
        assert has_grounding_evidence(research, "The moon's atmospheric scattering claim is wrong") is True

    def test_uncovered_claim_lacks_grounding_evidence(self) -> None:
        research = self._research()
        assert has_grounding_evidence(research, "The video's title overstates the content") is False

    def test_no_research_lacks_grounding_evidence(self) -> None:
        assert has_grounding_evidence(None, "atmospheric scattering moon claim") is False
