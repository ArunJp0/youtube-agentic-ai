# Tests for ComplianceAgent (deterministic checks -> optional semantic
# review -> centralized decision rules orchestration). Uses fake LLM/
# catalog test doubles and local Pillow-generated images only - no real
# Gemini/Pexels/network calls.
from __future__ import annotations

import json

import pytest
from PIL import Image

from src.agents.compliance_agent import ComplianceAgent, ComplianceAgentError
from src.llm.provider import LLMProvider
from src.models.metadata import MetadataResult
from src.models.music import BGMTrack
from src.models.provenance import BGMProvenance, ProvenanceManifest, ThumbnailProvenance, VisualAssetProvenance
from src.models.script import ScriptResult, ScriptSection
from src.tools.music_catalog_provider import MockMusicCatalogProvider


# ---- fixtures/helpers -------------------------------------------------------


def _video(tmp_path, name="video.mp4", empty=False) -> str:
    path = tmp_path / name
    path.write_bytes(b"" if empty else b"FAKE MP4 BYTES")
    return str(path)


def _thumbnail_image(tmp_path, name="thumb.jpg", size=(1280, 720)) -> str:
    path = tmp_path / name
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return str(path)


def _metadata(**overrides) -> MetadataResult:
    defaults = dict(
        success=True,
        topic="Why do humans dream?",
        title="Why Do We Dream?",
        description="A grounded look at the science of dreaming.",
        output_path=None,
    )
    defaults.update(overrides)
    return MetadataResult(**defaults)


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


def _visual_asset_provenance(provider="pexels", source_url="https://pexels.com/1") -> VisualAssetProvenance:
    return VisualAssetProvenance(section_index=0, section_heading="S", provider=provider, source_url=source_url)


def _thumbnail_provenance(**overrides) -> ThumbnailProvenance:
    defaults = dict(provider="pexels", provider_asset_id="1", source_url="https://pexels.com/1", attribution="Author")
    defaults.update(overrides)
    return ThumbnailProvenance(**defaults)


def _bgm_track(**overrides) -> BGMTrack:
    defaults = dict(
        track_id="calm-music",
        file_path="tracks/calm.mp3",
        title="Calm Music",
        source="YouTube Audio Library",
        license_type="youtube_audio_library_no_attribution",
        attribution_required=False,
        attribution_text=None,
    )
    defaults.update(overrides)
    return BGMTrack(**defaults)


def _manifest_bgm_from_track(track: BGMTrack) -> BGMProvenance:
    return BGMProvenance(
        track_id=track.track_id,
        title=track.title,
        source=track.source,
        license_type=track.license_type,
        attribution_required=track.attribution_required,
        attribution_text=track.attribution_text,
    )


def _manifest(video_path: str, bgm_track: BGMTrack = None, include_visual=True, include_thumbnail=True) -> ProvenanceManifest:
    return ProvenanceManifest(
        run_id="run-1",
        topic="Why do humans dream?",
        final_video_path=video_path,
        created_at="2026-01-01T00:00:00+00:00",
        visual_assets=[_visual_asset_provenance()] if include_visual else [],
        thumbnail=_thumbnail_provenance() if include_thumbnail else None,
        bgm=_manifest_bgm_from_track(bgm_track) if bgm_track is not None else None,
    )


def _valid_semantic_response(findings=None, summary="No issues found") -> str:
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


# ---- tests -------------------------------------------------------------------


class TestFullyCompliantPackage:
    def test_known_good_package_with_manifest_passes(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            script=_script(),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )

        assert result.success is True
        assert result.publish_decision == "PASS"
        assert result.risk_level == "low"
        assert result.blockers == []
        assert result.warnings == []
        assert "not a legal" in result.disclaimer.lower()


class TestVisualProvenance:
    def test_missing_visual_provenance_is_review(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track, include_visual=False),
        )
        assert result.publish_decision == "REVIEW"
        assert result.blockers == []
        assert any("visual" in w.lower() for w in result.warnings)

    def test_unknown_visual_provider_blocks(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        manifest = _manifest(video_path, bgm_track=track)
        manifest = manifest.model_copy(
            update={"visual_assets": [_visual_asset_provenance(provider="random-scraper", source_url="https://x.com")]}
        )
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=manifest,
        )
        assert result.publish_decision == "BLOCK"
        assert result.risk_level == "high"
        assert any("provider" in b.lower() for b in result.blockers)


class TestThumbnailProvenance:
    def test_missing_thumbnail_provenance_is_review(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track, include_thumbnail=False),
        )
        assert result.publish_decision == "REVIEW"
        assert any("thumbnail" in w.lower() for w in result.warnings)


class TestBgmProvenance:
    def test_approved_catalog_track_ok(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.attribution_required is False
        assert result.required_attributions == []
        assert result.publish_decision == "PASS"

    def test_selected_bgm_missing_from_catalog_blocks(self, tmp_path) -> None:
        selected = _bgm_track(track_id="unapproved-track")
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[_bgm_track()]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=selected),
        )
        assert result.publish_decision == "BLOCK"
        assert result.risk_level == "high"
        assert any("approved catalog" in b.lower() for b in result.blockers)

    def test_attribution_required_with_attribution_available(self, tmp_path) -> None:
        track = _bgm_track(attribution_required=True, attribution_text="Music by Example Artist")
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.publish_decision == "PASS"
        assert result.attribution_required is True
        assert len(result.required_attributions) == 1
        assert result.required_attributions[0].attribution_text == "Music by Example Artist"
        assert result.required_attributions[0].asset_type == "bgm"

    def test_attribution_required_but_missing_blocks(self, tmp_path) -> None:
        track = _bgm_track(attribution_required=True, attribution_text=None)
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.publish_decision == "BLOCK"
        assert result.attribution_required is True
        assert result.required_attributions == []

    def test_manifest_catalog_disagreement_surfaces_as_warning(self, tmp_path) -> None:
        """The manifest recorded no-attribution at run time, but the CURRENT
        catalog now requires it - the catalog wins, and the disagreement is
        surfaced (never silently resolved either way)."""
        video_path = _video(tmp_path)
        stale_manifest_bgm = BGMProvenance(
            track_id="calm-music", title="Calm Music", source="YouTube Audio Library",
            license_type="youtube_audio_library_no_attribution", attribution_required=False, attribution_text=None,
        )
        current_catalog_track = _bgm_track(attribution_required=True, attribution_text="Credit required now")
        manifest = ProvenanceManifest(
            run_id="run-1", topic="Why do humans dream?", final_video_path=video_path,
            created_at="2026-01-01T00:00:00+00:00",
            visual_assets=[_visual_asset_provenance()], thumbnail=_thumbnail_provenance(), bgm=stale_manifest_bgm,
        )
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[current_catalog_track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=manifest,
        )
        assert result.attribution_required is True
        assert result.required_attributions[0].attribution_text == "Credit required now"
        assert result.publish_decision == "REVIEW"
        assert any("disagreement" in w.lower() for w in result.warnings)


class TestMissingArtifacts:
    def test_missing_final_video_blocks(self, tmp_path) -> None:
        track = _bgm_track()
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=None,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision == "BLOCK"

    def test_missing_thumbnail_blocks(self, tmp_path) -> None:
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=None,
        )
        assert result.publish_decision == "BLOCK"

    def test_missing_metadata_blocks(self, tmp_path) -> None:
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=None,
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision == "BLOCK"


class TestLegacyArtifactWithoutManifest:
    def test_no_manifest_at_all_never_invents_provenance(self, tmp_path) -> None:
        """A legacy artifact with no ProvenanceManifest must never be
        treated as having clean provenance - the standalone agent must
        never infer/fabricate it, only report REVIEW."""
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[_bgm_track()]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=None,
        )
        assert result.publish_decision == "REVIEW"
        assert result.publish_decision != "PASS"
        visual_check = next(c for c in result.checks if c.check_name == "visual_provenance")
        bgm_check = next(c for c in result.checks if c.check_name == "bgm_provenance")
        thumb_prov_check = next(c for c in result.checks if c.check_name == "thumbnail_provenance")
        assert visual_check.status == "unavailable"
        assert bgm_check.status == "unavailable"
        assert thumb_prov_check.status == "unavailable"
        assert result.attribution_required is False
        assert result.required_attributions == []


class TestSemanticReviewFindingsCauseReview:
    def test_title_content_mismatch_is_review(self, tmp_path) -> None:
        findings = [{"category": "title_content_mismatch", "description": "Title claims something the script doesn't cover"}]
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response(findings=findings)),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            script=_script(),
        )
        assert result.publish_decision == "REVIEW"
        assert result.blockers == []

    def test_misleading_thumbnail_content_mismatch_is_review(self, tmp_path) -> None:
        findings = [{"category": "thumbnail_content_mismatch", "description": "Thumbnail hook implies unrelated content"}]
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response(findings=findings)),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision == "REVIEW"

    def test_unsupported_strong_claim_is_review(self, tmp_path) -> None:
        findings = [{"category": "unsupported_claim", "description": "Description claims a cure with no support", "severity": "high"}]
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response(findings=findings)),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision == "REVIEW"
        assert result.semantic_review.findings[0].severity == "high"
        # A high-severity semantic finding never escalates to BLOCK on its own.
        assert result.blockers == []


class TestLlmFailureBehavior:
    def test_llm_failure_preserves_deterministic_checks_and_returns_review(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(raise_error=RuntimeError("simulated 503")),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.publish_decision == "REVIEW"
        # Deterministic checks still ran and are reported.
        assert any(c.check_name == "final_video_present" and c.status == "ok" for c in result.checks)
        assert any(c.check_name == "bgm_provenance" and c.status == "ok" for c in result.checks)
        assert result.llm_used is False

    def test_malformed_llm_response_returns_review_not_pass(self, tmp_path) -> None:
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response="not valid json"),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision != "PASS"
        assert result.publish_decision == "REVIEW"

    def test_no_fake_pass_after_llm_failure_even_with_clean_deterministic_checks(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(raise_error=TimeoutError("timed out")),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.publish_decision != "PASS"

    def test_no_llm_provider_configured_also_returns_review(self, tmp_path) -> None:
        agent = ComplianceAgent(music_catalog_provider=MockMusicCatalogProvider(tracks=[]), llm_provider=None)
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
        )
        assert result.publish_decision == "REVIEW"
        assert result.llm_used is False


class TestDeterministicBlockerOverridesLlm:
    def test_blocker_overrides_clean_semantic_review(self, tmp_path) -> None:
        """Even when the LLM finds nothing wrong, a deterministic blocker
        (missing thumbnail) must still force BLOCK."""
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=_metadata(),
            thumbnail_path=None,
        )
        assert result.semantic_review.performed is True
        assert result.semantic_review.findings == []
        assert result.publish_decision == "BLOCK"

    def test_blocker_overrides_manifest_backed_clean_review(self, tmp_path) -> None:
        """Same guarantee even with a full, otherwise-clean provenance
        manifest present - a deterministic BGM blocker still wins."""
        selected = _bgm_track(track_id="unapproved-track")
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[_bgm_track()]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=selected),
        )
        assert result.semantic_review.performed is True
        assert result.semantic_review.findings == []
        assert result.publish_decision == "BLOCK"


class TestWarningsAggregation:
    def test_multiple_warnings_all_reported(self, tmp_path) -> None:
        metadata = _metadata(topic="A different topic entirely")
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=_video(tmp_path),
            metadata_result=metadata,
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=None,
        )
        # topic mismatch + visual provenance unavailable + thumbnail provenance
        # unavailable + bgm provenance unavailable
        assert len(result.warnings) >= 3
        assert result.publish_decision == "REVIEW"


class TestRequiredAttributionsAreMachineReadable:
    def test_required_attributions_are_structured_objects(self, tmp_path) -> None:
        track = _bgm_track(attribution_required=True, attribution_text="Credit: Example")
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        attribution = result.required_attributions[0]
        assert attribution.asset_id == "calm-music"
        assert attribution.source == "YouTube Audio Library"
        assert attribution.attribution_text == "Credit: Example"
        # Confirm it round-trips as structured data, not just a string.
        dumped = attribution.model_dump()
        assert dumped["asset_type"] == "bgm"


class TestDisclaimerWording:
    def test_pass_result_disclaimer_does_not_claim_guarantee(self, tmp_path) -> None:
        track = _bgm_track()
        video_path = _video(tmp_path)
        agent = ComplianceAgent(
            music_catalog_provider=MockMusicCatalogProvider(tracks=[track]),
            llm_provider=FakeLLMProvider(response=_valid_semantic_response()),
        )
        result = agent.review_compliance(
            "Why do humans dream?",
            final_video_path=video_path,
            metadata_result=_metadata(),
            thumbnail_path=_thumbnail_image(tmp_path),
            provenance_manifest=_manifest(video_path, bgm_track=track),
        )
        assert result.publish_decision == "PASS"
        lowered = result.disclaimer.lower()
        assert "legally safe" not in lowered
        assert "copyright safe" not in lowered
        assert "guarantee" in lowered


class TestConfigurationErrors:
    def test_missing_topic_raises(self, tmp_path) -> None:
        agent = ComplianceAgent(music_catalog_provider=MockMusicCatalogProvider(tracks=[]))
        with pytest.raises(ComplianceAgentError):
            agent.review_compliance("", final_video_path=_video(tmp_path), metadata_result=_metadata(), thumbnail_path=None)
