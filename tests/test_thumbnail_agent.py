# Tests for ThumbnailAgent (plan -> select -> download -> render ->
# validate orchestration). Uses fake LLM/media-provider test doubles and
# real Pillow operations on locally-generated images only - no real
# Gemini/Pexels/network calls.
from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from src.agents.thumbnail_agent import ThumbnailAgent, ThumbnailAgentError
from src.llm.provider import LLMProvider
from src.models.script import ScriptResult, ScriptSection
from src.models.thumbnail import ThumbnailResult
from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError


def _script(**overrides) -> ScriptResult:
    defaults = dict(
        topic="Why do humans dream?",
        video_title="Why Do We Dream?",
        hook="HOOK",
        introduction="INTRO",
        sections=[
            ScriptSection(heading="REM Sleep", narration="Dreams occur mainly during REM sleep."),
            ScriptSection(heading="Memory", narration="The brain consolidates memories while dreaming."),
        ],
        conclusion="CONCLUSION",
        call_to_action="CTA",
    )
    defaults.update(overrides)
    return ScriptResult(**defaults)


def _valid_plan_response(**overrides) -> str:
    payload = {
        "hook_text": "WHY DO WE DREAM?",
        "visual_concept": "A person sleeping with abstract dream imagery",
        "search_query": "person sleeping peacefully",
        "mood": "curious",
        "subject": "a sleeping person",
        "composition": "subject_left",
        "text_position": "right",
        "avoid_concepts": ["nightmare horror imagery"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _candidate(content_hint="person sleeping in bed", asset_id="101", url_suffix=".jpg") -> MediaCandidate:
    return MediaCandidate(
        asset_type="image",
        download_url=f"https://mock.media/{asset_id}{url_suffix}",
        source_url=f"https://mock.media/page/{asset_id}",
        provider_asset_id=asset_id,
        attribution="Mock Photographer",
        width=1920,
        height=1080,
        content_hint=content_hint,
    )


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


class FakeMediaProvider(MediaProvider):
    """Test double: records calls, returns configured candidates, and
    writes a real (or deliberately corrupt) local image on download - no
    network involved."""

    def __init__(
        self,
        candidates: list | None = None,
        search_error: Exception | None = None,
        download_error: Exception | None = None,
        write_valid_image: bool = True,
        image_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        self.candidates = candidates if candidates is not None else [_candidate()]
        self.search_error = search_error
        self.download_error = download_error
        self.write_valid_image = write_valid_image
        self.image_size = image_size
        self.search_calls: list[dict] = []
        self.download_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fake-media"

    async def search(self, query: str, prefer_video: bool = True, max_results: int = 5):
        self.search_calls.append({"query": query, "prefer_video": prefer_video, "max_results": max_results})
        if self.search_error:
            raise self.search_error
        return self.candidates

    async def download(self, candidate: MediaCandidate, output_path: str) -> None:
        self.download_calls.append({"candidate": candidate, "output_path": output_path})
        if self.download_error:
            raise self.download_error
        if self.write_valid_image:
            Image.new("RGB", self.image_size, (60, 90, 140)).save(output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(b"not a real image")


class TestThumbnailAgentValidation:
    @pytest.mark.asyncio
    async def test_missing_topic_raises(self, tmp_path) -> None:
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(ThumbnailAgentError, match="required"):
            await agent.generate_thumbnail("", _script())

    @pytest.mark.asyncio
    async def test_missing_script_raises(self, tmp_path) -> None:
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), output_dir=str(tmp_path))
        with pytest.raises(ThumbnailAgentError, match="required"):
            await agent.generate_thumbnail("dreams", None)


class TestThumbnailAgentOneCallMaximum:
    @pytest.mark.asyncio
    async def test_exactly_one_llm_call(self, tmp_path) -> None:
        llm = FakeLLMProvider(response=_valid_plan_response())
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), llm_provider=llm, output_dir=str(tmp_path))

        await agent.generate_thumbnail("Why do humans dream?", _script())

        assert len(llm.calls) == 1


class TestThumbnailAgentMediaProviderReuse:
    @pytest.mark.asyncio
    async def test_search_requests_photos_not_video(self, tmp_path) -> None:
        media = FakeMediaProvider()
        agent = ThumbnailAgent(media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path))

        await agent.generate_thumbnail("dreams", _script())

        assert media.search_calls[0]["prefer_video"] is False

    @pytest.mark.asyncio
    async def test_search_uses_plan_search_query(self, tmp_path) -> None:
        media = FakeMediaProvider()
        agent = ThumbnailAgent(media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path))

        await agent.generate_thumbnail("dreams", _script())

        assert media.search_calls[0]["query"] == "person sleeping peacefully"


class TestThumbnailAgentSuccess:
    @pytest.mark.asyncio
    async def test_valid_response_produces_success_result(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response()),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("Why do humans dream?", _script())

        assert isinstance(result, ThumbnailResult)
        assert result.success is True
        assert (result.width, result.height) == (1280, 720)
        assert os.path.exists(result.output_path)

    @pytest.mark.asyncio
    async def test_selected_asset_metadata_preserved(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response()),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.selected_asset.provider == "fake-media"
        assert result.selected_asset.provider_asset_id == "101"
        assert result.selected_asset.attribution == "Mock Photographer"

    @pytest.mark.asyncio
    async def test_llm_provider_diagnostics_captured(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response()),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.llm_provider == "fake-llm"
        assert result.llm_model == "fake-model"
        assert result.used_fallback_model is False

    @pytest.mark.asyncio
    async def test_output_path_uses_video_slug_when_given(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response()),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("dreams", _script(), video_slug="why-do-humans-dream-abcd1234")

        assert result.output_path == os.path.join(str(tmp_path), "why-do-humans-dream-abcd1234.jpg")

    @pytest.mark.asyncio
    async def test_output_path_falls_back_to_hook_slug(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response(hook_text="Dream Big")),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.output_path == os.path.join(str(tmp_path), "dream-big.jpg")

    @pytest.mark.asyncio
    async def test_portrait_source_image_still_produces_valid_thumbnail(self, tmp_path) -> None:
        media = FakeMediaProvider(image_size=(900, 1600))
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is True
        assert (result.width, result.height) == (1280, 720)


class TestThumbnailAgentPlanningFallback:
    @pytest.mark.asyncio
    async def test_no_llm_provider_uses_deterministic_plan_and_still_succeeds(self, tmp_path) -> None:
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), llm_provider=None, output_dir=str(tmp_path))

        result = await agent.generate_thumbnail("Why do humans dream?", _script())

        assert result.success is True
        assert result.plan.used_semantic_planning is False
        assert result.plan.hook_text == "WHY DO HUMANS DREAM"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_and_still_succeeds(self, tmp_path) -> None:
        llm = FakeLLMProvider(raise_error=RuntimeError("Gemini outage"))
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), llm_provider=llm, output_dir=str(tmp_path))

        result = await agent.generate_thumbnail("Why do humans dream?", _script())

        assert result.success is True
        assert result.plan.used_semantic_planning is False
        assert "Gemini outage" in result.plan.fallback_reason

    @pytest.mark.asyncio
    async def test_malformed_llm_json_falls_back_and_still_succeeds(self, tmp_path) -> None:
        llm = FakeLLMProvider(response="not json at all")
        agent = ThumbnailAgent(media_provider=FakeMediaProvider(), llm_provider=llm, output_dir=str(tmp_path))

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is True
        assert result.plan.used_semantic_planning is False


class TestThumbnailAgentHookTextHandling:
    @pytest.mark.asyncio
    async def test_over_length_hook_normalized_not_failed(self, tmp_path) -> None:
        long_hook = "WHY " * 30
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response(hook_text=long_hook)),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is True
        assert len(result.plan.hook_text) <= 60

    @pytest.mark.asyncio
    async def test_ambiguous_isolated_statistic_hook_is_replaced_end_to_end(self, tmp_path) -> None:
        """The real failure pattern found in manual review: an LLM-provided
        hook that's factually grounded but reads as an ambiguous isolated
        statistic with no topical connection must be deterministically
        replaced, not shipped as-is."""
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response(hook_text="Two Hours Every Night")),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail(
            "Why do humans dream?", _script(), metadata_title="Why Do Humans Dream? The Science of Sleep"
        )

        assert result.success is True
        assert result.plan.hook_text != "Two Hours Every Night"
        assert any("isolated statistic" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_ambiguous_hook_guard_is_generic_not_hardcoded_to_dreams_topic(self, tmp_path) -> None:
        """The exact same guard applied to a completely unrelated topic/
        script must behave the same way - proves the rule isn't a special
        case for the dreams example that motivated it."""
        volcano_script = _script(
            topic="How volcanoes erupt",
            video_title="How Volcanoes Erupt",
            sections=[
                ScriptSection(heading="Magma", narration="Molten rock rises through the crust."),
                ScriptSection(heading="Pressure", narration="Gas pressure builds until it is released."),
            ],
        )
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response(hook_text="300 Degrees Hotter")),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail(
            "How volcanoes erupt", volcano_script, metadata_title="How Volcanoes Erupt Explained"
        )

        assert result.success is True
        assert result.plan.hook_text != "300 Degrees Hotter"
        assert "volcano" in result.plan.hook_text.lower() or "erupt" in result.plan.hook_text.lower()

    @pytest.mark.asyncio
    async def test_clear_hook_is_not_replaced(self, tmp_path) -> None:
        agent = ThumbnailAgent(
            media_provider=FakeMediaProvider(),
            llm_provider=FakeLLMProvider(response=_valid_plan_response(hook_text="WHY DO WE DREAM?")),
            output_dir=str(tmp_path),
        )

        result = await agent.generate_thumbnail("Why do humans dream?", _script())

        assert result.success is True
        assert result.plan.hook_text == "WHY DO WE DREAM?"
        assert result.warnings == []


class TestThumbnailAgentImageSelection:
    @pytest.mark.asyncio
    async def test_avoid_concept_candidate_skipped_in_favor_of_safe_one(self, tmp_path) -> None:
        bad = _candidate(content_hint="nightmare horror imagery", asset_id="1")
        good = _candidate(content_hint="person sleeping peacefully", asset_id="2")
        media = FakeMediaProvider(candidates=[bad, good])
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is True
        assert result.selected_asset.provider_asset_id == "2"

    @pytest.mark.asyncio
    async def test_no_candidates_found_fails_cleanly(self, tmp_path) -> None:
        media = FakeMediaProvider(candidates=[])
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is False
        assert "no suitable" in result.error.lower()


class TestThumbnailAgentFailures:
    @pytest.mark.asyncio
    async def test_search_failure_returns_clean_failure(self, tmp_path) -> None:
        media = FakeMediaProvider(search_error=MediaProviderError("Pexels outage"))
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is False
        assert "image search failed" in result.error.lower()
        assert result.plan is not None  # plan is preserved even on downstream failure

    @pytest.mark.asyncio
    async def test_download_failure_returns_clean_failure(self, tmp_path) -> None:
        media = FakeMediaProvider(download_error=MediaProviderError("network error"))
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is False
        assert "download failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_corrupt_downloaded_image_returns_clean_failure(self, tmp_path) -> None:
        media = FakeMediaProvider(write_valid_image=False)
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is False
        assert "rendering failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_failure_never_returns_fake_success(self, tmp_path) -> None:
        media = FakeMediaProvider(search_error=MediaProviderError("outage"))
        agent = ThumbnailAgent(
            media_provider=media, llm_provider=FakeLLMProvider(response=_valid_plan_response()), output_dir=str(tmp_path)
        )

        result = await agent.generate_thumbnail("dreams", _script())

        assert result.success is False
        assert result.output_path is None
        assert result.width is None
        assert result.height is None
