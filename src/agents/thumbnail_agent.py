# Thumbnail Agent: orchestrates the two-part thumbnail architecture -
# semantic planning (ThumbnailPlanner, one LLM call) and deterministic
# rendering (thumbnail_renderer) - into one typed ThumbnailResult, plus
# stock-image selection/download and output validation in between.
#
# Reuses the existing MediaProvider abstraction (the same one
# VisualMediaService already depends on) for stock-photo search/download -
# no second HTTP/Pexels implementation. Like AudioMixingService/
# MetadataAgent, this is deterministic orchestration, not a reasoning
# component itself: semantic judgment stays inside the injected
# ThumbnailPlanner, and image selection/rendering/validation are all
# fixed-rule deterministic steps.
from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional

from src.agents.thumbnail_planner import ThumbnailPlanner
from src.llm.provider import LLMProvider
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailPlan, ThumbnailResult, ThumbnailSourceAsset
from src.services.thumbnail_planning import build_deterministic_thumbnail_plan
from src.services.thumbnail_renderer import ThumbnailRenderError, render_thumbnail
from src.services.thumbnail_selection import select_thumbnail_candidate
from src.services.thumbnail_validation import ThumbnailValidationError, resolve_hook_text, validate_output_image
from src.tools.media_provider import MediaCandidate, MediaProvider, MediaProviderError

DEFAULT_THUMBNAIL_OUTPUT_DIR = os.path.join("output", "thumbnails")

_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")
_EXTENSION_RE = re.compile(r"\.(jpg|jpeg|png)(?:\?|$)", re.IGNORECASE)


class ThumbnailAgentError(Exception):
    """Raised only for configuration/programmer errors (e.g. missing topic
    or ScriptResult).

    Image-search/selection/rendering/validation failures are NEVER raised -
    they are captured in the returned ThumbnailResult (success=False,
    error=...) so callers always get a structured result back.
    """


class ThumbnailAgent:
    """Produces a 1280x720 YouTube thumbnail for a video from its real
    topic/ScriptResult (and, when available, already-generated metadata),
    using exactly one LLM call for planning plus deterministic image
    selection/rendering/validation."""

    def __init__(
        self,
        media_provider: MediaProvider,
        llm_provider: Optional[LLMProvider] = None,
        output_dir: str = DEFAULT_THUMBNAIL_OUTPUT_DIR,
        font_path: Optional[str] = None,
    ) -> None:
        """Initialize the Thumbnail Agent.

        Args:
            media_provider: MediaProvider implementation for stock-photo
                search/download (the same abstraction VisualMediaService
                already uses - real Pexels, or a mock for tests)
            llm_provider: Optional LLMProvider for real semantic planning
                via ThumbnailPlanner. If None, every plan uses the
                deterministic fallback directly (no LLM call at all).
            output_dir: Local directory to write the rendered thumbnail
                into (expected to be excluded from version control)
            font_path: Optional explicit font file path, tried before the
                renderer's built-in local/system fallbacks
        """
        self.media_provider = media_provider
        self.planner = ThumbnailPlanner(llm_provider) if llm_provider is not None else None
        self.output_dir = output_dir
        self.font_path = font_path

    async def generate_thumbnail(
        self,
        topic: str,
        script: ScriptResult,
        metadata_title: Optional[str] = None,
        seo_summary: Optional[str] = None,
        video_slug: Optional[str] = None,
    ) -> ThumbnailResult:
        """Plan, select an image for, render, and validate a thumbnail for
        ``topic``/``script``.

        Never raises for processing failures (search/selection/rendering/
        validation) - those are captured in the returned ThumbnailResult.
        Only raises ThumbnailAgentError for configuration/programmer errors.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent (never
                regenerated here - Research/Script are the caller's job)
            metadata_title: Optional already-generated video title (from
                MetadataResult), used as extra planning context
            seo_summary: Optional already-generated SEO summary (from
                MetadataResult), used as extra planning context
            video_slug: Optional stable slug (e.g. the video's own output
                filename base) used for the thumbnail's filename; falls
                back to a slug of the generated hook text

        Returns:
            Structured ThumbnailResult describing the outcome
        """
        if not topic or script is None:
            raise ThumbnailAgentError("topic and ScriptResult are both required")

        plan = self._plan(topic, script, metadata_title, seo_summary)
        # Normalizes the hook and, if it reads as an ambiguous isolated
        # statistic/duration with no clear connection to the video's own
        # topic/title (e.g. "Two Hours Every Night" for a video about why
        # humans dream), deterministically replaces it with a topic/title-
        # derived hook instead - never a second LLM call.
        final_hook, hook_warning = resolve_hook_text(
            plan.hook_text, topic, metadata_title=metadata_title, extra_reference_texts=(script.video_title,)
        )
        if not final_hook:
            return self._failure(topic, "Thumbnail plan produced an empty hook text after normalization", plan)
        if final_hook != plan.hook_text:
            plan = plan.model_copy(update={"hook_text": final_hook})
        hook_warnings = [hook_warning] if hook_warning else []

        try:
            candidates = await self.media_provider.search(plan.search_query, prefer_video=False, max_results=5)
        except MediaProviderError as e:
            return self._failure(topic, f"Thumbnail image search failed: {e}", plan)
        except Exception as e:
            return self._failure(topic, f"Unexpected thumbnail image search error: {e}", plan)

        candidate, selection_warnings = select_thumbnail_candidate(candidates, plan.avoid_concepts)
        if candidate is None:
            return self._failure(
                topic, "No suitable thumbnail image found: " + "; ".join(selection_warnings), plan
            )

        with tempfile.TemporaryDirectory(prefix="thumbnail_") as tmp_dir:
            source_path = os.path.join(tmp_dir, f"source{_extension_for(candidate)}")
            try:
                await self.media_provider.download(candidate, source_path)
            except MediaProviderError as e:
                return self._failure(topic, f"Thumbnail image download failed: {e}", plan)
            except Exception as e:
                return self._failure(topic, f"Unexpected thumbnail image download error: {e}", plan)

            output_path = self._build_output_path(video_slug or plan.hook_text)
            try:
                render_thumbnail(source_path, plan, output_path, font_path=self.font_path)
            except ThumbnailRenderError as e:
                return self._failure(topic, f"Thumbnail rendering failed: {e}", plan)

        try:
            width, height = validate_output_image(output_path)
        except ThumbnailValidationError as e:
            return self._failure(topic, f"Thumbnail validation failed: {e}", plan)

        llm_provider_obj = self.planner.llm_provider if self.planner else None
        return ThumbnailResult(
            success=True,
            topic=topic,
            output_path=output_path,
            width=width,
            height=height,
            plan=plan,
            selected_asset=ThumbnailSourceAsset(
                provider=self.media_provider.name,
                provider_asset_id=candidate.provider_asset_id,
                source_url=candidate.source_url,
                attribution=candidate.attribution,
                width=candidate.width,
                height=candidate.height,
            ),
            llm_provider=(
                getattr(llm_provider_obj, "name", None) or type(llm_provider_obj).__name__
                if llm_provider_obj is not None
                else None
            ),
            llm_model=getattr(llm_provider_obj, "last_model_used", None),
            used_fallback_model=getattr(llm_provider_obj, "last_used_fallback", None),
            warnings=hook_warnings + selection_warnings,
        )

    # ---- helpers ------------------------------------------------------------

    def _plan(
        self, topic: str, script: ScriptResult, metadata_title: Optional[str], seo_summary: Optional[str]
    ) -> ThumbnailPlan:
        if self.planner is not None:
            return self.planner.plan_thumbnail(topic, script, metadata_title, seo_summary)
        return build_deterministic_thumbnail_plan(topic, "No LLM provider configured")

    def _build_output_path(self, slug_source: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        slug = _slugify(slug_source)
        return os.path.join(self.output_dir, f"{slug}.jpg")

    @staticmethod
    def _failure(topic: str, error: str, plan: Optional[ThumbnailPlan]) -> ThumbnailResult:
        return ThumbnailResult(success=False, topic=topic, error=error, plan=plan)


def _extension_for(candidate: MediaCandidate) -> str:
    match = _EXTENSION_RE.search(candidate.download_url or "")
    return f".{match.group(1).lower()}" if match else ".jpg"


def _slugify(text: str) -> str:
    slug = _SLUG_INVALID_RE.sub("-", text.lower()).strip("-")
    return slug or "thumbnail"
