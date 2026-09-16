# AI Video Generation fallback-routing seam.
#
# This is the "AI Video Prompt/Request -> AIVideoProvider -> generated clip
# -> bounded retry -> Pexels fallback" step of the intended future visual
# architecture: Visual Context Planner -> AI Video Prompt/Request ->
# AIVideoProvider -> generated clip -> Visual QC -> bounded retry -> Pexels
# fallback when configured -> existing FFmpeg assembly.
#
# Deliberately standalone and NOT wired into VisualMediaService's existing
# slot-acquisition loop (_acquire_slot_asset) yet - that loop is the
# heavily-tested, real-provider-validated core of the current production
# pipeline, and rewiring it is a dedicated future integration step (a
# provider/configuration change, not a redesign) once a real paid AI video
# provider is actually being adopted. This module builds and fully tests
# the routing decision itself in isolation first: try AI generation
# (bounded retries) -> Pexels fallback (when enabled) -> structured
# failure - so that later integration is "call this function", not "design
# this logic".
#
# Returns the SAME MediaAsset model VisualMediaService already produces
# (never a new/parallel result type), so a successful AI-generated clip is
# indistinguishable in shape from a Pexels asset to every downstream stage
# (Visual QC, Video Assembly, provenance, compliance).
from __future__ import annotations

from typing import Awaitable, Callable, Optional, Tuple

from src.models.ai_video import AIVideoGenerationRequest, AIVideoGenerationResult
from src.models.media import MediaAsset
from src.tools.ai_video_provider import AIVideoProvider

# Returns (asset, query_or_prompt_used) - the exact same shape
# VisualMediaService._acquire_slot_asset already returns, reused as-is.
PexelsFallback = Callable[[], Awaitable[Tuple[MediaAsset, str]]]


async def acquire_visual_asset_with_ai_fallback(
    request: AIVideoGenerationRequest,
    ai_video_provider: Optional[AIVideoProvider],
    pexels_fallback: PexelsFallback,
    max_retries: int,
    stock_fallback_enabled: bool,
) -> Tuple[MediaAsset, str]:
    """Acquire one visual slot's asset, preferring AI generation when
    configured, falling back to the existing Pexels/mock selection chain.

    Args:
        request: The AI generation request for this specific slot (prompt,
            duration, section/slot index, etc.) - built upstream from the
            Visual Context Planner's own plan, never invented here.
        ai_video_provider: Injected AIVideoProvider, or ``None`` to skip AI
            generation entirely and go straight to ``pexels_fallback``
            (the exact current, unmodified production behavior).
        pexels_fallback: Zero-argument async callable that performs the
            EXISTING stock-media search/selection/download chain (e.g. a
            closure over ``VisualMediaService._acquire_slot_asset``) -
            never re-implemented here.
        max_retries: Bounded number of ADDITIONAL AI generation attempts
            after the first (so ``max_retries=2`` means up to 3 total
            calls) - never an unbounded retry loop.
        stock_fallback_enabled: When AI generation exhausts its retries
            without success, fall back to ``pexels_fallback`` if True;
            otherwise return a structured failure MediaAsset (never a
            silent empty slot, and never a crash).

    Returns:
        (asset, query_or_prompt_used) - the same shape
        VisualMediaService's own slot-acquisition already returns.
    """
    if ai_video_provider is None:
        return await pexels_fallback()

    last_result: Optional[AIVideoGenerationResult] = None
    for _ in range(max(max_retries, 0) + 1):
        last_result = await ai_video_provider.generate(request)
        if last_result.success and last_result.local_file_path:
            return (
                MediaAsset(
                    provider=last_result.provider,
                    asset_type="video",
                    local_file_path=last_result.local_file_path,
                    search_query=request.prompt,
                    section_index=request.section_index,
                    duration_seconds=last_result.duration_seconds,
                    width=last_result.width,
                    height=last_result.height,
                    reused=False,
                    relevance_tier="ai_generated",
                    success=True,
                ),
                request.prompt,
            )

    if stock_fallback_enabled:
        return await pexels_fallback()

    error_detail = last_result.error if last_result else "AI video generation unavailable"
    attempts = max(max_retries, 0) + 1
    return (
        MediaAsset(
            provider=ai_video_provider.name,
            search_query=request.prompt,
            section_index=request.section_index,
            success=False,
            error=f"AI video generation failed after {attempts} attempt(s): {error_detail}",
        ),
        request.prompt,
    )


def build_ai_video_request(
    prompt: str,
    section_index: int,
    slot_index: int,
    duration_seconds: float,
    aspect_ratio: str = "16:9",
    negative_prompt: Optional[str] = None,
) -> AIVideoGenerationRequest:
    """Small convenience constructor - never a second place that decides
    prompt CONTENT (that stays entirely the Visual Context Planner's job);
    this only assembles the typed request envelope around an
    already-decided prompt string."""
    return AIVideoGenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        section_index=section_index,
        slot_index=slot_index,
    )
