# AI Video Generation provider-agnostic data models.
#
# These models describe the CONTRACT between the visual pipeline and any
# AI-video-generation backend (a free/demo provider today, a paid one like
# fal.ai/Kling, Seedance, Veo, or PixVerse's official API later) - nothing
# here names a specific vendor. See src/tools/ai_video_provider.py for the
# abstract provider interface these models are the input/output of.
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AIVideoStatus = Literal[
    "not_started",
    "queued",
    "generating",
    "succeeded",
    "failed",
    "rejected",
]


class AIVideoGenerationRequest(BaseModel):
    """One request to generate a single short AI video clip for one visual
    slot. Deliberately provider-agnostic - a concrete AIVideoProvider
    implementation translates this into whatever request shape its own API
    actually expects (Kling/Seedance/Veo/PixVerse all differ), never the
    other way around."""

    prompt: str = Field(min_length=1, description="Visual generation prompt grounded in the section's content")
    negative_prompt: Optional[str] = Field(
        default=None, description="Concepts/elements to avoid in the generated clip, if the provider supports it"
    )
    aspect_ratio: str = Field(default="16:9", description="e.g. '16:9', '9:16', '1:1'")
    duration_seconds: float = Field(gt=0.0, description="Requested clip duration - providers may round to their own supported durations")
    resolution: Optional[str] = Field(
        default=None, description="Requested quality/resolution hint (e.g. '720p', '1080p'), when the provider supports a choice"
    )
    seed: Optional[int] = Field(default=None, description="Optional deterministic seed, when the provider supports one")
    section_index: int = Field(ge=0, description="Index of the ScriptSection this clip is for")
    slot_index: int = Field(ge=0, description="Index of the visual slot within the section this clip is for")


class AIVideoGenerationResult(BaseModel):
    """Structured output of one AIVideoProvider.generate() call. Mirrors
    every other provider result in this project (e.g. UploadResult): never
    raises for a generation/provider failure - the caller always gets a
    typed result back with ``status``/``error`` describing what happened."""

    success: bool = Field(description="Whether a usable local video file was produced")
    status: AIVideoStatus
    provider: str = Field(min_length=1, description="Provider name, e.g. 'local_ai_video' or a future 'fal_kling'")
    model: Optional[str] = Field(default=None, description="Underlying model/endpoint actually used, if reported")
    local_file_path: Optional[str] = Field(default=None, description="Local filesystem path to the generated clip, if successful")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Actual clip duration, if known")
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)
    provider_job_id: Optional[str] = Field(default=None, description="Provider-specific generation job/request ID, for traceability")
    cost_usd: Optional[float] = Field(default=None, ge=0.0, description="Reported generation cost, when the provider exposes usage/billing data")
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, description="Error message if generation failed/was rejected")
