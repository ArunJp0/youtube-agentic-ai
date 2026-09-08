# Provenance manifest data models - a small, deterministic, machine-readable
# record of exactly which external assets a completed pipeline run actually
# used, persisted to disk so a later STANDALONE ComplianceAgent (which has
# no access to the in-memory PipelineState that produced the run) can
# recover real provenance instead of guessing or fabricating it.
#
# Deliberately NOT a serialization of entire internal result objects
# (VisualResult/AudioMixResult/ThumbnailResult) - only the specific fields a
# compliance review actually needs. Never includes API keys, secrets,
# tokens, prompts, or other internal LLM data.
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

PROVENANCE_SCHEMA_VERSION = 1


class VisualAssetProvenance(BaseModel):
    """Provenance for one visual asset actually used in the assembled video."""

    section_index: int = Field(ge=0)
    section_heading: str = Field(default="")
    provider: Optional[str] = Field(default=None, description="e.g. 'pexels', 'mock'")
    provider_asset_id: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    attribution: Optional[str] = Field(default=None)
    local_file_path: Optional[str] = Field(default=None)


class ThumbnailProvenance(BaseModel):
    """Provenance for the source image the thumbnail was rendered from."""

    provider: Optional[str] = Field(default=None)
    provider_asset_id: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    attribution: Optional[str] = Field(default=None)
    output_path: Optional[str] = Field(default=None, description="The rendered thumbnail file this run produced")


class BGMProvenance(BaseModel):
    """Provenance for the BGM track actually mixed into this run's video.

    ``attribution_required``/``attribution_text`` here are a snapshot of the
    approved catalog AT THE TIME OF THE RUN - a later ComplianceAgent must
    still cross-check ``track_id`` against the CURRENT
    ``assets/bgm/catalog.json`` and treat the current catalog as
    authoritative (see ``src.services.compliance_checks.check_bgm_provenance``),
    reporting a disagreement rather than blindly trusting this snapshot.
    """

    track_id: str = Field(min_length=1)
    title: str = Field(default="")
    source: str = Field(default="")
    license_type: str = Field(default="")
    attribution_required: bool = Field(default=False)
    attribution_text: Optional[str] = Field(default=None)
    local_file_path: Optional[str] = Field(default=None)


class ProvenanceManifest(BaseModel):
    """One completed pipeline run's persisted, machine-readable provenance
    record - the single source of truth a standalone compliance review
    reads instead of requiring live PipelineState."""

    schema_version: int = Field(default=PROVENANCE_SCHEMA_VERSION)
    run_id: str = Field(min_length=1, description="Stable identifier shared with the final video's own filename")
    topic: str = Field(min_length=1)
    final_video_path: str = Field(min_length=1)
    created_at: str = Field(min_length=1, description="ISO 8601 UTC timestamp this manifest was written")
    visual_assets: List[VisualAssetProvenance] = Field(default_factory=list)
    thumbnail: Optional[ThumbnailProvenance] = Field(default=None)
    bgm: Optional[BGMProvenance] = Field(default=None)
