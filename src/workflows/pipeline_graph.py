# Full Research -> Script -> Voice -> Visual Media -> Visual QC -> Video
# Assembly -> Subtitle/Caption -> BGM/Audio Mixing -> Metadata -> Thumbnail ->
# Copyright/Compliance pipeline using LangGraph.
#
# This module does not implement any research, scripting, voice-synthesis,
# visual-media, QC-evaluation, video-encoding, transcription, subtitle-
# rendering, music-planning/selection/mixing, metadata-generation,
# thumbnail planning/rendering, or compliance-checking logic itself - it
# only wires the existing ResearchAgent, ScriptAgent, VoiceService,
# VisualMediaService, VisualQCService, VideoAssemblyService,
# CaptionService, AudioMixingService, MetadataAgent, ThumbnailAgent, and
# ComplianceAgent together into a single LangGraph state machine, passing
# each stage's output directly into the next stage's input.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from langgraph.graph import END, StateGraph

from src.agents.compliance_agent import ComplianceAgent, ComplianceAgentError
from src.agents.metadata_agent import DEFAULT_METADATA_OUTPUT_DIR, MetadataAgent, MetadataAgentError
from src.agents.research import ResearchAgent, ResearchAgentError
from src.agents.script import ScriptAgent, ScriptAgentError
from src.agents.thumbnail_agent import DEFAULT_THUMBNAIL_OUTPUT_DIR, ThumbnailAgent, ThumbnailAgentError
from src.agents.visual_context_planner import VisualContextPlanner
from src.models.captions import CaptionResult
from src.models.compliance import ComplianceResult
from src.models.media import VisualResult
from src.models.metadata import MetadataResult
from src.models.music import AudioMixResult
from src.models.remediation import (
    DEFAULT_MAX_REMEDIATION_ATTEMPTS,
    LocalizedFinding,
    RemediationAttemptRecord,
    ScriptCorrection,
)
from src.models.research import ResearchResult
from src.models.script import ScriptResult
from src.models.thumbnail import ThumbnailResult
from src.models.video import VideoAssemblyResult
from src.models.visual_plan import VisualPlan
from src.models.visual_qc import VisualQCResult
from src.models.voice import VoiceResult
from src.services.audio_mixing_service import AudioMixingService, AudioMixingServiceError
from src.services.caption_service import DEFAULT_SUBTITLE_OUTPUT_DIR, CaptionService, CaptionServiceError
from src.services.compliance_record_store import DEFAULT_COMPLIANCE_RECORD_DIR, ComplianceRecordStore
from src.services.finding_localizer import has_grounding_evidence, localize_findings
from src.services.provenance_collection import persist_provenance_if_completed
from src.services.provenance_store import DEFAULT_PROVENANCE_OUTPUT_DIR, ProvenanceManifestStore, ProvenanceStoreError
from src.services.script_context_reconstruction import original_base_name
from src.services.video_assembly_service import (
    DEFAULT_VIDEO_OUTPUT_DIR,
    VideoAssemblyService,
    VideoAssemblyServiceError,
)
from src.services.visual_media_service import (
    DEFAULT_MEDIA_OUTPUT_DIR,
    VisualMediaService,
    VisualMediaServiceError,
)
from src.services.visual_qc_service import VisualQCService, VisualQCServiceError
from src.services.voice_service import DEFAULT_OUTPUT_DIR, VoiceService, VoiceServiceError
from src.tools.ffmpeg_video_assembler import VideoAssembler
from src.tools.media_provider import MediaProvider
from src.tools.music_catalog_provider import MusicCatalogProvider
from src.tools.transcription_provider import TranscriptionProvider
from src.tools.visual_relevance_evaluator import VisualRelevanceEvaluator
from src.tools.voice_provider import VoiceProvider


def _captioned_video_result(caption_result: CaptionResult) -> VideoAssemblyResult:
    """Adapt CaptionService's captioned MP4 into the VideoAssemblyResult
    shape AudioMixingService already expects as its source video.

    BGM is mixed onto the captioned output (what viewers will actually
    see), not the pre-caption assembly - the same file the standalone
    ``bgm_demo.py`` targets for real validation. This is a thin adapter,
    not new business logic: AudioMixingService itself only reads
    ``output_path``/``success`` off whatever VideoAssemblyResult-shaped
    object it's given.
    """
    return VideoAssemblyResult(
        success=True,
        output_path=caption_result.captioned_video_path,
        duration_seconds=caption_result.captioned_duration_seconds,
        format="mp4",
    )


@dataclass
class PipelineState:
    """Shared state for the Research -> Script -> Voice -> Visual Media ->
    Visual QC -> Video Assembly pipeline."""

    topic: str = ""
    research_result: Optional[ResearchResult] = None
    script_result: Optional[ScriptResult] = None
    voice_result: Optional[VoiceResult] = None
    visual_plan: Optional[VisualPlan] = None
    visual_result: Optional[VisualResult] = None
    visual_qc_result: Optional[VisualQCResult] = None
    # The VisualResult Video Assembly actually consumes: visual_result
    # (above) is never overwritten/silently replaced - it stays exactly
    # what VisualMediaService produced, pre-QC - so this field is always
    # populated separately once Visual QC completes acceptably, and it's
    # the only one that may differ from visual_result (when QC replaced a
    # weak/misleading asset).
    qc_approved_visual_result: Optional[VisualResult] = None
    video_assembly_result: Optional[VideoAssemblyResult] = None
    caption_result: Optional[CaptionResult] = None
    # AudioMixResult already carries the mood plan (.music_plan), the
    # selected track (.selected_track), and the final mixed MP4 path
    # (.output_path) - no separate top-level fields for those, consistent
    # with how VisualQCResult/CaptionResult are each the single source of
    # truth for their own stage's structured data.
    audio_mix_result: Optional[AudioMixResult] = None
    # MetadataResult already carries the generated title/description/tags/
    # hashtags/chapters and the written JSON artifact path - no separate
    # top-level fields, consistent with every other stage's result.
    metadata_result: Optional[MetadataResult] = None
    # ThumbnailResult already carries the generated plan, selected source
    # image, and final rendered thumbnail path - no separate top-level
    # fields, consistent with every other stage's result.
    thumbnail_result: Optional[ThumbnailResult] = None
    # ComplianceResult already carries the publish_decision/risk_level/
    # checks/warnings/blockers/required_attributions/semantic_review - no
    # separate top-level PASS/REVIEW/BLOCK fields, consistent with every
    # other stage's result. A future Upload Agent reads this directly.
    compliance_result: Optional[ComplianceResult] = None
    # Compliance Remediation state (see script_revision_node/
    # route_after_compliance below). remediation_attempt counts corrected
    # candidates actually produced so far (0 = the original, never-
    # remediated evaluation). remediation_parent_run_id is set once, on the
    # first remediation attempt, to the ORIGINAL run's own run_id - every
    # subsequent attempt's PersistedComplianceRecord links back to that
    # same parent, never to the immediately-prior attempt. remediation_history
    # is this run's full audit trail (typed, not just log lines) - never
    # cleared, appended to only.
    remediation_attempt: int = 0
    remediation_parent_run_id: Optional[str] = None
    remediation_history: List[RemediationAttemptRecord] = field(default_factory=list)
    # pending -> researching -> researched -> scripted -> voiced ->
    # visualized -> qc_passed -> assembled -> captioned -> mixed ->
    # metadata_generated -> thumbnail_generated -> revising (mid-
    # remediation, transient) -> completed (Compliance PASS) ->
    # review_required (Compliance REVIEW, no actionable findings or
    # remediation disabled) -> review_exhausted (Compliance REVIEW,
    # actionable findings existed but max_remediation_attempts was used up)
    # -> blocked (Compliance BLOCK, never remediated) -> failed (a
    # technical failure at any stage)
    status: str = "pending"
    error: Optional[str] = None


def build_pipeline_graph(
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    visual_relevance_evaluator: VisualRelevanceEvaluator,
    transcription_provider: TranscriptionProvider,
    music_catalog_provider: MusicCatalogProvider,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    subtitle_output_dir: str = DEFAULT_SUBTITLE_OUTPUT_DIR,
    thumbnail_output_dir: str = DEFAULT_THUMBNAIL_OUTPUT_DIR,
    metadata_output_dir: str = DEFAULT_METADATA_OUTPUT_DIR,
    provenance_output_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR,
    compliance_output_dir: str = DEFAULT_COMPLIANCE_RECORD_DIR,
    max_remediation_attempts: int = DEFAULT_MAX_REMEDIATION_ATTEMPTS,
) -> StateGraph:
    """Build the LangGraph state machine chaining Research -> Script -> Voice
    -> Visual Media -> Visual QC -> Video Assembly -> Subtitle/Caption ->
    BGM/Audio Mixing -> Metadata -> Thumbnail -> Copyright/Compliance.

    Reuses the existing ResearchAgent, ScriptAgent, VoiceService,
    VisualMediaService, VisualQCService, VideoAssemblyService,
    CaptionService, AudioMixingService, MetadataAgent, ThumbnailAgent, and
    ComplianceAgent as-is (no duplicated business logic, no reimplemented
    FFmpeg calls, no re-implemented QC evaluation/replacement, no
    re-implemented transcription/SRT/subtitle-rendering, music-planning/
    selection/mixing, metadata-generation/validation, thumbnail planning/
    rendering/validation, or compliance-checking/provenance logic); this
    graph only wires their existing interfaces together and shares one
    PipelineState across all eleven.
    VoiceService, VisualMediaService, VisualQCService, VideoAssemblyService,
    CaptionService, AudioMixingService, MetadataAgent, ThumbnailAgent, and
    ComplianceAgent all remain deterministic orchestration here - each is
    invoked directly, not treated as a reasoning agent (semantic judgment
    stays inside VisualContextPlanner/VisualQCService's injected evaluator,
    AudioMixingService's injected MusicContextPlanner, MetadataAgent's own
    single LLM call, ThumbnailAgent's injected ThumbnailPlanner's own
    single LLM call, and ComplianceAgent's injected ComplianceReviewer's
    own single advisory LLM call; speech-to-text stays inside
    CaptionService's injected transcription provider).
    VideoAssemblyService receives the exact ScriptResult/VoiceResult
    already produced earlier in this same run, and the post-QC
    ``qc_approved_visual_result`` (never the raw, pre-QC ``visual_result``).
    CaptionService receives the exact VoiceResult/VideoAssemblyResult
    already produced earlier in this same run. AudioMixingService receives
    the exact ScriptResult already produced earlier in this same run
    (Research/Script are never re-run for BGM mood planning) and the
    captioned MP4 CaptionService just produced. MetadataAgent likewise
    receives the exact ScriptResult already produced earlier in this same
    run (never reconstructed from an .srt transcript - that fallback is
    only for the standalone demo, which has no PipelineState to read a
    real ScriptResult from) and the real final duration from the BGM-mixed
    MP4. ThumbnailAgent likewise receives the exact ScriptResult already
    produced earlier in this same run, plus the real MetadataResult's
    title/SEO summary as extra planning context (Research/Script/Metadata
    are never re-run for thumbnail planning). ComplianceAgent receives the
    exact ScriptResult/MetadataResult/ThumbnailResult already produced
    earlier in this same run, the real final video path from
    ``audio_mix_result``, and the current run's own ProvenanceManifest
    (written by ``thumbnail_node`` on its own success, then looked up by
    ``compliance_node`` via ``ProvenanceManifestStore.find_for_video`` -
    never reconstructed from filenames, never a previous run's manifest,
    since both steps derive the same run identifier from the same real
    ``audio_mix_result.output_path``) - nothing is regenerated,
    re-synthesized, re-downloaded, or re-assembled.

    Graph structure:
        START → research ─(ok)─→ script ─(ok)─→ voice ─(ok)─→ media ─(ok)─→ visual_qc ─(ok)─→ video_assembly ─(ok)─→ captions ─(ok)─→ bgm ─(ok)─→ metadata ─(ok)─→ thumbnail ─(ok)─→ compliance ─(PASS/BLOCK)─→ END
                    │                  │               │              │                │                  │                  │              │                  │                  │
                    └──────(error)─────┴──────(error)───┴───(error)────┴────(error)─────┴──────(error)─────┴──────(error)─────┴──────(error)──┴──────(error)──────┴──────(error)────┴──────(error)──→ END

        compliance ─(REVIEW, actionable finding(s), attempts remain)─→ script_revision ─(corrected)─→ voice (loop)
        compliance ─(REVIEW, no actionable finding, or attempts exhausted)─→ END
        script_revision ─(correction failed)─→ END

    Args:
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research, Script,
            Visual Context Planning, Visual QC, BGM mood planning,
            Thumbnail planning, and the Compliance semantic review
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation shared by the Visual
            Media Service and the Thumbnail Agent (stock-photo search/download)
        assembler: VideoAssembler implementation, shared by Visual QC (frame
            extraction/probing), the Video Assembly Service (encoding), the
            Caption Service (subtitle burning), and Audio Mixing (background
            music mixing)
        visual_relevance_evaluator: VisualRelevanceEvaluator implementation
            for the Visual QC Service
        transcription_provider: TranscriptionProvider implementation for
            the Caption Service
        music_catalog_provider: MusicCatalogProvider implementation (the
            approved BGM catalog) shared by the Audio Mixing Service and
            the Compliance Agent's BGM provenance cross-check
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into
        subtitle_output_dir: Directory the Caption Service writes .srt files into
        thumbnail_output_dir: Directory the Thumbnail Agent writes the rendered thumbnail into
        metadata_output_dir: Directory the Metadata Agent writes its JSON artifact into
        provenance_output_dir: Directory the provenance manifest is written into
            (by ``thumbnail_node``, on its own success) and read from (by
            ``compliance_node``)
        compliance_output_dir: Directory ComplianceRecordStore persists a
            durable record into for every PASS/REVIEW/BLOCK decision
            (written by ``compliance_node``, for every attempt)
        max_remediation_attempts: Bounded retry limit for Compliance
            Remediation (see ``script_revision_node``/
            ``route_after_compliance``) - 0 disables remediation entirely
            (a REVIEW always stops immediately, exactly as before this
            milestone)

    Returns:
        StateGraph ready to be ``.compile()``d
    """
    research_agent = ResearchAgent(search_provider=search_provider, llm_provider=llm_provider)
    script_agent = ScriptAgent(llm_provider=llm_provider)
    voice_service = VoiceService(
        voice_provider=voice_provider, voice_name=voice_name, output_dir=voice_output_dir
    )
    # Reuses the same LLMProvider Research/Script already depend on - one
    # extra call per pipeline run (for the whole script at once), not a new
    # provider/setting. If that call fails, VisualContextPlanner falls back
    # to the deterministic query-generation path on its own.
    visual_planner = VisualContextPlanner(llm_provider=llm_provider)
    visual_service = VisualMediaService(
        media_provider=media_provider, visual_planner=visual_planner, output_dir=media_output_dir
    )
    # visual_media_service=visual_service lets Visual QC request bounded
    # replacements through VisualMediaService's own existing selection
    # logic (see visual_qc_node) - VisualQCService never re-implements
    # candidate search itself.
    visual_qc_service = VisualQCService(
        evaluator=visual_relevance_evaluator, assembler=assembler, visual_media_service=visual_service
    )
    video_service = VideoAssemblyService(assembler=assembler, output_dir=video_output_dir)
    # Shares the same VideoAssembler as Visual QC/Video Assembly (subtitle
    # burning reuses its burn_subtitles method) - no second FFmpeg wrapper.
    caption_service = CaptionService(
        transcription_provider=transcription_provider, assembler=assembler, output_dir=subtitle_output_dir
    )
    # Reuses the same LLMProvider (MusicContextPlanner's single optional
    # mood-planning call per run) and the same VideoAssembler (background-
    # audio mixing reuses its mix_background_audio method) - no new
    # provider/setting and no second FFmpeg wrapper.
    audio_mixing_service = AudioMixingService(
        catalog_provider=music_catalog_provider, assembler=assembler, llm_provider=llm_provider
    )
    # Reuses the same LLMProvider (its single metadata-generation call per
    # run) - no new provider/API key path. Chapter timestamps come from
    # MetadataAgent's own deterministic section-timing derivation, never
    # from this LLM call.
    metadata_agent = MetadataAgent(llm_provider=llm_provider, output_dir=metadata_output_dir)
    # Reuses the same LLMProvider (its single thumbnail-planning call per
    # run, via the injected ThumbnailPlanner) and the same MediaProvider
    # (stock-photo search/download) already used by Visual Media - no new
    # provider/API key path and no second Pexels/HTTP implementation.
    thumbnail_agent = ThumbnailAgent(
        media_provider=media_provider, llm_provider=llm_provider, output_dir=thumbnail_output_dir
    )
    # Reuses the same MusicCatalogProvider AudioMixingService already
    # depends on (the trusted, CURRENT source of BGM licensing/attribution
    # data) and the same LLMProvider (its single advisory semantic-review
    # call per run, via the injected ComplianceReviewer) - no new provider/
    # API key path and no duplicated catalog-reading logic.
    compliance_agent = ComplianceAgent(music_catalog_provider=music_catalog_provider, llm_provider=llm_provider)
    compliance_record_store = ComplianceRecordStore(compliance_output_dir)

    async def research_node(state: PipelineState) -> dict:
        try:
            result = await research_agent.research(state.topic)
            return {"research_result": result, "status": "researched", "error": None}
        except ResearchAgentError as e:
            return {"research_result": None, "status": "failed", "error": f"Research failed: {e}"}
        except Exception as e:
            return {
                "research_result": None,
                "status": "failed",
                "error": f"Unexpected research error: {e}",
            }

    async def script_node(state: PipelineState) -> dict:
        try:
            result = await script_agent.generate_script(state.research_result)
            return {"script_result": result, "status": "scripted", "error": None}
        except ScriptAgentError as e:
            return {"script_result": None, "status": "failed", "error": f"Script generation failed: {e}"}
        except Exception as e:
            return {
                "script_result": None,
                "status": "failed",
                "error": f"Unexpected script error: {e}",
            }

    async def voice_node(state: PipelineState) -> dict:
        try:
            result = await voice_service.generate_voice(state.script_result)
        except VoiceServiceError as e:
            return {"voice_result": None, "status": "failed", "error": f"Voice generation failed: {e}"}
        except Exception as e:
            return {
                "voice_result": None,
                "status": "failed",
                "error": f"Unexpected voice error: {e}",
            }

        if not result.success:
            # VoiceService never raises for synthesis failures - it reports
            # them in VoiceResult.error instead. Surface that in pipeline
            # state (and keep the VoiceResult itself for inspection).
            return {
                "voice_result": result,
                "status": "failed",
                "error": f"Voice generation failed: {result.error}",
            }

        return {"voice_result": result, "status": "voiced", "error": None}

    async def media_node(state: PipelineState) -> dict:
        try:
            # The same ScriptResult produced by the script stage is passed
            # straight through - the script is never regenerated or rewritten.
            # VoiceResult.duration_seconds (the real narration audio length)
            # is the upstream timing input for visual planning; fall back to
            # the script's own deterministic estimate only if a voice
            # provider couldn't report a duration.
            narration_duration = (
                state.voice_result.duration_seconds or state.script_result.estimated_duration_seconds
            )
            # Built once and threaded through (via visual_plan=) so Visual
            # QC can reuse the exact same plan later - one Gemini planning
            # call per run, never a second one for QC.
            plan = visual_service.build_plan(state.script_result)
            result = await visual_service.generate_visuals(
                state.script_result, narration_duration, visual_plan=plan
            )
        except VisualMediaServiceError as e:
            return {
                "visual_plan": None,
                "visual_result": None,
                "status": "failed",
                "error": f"Visual media generation failed: {e}",
            }
        except Exception as e:
            return {
                "visual_plan": None,
                "visual_result": None,
                "status": "failed",
                "error": f"Unexpected visual media error: {e}",
            }

        if not result.success:
            # VisualMediaService never raises for search/download failures -
            # it reports them in VisualResult.error instead. Surface that in
            # pipeline state (and keep the VisualResult itself for inspection).
            return {
                "visual_plan": plan,
                "visual_result": result,
                "status": "failed",
                "error": f"Visual media generation failed: {result.error}",
            }

        return {"visual_plan": plan, "visual_result": result, "status": "visualized", "error": None}

    async def visual_qc_node(state: PipelineState) -> dict:
        try:
            # The exact VisualPlan/VisualResult already produced by the
            # media stage are passed straight through - nothing is
            # re-planned or re-selected except bounded, QC-driven
            # replacements (via VisualMediaService.acquire_replacement_asset,
            # VisualQCService's own existing mechanism - not duplicated here).
            qc_result, qc_visual_result = await visual_qc_service.run_qc(
                state.topic, state.script_result, state.visual_plan, state.visual_result
            )
        except VisualQCServiceError as e:
            return {
                "visual_qc_result": None,
                "qc_approved_visual_result": None,
                "status": "failed",
                "error": f"Visual QC failed: {e}",
            }
        except Exception as e:
            return {
                "visual_qc_result": None,
                "qc_approved_visual_result": None,
                "status": "failed",
                "error": f"Unexpected visual QC error: {e}",
            }

        if not qc_result.success:
            # VisualQCService itself couldn't run (e.g. it was handed an
            # already-unsuccessful VisualResult) - surfaced in
            # VisualQCResult.error. Keep both results for inspection.
            return {
                "visual_qc_result": qc_result,
                "qc_approved_visual_result": qc_visual_result,
                "status": "failed",
                "error": f"Visual QC failed: {qc_result.error}",
            }

        if qc_result.rejected_count > 0:
            # At least one asset is still flagged misleading/conflicting
            # after bounded replacement was exhausted - required media
            # could not be safely approved or replaced. Stop before Video
            # Assembly rather than risk a misleading clip reaching the
            # final video; earlier-stage results are preserved below.
            return {
                "visual_qc_result": qc_result,
                "qc_approved_visual_result": qc_visual_result,
                "status": "failed",
                "error": (
                    f"Visual QC rejected {qc_result.rejected_count} asset(s) as misleading "
                    "with no safe replacement available"
                ),
            }

        return {
            "visual_qc_result": qc_result,
            "qc_approved_visual_result": qc_visual_result,
            "status": "qc_passed",
            "error": None,
        }

    async def video_assembly_node(state: PipelineState) -> dict:
        try:
            # ScriptResult/VoiceResult already produced earlier in this run
            # are passed straight through, and Video Assembly consumes the
            # post-QC qc_approved_visual_result (never the raw visual_result)
            # - nothing is regenerated, re-synthesized, or re-downloaded.
            result = await video_service.assemble_video(
                state.script_result, state.voice_result, state.qc_approved_visual_result
            )
        except VideoAssemblyServiceError as e:
            return {
                "video_assembly_result": None,
                "status": "failed",
                "error": f"Video assembly failed: {e}",
            }
        except Exception as e:
            return {
                "video_assembly_result": None,
                "status": "failed",
                "error": f"Unexpected video assembly error: {e}",
            }

        if not result.success:
            # VideoAssemblyService never raises for FFmpeg/processing
            # failures - it reports them in VideoAssemblyResult.error
            # instead. Surface that in pipeline state (and keep the
            # VideoAssemblyResult itself for inspection).
            return {
                "video_assembly_result": result,
                "status": "failed",
                "error": f"Video assembly failed: {result.error}",
            }

        return {"video_assembly_result": result, "status": "assembled", "error": None}

    async def caption_node(state: PipelineState) -> dict:
        try:
            # The exact VoiceResult/VideoAssemblyResult already produced
            # earlier in this run are passed straight through - narration
            # is never regenerated and video is never reassembled. Caption
            # timing comes entirely from CaptionService's own transcription
            # of the real narration audio, not from any script-derived estimate.
            result = await caption_service.generate_captions(state.voice_result, state.video_assembly_result)
        except CaptionServiceError as e:
            return {"caption_result": None, "status": "failed", "error": f"Caption generation failed: {e}"}
        except Exception as e:
            return {
                "caption_result": None,
                "status": "failed",
                "error": f"Unexpected caption error: {e}",
            }

        if not result.success:
            # CaptionService never raises for transcription/rendering
            # failures - it reports them in CaptionResult.error instead.
            # Surface that in pipeline state (and keep the CaptionResult
            # itself for inspection); the original assembled MP4 is
            # untouched regardless (CaptionService never writes to it).
            return {
                "caption_result": result,
                "status": "failed",
                "error": f"Caption generation failed: {result.error}",
            }

        return {"caption_result": result, "status": "captioned", "error": None}

    async def bgm_node(state: PipelineState) -> dict:
        try:
            # The exact ScriptResult already produced earlier in this run is
            # passed straight through for mood planning - Research/Script
            # are never re-run just to get BGM context (unlike the
            # standalone demo, which has no PipelineState to read a
            # ScriptResult from and must reconstruct one from an existing
            # .srt transcript instead). BGM mixes onto the captioned MP4
            # CaptionService just produced, not the pre-caption assembly.
            result = await audio_mixing_service.generate_mix(
                state.topic, state.script_result, _captioned_video_result(state.caption_result)
            )
        except AudioMixingServiceError as e:
            return {"audio_mix_result": None, "status": "failed", "error": f"BGM mixing failed: {e}"}
        except Exception as e:
            return {
                "audio_mix_result": None,
                "status": "failed",
                "error": f"Unexpected BGM mixing error: {e}",
            }

        if not result.success:
            # AudioMixingService never raises for catalog/selection/mixing
            # failures - it reports them in AudioMixResult.error instead
            # (including a semantic mood-planning failure, which is NOT a
            # mixing failure: MusicContextPlanner already fell back to a
            # deterministic MusicPlan internally and mixing still would
            # have been attempted - see AudioMixResult.music_plan.fallback_reason
            # for that case). Surface the actual failure here; the
            # captioned MP4 is untouched regardless (AudioMixingService
            # never writes to it, always to a new copy).
            return {
                "audio_mix_result": result,
                "status": "failed",
                "error": f"BGM mixing failed: {result.error}",
            }

        return {"audio_mix_result": result, "status": "mixed", "error": None}

    async def metadata_node(state: PipelineState) -> dict:
        try:
            # The exact ScriptResult already produced earlier in this run is
            # passed straight through (never reconstructed from an .srt
            # transcript - see MetadataAgent's docstring/module comment for
            # why that fallback only exists for the standalone demo).
            # Chapter timestamps are derived deterministically inside
            # MetadataAgent from this same multi-section ScriptResult and
            # the real BGM-mixed video's own probed duration - the LLM
            # supplies only chapter labels, never timestamps.
            duration = state.audio_mix_result.output_duration_seconds or state.audio_mix_result.source_duration_seconds
            # video_slug pins the metadata JSON's filename to THIS exact
            # run's own video (never a content-derived slug that could
            # collide with another run/remediation attempt whose LLM
            # happened to generate a similar title - see thumbnail_node's
            # identical fix for the corruption this previously caused).
            video_slug = (
                original_base_name(state.audio_mix_result.output_path) if state.audio_mix_result.output_path else None
            )
            result = metadata_agent.generate_metadata(
                state.topic, state.script_result, duration_seconds=duration, video_slug=video_slug
            )
        except MetadataAgentError as e:
            return {"metadata_result": None, "status": "failed", "error": f"Metadata generation failed: {e}"}
        except Exception as e:
            return {
                "metadata_result": None,
                "status": "failed",
                "error": f"Unexpected metadata error: {e}",
            }

        if not result.success:
            # MetadataAgent never raises for LLM/parsing/validation
            # failures - it reports them in MetadataResult.error instead
            # (a chapters-only issue does NOT reach here: MetadataAgent
            # already degrades that to chapters_available=False while
            # still returning success=True - see MetadataResult.chapters_
            # omitted_reason for that case). Surface the actual failure
            # here; the final BGM-mixed MP4 and every earlier result are
            # untouched regardless (MetadataAgent never writes to video
            # files, only its own JSON artifact).
            return {
                "metadata_result": result,
                "status": "failed",
                "error": f"Metadata generation failed: {result.error}",
            }

        return {"metadata_result": result, "status": "metadata_generated", "error": None}

    async def thumbnail_node(state: PipelineState) -> dict:
        try:
            # The exact ScriptResult already produced earlier in this run is
            # passed straight through (Research/Script are never re-run for
            # thumbnail planning), and the real MetadataResult already
            # produced by the metadata stage supplies extra title/SEO
            # context - Metadata is never re-run either.
            # video_slug pins the thumbnail's filename to THIS exact run's
            # own video (never a hook-text-derived slug, which two
            # different runs/remediation attempts can independently
            # generate identically and silently overwrite each other's
            # file on disk - a real corruption observed and root-caused in
            # an earlier milestone).
            video_slug = (
                original_base_name(state.audio_mix_result.output_path)
                if state.audio_mix_result and state.audio_mix_result.output_path
                else None
            )
            result = await thumbnail_agent.generate_thumbnail(
                state.topic,
                state.script_result,
                metadata_title=state.metadata_result.title,
                seo_summary=state.metadata_result.seo_summary,
                video_slug=video_slug,
            )
        except ThumbnailAgentError as e:
            return {"thumbnail_result": None, "status": "failed", "error": f"Thumbnail generation failed: {e}"}
        except Exception as e:
            return {
                "thumbnail_result": None,
                "status": "failed",
                "error": f"Unexpected thumbnail error: {e}",
            }

        if not result.success:
            # ThumbnailAgent never raises for search/selection/rendering/
            # validation failures - it reports them in ThumbnailResult.error
            # instead. Surface that in pipeline state (and keep the
            # ThumbnailResult itself for inspection); the final BGM-mixed
            # MP4, metadata, and every earlier result are untouched
            # regardless (ThumbnailAgent never writes to them, only its own
            # thumbnail image).
            return {
                "thumbnail_result": result,
                "status": "failed",
                "error": f"Thumbnail generation failed: {result.error}",
            }

        # Persist provenance for THIS exact run immediately, right after the
        # last stage that contributes to it succeeds - never deferred until
        # the whole pipeline finishes. compliance_node (next) looks this
        # manifest back up by the same real audio_mix_result.output_path,
        # so it can never accidentally read a previous run's manifest. A
        # write failure here is best-effort/non-fatal (see
        # persist_provenance_if_completed) and never fails thumbnail
        # generation, which has already genuinely succeeded.
        manifest_state = PipelineState(
            topic=state.topic,
            audio_mix_result=state.audio_mix_result,
            qc_approved_visual_result=state.qc_approved_visual_result,
            thumbnail_result=result,
        )
        persist_provenance_if_completed(manifest_state, provenance_output_dir)

        return {"thumbnail_result": result, "status": "thumbnail_generated", "error": None}

    async def compliance_node(state: PipelineState) -> dict:
        # The current run's own provenance manifest, written by
        # thumbnail_node above using this exact same audio_mix_result -
        # never a previous run's manifest, and never reconstructed from
        # filenames. A missing manifest (write failed, or somehow absent)
        # is treated as unavailable, never raised - ComplianceAgent already
        # reports that as a conservative REVIEW-contributing warning rather
        # than fabricating provenance.
        video_path = state.audio_mix_result.output_path if state.audio_mix_result else None
        try:
            provenance_manifest = (
                ProvenanceManifestStore(provenance_output_dir).find_for_video(video_path) if video_path else None
            )
        except ProvenanceStoreError:
            provenance_manifest = None

        try:
            # The exact ScriptResult/MetadataResult/ThumbnailResult already
            # produced earlier in this same run are passed straight through
            # - Research/Script/Metadata/Thumbnail are never re-run for
            # compliance review.
            result = compliance_agent.review_compliance(
                state.topic,
                final_video_path=video_path,
                metadata_result=state.metadata_result,
                thumbnail_path=state.thumbnail_result.output_path if state.thumbnail_result else None,
                script=state.script_result,
                thumbnail_result=state.thumbnail_result,
                provenance_manifest=provenance_manifest,
            )
        except ComplianceAgentError as e:
            return {"compliance_result": None, "status": "failed", "error": f"Compliance review failed: {e}"}
        except Exception as e:
            return {
                "compliance_result": None,
                "status": "failed",
                "error": f"Unexpected compliance error: {e}",
            }

        # Persist a durable compliance record for THIS exact candidate -
        # for every decision (PASS/REVIEW/BLOCK), not just PASS. Best-
        # effort/non-fatal (never retroactively fails an already-completed
        # review), exactly like provenance persistence above. run_id is the
        # same identifier ProvenanceManifestStore/PublishingRecordStore
        # already key by (original_base_name of this exact video), so a
        # later Upload Agent's compliance-gate lookup for this candidate
        # always finds the record this exact evaluation just wrote.
        run_id = original_base_name(video_path) if video_path else None
        if run_id:
            try:
                compliance_record_store.write(
                    run_id,
                    result,
                    parent_run_id=state.remediation_parent_run_id,
                    attempt_number=state.remediation_attempt,
                    evaluated_at=datetime.now(timezone.utc).isoformat(),
                )
            except OSError:
                pass

        # Backfill the outcome of whichever remediation attempt led to THIS
        # evaluation (if any) - the attempt record itself was created by
        # script_revision_node before this candidate existed, and only now
        # do we know what it actually produced.
        remediation_history = state.remediation_history
        if remediation_history and remediation_history[-1].resulting_decision is None:
            updated_last = remediation_history[-1].model_copy(
                update={
                    "resulting_run_id": run_id,
                    "resulting_decision": result.publish_decision,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            remediation_history = remediation_history[:-1] + [updated_last]

        if result.publish_decision == "BLOCK":
            # A known deterministic compliance blocker was found - never
            # treated as PASS, and NEVER remediated (a blocker is outside
            # what a script rewrite can fix - see route_after_compliance).
            # Every earlier artifact (final video, metadata, thumbnail) is
            # preserved untouched; only the pipeline's own status/error
            # reflect the blocked outcome.
            blocker_summary = "; ".join(result.blockers) or "see compliance_result for details"
            return {
                "compliance_result": result,
                "remediation_history": remediation_history,
                "status": "blocked",
                "error": f"Compliance blocked publishing: {blocker_summary}",
            }

        if result.publish_decision == "REVIEW":
            # Uncertainty or a non-blocking concern requiring human/another
            # system review - never silently treated as approved.
            # route_after_compliance (below) independently makes the exact
            # same actionability/attempt-budget check to decide whether to
            # loop back into script_revision; this only determines the
            # STATUS this evaluation reports if the pipeline stops here
            # (either because it's about to loop - in which case a LATER
            # compliance_node call overwrites this - or because it's truly
            # terminal).
            actionable, _ = _remediation_eligibility(result, state.script_result)
            attempts_exhausted = state.remediation_attempt >= max_remediation_attempts
            final_status = "review_exhausted" if (actionable and attempts_exhausted) else "review_required"
            warning_summary = "; ".join(result.warnings) or "see compliance_result for details"
            return {
                "compliance_result": result,
                "remediation_history": remediation_history,
                "status": final_status,
                "error": f"Compliance requires review ({final_status}): {warning_summary}",
            }

        return {
            "compliance_result": result,
            "remediation_history": remediation_history,
            "status": "completed",
            "error": None,
        }

    async def script_revision_node(state: PipelineState) -> dict:
        """Targeted Compliance Remediation: revise ONLY the ScriptSection(s)
        route_after_compliance already confirmed are confidently localized
        and actionable, grounded in the original ResearchResult (or one
        bounded, single-claim research refresh when that evidence is
        insufficient), then loop back into Voice so every downstream
        artifact this correction invalidates gets a genuinely fresh
        candidate - never a patched/spliced one.

        Never fabricates a correction: if the LLM revision call itself
        fails, this attempt stops here (routes to END via
        route_after_script_revision) with the prior REVIEW result/status
        left exactly as compliance_node reported it - never retried
        silently, never treated as fixed.
        """
        result = state.compliance_result
        findings = result.semantic_review.findings if result and result.semantic_review else []
        localized = localize_findings(findings, state.script_result)
        actionable = [f for f in localized if f.actionable]

        current_run_id = (
            original_base_name(state.audio_mix_result.output_path)
            if state.audio_mix_result and state.audio_mix_result.output_path
            else None
        )
        # parent_run_id always points at the ORIGINAL (attempt 0) run,
        # never at the immediately-prior attempt - set once, on the first
        # remediation attempt, and carried forward unchanged after that.
        parent_run_id = state.remediation_parent_run_id or current_run_id
        attempt_number = state.remediation_attempt + 1
        started_at = datetime.now(timezone.utc).isoformat()

        revised_script = state.script_result
        corrections: List[ScriptCorrection] = []
        for localized_finding in actionable:
            section = revised_script.sections[localized_finding.section_index]
            refreshed_fact = None
            used_refresh = False
            refresh_reason = None
            if not has_grounding_evidence(state.research_result, localized_finding.finding_description):
                # Existing research doesn't obviously cover this claim -
                # one bounded, single-claim refresh (never a full
                # re-research) before attempting the correction.
                refreshed_fact = await research_agent.research_focused_claim(
                    state.topic, localized_finding.finding_description
                )
                used_refresh = True
                if refreshed_fact is None:
                    refresh_reason = "Bounded research refresh found no usable evidence for this claim"

            try:
                revised_script = script_agent.revise_section(
                    revised_script,
                    state.research_result,
                    localized_finding.section_index,
                    localized_finding.finding_description,
                    refreshed_fact=refreshed_fact,
                )
            except ScriptAgentError as e:
                # Cannot safely correct this finding - stop remediation for
                # this attempt cleanly rather than fabricate a fix. The
                # prior REVIEW compliance_result/status are left untouched,
                # so the pipeline still reports a genuine, real outcome.
                return {
                    "remediation_attempt": attempt_number,
                    "remediation_parent_run_id": parent_run_id,
                    "error": f"Compliance remediation attempt {attempt_number} could not revise "
                    f"section '{section.heading}': {e}",
                }

            corrections.append(
                ScriptCorrection(
                    section_index=localized_finding.section_index,
                    section_heading=section.heading,
                    original_narration=section.narration,
                    revised_narration=revised_script.sections[localized_finding.section_index].narration,
                    finding_description=localized_finding.finding_description,
                    used_research_refresh=used_refresh,
                    research_refresh_reason=refresh_reason,
                )
            )

        record = RemediationAttemptRecord(
            attempt_number=attempt_number,
            parent_run_id=parent_run_id,
            triggering_decision="REVIEW",
            findings_considered=len(findings),
            findings_localized=len(actionable),
            corrections=corrections,
            started_at=started_at,
        )

        return {
            "script_result": revised_script,
            "remediation_attempt": attempt_number,
            "remediation_parent_run_id": parent_run_id,
            "remediation_history": state.remediation_history + [record],
            "status": "revising",
            "error": None,
        }

    def _remediation_eligibility(
        result: Optional[ComplianceResult], script_result: Optional[ScriptResult]
    ) -> Tuple[bool, List[LocalizedFinding]]:
        """Whether ``result`` has at least one confidently-localized,
        actionable semantic finding a script correction could address -
        the single shared check compliance_node/route_after_compliance
        both use, so "is this remediable" is decided in exactly one place."""
        if result is None or result.publish_decision != "REVIEW" or script_result is None:
            return False, []
        findings = result.semantic_review.findings if result.semantic_review else []
        localized = localize_findings(findings, script_result)
        return any(f.actionable for f in localized), localized

    def route_after_research(state: PipelineState) -> str:
        """Only proceed to scripting if research actually produced a result."""
        return "script" if state.research_result is not None else END

    def route_after_script(state: PipelineState) -> str:
        """Only proceed to voice generation if scripting actually succeeded."""
        return "voice" if state.script_result is not None else END

    def route_after_voice(state: PipelineState) -> str:
        """Only proceed to visual media generation if voice actually succeeded."""
        return "media" if state.voice_result is not None and state.voice_result.success else END

    def route_after_media(state: PipelineState) -> str:
        """Only proceed to Visual QC if visual media actually succeeded."""
        return (
            "visual_qc"
            if state.visual_result is not None and state.visual_result.success
            else END
        )

    def route_after_visual_qc(state: PipelineState) -> str:
        """Only proceed to video assembly once Visual QC has approved (or
        safely replaced) every section's media - never after a hard QC
        failure (see visual_qc_node's rejected_count policy above)."""
        return "video_assembly" if state.status == "qc_passed" else END

    def route_after_video_assembly(state: PipelineState) -> str:
        """Only proceed to captioning if video assembly actually succeeded -
        the caption node must never run on a missing/failed assembled MP4."""
        return (
            "captions"
            if state.video_assembly_result is not None and state.video_assembly_result.success
            else END
        )

    def route_after_captions(state: PipelineState) -> str:
        """Only proceed to BGM mixing if captioning actually succeeded - the
        bgm node must never run on a missing/failed captioned MP4."""
        return "bgm" if state.caption_result is not None and state.caption_result.success else END

    def route_after_bgm(state: PipelineState) -> str:
        """Only proceed to metadata generation if BGM mixing actually
        succeeded - the metadata node must never run on a missing/failed
        final mixed MP4."""
        return "metadata" if state.audio_mix_result is not None and state.audio_mix_result.success else END

    def route_after_metadata(state: PipelineState) -> str:
        """Only proceed to thumbnail generation if metadata generation
        actually succeeded - the thumbnail node must never run on
        missing/failed metadata."""
        return "thumbnail" if state.metadata_result is not None and state.metadata_result.success else END

    def route_after_thumbnail(state: PipelineState) -> str:
        """Only proceed to compliance review if thumbnail generation
        actually succeeded - the compliance node must never run on
        missing/failed thumbnail generation."""
        return "compliance" if state.thumbnail_result is not None and state.thumbnail_result.success else END

    def route_after_compliance(state: PipelineState) -> str:
        """PASS or BLOCK always end here - a BLOCK is never remediated
        (see compliance_node). A REVIEW loops back into script_revision
        only when at least one finding is confidently localized/actionable
        AND the attempt budget isn't exhausted yet; otherwise it ends here
        too (compliance_node has already set the correct terminal status -
        review_required or review_exhausted)."""
        if state.remediation_attempt >= max_remediation_attempts:
            return END
        actionable, _ = _remediation_eligibility(state.compliance_result, state.script_result)
        return "script_revision" if actionable else END

    def route_after_script_revision(state: PipelineState) -> str:
        """Only loop back into Voice if script_revision_node actually
        produced a corrected candidate (status == "revising") - a revision
        failure routes straight to END, preserving the prior genuine
        REVIEW result/status untouched rather than looping on a fabricated
        or partial correction."""
        return "voice" if state.status == "revising" else END

    graph = StateGraph(PipelineState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("voice", voice_node)
    graph.add_node("media", media_node)
    graph.add_node("visual_qc", visual_qc_node)
    graph.add_node("video_assembly", video_assembly_node)
    graph.add_node("captions", caption_node)
    graph.add_node("bgm", bgm_node)
    graph.add_node("metadata", metadata_node)
    graph.add_node("thumbnail", thumbnail_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("script_revision", script_revision_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"script": "script", END: END})
    graph.add_conditional_edges("script", route_after_script, {"voice": "voice", END: END})
    graph.add_conditional_edges("voice", route_after_voice, {"media": "media", END: END})
    graph.add_conditional_edges("media", route_after_media, {"visual_qc": "visual_qc", END: END})
    graph.add_conditional_edges(
        "visual_qc", route_after_visual_qc, {"video_assembly": "video_assembly", END: END}
    )
    graph.add_conditional_edges(
        "video_assembly", route_after_video_assembly, {"captions": "captions", END: END}
    )
    graph.add_conditional_edges("captions", route_after_captions, {"bgm": "bgm", END: END})
    graph.add_conditional_edges("bgm", route_after_bgm, {"metadata": "metadata", END: END})
    graph.add_conditional_edges("metadata", route_after_metadata, {"thumbnail": "thumbnail", END: END})
    graph.add_conditional_edges("thumbnail", route_after_thumbnail, {"compliance": "compliance", END: END})
    # Bounded cycle: compliance -> script_revision -> voice -> ... ->
    # compliance again, at most max_remediation_attempts times (see
    # route_after_compliance/route_after_script_revision) - reuses every
    # existing downstream node unchanged, since a corrected script_result
    # flowing back through voice_node/media_node/etc. is transparent to them.
    graph.add_conditional_edges(
        "compliance", route_after_compliance, {"script_revision": "script_revision", END: END}
    )
    graph.add_conditional_edges("script_revision", route_after_script_revision, {"voice": "voice", END: END})

    return graph


async def run_pipeline(
    topic: str,
    search_provider,
    llm_provider,
    voice_provider: VoiceProvider,
    voice_name: str,
    media_provider: MediaProvider,
    assembler: VideoAssembler,
    visual_relevance_evaluator: VisualRelevanceEvaluator,
    transcription_provider: TranscriptionProvider,
    music_catalog_provider: MusicCatalogProvider,
    voice_output_dir: str = DEFAULT_OUTPUT_DIR,
    media_output_dir: str = DEFAULT_MEDIA_OUTPUT_DIR,
    video_output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    subtitle_output_dir: str = DEFAULT_SUBTITLE_OUTPUT_DIR,
    thumbnail_output_dir: str = DEFAULT_THUMBNAIL_OUTPUT_DIR,
    metadata_output_dir: str = DEFAULT_METADATA_OUTPUT_DIR,
    provenance_output_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR,
    compliance_output_dir: str = DEFAULT_COMPLIANCE_RECORD_DIR,
    max_remediation_attempts: int = DEFAULT_MAX_REMEDIATION_ATTEMPTS,
) -> PipelineState:
    """Run the full Research -> Script -> Voice -> Visual Media -> Visual QC
    -> Video Assembly -> Subtitle/Caption -> BGM/Audio Mixing -> Metadata ->
    Thumbnail -> Copyright/Compliance pipeline and return the final state.

    Provenance persistence happens INSIDE the graph, not as post-run
    processing: ``thumbnail_node`` persists a machine-readable
    ProvenanceManifest for the current run immediately on its own success
    (via ``persist_provenance_if_completed``), and ``compliance_node`` looks
    that exact manifest back up (via ``ProvenanceManifestStore.find_for_video``)
    before running its checks - this guarantees Compliance always reviews
    the current run's real provenance, never a previous run's, and never a
    filename-guessed reconstruction. No individual node writes provenance
    for any run but its own, and a write failure never retroactively fails
    an already-succeeded stage (see src.services.provenance_collection).

    Unlike the individual research/script workflow convenience functions
    (which raise on failure), this returns the full PipelineState so callers
    can inspect ``status``/``error`` directly, including on partial failure
    (e.g. every stage through thumbnail generation succeeded but compliance
    review returned BLOCK).

    Each stage only runs if the previous one succeeded - a failure at any
    stage short-circuits the rest of the pipeline and it never silently
    continues (see route_after_research/route_after_script/route_after_voice/
    route_after_media/route_after_visual_qc/route_after_video_assembly/
    route_after_captions/route_after_bgm/route_after_metadata/
    route_after_thumbnail). Visual QC itself fails the pipeline (status
    "failed", Video Assembly never runs) if any asset is still flagged
    misleading/conflicting after bounded replacement is exhausted - see
    visual_qc_node. Pipeline status only becomes "completed" once
    compliance review itself returns a PASS decision; a semantic mood-
    planning failure inside BGM, or a chapters-only issue inside metadata
    generation, does not itself fail the pipeline (both degrade to a safe
    fallback and continue) - only an actual mixing/selection/catalog
    failure, a total metadata generation failure, a total thumbnail
    generation failure, or a total compliance-review failure (as opposed to
    a completed review that itself returns REVIEW/BLOCK - see
    compliance_node), does. A compliance REVIEW decision sets status
    "review_required" and a BLOCK decision sets status "blocked" - neither
    is ever reported as "completed", and every earlier-stage result
    (including the final BGM-mixed MP4, generated metadata, and rendered
    thumbnail) is preserved unchanged in all three outcomes.

    Args:
        topic: Research topic to investigate, script, narrate, illustrate, and assemble
        search_provider: SearchProvider implementation for the Research Agent
        llm_provider: LLMProvider implementation shared by Research, Script,
            Visual Context Planning, Visual QC, BGM mood planning,
            Thumbnail planning, and the Compliance semantic review
        voice_provider: VoiceProvider implementation for the Voice Service
        voice_name: Provider-specific voice identifier for the Voice Service
        media_provider: MediaProvider implementation shared by the Visual
            Media Service and the Thumbnail Agent
        assembler: VideoAssembler implementation, shared by Visual QC, the
            Video Assembly Service, the Caption Service, and Audio Mixing
        visual_relevance_evaluator: VisualRelevanceEvaluator implementation
            for the Visual QC Service
        transcription_provider: TranscriptionProvider implementation for
            the Caption Service
        music_catalog_provider: MusicCatalogProvider implementation (the
            approved BGM catalog) shared by the Audio Mixing Service and
            the Compliance Agent's BGM provenance cross-check
        voice_output_dir: Directory the Voice Service writes audio files into
        media_output_dir: Directory the Visual Media Service writes assets into
        video_output_dir: Directory the Video Assembly Service writes the final MP4 into
        subtitle_output_dir: Directory the Caption Service writes .srt files into
        thumbnail_output_dir: Directory the Thumbnail Agent writes the rendered thumbnail into
        metadata_output_dir: Directory the Metadata Agent writes its JSON artifact into
        provenance_output_dir: Directory the provenance manifest is written into
            (by thumbnail_node) and read from (by compliance_node)
        compliance_output_dir: Directory ComplianceRecordStore persists a
            durable record into for every PASS/REVIEW/BLOCK decision
        max_remediation_attempts: Bounded retry limit for Compliance
            Remediation - 0 disables remediation entirely

    Returns:
        Final PipelineState (check ``.status``/``.error`` for outcome)
    """
    graph = build_pipeline_graph(
        search_provider,
        llm_provider,
        voice_provider,
        voice_name,
        media_provider,
        assembler,
        visual_relevance_evaluator,
        transcription_provider,
        music_catalog_provider,
        voice_output_dir,
        media_output_dir,
        video_output_dir,
        subtitle_output_dir,
        thumbnail_output_dir,
        metadata_output_dir,
        provenance_output_dir,
        compliance_output_dir,
        max_remediation_attempts,
    ).compile()
    initial_state = PipelineState(topic=topic, status="researching")

    raw_result = await graph.ainvoke(initial_state)

    return PipelineState(
        topic=raw_result.get("topic", topic),
        research_result=raw_result.get("research_result"),
        script_result=raw_result.get("script_result"),
        voice_result=raw_result.get("voice_result"),
        visual_plan=raw_result.get("visual_plan"),
        visual_result=raw_result.get("visual_result"),
        visual_qc_result=raw_result.get("visual_qc_result"),
        qc_approved_visual_result=raw_result.get("qc_approved_visual_result"),
        video_assembly_result=raw_result.get("video_assembly_result"),
        caption_result=raw_result.get("caption_result"),
        audio_mix_result=raw_result.get("audio_mix_result"),
        metadata_result=raw_result.get("metadata_result"),
        thumbnail_result=raw_result.get("thumbnail_result"),
        compliance_result=raw_result.get("compliance_result"),
        remediation_attempt=raw_result.get("remediation_attempt", 0),
        remediation_parent_run_id=raw_result.get("remediation_parent_run_id"),
        remediation_history=raw_result.get("remediation_history", []),
        status=raw_result.get("status", "unknown"),
        error=raw_result.get("error"),
    )
