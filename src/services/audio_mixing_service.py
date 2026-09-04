# Audio Mixing Service: plans BGM mood/context, selects an approved local
# track, and mixes it under an already-captioned video's existing
# narration audio.
#
# This is a deterministic orchestration service, not an LLM-driven
# reasoning agent - the only "intelligence" involved is mood/context
# planning (delegated to MusicContextPlanner, one LLM call, with a safe
# deterministic fallback); track selection is fixed-rule scoring (see
# MusicSelectionService), and mixing itself is delegated to the existing
# VideoAssembler/FFmpeg infrastructure - no second audio-processing stack.
#
# Standalone for this milestone: not wired into the main LangGraph
# pipeline yet (see src/bgm_demo.py).
from __future__ import annotations

import os
from typing import Optional

from src.agents.music_context_planner import MusicContextPlanner
from src.llm.provider import LLMProvider
from src.models.music import AudioMixResult, BGMTrack, MusicPlan
from src.models.script import ScriptResult
from src.models.video import VideoAssemblyResult
from src.services.music_planning import build_deterministic_music_plan
from src.services.music_selection_service import MusicSelectionService
from src.tools.ffmpeg_video_assembler import VideoAssembler, VideoAssemblerError
from src.tools.music_catalog_provider import MusicCatalogProvider, MusicCatalogProviderError

# Conservative default: markedly quieter than the narration, so the music
# stays a subtle bed rather than competing for attention.
DEFAULT_BGM_GAIN_DB = -24.0
DEFAULT_FADE_IN_SECONDS = 2.0
DEFAULT_FADE_OUT_SECONDS = 3.0
DEFAULT_USE_DUCKING = True
OUTPUT_SUFFIX = "-bgm"

# How far the final mixed duration is allowed to drift from the source
# video's duration before it's surfaced as a warning (not a failure - a
# successfully produced video is still returned).
DURATION_TOLERANCE_SECONDS = 1.0


class AudioMixingServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Processing failures (missing files, no eligible track, mixing errors)
    are NOT raised - they are captured in the returned AudioMixResult
    (success=False, error=...) so callers always get a structured result
    back, and the source video is never touched on failure.
    """


class AudioMixingService:
    """Deterministic orchestration service that produces a BGM-mixed copy
    of an already-assembled/captioned video, from its real ScriptResult's
    mood/context - never a hardcoded or topic-specific music choice."""

    def __init__(
        self,
        catalog_provider: MusicCatalogProvider,
        assembler: VideoAssembler,
        llm_provider: Optional[LLMProvider] = None,
        selection_service: Optional[MusicSelectionService] = None,
        bgm_gain_db: float = DEFAULT_BGM_GAIN_DB,
        fade_in_seconds: float = DEFAULT_FADE_IN_SECONDS,
        fade_out_seconds: float = DEFAULT_FADE_OUT_SECONDS,
        use_ducking: bool = DEFAULT_USE_DUCKING,
    ) -> None:
        """Initialize the Audio Mixing Service.

        Args:
            catalog_provider: Source of the approved local BGM catalog
            assembler: VideoAssembler implementation, used for duration
                probing and mixing - the same FFmpeg infrastructure Video
                Assembly/Captions already use
            llm_provider: Optional LLMProvider for real mood/context
                planning via MusicContextPlanner. If None, every plan uses
                the deterministic fallback directly (no LLM call at all).
            selection_service: Optional MusicSelectionService override
                (e.g. to configure a fallback_track_id); defaults to one
                with no configured fallback track.
            bgm_gain_db: Gain applied to the selected track before mixing
            fade_in_seconds: Music fade-in duration at the start
            fade_out_seconds: Music fade-out duration, ending at the
                source video's own end
            use_ducking: Whether to sidechain-compress the music under
                narration
        """
        if bgm_gain_db > 0:
            raise AudioMixingServiceError(
                "bgm_gain_db must not amplify music above its source level (use <= 0 dB) - "
                "narration must remain dominant and this avoids clipping risk"
            )

        self.catalog_provider = catalog_provider
        self.assembler = assembler
        self.planner = MusicContextPlanner(llm_provider) if llm_provider is not None else None
        self.selection_service = selection_service or MusicSelectionService()
        self.bgm_gain_db = bgm_gain_db
        self.fade_in_seconds = fade_in_seconds
        self.fade_out_seconds = fade_out_seconds
        self.use_ducking = use_ducking

    async def generate_mix(
        self, topic: str, script: ScriptResult, video_result: VideoAssemblyResult
    ) -> AudioMixResult:
        """Plan mood/context, select an approved track, and mix it under
        ``video_result``'s existing narration audio.

        Never raises for processing failures (missing files, empty/no-
        match catalog, mixing errors) - those are captured in the returned
        AudioMixResult. Only raises AudioMixingServiceError for
        configuration/programmer errors. The source video at
        ``video_result.output_path`` is never modified, on success or
        failure.

        Args:
            topic: Overall video topic
            script: Structured script produced by the Script Agent
            video_result: Already-assembled/captioned MP4 to mix BGM onto

        Returns:
            Structured AudioMixResult describing the outcome
        """
        if script is None or video_result is None:
            raise AudioMixingServiceError("ScriptResult and VideoAssemblyResult are both required")

        if not video_result.success or not video_result.output_path:
            return self._failure(
                "VideoAssemblyResult was not successful; cannot mix BGM without a source video"
            )
        if not os.path.exists(video_result.output_path):
            return self._failure(f"Source video file not found: {video_result.output_path}")

        music_plan = self._plan_music(topic, script)

        try:
            catalog = self.catalog_provider.list_tracks()
        except MusicCatalogProviderError as e:
            return self._failure(f"Failed to load BGM catalog: {e}", music_plan=music_plan)

        track, selection_warnings = self.selection_service.select_track(music_plan, catalog)
        if track is None:
            return self._failure(
                "No suitable licensed BGM track available: " + "; ".join(selection_warnings),
                music_plan=music_plan,
            )
        if not os.path.exists(track.file_path):
            return self._failure(
                f"Selected BGM track's audio file not found: {track.file_path}",
                music_plan=music_plan,
                selected_track=track,
            )

        try:
            source_duration = self.assembler.probe_duration_seconds(video_result.output_path)
        except VideoAssemblerError as e:
            return self._failure(
                f"Failed to probe source video duration: {e}", music_plan=music_plan, selected_track=track
            )

        looped = self._is_looped(track, source_duration)
        output_path = self._build_output_path(video_result.output_path)
        fade_out_seconds = min(self.fade_out_seconds, max(source_duration / 2.0, 0.1))

        try:
            self.assembler.mix_background_audio(
                input_video_path=video_result.output_path,
                music_path=track.file_path,
                output_path=output_path,
                target_duration_seconds=source_duration,
                music_gain_db=self.bgm_gain_db,
                fade_in_seconds=self.fade_in_seconds,
                fade_out_seconds=fade_out_seconds,
                use_ducking=self.use_ducking,
            )
        except VideoAssemblerError as e:
            return self._failure(f"Audio mixing failed: {e}", music_plan=music_plan, selected_track=track)

        if not os.path.exists(output_path):
            return self._failure(
                "Audio mixing reported success but the output file is missing",
                music_plan=music_plan,
                selected_track=track,
            )

        try:
            output_duration = self.assembler.probe_duration_seconds(output_path)
        except VideoAssemblerError:
            output_duration = None

        warnings = list(selection_warnings)
        if output_duration is not None and abs(output_duration - source_duration) > DURATION_TOLERANCE_SECONDS:
            warnings.append(
                f"Output duration ({output_duration:.2f}s) differs from source video duration "
                f"({source_duration:.2f}s) by more than {DURATION_TOLERANCE_SECONDS}s"
            )

        return AudioMixResult(
            success=True,
            output_path=output_path,
            source_video_path=video_result.output_path,
            music_plan=music_plan,
            selected_track=track,
            bgm_gain_db=self.bgm_gain_db,
            ducking_used=self.use_ducking,
            looped=looped,
            source_duration_seconds=source_duration,
            output_duration_seconds=output_duration,
            warnings=warnings,
        )

    # ---- helpers ------------------------------------------------------------

    def _plan_music(self, topic: str, script: ScriptResult) -> MusicPlan:
        if self.planner is not None:
            return self.planner.plan_music(topic, script)
        return build_deterministic_music_plan(topic, script)

    def _is_looped(self, track: BGMTrack, source_duration: float) -> bool:
        try:
            track_duration = self.assembler.probe_duration_seconds(track.file_path)
        except VideoAssemblerError:
            track_duration = track.duration_seconds
        return bool(track_duration and track_duration < source_duration)

    @staticmethod
    def _build_output_path(video_output_path: str) -> str:
        root, ext = os.path.splitext(video_output_path)
        return f"{root}{OUTPUT_SUFFIX}{ext}"

    @staticmethod
    def _failure(
        error: str, music_plan: Optional[MusicPlan] = None, selected_track: Optional[BGMTrack] = None
    ) -> AudioMixResult:
        return AudioMixResult(success=False, error=error, music_plan=music_plan, selected_track=selected_track)
