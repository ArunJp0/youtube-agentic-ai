# Wires the real AutonomousRunController together from Settings for
# production use (the scheduler and `--run-once`/`--dry-run` CLI).
#
# Deliberately NOT business logic - every provider here comes from the same
# `src.config.providers` factories `pipeline_demo.py`/`topic_planner_demo.py`
# already use, and the pipeline call itself is exactly `run_pipeline(...)`.
# This module only assembles those existing pieces; it must never grow its
# own Research/Script/Video/Compliance/YouTube behavior.
from __future__ import annotations

from typing import List, Optional

from src.agents.research import ResearchAgent
from src.agents.topic_planner_agent import TopicPlannerAgent
from src.agents.topic_ranking_planner import TopicRankingPlanner
from src.config.providers import (
    get_ai_video_provider,
    get_current_news_search_provider,
    get_llm_provider,
    get_media_provider,
    get_search_provider,
    get_topic_planner_source,
    get_topic_source_provider,
    get_transcription_provider,
    get_visual_relevance_evaluator,
    get_voice_provider,
)
from src.config.settings import Settings
from src.models.youtube_upload import PublishingIntent, YOUTUBE_UPLOAD_SCOPES
from src.orchestration.autonomous_run_controller import AutonomousRunController, PipelineRunner
from src.orchestration.topic_continuity_orchestrator import TopicContinuityOrchestrator
from src.services.autonomous_run_store import AutonomousRunStore, DEFAULT_AUTONOMOUS_RUN_DIR
from src.services.run_lock import DEFAULT_LOCK_PATH, RunLock
from src.services.script_duration import calculate_word_budget, resolve_target_duration_minutes
from src.services.topic_reserve_store import QualifiedTopicReserveStore
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler
from src.tools.neutral_visual_generator import FFmpegNeutralVisualGenerator
from src.tools.music_catalog_provider import LocalMusicCatalogProvider
from src.tools.youtube_client import GoogleYouTubeClient, YouTubeClient
from src.workflows.pipeline_graph import PipelineState, run_pipeline


def _parse_csv_setting(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def build_publishing_intent(settings: Settings) -> PublishingIntent:
    """Build the ``PublishingIntent`` driven entirely by
    ``AUTONOMOUS_PUBLISHING_MODE`` - defaults to ``"disabled"``, so a fresh
    deployment never accidentally publishes publicly. ``"scheduled"`` mode
    is not derivable from config alone (it needs a concrete future
    timestamp the operator supplies per-run), so it is intentionally not
    auto-selected here - callers wanting a scheduled autonomous publish
    must construct that ``PublishingIntent`` themselves.
    """
    mode = settings.autonomous_publishing_mode
    if mode not in ("disabled", "private"):
        mode = "disabled"
    return PublishingIntent(mode=mode)


def build_topic_planner_agent(settings: Settings) -> TopicPlannerAgent:
    """Construct the real ``TopicPlannerAgent`` from ``Settings``, mirroring
    ``topic_planner_demo.py`` exactly - not a second implementation."""
    topic_source_provider = get_topic_planner_source(settings)
    llm_provider = get_llm_provider(settings)
    ranking_planner = TopicRankingPlanner(llm_provider) if settings.llm_provider != "mock" else None

    target_markets = _parse_csv_setting(settings.topic_target_markets)
    preferred_categories = _parse_csv_setting(settings.topic_categories)

    return TopicPlannerAgent(
        topic_source_provider=topic_source_provider,
        ranking_planner=ranking_planner,
        niche=settings.topic_planner_niche,
        region=settings.topic_planner_region,
        candidate_limit=settings.topic_planner_candidate_limit,
        history_limit=settings.topic_planner_history_limit,
        mode=settings.topic_mode,
        target_markets=target_markets or None,
        preferred_categories=preferred_categories or None,
        freshness_hours=settings.topic_freshness_hours,
    )


def build_evergreen_topic_planner_agent(settings: Settings) -> TopicPlannerAgent:
    """Construct a SEPARATE, dedicated evergreen-only ``TopicPlannerAgent``
    for TIER 2 of the content-continuity strategy - mirrors
    ``build_topic_planner_agent`` exactly except it always uses
    ``get_topic_source_provider`` (the plain evergreen source, bypassing
    ``TOPIC_MODE``'s mixed/trending composition) and forces
    ``mode="evergreen"``, so it is never affected by whatever ``TOPIC_MODE``
    the PRIMARY planner is configured with.
    """
    topic_source_provider = get_topic_source_provider(settings)
    llm_provider = get_llm_provider(settings)
    ranking_planner = TopicRankingPlanner(llm_provider) if settings.llm_provider != "mock" else None

    target_markets = _parse_csv_setting(settings.topic_target_markets)
    preferred_categories = _parse_csv_setting(settings.topic_categories)

    return TopicPlannerAgent(
        topic_source_provider=topic_source_provider,
        ranking_planner=ranking_planner,
        niche=settings.topic_planner_niche,
        region=settings.topic_planner_region,
        candidate_limit=settings.topic_planner_candidate_limit,
        history_limit=settings.topic_planner_history_limit,
        mode="evergreen",
        target_markets=target_markets or None,
        preferred_categories=preferred_categories or None,
        freshness_hours=settings.topic_freshness_hours,
    )


def build_topic_continuity_orchestrator(settings: Settings) -> TopicContinuityOrchestrator:
    """Assemble the real ``TopicContinuityOrchestrator`` (see
    src.orchestration.topic_continuity_orchestrator) from ``Settings`` -
    the SAME real providers ``build_pipeline_runner``'s own internal
    Research step uses, so a TIER 1/2/3 candidate is validated with
    exactly the production Research quality contract before the content
    pipeline is ever invoked.
    """
    research_agent = ResearchAgent(
        search_provider=get_search_provider(settings),
        llm_provider=get_llm_provider(settings),
        current_news_search_provider=get_current_news_search_provider(settings),
    )
    return TopicContinuityOrchestrator(
        primary_topic_planner=build_topic_planner_agent(settings),
        research_agent=research_agent,
        evergreen_topic_planner=build_evergreen_topic_planner_agent(settings),
        reserve_store=QualifiedTopicReserveStore(),
        max_primary_candidates=settings.topic_continuity_max_primary_candidates,
        max_evergreen_candidates=settings.topic_continuity_max_evergreen_candidates,
        max_reserve_attempts=settings.topic_continuity_max_reserve_attempts,
        reserve_seed_attempts=settings.topic_continuity_reserve_seed_attempts,
    )


def _build_youtube_client(settings: Settings) -> Optional[YouTubeClient]:
    from src.tools.youtube_oauth import get_credentials

    credentials = get_credentials(
        client_secret_path=settings.youtube_oauth_client_secret_path,
        token_path=settings.youtube_oauth_token_path,
        scopes=YOUTUBE_UPLOAD_SCOPES,
        interactive=False,
    )
    return GoogleYouTubeClient(credentials)


def build_pipeline_runner(settings: Settings) -> PipelineRunner:
    """Return a closure calling the existing, unchanged ``run_pipeline(...)``
    with real providers built from ``Settings`` - the only thing
    ``AutonomousRunController`` calls to run the content pipeline.

    A ``YouTubeClient`` is only constructed (performing OAuth token
    load/refresh) when ``publishing_intent.mode`` actually requires one, so
    a disabled-publishing autonomous run never touches YouTube credentials
    at all.
    """
    assembler = FFmpegVideoAssembler()
    # Reuses the same real ffmpeg binary the assembler above already
    # requires - a locally-generated neutral background clip, never an
    # external service, used only as Visual QC's absolute last-resort
    # recovery when a section would otherwise have zero safe usable
    # visual coverage (see src.tools.neutral_visual_generator).
    neutral_visual_generator = FFmpegNeutralVisualGenerator()
    search_provider = get_search_provider(settings)
    llm_provider = get_llm_provider(settings)
    voice_provider = get_voice_provider(settings)
    media_provider = get_media_provider(settings)
    visual_relevance_evaluator = get_visual_relevance_evaluator(settings)
    transcription_provider = get_transcription_provider(settings)
    music_catalog_provider = LocalMusicCatalogProvider()
    script_word_budget = calculate_word_budget(
        resolve_target_duration_minutes(settings.script_duration_profile, settings.script_target_duration_minutes)
    )
    ai_video_provider = get_ai_video_provider(settings)
    current_news_search_provider = get_current_news_search_provider(settings)

    async def _run(
        topic: str, topic_source: Optional[str], publishing_intent: Optional[PublishingIntent]
    ) -> PipelineState:
        youtube_client = _build_youtube_client(settings) if publishing_intent and publishing_intent.mode != "disabled" else None
        return await run_pipeline(
            topic,
            search_provider,
            llm_provider,
            voice_provider,
            settings.voice_name,
            media_provider,
            assembler,
            visual_relevance_evaluator,
            transcription_provider,
            music_catalog_provider,
            youtube_client=youtube_client,
            publishing_intent=publishing_intent,
            script_word_budget=script_word_budget,
            ai_video_provider=ai_video_provider,
            ai_video_max_retries=settings.ai_video_max_retries,
            stock_fallback_enabled=settings.stock_fallback_enabled,
            topic_source=topic_source,
            current_news_search_provider=current_news_search_provider,
            neutral_visual_generator=neutral_visual_generator,
        )

    return _run


def build_autonomous_run_controller(settings: Optional[Settings] = None) -> AutonomousRunController:
    """Assemble the production ``AutonomousRunController`` - the single
    object both the scheduler and the manual ``--run-once``/``--dry-run``
    CLI invoke (STEP 10: no separate code path for manual vs scheduled
    execution)."""
    settings = settings or Settings()
    return AutonomousRunController(
        topic_planner_agent=build_topic_planner_agent(settings),
        pipeline_runner=build_pipeline_runner(settings),
        run_store=AutonomousRunStore(DEFAULT_AUTONOMOUS_RUN_DIR),
        run_lock=RunLock(DEFAULT_LOCK_PATH, timeout_seconds=settings.autonomous_lock_timeout),
        publishing_intent=build_publishing_intent(settings),
        topic_continuity=build_topic_continuity_orchestrator(settings) if settings.topic_continuity_enabled else None,
    )
