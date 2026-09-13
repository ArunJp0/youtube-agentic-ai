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

from src.agents.topic_planner_agent import TopicPlannerAgent
from src.agents.topic_ranking_planner import TopicRankingPlanner
from src.config.providers import (
    get_llm_provider,
    get_media_provider,
    get_search_provider,
    get_topic_planner_source,
    get_transcription_provider,
    get_visual_relevance_evaluator,
    get_voice_provider,
)
from src.config.settings import Settings
from src.models.youtube_upload import PublishingIntent, YOUTUBE_UPLOAD_SCOPES
from src.orchestration.autonomous_run_controller import AutonomousRunController, PipelineRunner
from src.services.autonomous_run_store import AutonomousRunStore, DEFAULT_AUTONOMOUS_RUN_DIR
from src.services.run_lock import DEFAULT_LOCK_PATH, RunLock
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler
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
    search_provider = get_search_provider(settings)
    llm_provider = get_llm_provider(settings)
    voice_provider = get_voice_provider(settings)
    media_provider = get_media_provider(settings)
    visual_relevance_evaluator = get_visual_relevance_evaluator(settings)
    transcription_provider = get_transcription_provider(settings)
    music_catalog_provider = LocalMusicCatalogProvider()

    async def _run(topic: str, publishing_intent: Optional[PublishingIntent]) -> PipelineState:
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
    )
