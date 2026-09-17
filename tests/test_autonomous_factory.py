# Tests for src.orchestration.autonomous_factory - proving AUTONOMOUS
# execution wires the SAME real TOPIC_MODE/TOPIC_PLANNER_SOURCE/
# TOPIC_NEWS_SOURCES_ENABLED settings a manual `topic_planner_demo.py` run
# would use, never a hidden hardcoded/mock-only path. No real Gemini/
# YouTube/Google News network calls - provider construction is lazy
# (network only touched inside discover_candidates(), never __init__), so
# these tests only ever construct objects, never call them.
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.agents.topic_planner_agent import TopicPlannerAgent
from src.config.settings import Settings
from src.orchestration.autonomous_factory import (
    build_pipeline_runner,
    build_publishing_intent,
    build_topic_planner_agent,
)
from src.services.script_duration import calculate_word_budget, resolve_target_duration_minutes
from src.tools.topic_source_provider import CompositeTopicSourceProvider, MockTopicSourceProvider


class TestBuildTopicPlannerAgent:
    def test_returns_a_real_topic_planner_agent(self) -> None:
        settings = Settings(topic_mode="evergreen", topic_planner_source="mock")
        agent = build_topic_planner_agent(settings)
        assert isinstance(agent, TopicPlannerAgent)

    def test_default_settings_produce_mock_evergreen_only(self) -> None:
        """Documents the root cause this milestone fixed: with Settings()'s
        own defaults (topic_mode='evergreen', topic_planner_source='mock'),
        autonomous execution only ever sees the fixed mock candidate list -
        this is exactly why .env now explicitly overrides these."""
        settings = Settings(topic_mode="evergreen", topic_planner_source="mock")
        agent = build_topic_planner_agent(settings)
        assert isinstance(agent.topic_source_provider, MockTopicSourceProvider)
        assert not isinstance(agent.topic_source_provider, CompositeTopicSourceProvider)
        assert agent.mode == "evergreen"

    def test_recommended_mixed_configuration_wires_composite_real_sources(self) -> None:
        """The exact .env configuration this milestone sets
        (TOPIC_MODE=mixed, TOPIC_NEWS_SOURCES_ENABLED=true,
        TOPIC_PLANNER_SOURCE=youtube) must produce a real composite source
        when passed through the autonomous factory, not just the plain
        providers.py factory in isolation."""
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider
        from src.tools.youtube_topic_source_provider import YouTubeTopicSourceProvider

        settings = Settings(topic_mode="mixed", topic_news_sources_enabled=True, topic_planner_source="youtube")
        agent = build_topic_planner_agent(settings)

        assert isinstance(agent.topic_source_provider, CompositeTopicSourceProvider)
        component_types = [type(p) for p in agent.topic_source_provider.providers]
        assert YouTubeTopicSourceProvider in component_types
        assert CurrentNewsTopicSourceProvider in component_types
        assert agent.mode == "mixed"

    def test_trending_mode_wires_real_news_source(self) -> None:
        from src.tools.current_news_topic_source_provider import CurrentNewsTopicSourceProvider

        settings = Settings(topic_mode="trending", topic_news_sources_enabled=True, topic_planner_source="youtube")
        agent = build_topic_planner_agent(settings)

        assert isinstance(agent.topic_source_provider, CurrentNewsTopicSourceProvider)
        assert agent.mode == "trending"

    def test_target_markets_and_categories_pass_through(self) -> None:
        settings = Settings(
            topic_mode="evergreen",
            topic_planner_source="mock",
            topic_target_markets="global,IN",
            topic_categories="technology,science",
            topic_freshness_hours=72.0,
        )
        agent = build_topic_planner_agent(settings)

        assert agent.target_markets == ["global", "IN"]
        assert agent.preferred_categories == ["technology", "science"]
        assert agent.freshness_hours == 72.0

    def test_niche_and_region_pass_through(self) -> None:
        settings = Settings(topic_planner_niche="education", topic_planner_region="GB")
        agent = build_topic_planner_agent(settings)

        assert agent.niche == "education"
        assert agent.region == "GB"

    def test_no_topic_text_hardcoded_in_agent_construction(self) -> None:
        """Genericity guard: nothing about the assembled agent depends on
        any specific topic/story string - only on configuration."""
        settings = Settings(topic_mode="evergreen", topic_planner_source="mock")
        agent = build_topic_planner_agent(settings)
        assert agent.niche is None
        assert agent.region is None


class TestBuildPipelineRunnerDurationWiring:
    """Proves the autonomous pipeline runner passes a real, Settings-driven
    script_word_budget into run_pipeline - never leaving Script Agent's
    depth stuck at a hardcoded default regardless of configuration. Patches
    run_pipeline itself so no provider is ever actually constructed/called
    (no real Gemini/Wikipedia/Edge-TTS/Pexels/Whisper/FFmpeg/network)."""

    def test_settings_driven_word_budget_reaches_run_pipeline(self) -> None:
        settings = Settings(
            llm_provider="mock",
            search_provider="mock",
            voice_provider="mock",
            media_provider="mock",
            transcription_provider="mock",
            script_duration_profile="long",
        )
        expected_budget = calculate_word_budget(
            resolve_target_duration_minutes(settings.script_duration_profile, settings.script_target_duration_minutes)
        )

        with patch(
            "src.orchestration.autonomous_factory.run_pipeline", new_callable=AsyncMock
        ) as mock_run_pipeline:
            runner = build_pipeline_runner(settings)
            asyncio.run(runner("Some topic", None, None))

        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["script_word_budget"] == expected_budget

    def test_ai_video_settings_reach_run_pipeline_when_enabled(self) -> None:
        settings = Settings(
            llm_provider="mock",
            search_provider="mock",
            voice_provider="mock",
            media_provider="mock",
            transcription_provider="mock",
            ai_video_enabled=True,
            ai_video_provider="mock",
            ai_video_max_retries=3,
            stock_fallback_enabled=False,
        )

        with patch(
            "src.orchestration.autonomous_factory.run_pipeline", new_callable=AsyncMock
        ) as mock_run_pipeline:
            runner = build_pipeline_runner(settings)
            asyncio.run(runner("Some topic", None, None))

        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["ai_video_provider"] is not None
        assert kwargs["ai_video_provider"].name == "mock"
        assert kwargs["ai_video_max_retries"] == 3
        assert kwargs["stock_fallback_enabled"] is False

    def test_ai_video_disabled_by_default_passes_none(self) -> None:
        settings = Settings(
            llm_provider="mock",
            search_provider="mock",
            voice_provider="mock",
            media_provider="mock",
            transcription_provider="mock",
        )

        with patch(
            "src.orchestration.autonomous_factory.run_pipeline", new_callable=AsyncMock
        ) as mock_run_pipeline:
            runner = build_pipeline_runner(settings)
            asyncio.run(runner("Some topic", None, None))

        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["ai_video_provider"] is None
        assert kwargs["stock_fallback_enabled"] is True

    def test_explicit_target_minutes_override_reaches_run_pipeline(self) -> None:
        settings = Settings(
            llm_provider="mock",
            search_provider="mock",
            voice_provider="mock",
            media_provider="mock",
            transcription_provider="mock",
            script_target_duration_minutes=9.0,
        )

        with patch(
            "src.orchestration.autonomous_factory.run_pipeline", new_callable=AsyncMock
        ) as mock_run_pipeline:
            runner = build_pipeline_runner(settings)
            asyncio.run(runner("Some topic", None, None))

        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["script_word_budget"].target_duration_minutes == 9.0


class TestBuildPublishingIntent:
    """Regression guard: this milestone's topic-planner-only changes must
    not alter the existing safe-by-default publishing behavior."""

    def test_defaults_to_disabled(self) -> None:
        import os

        # Guard against a developer's/deployment's local .env overriding
        # AUTONOMOUS_PUBLISHING_MODE (e.g. during a controlled real
        # validation run) and masking this default-value assertion.
        previous = os.environ.pop("AUTONOMOUS_PUBLISHING_MODE", None)
        try:
            settings = Settings()
            intent = build_publishing_intent(settings)
            assert intent.mode == "disabled"
        finally:
            if previous is not None:
                os.environ["AUTONOMOUS_PUBLISHING_MODE"] = previous

    def test_private_mode_passes_through(self) -> None:
        settings = Settings(autonomous_publishing_mode="private")
        intent = build_publishing_intent(settings)
        assert intent.mode == "private"
