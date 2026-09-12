# Standalone live demo/CLI for the Topic Planner Agent.
#
# NOT wired into the main LangGraph pipeline (src/workflows/pipeline_graph.py)
# yet - deliberately standalone, per this milestone's scope. Prints the
# selected topic (or explicit failure) - never performs Research/Script/
# Voice/.../Compliance/Upload itself.
from __future__ import annotations

import asyncio
import sys

from src.agents.topic_planner_agent import TopicPlannerAgent
from src.agents.topic_ranking_planner import TopicRankingPlanner
from src.config.providers import ProviderConfigError, get_llm_provider, get_topic_planner_source
from src.config.settings import Settings
from src.models.topic_planner import TopicSelectionResult


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def print_topic_selection_result(result: TopicSelectionResult) -> None:
    print("\n" + "=" * 60)
    print("TOPIC PLANNER RESULT")
    print("=" * 60)
    print(f"Status:  {result.status}")
    print(f"Success: {result.success}")
    if result.success:
        print(f"Selected topic: {result.selected_topic}")
        print(f"Category:       {result.category}")
        print(f"Score:          {result.score}")
        print(f"Rationale:      {result.rationale}")
        print(f"Semantic ranking used: {result.used_semantic_ranking} (fallback_used={result.fallback_used})")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Sources used:   {result.sources_used}")
    print(f"Candidates:     {result.candidate_count} ({result.duplicate_count} rejected as duplicates)")
    print(f"Selected at:    {result.selected_at}")
    if result.is_sensitive:
        print(f"\nSENSITIVE STORY FLAG: {result.sensitivity_reasons} (downstream Research/Compliance should apply stricter treatment)")
    if result.top_candidates:
        print("\nTop candidates considered:")
        for score in result.top_candidates:
            trend_bits = []
            if score.freshness_score is not None:
                trend_bits.append(f"freshness={score.freshness_score:.2f}")
            if score.source_confidence_score is not None:
                trend_bits.append(f"source_confidence={score.source_confidence_score:.2f}")
            if score.market_relevance_score is not None:
                trend_bits.append(f"market={score.market_relevance_score:.2f}")
            if score.category_relevance_score is not None:
                trend_bits.append(f"category={score.category_relevance_score:.2f}")
            trend_suffix = f" [{', '.join(trend_bits)}]" if trend_bits else ""
            print(
                f"   {score.total_score:.3f}  {score.candidate.raw_title!r} "
                f"(relevance={score.relevance_score:.2f} novelty={score.novelty_score:.2f} "
                f"evergreen={score.evergreen_score:.2f} suitability={score.suitability_score:.2f} "
                f"popularity={score.popularity_score:.2f}){trend_suffix}"
                + (f" [sources: {score.candidate.distinct_source_count}]" if score.candidate.distinct_source_count > 1 else "")
            )


def _parse_csv_setting(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


async def run_topic_planner_demo() -> TopicSelectionResult:
    settings = Settings()
    topic_source_provider = get_topic_planner_source(settings)
    llm_provider = get_llm_provider(settings)
    ranking_planner = TopicRankingPlanner(llm_provider) if settings.llm_provider != "mock" else None

    target_markets = _parse_csv_setting(settings.topic_target_markets)
    preferred_categories = _parse_csv_setting(settings.topic_categories)

    print(
        f"Topic Planner: mode={settings.topic_mode} | source={topic_source_provider.name} | "
        f"niche={settings.topic_planner_niche} | markets={target_markets or [settings.topic_planner_region]} | "
        f"language={settings.topic_language} | categories={preferred_categories or 'none configured'} | "
        f"freshness_hours={settings.topic_freshness_hours}"
    )
    print(f"Semantic ranking: {'enabled (' + settings.llm_provider + ')' if ranking_planner else 'disabled (mock LLM configured)'}")
    print("=" * 60)

    agent = TopicPlannerAgent(
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
    return await agent.plan_topic()


async def main() -> None:
    _ensure_utf8_stdout()
    try:
        result = await run_topic_planner_demo()
    except ProviderConfigError as e:
        print(f"Provider configuration error: {e}")
        sys.exit(1)

    print_topic_selection_result(result)

    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
