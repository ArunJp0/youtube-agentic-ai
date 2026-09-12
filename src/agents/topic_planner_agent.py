# Topic Planner Agent: chooses ONE video topic automatically, removing the
# need for a human to type a topic before each pipeline run.
#
# Deterministic orchestration only (candidate discovery -> normalization ->
# duplicate/history filtering -> scoring -> selection); the one piece of
# semantic judgment (relevance/evergreen/suitability) is delegated entirely
# to the injected TopicRankingPlanner, exactly like ComplianceAgent
# delegates to ComplianceReviewer. This agent never researches, scripts, or
# produces any video content itself - its only output is a typed
# TopicSelectionResult whose ``selected_topic`` is a plain string ready to
# pass into the existing ResearchAgent/run_pipeline, unchanged.
#
# Current/trending-news extension: this agent has NO awareness of any
# specific news source's HTTP/API details (that stays entirely inside
# TopicSourceProvider implementations, including CompositeTopicSourceProvider
# for merging multiple sources) and no awareness of any specific
# region/language/category name - ``mode``/``target_markets``/
# ``preferred_categories``/``freshness_hours`` are all plain configuration
# the caller supplies, never hardcoded here.
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

from src.agents.topic_ranking_planner import TopicRankingPlanner
from src.models.topic_planner import TopicCandidate, TopicRankingOutcome, TopicScore, TopicSelectionResult
from src.services.topic_freshness import freshness_score, is_stale
from src.services.topic_history import load_recent_produced_topics
from src.services.topic_normalization import (
    DUPLICATE_SIMILARITY_THRESHOLD,
    is_duplicate_topic,
    normalize_topic,
    topic_similarity,
)
from src.services.topic_plan_store import TopicPlanStore
from src.services.topic_sensitivity import detect_sensitivity
from src.services.provenance_store import DEFAULT_PROVENANCE_OUTPUT_DIR
from src.tools.topic_source_provider import TopicSourceProvider, TopicSourceProviderError

DEFAULT_CANDIDATE_LIMIT = 15
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_FRESHNESS_HOURS = 48.0

# Centralized, bounded, named weights (never scattered magic numbers) -
# sums to 1.0 exactly. Relevance/suitability weighted slightly higher
# since they most directly determine whether the pipeline can actually
# make a good video from the topic; popularity lowest, since it's an
# optional signal not every source provides. UNCHANGED from the original
# evergreen-only implementation - see _combine_scores for how trending
# signals are blended in without altering this formula for candidates that
# don't carry them.
SCORE_WEIGHTS = {
    "relevance": 0.30,
    "suitability": 0.25,
    "novelty": 0.20,
    "evergreen": 0.15,
    "popularity": 0.10,
}

# Additional trending-mode-only signals - only blended in for candidates
# that actually carry the underlying data (a pure evergreen candidate has
# none of these, so its total_score is computed with EXACTLY the original
# 5-signal SCORE_WEIGHTS formula above, unchanged). Sum of any subset of
# these is always < 1.0, and the base 5-signal weights are proportionally
# scaled down by (1 - applicable trend weight sum) so total_score always
# stays bounded [0, 1].
TREND_SCORE_WEIGHTS = {
    "freshness": 0.15,
    "source_confidence": 0.10,
    "market_relevance": 0.05,
    "category_relevance": 0.05,
}

# Deterministic step function for cross-source corroboration confidence -
# never a named-outlet trust list (which would require hardcoding specific
# publishers), purely "how many independent sources reported this same
# story" (STEP 5).
_SOURCE_CONFIDENCE_BY_COUNT = {1: 0.4, 2: 0.7}
_SOURCE_CONFIDENCE_MAX = 1.0

# Category/market preference scoring: a configured-but-unmatched
# category/market is a mild neutral-to-slightly-below score, never a hard
# penalty - "without making every other important story impossible to
# select" (STEP 7) / "a globally important story should still be
# selectable" (STEP 6).
_PREFERENCE_MATCH_SCORE = 1.0
_PREFERENCE_UNKNOWN_SCORE = 0.5
_PREFERENCE_UNMATCHED_SCORE = 0.3


class TopicPlannerAgentError(Exception):
    """Raised only for configuration/programmer errors. Source/ranking
    failures are NEVER raised - they are captured in the returned
    TopicSelectionResult (success=False) so callers always get a
    structured result back.
    """


class TopicPlannerAgent:
    """Selects one topic for the existing pipeline to research, script, and
    produce - reusing the existing history/provenance/LLM infrastructure,
    never duplicating Research Agent's own job."""

    def __init__(
        self,
        topic_source_provider: TopicSourceProvider,
        ranking_planner: Optional[TopicRankingPlanner] = None,
        topic_plan_store: Optional[TopicPlanStore] = None,
        niche: Optional[str] = None,
        region: Optional[str] = None,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        provenance_output_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR,
        mode: str = "evergreen",
        target_markets: Optional[List[str]] = None,
        preferred_categories: Optional[List[str]] = None,
        freshness_hours: float = DEFAULT_FRESHNESS_HOURS,
    ) -> None:
        """Initialize the Topic Planner Agent.

        Args:
            topic_source_provider: TopicSourceProvider implementation
                (real, mock, or a CompositeTopicSourceProvider merging
                several) - dependency injection, exactly like every other
                provider in this project.
            ranking_planner: Optional TopicRankingPlanner for the single
                bounded semantic-ranking call. If None, scoring is fully
                deterministic (relevance/evergreen/suitability all neutral)
                - never a fabricated semantic opinion.
            topic_plan_store: Persistence for this run's own decision
                history/traceability; defaults to the real
                ``output/topic_plans/`` location.
            niche: Optional configured content niche/category
            region: Optional single configured region/locale hint - used
                as the sole market when ``target_markets`` is not given
                (preserves the exact prior single-region behavior).
            candidate_limit: Maximum candidates to request per source/market
            history_limit: Maximum recent history entries to check against
            provenance_output_dir: Directory real completed-run provenance
                manifests are read from for duplicate prevention
            mode: "evergreen" (default, unchanged prior behavior),
                "trending", or "mixed" - purely informational here (the
                caller is responsible for constructing an appropriate
                ``topic_source_provider`` for the mode; see
                ``src.config.providers.get_topic_planner_source``) except
                that it gates the freshness hard-filter (only applied in
                "trending"/"mixed" mode, never "evergreen").
            target_markets: Optional list of configured markets (e.g.
                ["global", "GB", "IN"]) - when given, ``discover_candidates``
                is called once per market and results are merged; when
                omitted, falls back to the single ``region`` (or no region
                at all), exactly as before this extension.
            preferred_categories: Optional list of preferred category
                names - a scoring PREFERENCE only, never a hard filter.
            freshness_hours: Configured freshness window for trending-mode
                scoring/filtering (see src.services.topic_freshness).
        """
        self.topic_source_provider = topic_source_provider
        self.ranking_planner = ranking_planner
        self.topic_plan_store = topic_plan_store or TopicPlanStore()
        self.niche = niche
        self.region = region
        self.candidate_limit = candidate_limit
        self.history_limit = history_limit
        self.provenance_output_dir = provenance_output_dir
        self.mode = (mode or "evergreen").strip().lower()
        self.target_markets = target_markets
        self.preferred_categories = [c.strip().lower() for c in (preferred_categories or []) if c.strip()]
        self.freshness_hours = freshness_hours

    async def plan_topic(self) -> TopicSelectionResult:
        """Select one topic, or return an explicit failure result.

        Never raises for source/ranking failures or an empty/all-duplicate/
        all-stale candidate set - only raises TopicPlannerAgentError for
        configuration/programmer errors (there are none currently, reserved
        for future use exactly like every other agent in this project).

        Returns:
            Structured TopicSelectionResult describing the outcome.
        """
        markets = self.target_markets if self.target_markets else [self.region]

        raw_candidates: List[TopicCandidate] = []
        any_market_succeeded = False
        last_error: Optional[str] = None
        for market in markets:
            try:
                batch = await self.topic_source_provider.discover_candidates(
                    category=self.niche, region=market, limit=self.candidate_limit
                )
            except TopicSourceProviderError as e:
                last_error = str(e)
                continue
            any_market_succeeded = True
            raw_candidates.extend(batch)

        sources_used = self._resolve_sources_used()

        if not any_market_succeeded:
            return self._persist(
                TopicSelectionResult(
                    success=False,
                    status="source_unavailable",
                    error=last_error,
                    sources_used=sources_used,
                    selected_at=_now_iso(),
                )
            )

        if not raw_candidates:
            return self._persist(
                TopicSelectionResult(
                    success=False, status="no_candidates", sources_used=sources_used, selected_at=_now_iso()
                )
            )

        # Hard freshness filter - trending/mixed mode only, timestamps
        # only, never LLM judgment (STEP 4). Evergreen candidates (no
        # published_at) are never affected regardless of mode.
        if self.mode in ("trending", "mixed"):
            fresh_candidates = [c for c in raw_candidates if not is_stale(c.published_at, self.freshness_hours)]
            if not fresh_candidates:
                return self._persist(
                    TopicSelectionResult(
                        success=False,
                        status="all_candidates_stale",
                        candidate_count=len(raw_candidates),
                        sources_used=sources_used,
                        selected_at=_now_iso(),
                    )
                )
            raw_candidates = fresh_candidates

        candidates = self._deduplicate_candidates(raw_candidates)
        candidates = [self._flag_sensitivity(c) for c in candidates]

        produced_history = [normalize_topic(topic) for topic, _ in load_recent_produced_topics(
            self.provenance_output_dir, limit=self.history_limit
        )]
        planned_history = [
            normalize_topic(r.selected_topic)
            for r in self.topic_plan_store.list_recent(limit=self.history_limit)
            if r.selected_topic
        ]
        history = produced_history + planned_history

        survivors = [c for c in candidates if not is_duplicate_topic(c.normalized_title, history)]
        duplicate_count = len(candidates) - len(survivors)

        if not survivors:
            return self._persist(
                TopicSelectionResult(
                    success=False,
                    status="all_candidates_duplicate",
                    candidate_count=len(candidates),
                    duplicate_count=duplicate_count,
                    sources_used=sources_used,
                    selected_at=_now_iso(),
                )
            )

        ranking = self._rank(survivors)
        scores = self._combine_scores(survivors, history, ranking)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        best = scores[0]

        result = TopicSelectionResult(
            success=True,
            status="selected",
            selected_topic=best.candidate.raw_title,
            rationale=best.rationale,
            category=best.candidate.category,
            score=best.total_score,
            sources_used=sources_used,
            candidate_count=len(candidates),
            duplicate_count=duplicate_count,
            fallback_used=not (ranking is not None and ranking.performed),
            used_semantic_ranking=bool(ranking is not None and ranking.performed),
            selected_at=_now_iso(),
            top_candidates=scores[:5],
            is_sensitive=best.candidate.is_sensitive,
            sensitivity_reasons=best.candidate.sensitivity_reasons,
        )
        return self._persist(result)

    # ---- helpers ------------------------------------------------------------

    def _resolve_sources_used(self) -> List[str]:
        """Prefer the provider's own record of which component sources
        actually succeeded on the most recent call (see
        CompositeTopicSourceProvider.last_successful_sources) - falls back
        to the provider's own ``name`` for a plain (non-composite) provider."""
        successful = getattr(self.topic_source_provider, "last_successful_sources", None)
        if successful:
            seen: List[str] = []
            for name in successful:
                if name not in seen:
                    seen.append(name)
            return seen
        return [self.topic_source_provider.name]

    def _deduplicate_candidates(self, raw_candidates: List[TopicCandidate]) -> List[TopicCandidate]:
        """Normalize every candidate and merge later ones that are the
        same/near-duplicate of an already-kept candidate (e.g. two sources
        returning near-identical titles) - first occurrence's title/
        category is kept, but ``distinct_source_count`` is incremented
        whenever a genuinely different publisher/source corroborates it
        (STEP 5's cross-source confirmation signal), never double-counted
        for the same outlet appearing twice."""
        kept: List[TopicCandidate] = []
        kept_normalized: List[str] = []
        kept_source_names: List[Set[str]] = []

        for candidate in raw_candidates:
            normalized = candidate.normalized_title or normalize_topic(candidate.raw_title)
            if candidate.normalized_title != normalized:
                candidate = candidate.model_copy(update={"normalized_title": normalized})

            match_index = _find_duplicate_index(normalized, kept_normalized)
            if match_index is not None:
                this_source = candidate.source_name or candidate.source
                if this_source not in kept_source_names[match_index]:
                    kept_source_names[match_index].add(this_source)
                    kept[match_index] = kept[match_index].model_copy(
                        update={"distinct_source_count": kept[match_index].distinct_source_count + 1}
                    )
                continue

            kept.append(candidate)
            kept_normalized.append(normalized)
            kept_source_names.append({candidate.source_name or candidate.source})

        return kept

    def _flag_sensitivity(self, candidate: TopicCandidate) -> TopicCandidate:
        is_sensitive, reasons = detect_sensitivity(candidate.raw_title)
        if not is_sensitive:
            return candidate
        return candidate.model_copy(update={"is_sensitive": True, "sensitivity_reasons": reasons})

    def _rank(self, candidates: List[TopicCandidate]) -> Optional[TopicRankingOutcome]:
        if self.ranking_planner is None:
            return None
        return self.ranking_planner.rank(candidates, niche=self.niche)

    def _combine_scores(
        self, candidates: List[TopicCandidate], history: List[str], ranking: Optional[TopicRankingOutcome]
    ) -> List[TopicScore]:
        judgments_by_title = {j.normalized_title: j for j in ranking.judgments} if ranking is not None else {}

        scores: List[TopicScore] = []
        for candidate in candidates:
            novelty_score = 1.0 - max(
                (topic_similarity(candidate.normalized_title, past) for past in history), default=0.0
            )
            popularity_score = candidate.popularity_signal if candidate.popularity_signal is not None else 0.5

            judgment = judgments_by_title.get(candidate.normalized_title)
            if judgment is not None:
                relevance_score = judgment.relevance_score
                evergreen_score = judgment.evergreen_score
                suitability_score = judgment.suitability_score
                rationale = judgment.rationale
            else:
                relevance_score = evergreen_score = suitability_score = 0.5
                rationale = "Semantic ranking unavailable - neutral deterministic score"

            base_total = (
                SCORE_WEIGHTS["relevance"] * relevance_score
                + SCORE_WEIGHTS["suitability"] * suitability_score
                + SCORE_WEIGHTS["novelty"] * novelty_score
                + SCORE_WEIGHTS["evergreen"] * evergreen_score
                + SCORE_WEIGHTS["popularity"] * popularity_score
            )

            trend_freshness = freshness_score(candidate.published_at, self.freshness_hours)
            trend_source_confidence = _source_confidence(candidate) if candidate.published_at else None
            trend_market = self._market_relevance(candidate)
            trend_category = self._category_relevance(candidate)

            trend_signals = {
                "freshness": trend_freshness,
                "source_confidence": trend_source_confidence,
                "market_relevance": trend_market,
                "category_relevance": trend_category,
            }
            applicable = {k: v for k, v in trend_signals.items() if v is not None}
            trend_weight_sum = sum(TREND_SCORE_WEIGHTS[k] for k in applicable)
            trend_contribution = sum(TREND_SCORE_WEIGHTS[k] * v for k, v in applicable.items())

            # trend_weight_sum == 0 (no trending signals apply, e.g. a
            # pure evergreen candidate) -> base_scale == 1.0 -> total_score
            # is EXACTLY base_total, the original unmodified formula.
            base_scale = 1.0 - trend_weight_sum
            total_score = base_scale * base_total + trend_contribution

            scores.append(
                TopicScore(
                    candidate=candidate,
                    relevance_score=relevance_score,
                    novelty_score=novelty_score,
                    evergreen_score=evergreen_score,
                    suitability_score=suitability_score,
                    popularity_score=popularity_score,
                    total_score=round(max(0.0, min(1.0, total_score)), 4),
                    rationale=rationale,
                    freshness_score=trend_freshness,
                    source_confidence_score=trend_source_confidence,
                    market_relevance_score=trend_market,
                    category_relevance_score=trend_category,
                )
            )
        return scores

    def _market_relevance(self, candidate: TopicCandidate) -> Optional[float]:
        """None (not applicable) only when no market is configured at all.
        A candidate market of "global" (or unknown) always scores maximally
        relevant, REGARDLESS of what's configured - including when the
        configuration itself is exactly ``["global"]`` (a deliberate
        "world feed only" configuration, not "no preference")."""
        if not self.target_markets:
            return None
        configured = [m.strip().lower() for m in self.target_markets if m and m.strip()]
        if not configured:
            return None
        candidate_market = (candidate.market or "").strip().lower()
        if not candidate_market or candidate_market == "global":
            return _PREFERENCE_MATCH_SCORE  # a global story is always maximally relevant to any configured market
        if candidate_market in configured:
            return _PREFERENCE_MATCH_SCORE
        return _PREFERENCE_UNMATCHED_SCORE

    def _category_relevance(self, candidate: TopicCandidate) -> Optional[float]:
        if not self.preferred_categories:
            return None
        if not candidate.category:
            return _PREFERENCE_UNKNOWN_SCORE
        if candidate.category.strip().lower() in self.preferred_categories:
            return _PREFERENCE_MATCH_SCORE
        return _PREFERENCE_UNMATCHED_SCORE

    def _persist(self, result: TopicSelectionResult) -> TopicSelectionResult:
        try:
            self.topic_plan_store.write(result)
        except OSError:
            pass  # best-effort, never fails an already-decided result
        return result


def _find_duplicate_index(
    normalized_title: str, kept_normalized: List[str], threshold: float = DUPLICATE_SIMILARITY_THRESHOLD
) -> Optional[int]:
    for index, past in enumerate(kept_normalized):
        if normalized_title == past or topic_similarity(normalized_title, past) >= threshold:
            return index
    return None


def _source_confidence(candidate: TopicCandidate) -> float:
    return _SOURCE_CONFIDENCE_BY_COUNT.get(candidate.distinct_source_count, _SOURCE_CONFIDENCE_MAX)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
