# Music Selection Service: deterministically ranks the approved local BGM
# catalog against a MusicPlan and picks one best track. No LLM call here -
# matching a mood/energy/genre profile against typed catalog metadata is a
# fixed-rule scoring problem, not a reasoning task.
#
# Scoped to single-track selection for the current MVP (one BGM track for
# the whole video); the scoring function is kept as a small, isolated
# method specifically so a future multi-track/scene-level selector can
# reuse it without a rewrite.
from __future__ import annotations

from typing import List, Optional, Tuple

from src.models.music import BGMTrack, MusicPlan

# Ordered so a numeric "distance" between levels can be computed.
_ENERGY_LEVELS = ["low", "medium", "high"]

# Scoring weights - centralized here rather than scattered magic numbers.
MOOD_MATCH_WEIGHT = 2.0
GENRE_MATCH_WEIGHT = 1.5
ENERGY_EXACT_MATCH_WEIGHT = 1.0
ENERGY_DISTANCE_PENALTY = 0.5
NEUTRAL_SUBTLE_BONUS = 1.0


class MusicSelectionService:
    """Ranks only tracks already present in the approved catalog - never
    considers, searches for, or invents anything outside it.

    For narrated videos, instrumental tracks are strongly preferred: any
    track with vocals/lyrics is excluded by default (``instrumental=False``
    in the catalog), consistent with narration needing to stay the clearly
    dominant audio.
    """

    def __init__(self, fallback_track_id: Optional[str] = None) -> None:
        """Initialize the selector.

        Args:
            fallback_track_id: Optional catalog ``track_id`` to use when no
                track matches the plan well enough to be considered
                eligible (e.g. every candidate hits an avoid_styles term).
                If None (default), an unmatched plan results in an
                explicit "no suitable track" outcome rather than a silent
                guess.
        """
        self.fallback_track_id = fallback_track_id

    def select_track(
        self, plan: MusicPlan, catalog: List[BGMTrack]
    ) -> Tuple[Optional[BGMTrack], List[str]]:
        """Select the single best-matching track for ``plan`` from ``catalog``.

        Never raises - an empty catalog or no eligible track is reported
        via the returned ``(None, warnings)`` rather than an exception, so
        callers always get a structured outcome.

        Args:
            plan: The desired music characteristics
            catalog: Every track currently in the approved catalog

        Returns:
            (selected track or None, list of warnings/explanations)
        """
        if not catalog:
            return None, ["Approved BGM catalog is empty"]

        warnings: List[str] = []

        instrumental_only = [t for t in catalog if t.instrumental]
        if not instrumental_only:
            warnings.append("No instrumental tracks in the catalog; tracks with vocals are excluded by default")

        eligible = [t for t in instrumental_only if not self._matches_avoid(t, plan.avoid_styles)]
        if instrumental_only and not eligible:
            warnings.append("Every instrumental track matched an avoid_styles term in the music plan")

        if not eligible:
            fallback = self._resolve_fallback(catalog)
            if fallback is not None:
                warnings.append(
                    f"No eligible track matched the music plan; used configured fallback track '{fallback.track_id}'"
                )
                return fallback, warnings
            warnings.append("No eligible instrumental track found and no fallback track is configured")
            return None, warnings

        # Deterministic for equal inputs: ties broken by track_id, not
        # insertion order or dict iteration.
        ranked = sorted(eligible, key=lambda t: (-self._score(t, plan), t.track_id))
        return ranked[0], warnings

    # ---- ranking ------------------------------------------------------------

    @staticmethod
    def _matches_avoid(track: BGMTrack, avoid_styles: List[str]) -> bool:
        avoid_terms = {s.strip().lower() for s in avoid_styles if s and s.strip()}
        if not avoid_terms:
            return False
        track_terms = {tag.strip().lower() for tag in track.mood_tags if tag}
        if track.genre:
            track_terms.add(track.genre.strip().lower())
        return bool(avoid_terms & track_terms)

    @staticmethod
    def _score(track: BGMTrack, plan: MusicPlan) -> float:
        score = 0.0
        track_moods = {tag.strip().lower() for tag in track.mood_tags if tag}

        plan_moods = {m.strip().lower() for m in (plan.primary_mood, plan.secondary_mood) if m}
        score += MOOD_MATCH_WEIGHT * len(plan_moods & track_moods)

        preferred_genres = {g.strip().lower() for g in plan.preferred_genres if g}
        if track.genre and track.genre.strip().lower() in preferred_genres:
            score += GENRE_MATCH_WEIGHT

        if track.energy_level == plan.energy_level:
            score += ENERGY_EXACT_MATCH_WEIGHT
        else:
            try:
                distance = abs(_ENERGY_LEVELS.index(track.energy_level) - _ENERGY_LEVELS.index(plan.energy_level))
                score += max(0.0, ENERGY_EXACT_MATCH_WEIGHT - ENERGY_DISTANCE_PENALTY * distance)
            except ValueError:
                pass  # unknown energy level string - no distance credit

        if plan.requires_neutral_subtle and "subtle" in track_moods:
            score += NEUTRAL_SUBTLE_BONUS

        return score

    def _resolve_fallback(self, catalog: List[BGMTrack]) -> Optional[BGMTrack]:
        if not self.fallback_track_id:
            return None
        return next((t for t in catalog if t.track_id == self.fallback_track_id), None)
