# Tests for deterministic freshness scoring/filtering
# (src/services/topic_freshness.py). Timestamps only - no LLM/network calls.
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.services.topic_freshness import freshness_score, is_stale, parse_iso_timestamp

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


class TestParseIsoTimestamp:
    def test_parses_valid_timestamp(self) -> None:
        parsed = parse_iso_timestamp("2026-09-10T10:00:00+00:00")
        assert parsed == datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)

    def test_parses_z_suffix(self) -> None:
        parsed = parse_iso_timestamp("2026-09-10T10:00:00Z")
        assert parsed == datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)

    def test_naive_timestamp_assumed_utc(self) -> None:
        parsed = parse_iso_timestamp("2026-09-10T10:00:00")
        assert parsed.tzinfo is not None

    def test_none_returns_none(self) -> None:
        assert parse_iso_timestamp(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_iso_timestamp("") is None

    def test_malformed_returns_none_never_raises(self) -> None:
        assert parse_iso_timestamp("not a timestamp") is None


class TestFreshnessScore:
    def test_just_published_scores_near_one(self) -> None:
        published = (NOW - timedelta(minutes=1)).isoformat()
        score = freshness_score(published, freshness_hours=48, now=NOW)
        assert score > 0.99

    def test_published_in_future_clock_skew_clamped_to_one(self) -> None:
        published = (NOW + timedelta(minutes=5)).isoformat()
        score = freshness_score(published, freshness_hours=48, now=NOW)
        assert score == 1.0

    def test_half_window_scores_roughly_half(self) -> None:
        published = (NOW - timedelta(hours=24)).isoformat()
        score = freshness_score(published, freshness_hours=48, now=NOW)
        assert abs(score - 0.5) < 0.01

    def test_at_window_edge_scores_zero(self) -> None:
        published = (NOW - timedelta(hours=48)).isoformat()
        score = freshness_score(published, freshness_hours=48, now=NOW)
        assert score == 0.0

    def test_beyond_window_still_floored_at_zero(self) -> None:
        published = (NOW - timedelta(hours=200)).isoformat()
        score = freshness_score(published, freshness_hours=48, now=NOW)
        assert score == 0.0

    def test_fresh_story_scores_higher_than_stale_equivalent(self) -> None:
        """The core requirement: a fresh story ranks above a stale one."""
        fresh = freshness_score((NOW - timedelta(hours=1)).isoformat(), freshness_hours=48, now=NOW)
        stale = freshness_score((NOW - timedelta(hours=40)).isoformat(), freshness_hours=48, now=NOW)
        assert fresh > stale

    def test_missing_published_at_returns_none(self) -> None:
        assert freshness_score(None, freshness_hours=48, now=NOW) is None

    def test_zero_freshness_window_returns_none(self) -> None:
        assert freshness_score(NOW.isoformat(), freshness_hours=0, now=NOW) is None


class TestIsStale:
    def test_recent_story_not_stale(self) -> None:
        published = (NOW - timedelta(hours=1)).isoformat()
        assert is_stale(published, freshness_hours=48, now=NOW) is False

    def test_within_rejection_multiplier_not_stale(self) -> None:
        # 48h window * 3.0 multiplier = 144h cutoff
        published = (NOW - timedelta(hours=140)).isoformat()
        assert is_stale(published, freshness_hours=48, now=NOW) is False

    def test_beyond_rejection_multiplier_is_stale(self) -> None:
        published = (NOW - timedelta(hours=200)).isoformat()
        assert is_stale(published, freshness_hours=48, now=NOW) is True

    def test_missing_published_at_never_stale(self) -> None:
        """An evergreen candidate with no timestamp is never filtered."""
        assert is_stale(None, freshness_hours=48, now=NOW) is False

    def test_configurable_window_changes_cutoff(self) -> None:
        published = (NOW - timedelta(hours=10)).isoformat()
        assert is_stale(published, freshness_hours=48, now=NOW) is False
        assert is_stale(published, freshness_hours=1, now=NOW) is True
