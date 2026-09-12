# Deterministic freshness scoring/filtering for current/trending topic
# candidates - timestamps only, never LLM judgment. See STEP 4: "Use
# timestamps deterministically."
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# A candidate older than this multiple of the configured freshness window
# is filtered out entirely in trending mode (a hard cutoff), not merely
# down-scored - "freshness scoring/filtering" per the spec, both words.
STALE_REJECTION_MULTIPLIER = 3.0


def parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime,
    or None if missing/unparseable - never raises."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def freshness_score(
    published_at: Optional[str], freshness_hours: float, now: Optional[datetime] = None
) -> Optional[float]:
    """Bounded 0-1 freshness score: 1.0 for "just published", linearly
    decaying to 0.0 at the configured freshness window, floored at 0.0
    beyond that (never negative). Returns None (not applicable) if
    ``published_at`` is missing/unparseable or ``freshness_hours`` <= 0 -
    the caller treats an absent score as "no freshness signal", not "stale".
    """
    published = parse_iso_timestamp(published_at)
    if published is None or freshness_hours <= 0:
        return None

    current = now or datetime.now(timezone.utc)
    age_hours = (current - published).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    if age_hours >= freshness_hours:
        return 0.0
    return round(1.0 - (age_hours / freshness_hours), 4)


def is_stale(published_at: Optional[str], freshness_hours: float, now: Optional[datetime] = None) -> bool:
    """Hard freshness filter: True if ``published_at`` is known and older
    than STALE_REJECTION_MULTIPLIER times the configured freshness window.
    A candidate with no known publish timestamp is never considered stale
    by this check (there's nothing to measure) - evergreen candidates are
    unaffected."""
    published = parse_iso_timestamp(published_at)
    if published is None or freshness_hours <= 0:
        return False

    current = now or datetime.now(timezone.utc)
    age_hours = (current - published).total_seconds() / 3600.0
    return age_hours > (freshness_hours * STALE_REJECTION_MULTIPLIER)
