# Tests for deterministic thumbnail image selection
# (src/services/thumbnail_selection.py). No LLM/network involved.
from __future__ import annotations

from src.services.thumbnail_selection import select_thumbnail_candidate
from src.tools.media_provider import MediaCandidate


def _candidate(content_hint=None, asset_id="1") -> MediaCandidate:
    return MediaCandidate(
        asset_type="image",
        download_url=f"https://mock.media/{asset_id}.jpg",
        source_url=f"https://mock.media/page/{asset_id}",
        provider_asset_id=asset_id,
        content_hint=content_hint,
    )


class TestSelectThumbnailCandidate:
    def test_empty_candidates_returns_none_with_warning(self) -> None:
        candidate, warnings = select_thumbnail_candidate([], [])
        assert candidate is None
        assert any("no candidate" in w.lower() for w in warnings)

    def test_no_avoid_concepts_returns_first_candidate(self) -> None:
        first = _candidate(asset_id="1")
        second = _candidate(asset_id="2")
        candidate, warnings = select_thumbnail_candidate([first, second], [])
        assert candidate is first
        assert warnings == []

    def test_candidate_matching_avoid_concept_is_skipped(self) -> None:
        bad = _candidate(content_hint="construction site workers", asset_id="1")
        good = _candidate(content_hint="person sleeping peacefully", asset_id="2")
        candidate, warnings = select_thumbnail_candidate([bad, good], ["construction site"])
        assert candidate is good
        assert warnings == []

    def test_candidate_without_content_hint_is_not_filtered(self) -> None:
        no_hint = _candidate(content_hint=None, asset_id="1")
        candidate, warnings = select_thumbnail_candidate([no_hint], ["construction"])
        assert candidate is no_hint

    def test_all_candidates_matching_avoid_falls_back_to_first_with_warning(self) -> None:
        first = _candidate(content_hint="construction site", asset_id="1")
        second = _candidate(content_hint="construction workers", asset_id="2")
        candidate, warnings = select_thumbnail_candidate([first, second], ["construction"])
        assert candidate is first
        assert any("last resort" in w.lower() for w in warnings)
