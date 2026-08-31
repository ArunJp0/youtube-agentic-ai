# Tests for lightweight asset-ID/timeline-based repetition QC.
from __future__ import annotations

from src.services.repetition_check import detect_repetition_warnings


class TestDetectRepetitionWarnings:
    def test_no_warnings_for_all_distinct_assets(self) -> None:
        assert detect_repetition_warnings(["a", "b", "c", "d"], min_gap=2) == []

    def test_back_to_back_repeat_flagged(self) -> None:
        warnings = detect_repetition_warnings(["a", "a", "b"], min_gap=2)
        assert len(warnings) == 1
        assert "back-to-back" in warnings[0]
        assert "'a'" in warnings[0]

    def test_short_interval_reuse_flagged(self) -> None:
        warnings = detect_repetition_warnings(["a", "b", "a"], min_gap=2)
        assert len(warnings) == 1
        assert "back-to-back" not in warnings[0]

    def test_reuse_beyond_min_gap_not_flagged(self) -> None:
        warnings = detect_repetition_warnings(["a", "b", "c", "a"], min_gap=2)
        assert warnings == []

    def test_reuse_exactly_at_min_gap_boundary_flagged(self) -> None:
        # gap of 2 (positions 0 and 2) with min_gap=2 should still flag.
        warnings = detect_repetition_warnings(["a", "b", "a"], min_gap=2)
        assert len(warnings) == 1

    def test_empty_list_returns_no_warnings(self) -> None:
        assert detect_repetition_warnings([], min_gap=2) == []

    def test_empty_asset_ids_ignored(self) -> None:
        assert detect_repetition_warnings(["", "", ""], min_gap=2) == []

    def test_controlled_long_video_reuse_pattern_flags_each_short_gap(self) -> None:
        # 0,1,2,0,1,2,... pattern (as VisualMediaService's own reuse-after-
        # gap policy produces) - every reuse here has a gap of 3, so with
        # min_gap=2 none should be flagged (acceptable controlled reuse).
        sequence = ["0", "1", "2", "0", "1", "2", "0"]
        assert detect_repetition_warnings(sequence, min_gap=2) == []

    def test_multiple_distinct_warnings_reported(self) -> None:
        warnings = detect_repetition_warnings(["a", "a", "b", "b"], min_gap=2)
        assert len(warnings) == 2
