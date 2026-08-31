# Tests for deterministic representative-frame sampling.
from __future__ import annotations

from src.services.frame_sampling import (
    MEDIUM_CLIP_MAX_SECONDS,
    SHORT_CLIP_MAX_SECONDS,
    calculate_sample_timestamps,
)


class TestCalculateSampleTimestamps:
    def test_short_clip_gets_single_midpoint_frame(self) -> None:
        timestamps = calculate_sample_timestamps(4.0)
        assert timestamps == [2.0]

    def test_at_short_boundary_still_single_frame(self) -> None:
        timestamps = calculate_sample_timestamps(SHORT_CLIP_MAX_SECONDS)
        assert len(timestamps) == 1

    def test_medium_clip_gets_two_frames(self) -> None:
        timestamps = calculate_sample_timestamps(10.0)
        assert len(timestamps) == 2
        assert timestamps == [2.5, 7.5]

    def test_at_medium_boundary_still_two_frames(self) -> None:
        timestamps = calculate_sample_timestamps(MEDIUM_CLIP_MAX_SECONDS)
        assert len(timestamps) == 2

    def test_long_clip_gets_three_frames(self) -> None:
        timestamps = calculate_sample_timestamps(30.0)
        assert len(timestamps) == 3
        assert timestamps == [7.5, 15.0, 22.5]

    def test_never_more_than_three_frames_regardless_of_length(self) -> None:
        assert len(calculate_sample_timestamps(600.0)) == 3

    def test_timestamps_are_ordered_earliest_first(self) -> None:
        timestamps = calculate_sample_timestamps(60.0)
        assert timestamps == sorted(timestamps)

    def test_zero_duration_returns_single_zero_timestamp(self) -> None:
        assert calculate_sample_timestamps(0.0) == [0.0]

    def test_negative_duration_returns_single_zero_timestamp(self) -> None:
        assert calculate_sample_timestamps(-5.0) == [0.0]

    def test_deterministic(self) -> None:
        assert calculate_sample_timestamps(12.0) == calculate_sample_timestamps(12.0)
