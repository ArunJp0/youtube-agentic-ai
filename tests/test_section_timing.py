# Tests for the shared deterministic section-duration calculation used by
# both VisualMediaService (slot planning) and VideoAssemblyService (clip
# sizing).
from __future__ import annotations

import pytest

from src.models.script import ScriptSection
from src.services.section_timing import calculate_section_durations


class TestCalculateSectionDurations:
    def test_durations_proportional_to_narration_length(self) -> None:
        sections = [
            ScriptSection(heading="A", narration="one two three four"),
            ScriptSection(
                heading="B", narration="one two three four five six seven eight ten eleven twelve"
            ),
        ]
        durations = calculate_section_durations(sections, 40.0)
        assert len(durations) == 2
        assert durations[1] > durations[0]
        assert sum(durations) == pytest.approx(40.0, abs=0.01)

    def test_durations_sum_exactly_matches_total_despite_rounding(self) -> None:
        sections = [ScriptSection(heading=f"S{i}", narration="word " * (i + 1)) for i in range(5)]
        durations = calculate_section_durations(sections, 123.456)
        assert sum(durations) == pytest.approx(123.456, abs=0.001)

    def test_single_section_gets_full_duration(self) -> None:
        sections = [ScriptSection(heading="Only", narration="Some narration text here.")]
        durations = calculate_section_durations(sections, 50.0)
        assert durations == pytest.approx([50.0])

    def test_equal_length_sections_get_equal_durations(self) -> None:
        sections = [
            ScriptSection(heading="A", narration="one two three four"),
            ScriptSection(heading="B", narration="five six seven eight"),
        ]
        durations = calculate_section_durations(sections, 20.0)
        assert durations[0] == pytest.approx(10.0)
        assert durations[1] == pytest.approx(10.0)

    def test_empty_sections_returns_empty_list(self) -> None:
        assert calculate_section_durations([], 60.0) == []
