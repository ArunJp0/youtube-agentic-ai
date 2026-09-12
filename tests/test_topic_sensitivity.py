# Tests for the deterministic sensitive-category keyword heuristic
# (src/services/topic_sensitivity.py). No LLM/network calls - pure
# function only. Never suppresses/rewrites - only flags.
from __future__ import annotations

from src.services.topic_sensitivity import detect_sensitivity


class TestDetectSensitivity:
    def test_non_sensitive_topic_not_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("Why do cats purr?")
        assert is_sensitive is False
        assert reasons == []

    def test_conflict_keyword_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("Military airstrike hits capital city")
        assert is_sensitive is True
        assert "active_conflict" in reasons

    def test_election_keyword_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("National election results announced today")
        assert is_sensitive is True
        assert "elections_politics" in reasons

    def test_crime_allegation_keyword_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("Executive arrested on fraud charges")
        assert is_sensitive is True
        assert "crime_allegations" in reasons

    def test_medical_emergency_keyword_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("New virus outbreak reported in region")
        assert is_sensitive is True
        assert "medical_emergency" in reasons

    def test_casualty_keyword_flagged(self) -> None:
        is_sensitive, reasons = detect_sensitivity("Death toll rises after building collapse")
        assert is_sensitive is True
        assert "casualties" in reasons

    def test_multiple_categories_can_match(self) -> None:
        is_sensitive, reasons = detect_sensitivity("War death toll climbs as ceasefire talks stall")
        assert is_sensitive is True
        assert "active_conflict" in reasons
        assert "casualties" in reasons

    def test_case_insensitive_matching(self) -> None:
        is_sensitive, _ = detect_sensitivity("BREAKING: WAR ESCALATES OVERNIGHT")
        assert is_sensitive is True

    def test_word_boundary_avoids_false_positive_substring(self) -> None:
        """'warehouse' must not match the 'war' keyword."""
        is_sensitive, reasons = detect_sensitivity("New warehouse robot improves shipping speed")
        assert is_sensitive is False
        assert reasons == []

    def test_empty_text_never_sensitive(self) -> None:
        assert detect_sensitivity("") == (False, [])

    def test_none_text_never_sensitive(self) -> None:
        assert detect_sensitivity(None) == (False, [])

    def test_never_suppresses_only_flags(self) -> None:
        """detect_sensitivity returns data only - it has no side effect and
        makes no decision about whether the story should be used."""
        result = detect_sensitivity("Election results contested amid allegations")
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)
