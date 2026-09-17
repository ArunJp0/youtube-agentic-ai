# Tests for src.services.research_relevance - the current-news relevance
# filter, query simplifier, and dedup helper. All mocked (no real network,
# no real LLM calls).
from __future__ import annotations

from src.services.research_relevance import (
    ResearchRelevanceFilter,
    dedupe_by_url,
    significant_terms,
    simplify_query,
)


class _RecordingLLMProvider:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _ExplodingLLMProvider:
    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")


def _result(title: str, snippet: str = "", url: str = "https://example.com/a") -> dict:
    return {"title": title, "snippet": snippet, "url": url}


class TestDeterministicPass:
    def test_unrelated_result_rejected_without_any_llm_call(self) -> None:
        llm = _RecordingLLMProvider()
        filt = ResearchRelevanceFilter(llm)
        results = [_result("Pop star clarifies tour rumor", "An unrelated entertainment story.")]

        outcome = filt.filter("Poll shows rising trust in global institutions", results)

        assert outcome.kept == []
        assert outcome.rejected_count == 1
        assert outcome.semantic_review_performed is False
        assert llm.calls == []

    def test_relevant_result_retained(self) -> None:
        filt = ResearchRelevanceFilter(None)
        results = [_result("Poll shows rising trust in global institutions", "New survey data on institutions.")]

        outcome = filt.filter("Poll shows rising trust in global institutions", results)

        assert len(outcome.kept) == 1
        assert outcome.rejected_count == 0

    def test_older_background_source_retained_regardless_of_age(self) -> None:
        """The filter never inspects published_at - age alone is never the
        relevance decision (see module docstring)."""
        filt = ResearchRelevanceFilter(None)
        results = [
            {
                "title": "Global institutions trust background report",
                "snippet": "Historical context on institutions and trust.",
                "url": "https://example.com/old",
                "published_at": "2019-01-01T00:00:00+00:00",
            }
        ]

        outcome = filt.filter("Poll shows rising trust in global institutions", results)

        assert len(outcome.kept) == 1
        assert outcome.kept[0]["published_at"] == "2019-01-01T00:00:00+00:00"

    def test_does_not_accept_a_source_merely_because_it_is_verbose(self) -> None:
        verbose_but_unrelated = _result(
            "Local team wins championship",
            " ".join(["championship"] * 200),
        )
        filt = ResearchRelevanceFilter(None)

        outcome = filt.filter("Poll shows rising trust in global institutions", [verbose_but_unrelated])

        assert outcome.kept == []

    def test_no_results_returns_empty_without_llm_call(self) -> None:
        llm = _RecordingLLMProvider()
        filt = ResearchRelevanceFilter(llm)

        outcome = filt.filter("Any topic", [])

        assert outcome.kept == []
        assert outcome.semantic_review_performed is False
        assert llm.calls == []

    def test_empty_topic_never_rejects_anything_deterministically(self) -> None:
        filt = ResearchRelevanceFilter(None)
        results = [_result("Anything at all", "Some snippet.")]

        outcome = filt.filter("", results)

        assert len(outcome.kept) == 1


class TestSemanticBatchPass:
    def test_single_batched_call_for_multiple_ambiguous_candidates(self) -> None:
        """Requirement: at most ONE bounded batch LLM call for ALL
        candidates, never one call per source."""
        response = (
            '{"classifications": ['
            '{"index": 0, "classification": "direct", "reason": "on topic"}, '
            '{"index": 1, "classification": "direct", "reason": "on topic"}, '
            '{"index": 2, "classification": "unrelated", "reason": "unrelated tangent"}'
            "]}"
        )
        llm = _RecordingLLMProvider(response)
        filt = ResearchRelevanceFilter(llm)
        topic = "Poll shows rising trust in global institutions"
        results = [
            _result("Poll on global institutions trust", "Survey data on trust.", url="https://example.com/1"),
            _result("Institutions trust survey follow-up", "More survey detail.", url="https://example.com/2"),
            # Shares the word "institutions" incidentally but is a different story -
            # passes the cheap deterministic pass, gets caught by the semantic one.
            _result("Local institutions host trust-building bake sale", "Community bake sale news.", url="https://example.com/3"),
        ]

        outcome = filt.filter(topic, results)

        assert len(llm.calls) == 1
        assert len(outcome.kept) == 2
        assert len(outcome.direct) == 2
        assert outcome.semantic_review_performed is True
        assert {r["url"] for r in outcome.kept} == {"https://example.com/1", "https://example.com/2"}

    def test_llm_failure_falls_back_to_deterministic_survivors(self) -> None:
        filt = ResearchRelevanceFilter(_ExplodingLLMProvider())
        topic = "Poll shows rising trust in global institutions"
        results = [_result("Poll on global institutions trust", "Survey data.", url="https://example.com/1")]

        outcome = filt.filter(topic, results)

        assert len(outcome.kept) == 1
        assert outcome.semantic_review_performed is False

    def test_malformed_llm_response_falls_back_to_deterministic_survivors(self) -> None:
        llm = _RecordingLLMProvider("not json at all")
        filt = ResearchRelevanceFilter(llm)
        topic = "Poll shows rising trust in global institutions"
        results = [_result("Poll on global institutions trust", "Survey data.", url="https://example.com/1")]

        outcome = filt.filter(topic, results)

        assert len(outcome.kept) == 1
        assert outcome.semantic_review_performed is False

    def test_missing_index_in_response_defaults_to_kept(self) -> None:
        llm = _RecordingLLMProvider('{"relevance": []}')
        filt = ResearchRelevanceFilter(llm)
        topic = "Poll shows rising trust in global institutions"
        results = [_result("Poll on global institutions trust", "Survey data.", url="https://example.com/1")]

        outcome = filt.filter(topic, results)

        assert len(outcome.kept) == 1
        assert outcome.semantic_review_performed is True

    def test_no_llm_provider_uses_deterministic_pass_only(self) -> None:
        filt = ResearchRelevanceFilter(None)
        topic = "Poll shows rising trust in global institutions"
        results = [_result("Poll on global institutions trust", "Survey data.", url="https://example.com/1")]

        outcome = filt.filter(topic, results)

        assert len(outcome.kept) == 1
        assert outcome.semantic_review_performed is False


class TestSimplifyQuery:
    def test_splits_on_first_comma_clause(self) -> None:
        result = simplify_query("Poll shows rising trust in global institutions, less comfort with US leadership")
        assert result == "Poll shows rising trust in global institutions"

    def test_falls_back_to_first_n_words_with_no_clause_break(self) -> None:
        result = simplify_query("One two three four five six seven eight nine ten eleven")
        assert result == "One two three four five six seven eight"
        assert len(result.split()) == 8

    def test_short_topic_returned_unchanged(self) -> None:
        assert simplify_query("Why do cats purr") == "Why do cats purr"

    def test_empty_topic_returns_empty(self) -> None:
        assert simplify_query("") == ""
        assert simplify_query("   ") == ""

    def test_never_hardcodes_any_specific_story_text(self) -> None:
        """Generic-across-topics guard: the same mechanical rule applies
        regardless of subject matter."""
        a = simplify_query("Company X announces merger, shares surge on the news")
        b = simplify_query("Scientists discover new exoplanet, mission called historic")
        assert a == "Company X announces merger"
        assert b == "Scientists discover new exoplanet"


class TestDedupeByUrl:
    def test_removes_later_duplicate_url(self) -> None:
        results = [
            {"url": "https://example.com/a", "title": "First"},
            {"url": "https://example.com/a", "title": "Duplicate"},
            {"url": "https://example.com/b", "title": "Second"},
        ]
        deduped = dedupe_by_url(results)
        assert [r["title"] for r in deduped] == ["First", "Second"]

    def test_entries_without_url_are_kept(self) -> None:
        results = [{"url": None, "title": "No URL"}, {"title": "Also no URL"}]
        assert dedupe_by_url(results) == results


class TestSignificantTerms:
    def test_strips_stopwords_and_short_tokens(self) -> None:
        terms = significant_terms("The poll is a big study of institutions")
        assert "the" not in terms
        assert "is" not in terms
        assert "a" not in terms
        assert "poll" in terms
        # Morphologically normalized (plural "s" stripped) - see
        # TestMorphologicalNormalization for dedicated coverage.
        assert "institution" in terms


class TestMorphologicalNormalization:
    """Conservative singular/plural normalization - the real gap a
    controlled validation exposed: a topic's own plural form ("Houthis")
    failed to overlap a candidate using the singular ("Houthi")."""

    def test_singular_plural_mismatch_resolved(self) -> None:
        topic_terms = significant_terms("Why the Houthis have grown stronger")
        candidate_terms = significant_terms("Analysts assess the Houthi military campaign")
        assert topic_terms & candidate_terms

    def test_possessive_form_resolved(self) -> None:
        topic_terms = significant_terms("The nation's economy is struggling")
        candidate_terms = significant_terms("Reports detail the nation economy in depth")
        assert topic_terms & candidate_terms

    def test_does_not_over_normalize_non_plural_endings(self) -> None:
        """Conservative guard: common non-plural "-us"/"-ss"/"-os"
        endings must never be stripped, avoiding an unrelated match.
        "-is" is deliberately NOT protected (see _NON_PLURAL_S_ENDINGS) -
        demonyms/names ending in "i" (Houthi, Israeli, Iraqi) pluralize to
        "-is" and are common in exactly this feature's real domain."""
        for word in ("focus", "class", "glass", "chaos", "business"):
            terms = significant_terms(word)
            assert word in terms, f"{word!r} should not have been altered"

    def test_short_words_never_stripped(self) -> None:
        # len<=4 words are protected from plural-stripping (e.g. "news",
        # "bus") - stripping "news" -> "new" would be a real regression.
        terms = significant_terms("Breaking news bus")
        assert "news" in terms

    def test_deterministic_pass_integration_end_to_end(self) -> None:
        """The morphological fix applied through the actual filter, not
        just the raw token function - the previously-observed real
        false-zero-overlap case now correctly passes the deterministic
        pass (never accepted merely because it's verbose - it must also
        genuinely share normalized terms)."""
        filt = ResearchRelevanceFilter(None)
        topic = "Why Yemen's Houthis have only grown stronger through years of war"
        candidate = _result(
            title="Analysts say the Houthi campaign shows no sign of weakening",
            snippet="Military analysts detail how the Houthi movement expanded its reach in Yemen.",
        )
        outcome = filt.filter(topic, [candidate])
        assert len(outcome.kept) == 1

    def test_ambiguous_case_still_deferred_to_semantic_batch_when_available(self) -> None:
        """Normalization only helps with trivial morphological variants -
        genuinely ambiguous relevance (shares one normalized term by
        coincidence, but is a different story) must still be decided by
        the semantic batch check, not assumed relevant just from overlap."""
        response = '{"classifications": [{"index": 0, "classification": "unrelated", "reason": "different story"}]}'
        llm = _RecordingLLMProvider(response)
        filt = ResearchRelevanceFilter(llm)
        topic = "Houthis seize control of strategic Red Sea islands"
        candidate = _result(
            title="Local council in a different country seizes control of an unrelated islands zoning dispute",
            snippet="A municipal council voted to seize control over an unrelated islands zoning matter.",
        )
        outcome = filt.filter(topic, [candidate])
        assert len(llm.calls) == 1  # deterministic pass alone did not decide
        assert outcome.kept == []
