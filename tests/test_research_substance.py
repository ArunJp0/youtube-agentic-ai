# Tests for src.services.research_substance - the distinct-content-aware
# substance gate, near-duplicate/title-only detection, and the key-point
# meta-refusal detector. All mocked/pure - no real network, no real LLM
# calls.
from __future__ import annotations

from src.services.research_substance import (
    SubstanceAssessment,
    assess_substance,
    dedupe_by_content,
    group_near_duplicate_content,
    is_meta_refusal_response,
    is_title_only_snippet,
)


def _result(title: str, snippet: str, url: str = "https://example.com/a") -> dict:
    return {"title": title, "snippet": snippet, "url": url}


def _substantive(url: str, sentence: str, words: int = 50) -> dict:
    """A genuinely distinct, substantive result - real per-source content
    (including a distinct title, derived from its own sentence, not a
    templated near-identical one), not a repeated headline."""
    return _result(
        title=sentence,
        snippet=sentence + " " + " ".join(f"{url.rsplit('/', 1)[-1]}-fact-{i}" for i in range(words)),
        url=url,
    )


class TestTitleOnlyDetection:
    def test_empty_snippet_is_title_only(self) -> None:
        assert is_title_only_snippet("Some Headline", "") is True

    def test_snippet_identical_to_title_is_title_only(self) -> None:
        assert is_title_only_snippet("Poll shows rising trust", "Poll shows rising trust") is True

    def test_snippet_is_title_plus_outlet_suffix_is_title_only(self) -> None:
        assert (
            is_title_only_snippet(
                "Poll shows rising trust in global institutions",
                "Poll shows rising trust in global institutions - Example Times",
            )
            is True
        )

    def test_genuinely_distinct_snippet_is_not_title_only(self) -> None:
        assert (
            is_title_only_snippet(
                "Poll shows rising trust in global institutions",
                "A new survey of 40,000 respondents across 30 countries found that "
                "confidence in multilateral organizations rose five points since last year.",
            )
            is False
        )

    def test_short_unrelated_snippet_is_not_title_only(self) -> None:
        # Below the fuzzy-match length floor and not an exact/near match -
        # never flagged just for being short.
        assert is_title_only_snippet("Poll shows rising trust", "Short update.") is False


class TestNearDuplicateGrouping:
    def test_four_identical_headlines_collapse_to_one_group(self) -> None:
        results = [
            _result("Why X grew stronger through years of Y", "Why X grew stronger through years of Y", url=f"https://outlet{i}.com/a")
            for i in range(4)
        ]
        groups = group_near_duplicate_content(results)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_near_duplicate_with_outlet_suffix_still_collapses(self) -> None:
        headline = "Why the region's economy has only grown stronger through years of change"
        results = [
            _result(headline, headline, url="https://a.com/1"),
            _result(f"{headline} - Outlet A", f"{headline} - Outlet A", url="https://b.com/2"),
            _result(f"{headline} | Outlet B", f"{headline} | Outlet B", url="https://c.com/3"),
        ]
        groups = group_near_duplicate_content(results)
        assert len(groups) == 1

    def test_genuinely_distinct_stories_do_not_collapse(self) -> None:
        results = [
            _substantive("https://a.com/1", "The first distinct real news story with its own details."),
            _substantive("https://b.com/2", "A completely different second story covering another angle."),
        ]
        groups = group_near_duplicate_content(results)
        assert len(groups) == 2

    def test_dedupe_by_content_keeps_most_informative_representative(self) -> None:
        thin = _result("Same Headline", "Same Headline", url="https://a.com/thin")
        rich = _result(
            "Same Headline",
            "Same Headline. " + " ".join(f"detail-{i}" for i in range(50)),
            url="https://b.com/rich",
        )
        deduped = dedupe_by_content([thin, rich])
        assert len(deduped) == 1
        assert deduped[0]["url"] == "https://b.com/rich"

    def test_dedupe_by_content_preserves_provenance_fields(self) -> None:
        r = _result("Headline", "Headline. " + " ".join(f"w{i}" for i in range(30)), url="https://a.com/1")
        r["published_at"] = "2026-09-16T00:00:00+00:00"
        r["source_name"] = "Example News"
        deduped = dedupe_by_content([r])
        assert deduped[0]["published_at"] == "2026-09-16T00:00:00+00:00"
        assert deduped[0]["source_name"] == "Example News"
        assert deduped[0]["url"] == "https://a.com/1"


class TestAssessSubstance:
    def test_four_identical_repeated_headlines_do_not_satisfy_threshold(self) -> None:
        results = [
            _result(
                "Why Yemen's Houthis have only grown stronger through years of war",
                "Why Yemen's Houthis have only grown stronger through years of war",
                url=f"https://outlet{i}.com/a",
            )
            for i in range(4)
        ]
        assessment = assess_substance(results, min_words=40)
        assert assessment.sufficient is False
        assert assessment.distinct_word_count == 0  # title-only content contributes nothing
        assert assessment.title_only_count == 4
        assert assessment.distinct_source_count == 1

    def test_near_duplicate_syndicated_snippets_do_not_inflate_substance(self) -> None:
        base = "A detailed real report about the ongoing situation with many specific facts and figures included here today."
        results = [_result("Headline", base, url=f"https://outlet{i}.com/a") for i in range(5)]
        single = assess_substance(results[:1], min_words=1)
        combined = assess_substance(results, min_words=1)
        assert combined.distinct_word_count == single.distinct_word_count
        assert combined.distinct_source_count == 1

    def test_genuinely_distinct_relevant_summaries_satisfy_substance(self) -> None:
        results = [
            _substantive("https://a.com/1", "First real distinct fact set about the story."),
            _substantive("https://b.com/2", "Second real distinct fact set covering another angle."),
        ]
        assessment = assess_substance(results, min_words=40)
        assert assessment.sufficient is True
        assert assessment.distinct_source_count == 2

    def test_title_only_rss_description_recognized_as_thin(self) -> None:
        results = [_result("A fairly long real headline about a real event", "", url="https://a.com/1")]
        assessment = assess_substance(results, min_words=5)
        assert assessment.title_only_count == 1
        assert assessment.distinct_word_count == 0
        assert assessment.sufficient is False

    def test_mixed_duplicate_and_substantive_sources_retain_substantive_information(self) -> None:
        duplicates = [
            _result("Same headline repeated", "Same headline repeated", url=f"https://outlet{i}.com/a")
            for i in range(3)
        ]
        substantive = _substantive("https://real.com/story", "A genuinely distinct, substantial report.", words=50)
        assessment = assess_substance(duplicates + [substantive], min_words=40)
        assert assessment.sufficient is True
        assert assessment.title_only_count == 3
        # The substantive source's own word count survives untouched.
        assert assessment.distinct_word_count == len(substantive["snippet"].split())

    def test_empty_results_are_insufficient(self) -> None:
        assessment = assess_substance([], min_words=1)
        assert assessment.sufficient is False
        assert assessment.distinct_word_count == 0

    def test_single_short_but_genuinely_distinct_snippet_below_threshold(self) -> None:
        results = [_substantive("https://a.com/1", "A real fact.", words=5)]
        assessment = assess_substance(results, min_words=40)
        assert assessment.sufficient is False
        assert assessment.distinct_word_count > 0  # counted, just not enough


class TestMetaRefusalDetection:
    def test_refusal_citing_missing_source_material_detected(self) -> None:
        text = (
            "Based on the provided titles, there is no underlying text or research data "
            "supplied to extract specific findings from."
        )
        assert is_meta_refusal_response(text) is True

    def test_request_for_more_input_detected(self) -> None:
        text = "If you can provide the full article text, I would be happy to extract the key points for you."
        assert is_meta_refusal_response(text) is True

    def test_genuine_factual_key_point_not_flagged(self) -> None:
        assert is_meta_refusal_response("Dreams primarily occur during REM sleep cycles at night.") is False

    def test_genuine_key_point_mentioning_data_not_flagged(self) -> None:
        """A real news fact mentioning 'data' must not be false-flagged -
        the detector requires a specific meta-discourse phrase, not just
        the bare word."""
        text = "New government data shows unemployment fell to its lowest level in a decade."
        assert is_meta_refusal_response(text) is False

    def test_empty_text_not_flagged(self) -> None:
        assert is_meta_refusal_response("") is False
        assert is_meta_refusal_response(None) is False

    def test_generic_across_any_topic(self) -> None:
        """Never references any specific topic/person/story - same
        detection logic regardless of subject matter."""
        a = is_meta_refusal_response("The source material only contains a title with no further details.")
        b = is_meta_refusal_response("The article text provided lacks any real information to summarize.")
        assert a is True
        assert b is True
