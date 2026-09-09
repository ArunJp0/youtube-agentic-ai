# Decisions

## Modular monolith architecture

The system is built as a single deployable codebase organized into clear internal modules (agents, workflows, services, tools, llm, models, config) rather than separate services. Keeps the MVP simple to develop, test, and deploy.

## LangGraph for orchestration

LangGraph is used to define and run multi-step agent workflows (e.g. the research workflow), giving explicit control over state and transitions between reasoning steps.

## Interchangeable LLM/search providers

LLM and search integrations are defined behind abstract provider interfaces (`src/llm/provider.py`, `src/tools/search_provider.py`), with mock implementations for development and testing. Real providers can be swapped in without changing agent or workflow logic.

## No paid APIs required for initial development

The MVP is built entirely on free/open-source or free-tier tools. Paid services are avoided until they are proven necessary.

## Database persistence postponed until real Research Agent validation

SQLAlchemy/Alembic scaffolding exists but is not yet wired into the workflow. Persistence will be added once the Research Agent has been validated against real LLM and search providers, to avoid designing schema around mock data.

## No unnecessary microservices/Celery/Redis/Kubernetes for MVP

The project intentionally avoids distributed-systems infrastructure until the MVP proves the need. This keeps local development and testing fast and simple.

## Wikipedia as the initial real search provider

Wikipedia's public MediaWiki API is free, requires no API key, and is sufficient for MVP research queries. It replaces the mock search provider behind the existing `SearchProvider` interface.

## Gemini as the primary LLM

Gemini is the primary LLM provider for the Research and Script Agents, chosen as a free/low-cost real provider behind the existing `LLMProvider` interface.

## Retry with exponential backoff for transient Gemini errors

429, 408, and 5xx responses (plus timeouts/network errors) are retried with exponential backoff and jitter, since these are transient and typically self-resolve on a subsequent attempt. Non-retryable errors (400/401/403, malformed/blocked responses) fail immediately.

## Fallback model support enabled

If the primary model remains unavailable after exhausting retries, the provider automatically retries against a configured fallback model before failing, improving reliability during upstream capacity issues.

## gemini-3.5-flash-lite preferred as primary for MVP testing

For MVP testing, `gemini-3.5-flash-lite` is configured as the primary model (with `gemini-3.6-flash` as fallback) due to higher free-tier RPM reliability - live pipeline runs against `gemini-3.6-flash` as primary hit sustained 429/503 rate limiting, whereas the lite tier's higher free quota keeps end-to-end validation runs from being dominated by retries/fallback.

## Research Agent remains provider-agnostic

All Wikipedia- and Gemini-specific logic (HTTP details, retry/backoff, fallback model selection) is contained within their respective provider implementations. `ResearchAgent` and the LangGraph workflow only depend on the abstract `LLMProvider`/`SearchProvider` interfaces and were not modified to add real-provider support.

## Script Agent consumes ResearchResult, reuses the LLMProvider abstraction

`ScriptAgent` takes a structured `ResearchResult` as input and depends only on the existing `LLMProvider` interface (the same abstraction the Research Agent uses) - it has no direct dependency on Gemini or any other concrete provider. This mirrors the Research Agent's design and lets the Script Agent run against mock or real LLM providers interchangeably.

## Script Agent narration is grounded in research, not free-generated

Every prompt sent by `ScriptAgent` includes the same compact research context (summary, key points, sourced facts) and explicitly instructs the LLM not to state claims beyond what was researched. Each script section carries the research's source URLs (`source_refs`) for traceability back to where its content came from.

## Script Agent output is structured for future Voice/Visual agents, not built as an article

`ScriptResult`/`ScriptSection` are shaped for downstream consumption: `narration` fields are written as natural spoken narration (not article prose), each section has an `estimated_duration_seconds` (derived deterministically from word count at a fixed speaking rate, not LLM-guessed) and an optional `visual_notes` placeholder for a future Visual Agent. The Voice and Visual/Video agents themselves are intentionally not implemented yet.

## ResearchResult passed directly into ScriptAgent through LangGraph orchestration

The Research → Script pipeline graph passes the `ResearchResult` produced by the research node directly into the script node's state, with no intermediate transformation or persistence step. This keeps the two agents decoupled (each still only depends on its own input/output models) while letting LangGraph own the hand-off between them.

## Automated tests use mocks; real API calls are reserved for manual validation

All automated/CI-run tests (pytest suite) exercise agents and workflows exclusively against mock LLM/search providers - no test consumes real Gemini or Wikipedia quota. Real-provider behavior (including live retry/backoff/fallback) is validated manually via the demo runners (`real_research_demo.py`, `script_demo.py`, `pipeline_demo.py`), not as part of the automated suite.

## Script generation remains provider-agnostic

`ScriptAgent` depends only on the existing `LLMProvider` abstraction, the same interface the Research Agent uses. It has no direct coupling to Gemini or any other concrete provider, so it runs unchanged against mock or real LLM providers.

## Voice generation is a deterministic service, not an LLM agent

Converting a `ScriptResult` into narration audio requires no reasoning or judgment - narration text extraction and ordering are fixed rules, and the actual speech synthesis is delegated to a provider. `VoiceService` is implemented as a plain deterministic service (mirroring the existing `services/` pattern) rather than a LangGraph agent, since there is no clear need for multi-step reasoning at this stage.

## VoiceProvider abstraction with Edge TTS as the free MVP implementation

Text-to-speech is defined behind a `VoiceProvider` interface (`src/tools/voice_provider.py`), with a `MockVoiceProvider` for tests and a free `EdgeVoiceProvider` (Microsoft Edge's neural TTS via the `edge-tts` package) for real narration - no API key required, suitable for local Windows development. `VoiceService` depends only on this interface, never on `edge-tts` directly.

## ElevenLabs deferred, not implemented

Paid ElevenLabs TTS is intentionally not implemented yet, consistent with the "no paid APIs for initial development" decision. The `VoiceProvider` abstraction is designed so an `ElevenLabsVoiceProvider` could be added later (as `EdgeVoiceProvider` was) without changing `VoiceService` or any calling code.

## Narration excludes non-spoken metadata

`VoiceService.extract_narration` only ever sends `hook`, `introduction`, each section's `narration`, `conclusion`, and `call_to_action` (in that order) to the voice provider. Source URLs, research/script notes, section headings, and visual notes are excluded, since they are metadata for other consumers (traceability, a future Visual Agent) and were never meant to be spoken.

## Duplicate prevention belongs primarily in ScriptAgent; VoiceService dedup is a final safety check

Investigating repeated narration in generated videos showed the root cause was `ScriptAgent` producing repetitive/boilerplate section content, not `VoiceService`'s narration assembly. `ScriptAgent` now actively avoids this at generation time (each section's prompt lists other points to avoid repeating, and a duplicate response is retried with a strengthened prompt before being dropped), with a `MIN_DISTINCT_SECTIONS` validation catching persistent failures. `VoiceService` keeps its own exact/near-duplicate check, but only as a last-resort safety net - it should rarely need to remove anything now.

## VoiceService is invoked as a pipeline stage, still deterministic

`VoiceService` is called directly from a `voice` node in the Research → Script → Voice LangGraph pipeline, running after `ScriptAgent` completes. Being wired into the pipeline doesn't change its nature: it remains a deterministic service with no reasoning/judgment of its own, invoked the same way whether called standalone or from the graph.

## Failed Research or Script stages short-circuit before voice generation

The pipeline routes conditionally after each stage: if Research fails, Script never runs; if Script fails, Voice never runs. This avoids wasting a TTS call (and writing an audio file) for a script that doesn't exist, and keeps failures attributable to the stage that actually failed rather than cascading into a misleading downstream error.

## Edge TTS is the current MVP voice provider

Consistent with the free/no-paid-API MVP constraint, Edge TTS remains the default real `VoiceProvider` used by the pipeline (`VOICE_PROVIDER=edge`). ElevenLabs or other paid providers remain deferred, swappable in later via the same `VoiceProvider` interface without pipeline changes.

## Generated media stays out of Git

Audio files produced by the pipeline (`output/audio/*.mp3`) are written to a directory already excluded via `.gitignore`. Generated media is treated as build output, not source - it's reproducible from a topic string plus the configured providers, so there's nothing to gain from tracking it and real cost (repo bloat) in doing so. The same applies to visual assets under `output/media/`.

## Visual media retrieval is a deterministic service, not an LLM agent

Preparing visual assets for a `ScriptResult` requires no reasoning or judgment - deriving a search query from a section's own text is fixed keyword-extraction logic, and asset selection (prefer video, then landscape, then first non-duplicate candidate) is a fixed rule. `VisualMediaService` is implemented as a plain deterministic service (mirroring `VoiceService`) rather than a LangGraph agent, consistent with "don't build an autonomous agent unless there's a clear need."

## MediaProvider abstraction with Pexels as the free MVP implementation

Stock image/video retrieval is defined behind a `MediaProvider` interface (`src/tools/media_provider.py`), with a `MockMediaProvider` for tests and a free `PexelsMediaProvider` for real assets. Pexels was chosen over Pixabay/Unsplash for the MVP because its free tier needs only a simple API key (no OAuth), all content is royalty-free and watermark-free by license, and its API supports server-side `orientation=landscape` filtering - directly serving the "prefer 16:9 landscape" requirement without extra client-side logic. `VisualMediaService` depends only on the `MediaProvider` interface, never on the Pexels API directly.

## Search queries are extracted deterministically, not via an LLM call

`VisualMediaService.build_search_queries` derives search queries from a section's own heading and narration using fixed keyword extraction and concept mapping (see below) - no LLM call per section. This avoids spending Gemini quota on a task that doesn't need reasoning, and keeps queries strictly derived from that section's actual content rather than any hardcoded topic assumption.

## Abstract/scientific narration is concept-mapped to concrete visual terms before search

An initial version of query extraction just kept the first few non-stopword words by position, which let scientific/abstract phrasing ("prefrontal cortex suppression," "evolutionary functions") pass straight into the Pexels query untranslated while visually concrete words later in the sentence got truncated off - manual review caught this producing irrelevant results (e.g. an industrial fire-extinguisher video for a section about brain activity). The fix: a phrase/word concept map translates abstract terms to concrete visual concepts (e.g. "prefrontal cortex" → "human brain neuroscience"), an expanded low-value word list drops non-visual filler, and remaining keywords are ranked by membership in a generic "concrete/visual" vocabulary before trimming to the query length limit - so a late-but-visual word survives ahead of an early-but-abstract one.

## Query generation uses a specific → broader → topic → last-resort fallback chain

`VisualMediaService` tries an ordered list of queries per section - most specific (concept-mapped heading+narration) first, then a broader heading-only query, then one derived from the script's overall topic, then a generic last resort ("background footage") - stopping at the first that yields a usable (non-duplicate) asset. This gives a section a real chance at a relevant match before falling back to something generic, without ever leaving a section with no visual at all.

## Duplicate visual assets are avoided by URL, not file content

`VisualMediaService` tracks each successfully-used asset's source/download URL across the whole script and skips any later candidate matching one already used, picking the next candidate instead (or recording a clean failure if none remain). Comparing by URL - not downloading and hashing file content - keeps duplicate detection cheap and matches what "the same stock photo/video" actually means for this use case.

## Visual Media Service is now a pipeline stage, not just a standalone service

`VisualMediaService` is called directly from a `media` node appended to the end of the Research → Script → Voice LangGraph pipeline, running after Voice succeeds. As with `VoiceService` joining the pipeline earlier, this doesn't change its nature: it remains a deterministic service invoked the same way whether called standalone (via the media demo) or from the graph, and it receives the same `ScriptResult` the script stage produced, unmodified.

## Media generation only runs after voice succeeds; its own failure ends the pipeline without discarding earlier results

The pipeline's conditional routing now covers four stages: Research → Script → Voice → Visual Media, each gated on the previous one's success. A Visual Media failure is surfaced as a structured `VisualResult` (success=False, error set) alongside the pipeline's overall `failed` status - the already-produced `ResearchResult`/`ScriptResult`/`VoiceResult` are preserved in the final state rather than discarded, so a downstream Video Assembly Service (or a human) can still inspect what succeeded.

## Full four-stage pipeline validated end-to-end with all real providers together

`python -m src.pipeline_demo "Why do humans dream?"` was run with every stage on its real provider simultaneously (Wikipedia search, Gemini LLM, Edge TTS, Pexels media) rather than each service only being validated in isolation. This confirmed the real providers compose correctly through the shared `PipelineState` - not just that each one works alone.

## Video assembly is a deterministic service, not an LLM agent

Combining a `VoiceResult`'s narration audio with a `VisualResult`'s section media into a final MP4 requires no reasoning or judgment - per-section timing is a fixed proportional calculation from narration word counts, and section-to-media mapping is taken directly from the given `VisualResult`. `VideoAssemblyService` is implemented as a plain deterministic service (mirroring `VoiceService`/`VisualMediaService`), consistent with "don't build an autonomous agent unless there's a clear need."

## FFmpeg as the video assembly backend, wrapped behind a VideoAssembler interface

Video encoding/muxing is delegated to the local FFmpeg/ffprobe binaries via a thin `VideoAssembler` interface (`src/tools/ffmpeg_video_assembler.py`), so `VideoAssemblyService` never shells out to FFmpeg directly and can be tested against a fake assembler. Unlike the LLM/search/voice/media providers, there's no "mock FFmpeg" shipped for real use - FFmpeg is a fixed local tool, not a swappable remote vendor - so the abstraction exists purely for testability, and availability is checked eagerly (constructing `FFmpegVideoAssembler` fails immediately, with a clear Windows install command, if `ffmpeg`/`ffprobe` aren't on PATH).

## Section video duration is proportional to narration word count, not an LLM estimate

`VideoAssemblyService.calculate_section_durations` allocates the narration audio's total duration across sections by each section's share of the combined narration word count (e.g. a section with 20% of the words gets ~20% of the timeline), with the last section absorbing rounding drift so durations always sum exactly to the audio duration. The audio file is the authoritative timeline, per the "voice controls total video duration" requirement - video length is never independently estimated.

## Downloaded/generated intermediate assets are temporary working files for now

Stock media (`output/media/`) and generated narration audio (`output/audio/`) are treated as temporary working assets for the current MVP - they may be safely cleaned up once downstream processing (video assembly, eventual upload) has consumed them. Final assembled videos (`output/video/`) should be retained according to a future retention policy once one exists. No automated cleanup or retention enforcement is implemented yet; this is deferred to a later milestone rather than guessed at now.

## Video Assembly Service is now the final pipeline stage, not just a standalone service

`VideoAssemblyService` is called directly from a `video_assembly` node appended to the end of the Research → Script → Voice → Visual Media LangGraph pipeline, running after Visual Media succeeds. As with `VoiceService` and `VisualMediaService` joining the pipeline earlier, this doesn't change its nature: it remains a deterministic service invoked the same way whether called standalone (via the video demo) or from the graph, and it receives the exact `ScriptResult`/`VoiceResult`/`VisualResult` already produced earlier in the same run - nothing is regenerated, re-synthesized, or re-downloaded.

## Pipeline status only becomes "completed" when video assembly itself succeeds

With five stages now chained, "completed" was moved from the Visual Media stage to the Video Assembly stage - a pipeline is only reported as completed once a real final MP4 exists. A Video Assembly failure is surfaced as a structured `VideoAssemblyResult` (success=False, error set) alongside the pipeline's overall `failed` status, with all earlier-stage results (`ResearchResult`/`ScriptResult`/`VoiceResult`/`VisualResult`) preserved rather than discarded.

## Visual slot count is duration-aware, not a fixed count per section/video

`VisualMediaService.calculate_slot_count` derives how many distinct visual slots a section needs from that section's own share of the real narration duration (`VoiceResult.duration_seconds`, passed in as an upstream timing input - never re-estimated independently), not from a fixed number per section or per video. A small tiered cadence policy (`CADENCE_POLICY`, a centralized, non-scattered list of constants) sets the target seconds-per-clip based on total video length: ~6-10s/clip for videos up to 5 minutes, ~8-12s/clip for 5-10 minutes, ~10-15s/clip beyond 10 minutes - so longer videos hold each clip slightly longer rather than needing linearly more clips. This directly replaces the earlier fixed one-clip-per-section approach that caused visible repetition on longer sections.

## A section may contain multiple ordered visual assets

`SectionMediaMapping.assets` is a list, not a single asset: `VisualMediaService` fills each section's planned slots in order, and `VideoAssemblyService` builds and concatenates one clip per slot (each sized to an even share of that section's planned duration) instead of exactly one clip per section. No FFmpeg wrapper/interface changes were needed - `VideoAssembler.build_section_clip` already operated per-single-clip; only the calling orchestration in `VideoAssemblyService` changed to loop over multiple assets per section. A section counts as usable for assembly if at least one of its planned slots produced a usable, on-disk asset - not all of them - since occasional single-slot fallback failures shouldn't discard an otherwise-fine section.

## Only the asset actually selected for a slot is downloaded

`MediaProvider.search()` (metadata only) and `.download()` (explicit, one candidate at a time) were already separate before this milestone; duration-aware planning does not change that interface. `VisualMediaService` inspects search results in-memory to pick a candidate and calls `.download()` only for the one candidate chosen per slot - never for rejected candidates, and never for a slot filled by reusing an already-downloaded asset (see below). This keeps storage/bandwidth proportional to what a video actually needs, even though there are now more slots than before.

## Global duplicate prevention with reuse only as a fallback, never the default

`VisualMediaService` tracks every asset it has downloaded across the *entire* video (keyed by `provider_asset_id`, falling back to URL) and applies a strict selection priority per slot: (1) a never-used asset for the slot's own query, (2) a never-used asset for a broader/topic-level query, (3) an already-downloaded asset that wasn't used in the last couple of slots (`RECENT_REUSE_LOOKBACK`) - reusing its local file with no re-download, (4) only if nothing else qualifies, the most recently used asset regardless of recency (immediate repetition, last resort). `MediaAsset.reused` records which of these happened. The system does not try to guarantee zero repetition at all cost: for long videos where Pexels genuinely lacks enough unique matching stock footage, controlled reuse (and, as a last resort, looping) is an accepted, deliberate fallback rather than a failure condition.

## Query-variant expansion stays deterministic and topic-generic

When a section needs multiple slots, `VisualMediaService.build_query_variants` generates several distinct-but-relevant search queries from that section's own heading+narration by chunking the same ranked/concept-mapped keyword pool (shared with the single broader-query builder via `_ranked_keywords`) into small groups, rather than repeating one query for every slot. This is still a fixed deterministic algorithm - no LLM call per visual slot - and was verified to generalize across unrelated topic domains (electric vehicles, space, volcanoes, ancient trade routes), not just the dream-narration examples that originally motivated it.

## Section timing logic is shared, not duplicated

The proportional (narration-word-count-based) section-duration calculation is needed by both `VisualMediaService` (to size slot counts) and `VideoAssemblyService` (to size clips) - it was extracted into a single `src/services/section_timing.py` module that both import, replacing what had been a duplicated/near-duplicated calculation living on `VideoAssemblyService` alone.

## VoiceService was intentionally left unchanged

Duration-aware visual planning consumes `VoiceResult.duration_seconds` as a read-only upstream input; no changes were made to voice synthesis, narration extraction, or `VoiceService` itself. This kept the change scoped to the visual/media/assembly layers it was meant to improve.

## Known limitation (resolved by Context-Aware Visual Planning): visual semantic relevance was bounded by deterministic query generation and stock availability

Manual review of a real end-to-end run confirmed the repetition problem was substantially improved (clips changed throughout the video, no obvious looping), but some individual stock clips were only loosely related to their section's narration. This was a content-quality limitation of deterministic (non-LLM) keyword/concept-mapped query generation combined with whatever Pexels actually had available for a given query - not a pipeline defect, since the pipeline still reliably produced a complete, correctly-timed, playable video. The Context-Aware Visual Planning and Semantic Media Filtering milestone below directly addresses the lexical-ambiguity part of this limitation; the entries under Known Limitations at the end of this document describe what remains.

## A dedicated VisualContextPlanner supplies semantic understanding, not VisualMediaService or an autonomous agent

Interpreting what a script section actually *means* (as opposed to which literal keywords it contains) requires judgment - exactly the kind of task an LLM is suited for and deterministic keyword extraction is not. Rather than growing `VisualMediaService` into something that reasons about meaning, or building a large autonomous multi-step agent, a small, focused `VisualContextPlanner` (`src/agents/visual_context_planner.py`) was added: it does one thing (read a whole script, return a structured per-section visual plan) and `VisualMediaService` remains a deterministic consumer of its output, unchanged in kind. This mirrors how `ResearchAgent`/`ScriptAgent` already own the LLM-reasoning steps in this codebase while the `services/` layer stays deterministic.

## Visual planning is one LLM call for the whole script, never per-section or per-slot

`VisualContextPlanner.plan_visuals` sends the entire script (topic, video title, every section's heading and narration, in order) to the LLM in a single request and expects one structured JSON response covering every section. This was a deliberate quota/cost constraint: a video's LLM spend for visual planning does not grow with its number of sections or, later, its number of duration-aware visual slots - it is always exactly one call, reusing the same `LLMProvider` already shared by Research and Script (no new provider or setting).

## The visual plan is a typed structured object, not free text

`SectionVisualPlan`/`VisualPlan` (`src/models/visual_plan.py`) give every section a `semantic_summary`, `visual_intents`, `search_queries`, `avoid_concepts`, and `neutral_fallback_queries`. Requiring the LLM to return this shape (parsed as strict JSON, with markdown-fence/commentary stripped defensively) rather than freeform prose keeps `VisualMediaService`'s consumption of it simple, typed, and testable, and gives the planner an explicit place to record disambiguation (`avoid_concepts`) rather than only a "better" keyword list.

## Prompting for contextual disambiguation, not hardcoding word-sense rules

The planner's prompt explicitly instructs the LLM to resolve ambiguous wording using the surrounding sentence and the video's overall topic (illustrated with one generic example - "constructs a narrative" meaning "mentally forms," not "builds" - inside the prompt text itself), rather than the codebase maintaining any dictionary of ambiguous words or topic-specific disambiguation rules. This keeps the mechanism fully generic: the same prompt structure was validated against an unrelated ambiguity (software "bug" vs. insect) in automated tests, and against the real "constructs ... humans exist" narration in a live run, without any code change between the two.

## Planner failure falls back to the exact same deterministic path used when no planner is configured

`VisualContextPlanner.plan_visuals` never raises for an LLM failure, a malformed response, or a schema/section-count mismatch - it catches all of these internally and returns `query_generation.build_deterministic_visual_plan(script)` instead, with `VisualPlan.used_semantic_planning=False` and a `fallback_reason` describing why. This is the identical deterministic plan `VisualMediaService` builds directly when constructed without a planner at all, so there is exactly one fallback implementation, not two divergent ones, and a Gemini outage degrades visual planning back to the already-validated pre-milestone behavior rather than failing the pipeline.

## Deterministic query-generation logic was extracted to a shared module, not duplicated

The keyword-extraction/concept-mapping/query-variant logic that previously lived on `VisualMediaService` was moved into `src/services/query_generation.py` as plain functions, used two ways: directly by `VisualMediaService` when no `VisualContextPlanner` is configured, and internally by `VisualContextPlanner`'s own fallback path. Both routes to "no semantic plan available" now share one implementation and one test suite (`tests/test_query_generation.py`) instead of risking two copies drifting apart.

## Semantic filtering is deterministic keyword overlap against lightweight metadata, not a classifier

`src/services/semantic_visual_filter.py` enforces the plan's `avoid_concepts` by checking a candidate's `content_hint` (a Pexels photo's "alt" text, or a descriptive words-only page-URL slug parsed with a small regex - see `MediaCandidate.content_hint`/`_slug_from_url`) for a substring match, and separately scores positive keyword overlap against `visual_intents`/`search_queries` for diagnostics. Deliberately not a classifier or an embeddings-based similarity check: the goal was a simple, explainable, unit-testable rule the LLM's plan can drive, not a second machine-learning layer. When a candidate has no usable metadata at all, it is not blocked - the filter only rejects on positive evidence of a mismatch, consistent with "a neutral relevant visual beats a specific but misleading one" without going so far as "reject anything unverified."

## Query selection rotates one plan query per slot rather than retrying the full query list every slot

Each visual slot searches its own specific query (rotated by slot index through the plan's `search_queries`, so different slots explore different facets of a section), then the plan's `neutral_fallback_queries`, then a shared last resort - mirroring the rotation approach already validated in the duration-aware milestone. An earlier draft of this change had every slot retry the *entire* specific-query list before falling back, which is unnecessary and would have significantly increased Pexels API calls once a section's stock pool is exhausted; per-slot rotation keeps search-call volume the same as before this milestone while adding semantic filtering on top.

## Selected-asset diagnostics: relevance_tier and relevance_score are informational, not gates

Each successfully selected `MediaAsset` records a `relevance_tier` (`"high"` for a specific-query match, `"neutral"` for a neutral/fallback-query match, `"reused"` for any form of asset reuse) and an optional `relevance_score` (keyword-overlap ratio against the plan, when metadata is available). Both are diagnostic only - printed in the demo runners and available for a future QC step - and are never used to reject a selection on their own; the pass/fail gate is `passes_avoid_filter`, which is explainable and testable, not a numeric threshold.

## VideoAssemblyService and VoiceService remain out of scope

Consistent with "VideoAssemblyService should not make semantic decisions," no changes were made to `VideoAssemblyService`, `FFmpegVideoAssembler`, or `VoiceService` for this milestone - `SectionMediaMapping`'s new `semantic_summary`/`avoid_concepts` fields are additive and optional, so `VideoAssemblyService`'s existing usable-asset mapping needed no changes at all.

## Known limitation (addressed by standalone Visual QC below): the semantic filter never inspected actual frame content

The semantic filter judges a candidate only by lightweight text metadata (alt text or a URL slug); it does not inspect actual video/image frames, so misleading or missing metadata could still let a mismatched candidate through. The Visual QC milestone below adds real frame inspection as a standalone capability; it is not yet wired into the main pipeline, so this limitation still applies to the end-to-end orchestration until the next (integration) milestone.

## Visual QC is a separate service, not logic added to VideoAssemblyService or VisualMediaService

Judging whether an already-selected asset's actual visual content fits a section is a distinct responsibility from selecting candidates (`VisualMediaService`) or assembling the final video (`VideoAssemblyService`), so it was implemented as its own `VisualQCService` (`src/services/visual_qc_service.py`) rather than folded into either. `VisualQCService` consumes what those two already produce/would consume (`ScriptResult`, `VisualPlan`, `VisualResult`) and can call back into `VisualMediaService` for a replacement, but neither existing service gained semantic-judgment logic of its own - `VideoAssemblyService` in particular remains untouched, consistent with "VideoAssemblyService should not make semantic decisions."

## Visual QC inspects actual representative frames, not just metadata

Metadata-only filtering (the prior milestone's `passes_avoid_filter`) can only judge what a provider's alt text or URL slug happens to describe - it cannot catch a mismatch the metadata doesn't mention. `VisualQCService` instead extracts a small number of real frames from each already-downloaded asset (via a new `VideoAssembler.extract_frames` method, using the same ffmpeg binary already relied on for assembly) and sends them to a vision-capable evaluator, so the QC decision is grounded in what the clip actually shows.

## Frame sampling is deterministic, duration-based, and capped at 3 frames - never every frame

`src/services/frame_sampling.py`'s `calculate_sample_timestamps` picks a fixed, small number of timestamps from a clip's own duration alone: 1 frame (50%) for clips up to 6s, 2 frames (25%/75%) up to 15s, 3 frames (25%/50%/75%) beyond that - never proportional to duration, never a full scan. This bounds both FFmpeg extraction cost and, more importantly, the number/size of images sent to the vision model per asset, regardless of how long a selected clip happens to be. An image asset skips extraction entirely - the downloaded file itself is the one "frame."

## VisualRelevanceEvaluator is a new abstraction, not an extension of LLMProvider

`LLMProvider.generate_text(prompt) -> str` has no way to carry images, and extending it would give every existing caller (Research/Script agents, `VisualContextPlanner`) a parameter they never use. Instead, `VisualRelevanceEvaluator` (`src/tools/visual_relevance_evaluator.py`) is a small, separate interface - `evaluate_section(context) -> List[RawAssetVerdict]` - with a `MockVisualRelevanceEvaluator` for tests and a real `GeminiVisualRelevanceEvaluator`. `GeminiLLMProvider`/`LLMProvider` and Research/Script's use of them were not modified.

## The real vision evaluator reuses Gemini's existing generateContent endpoint, not a new API

Before implementing the real evaluator, the existing `GeminiLLMProvider` (`src/llm/gemini.py`) was inspected: it already calls the `generateContent` REST endpoint, which Gemini's flash models support multimodally via `inline_data` image parts alongside `text` parts in the same request. `GeminiVisualRelevanceEvaluator` (`src/tools/gemini_visual_relevance_evaluator.py`) reuses that same endpoint shape (base64-encoded JPEG/PNG/WebP frames as `inline_data`, one request per section), rather than inventing an unsupported call or a second SDK dependency. It defaults to `settings.gemini_model` (the same model Research/Script/`VisualContextPlanner` already use, which is natively multimodal) via a new `get_visual_relevance_evaluator(settings)` factory that reuses `LLM_PROVIDER` rather than adding a separate setting - a mock `LLM_PROVIDER` gets the mock evaluator, a real one gets the real evaluator, automatically.

## A shared JSON-extraction helper replaces the duplicated one in VisualContextPlanner

`VisualContextPlanner` already had a private `_extract_json` (strip a markdown fence, find the outermost `{...}`) for parsing its own structured LLM response. `GeminiVisualRelevanceEvaluator` needs the identical logic for its own structured vision response. Rather than duplicating it, it was extracted into `src/services/llm_json.py` (`extract_json_object`/`JsonExtractionError`) and `VisualContextPlanner` was refactored (behavior-preserving, its own tests unchanged) to use the shared version - a small, justified refactor rather than a second copy.

## One vision request per section, never per frame or per slot

`VisualQCService` batches every asset selected for a section's visual slots into a single `evaluate_section` call, with all of that asset's representative frames attached and labeled by `asset_id`. This mirrors the same per-script-not-per-section discipline `VisualContextPlanner` already established for text planning, applied to vision calls: a section with 5 visual slots costs exactly 1 vision request (plus any bounded replacement re-checks), not 5.

## Relevance decisions come from centralized thresholds, not the evaluator's own opinion

`RawAssetVerdict` (the vision evaluator's raw output) carries only a `relevance_score` (0-1) and a `misleading_or_conflicting` flag - it does not decide "approved" vs "rejected" itself. `VisualQCService` owns that policy via two centralized constants, `APPROVE_SCORE_THRESHOLD` (0.70) and `NEUTRAL_SCORE_THRESHOLD` (0.50): `>= 0.70` approved, `0.50-0.69` neutral/acceptable (still kept - a relevant-but-not-literal B-roll clip is not penalized), `< 0.50` weak (replacement recommended), and `misleading_or_conflicting=True` always forces rejection regardless of score. Keeping this in one service-level place (not scattered per-call magic numbers, and not left to the evaluator to decide) makes the policy tunable and testable independent of the vision model's own behavior.

## Bounded replacement reuses VisualMediaService's own selection logic, not a second implementation

When Visual QC recommends replacing a weak/misleading asset, it doesn't re-implement candidate search - it calls a new public `VisualMediaService.acquire_replacement_asset`, which delegates to the exact same `_acquire_slot_asset` selection/reuse-fallback rules normal slot filling already uses, extended with an `exclude_ids` parameter so a QC-rejected id is never immediately reselected or reused as a fallback (only chosen again as an absolute last resort if literally nothing else was ever downloaded). This is capped at `max_replacement_attempts` (default 2, configurable) per slot - never an unbounded search - and if no acceptable replacement is found, the best-available asset is kept and explicitly flagged (`decision="weak"`/`"rejected"`, `replaced`/`replacement_attempts` recorded) rather than the pipeline stalling.

## VisualQCService reconstructs VisualMediaService's global dedup state from an already-finished VisualResult

Because Visual QC runs as a separate, standalone step after `generate_visuals()` has already returned, it doesn't have access to the live `downloaded_by_id`/`used_ids_in_order` maps `VisualMediaService` used internally during selection. `VisualQCService._reconstruct_global_state` rebuilds an equivalent state by walking the given `VisualResult`'s existing assets, so a QC-driven replacement still respects global duplicate prevention across the whole video, not just within one section.

## VisualMediaService.build_plan() is public so Visual QC never triggers a second LLM planning call

`VisualMediaService.generate_visuals()` now accepts an optional `visual_plan=` parameter; when omitted it still builds one internally exactly as before. The demo/QC flow calls the now-public `build_plan(script)` once, passes that same plan into `generate_visuals(visual_plan=plan)`, and later into `VisualQCService.run_qc(..., visual_plan=plan, ...)` - one planning call serves both selection and QC, rather than QC re-deriving (and potentially drifting from) a second plan.

## Repetition checking is asset-ID/position based, not frame-level computer vision

`src/services/repetition_check.py` flags an asset id that reappears within `RECENT_REUSE_LOOKBACK` slots of its own previous occurrence (the same policy constant `VisualMediaService` already uses for reuse-after-gap), by comparing ids and positions in the final selected sequence - not by comparing frame pixels. This is deliberately simple for this milestone: sufficient to catch the obvious case (the same stock clip showing up twice in a row or in quick succession) without building real computer-vision duplicate detection.

## Explicit vision/metadata-fallback/error distinction - never a silent vision-approved default

Every `AssetQCResult` records an `evaluation_source`: `"vision"` (a real evaluator verdict was used), `"metadata_fallback"` (the vision evaluator failed for that whole section - the asset is kept on the upstream metadata filter's prior approval, which already ran during selection), or `"error"` (the evaluator's response didn't include a verdict for this specific asset). A section-level evaluator failure never silently becomes "vision approved" - it's marked `metadata_fallback` and surfaced on `VisualQCResult.fallback_used`/`fallback_reason`, so a caller can always tell whether an asset was actually vision-checked.

## Standalone-first, then integration: Visual QC was validated alone before touching the pipeline

The prior milestone intentionally stopped short of integration - `src/visual_qc_demo.py` (mock script → real visual planning/media → real Visual QC) validated the capability against real downloaded media and a real vision model before the orchestration itself was touched. This milestone wires the already-validated `VisualQCService` into `src/workflows/pipeline_graph.py` as a `visual_qc` node between `media` and `video_assembly` - no QC evaluation/replacement/frame-sampling logic was reworked or duplicated to do this, only the graph wiring.

## Visual QC integration reuses VisualQCService as-is; the pipeline only adds routing policy

`build_pipeline_graph`'s new `visual_qc_node` calls `VisualQCService.run_qc(...)` exactly as `visual_qc_demo.py` already did - same method, same bounded-replacement mechanism, same fallback behavior. The only new logic added at the pipeline layer is deciding what to do with the result (continue vs. stop before Video Assembly) - see the hard-QC-failure decision below - keeping `VisualQCService` itself unaware of being called from a graph versus standalone, consistent with how `VoiceService`/`VisualMediaService`/`VideoAssemblyService` already join the pipeline unchanged in kind.

## Visual Media builds the plan once and threads it through state for Visual QC to reuse

`VisualMediaService.generate_visuals()`'s existing `visual_plan=` parameter (added specifically for this purpose in the standalone-QC milestone) is now used by the pipeline's `media_node`: it calls the now-public `build_plan(script)` once, stores the result on `PipelineState.visual_plan`, and passes it into `generate_visuals(visual_plan=plan)`. `visual_qc_node` reads `state.visual_plan` back out for `run_qc(...)`. One Gemini planning call serves both selection and QC for the whole pipeline run - Visual QC never triggers a second one.

## The pre-QC VisualResult is never silently overwritten; a separate field carries the QC-approved version

`PipelineState.visual_result` continues to mean exactly what it always has - `VisualMediaService`'s direct output, untouched by QC. A new `PipelineState.qc_approved_visual_result` field carries the post-QC version (with any replaced assets swapped in), populated by `visual_qc_node` once QC completes acceptably. `video_assembly_node` was changed to consume `qc_approved_visual_result`, never `visual_result` - so Video Assembly always builds the final MP4 from QC-approved media, while the original selection remains inspectable in state for diagnostics/comparison.

## Hard QC failure is defined at the pipeline level as "a misleading asset survived bounded replacement"

`VisualQCService.run_qc` itself reports `success=True` whenever the QC process completes without crashing, regardless of individual verdicts - by design, it always keeps *something* per slot (best-available-asset philosophy, matching `VisualMediaService`'s own selection fallback). The pipeline needed its own policy for "is this result actually acceptable to build a video from," so `visual_qc_node` treats `VisualQCResult.rejected_count > 0` (at least one asset still flagged `misleading_or_conflicting` after every bounded replacement attempt failed to find something safe) as a hard failure: pipeline status becomes `failed`, Video Assembly never runs, and every earlier-stage result is preserved in state rather than discarded. A `weak`-but-not-misleading asset (low score, no safe replacement found) does *not* block the pipeline - only content actively flagged misleading does, consistent with "prioritize avoiding misleading visuals over demanding literal footage for every abstract concept." This policy lives in the workflow node, not inside `VisualQCService`, since it's specifically about what the *pipeline* considers acceptable to publish, not a property of QC evaluation itself.

## Metadata-fallback approval is allowed to continue the pipeline

When the vision evaluator fails for a section (e.g. a Gemini outage), `VisualQCService` already falls back to approving that section's assets on the strength of the upstream metadata filter that ran during selection (`evaluation_source="metadata_fallback"`) - never silently marking them "vision approved." Since a metadata-fallback decision is never `rejected`, it doesn't trip the hard-failure policy above: the pipeline continues to Video Assembly, and `VisualQCResult.fallback_used`/`fallback_reason` (already part of the existing model) is preserved in `PipelineState.visual_qc_result` for inspection. This matches the pipeline's established pattern of degrading gracefully on a provider outage (see `VisualContextPlanner`'s own fallback) rather than treating every third-party failure as fatal.

## Pipeline demo progress stays at 6 top-level stages; Visual Context Planner is not a separate one

`src/pipeline_demo.py`'s stage labels became `[1/6] Research` … `[6/6] Video Assembly`, adding only `visual_qc` as a new top-level stage. The Visual Context Planner runs inside `media_node` (it's how `VisualMediaService` gets its plan) and was not given its own progress label - from outside the pipeline it isn't a separately observable step, and giving it one would suggest a 7-stage pipeline where there are really 6 orchestrated stages.

## Known limitations

- Semantic relevance still depends on the quality of the LLM's contextual understanding (when planning/QC succeed) or deterministic fallback logic (when they don't), combined with what stock footage Pexels actually has for a given query - perfect semantic matching is not guaranteed given finite provider inventory, even with Visual QC now integrated.
- A QC-driven replacement attempt costs one additional vision call per attempt (bounded by `max_replacement_attempts`, not free).
- Frame sampling inspects a small, fixed number of representative timestamps (not full scene detection) - a clip that changes content between sampled frames could still be judged on an unrepresentative moment.
- Repetition checking remains asset-ID/position based, not frame-level computer-vision duplicate detection - controlled visual reuse may still occur in longer videos.
- For long videos where Pexels lacks enough unique matching stock footage, controlled asset reuse (and, as a last resort, looping) remains an accepted fallback rather than a hard failure.
- The visual pipeline (planning → selection → QC → assembly) is considered complete for the current MVP; it is not planned to be further over-optimized without a new concrete problem to justify it.

## Captions are timed from the real narration audio, never estimated from script section durations

Script sections carry a deterministic `estimated_duration_seconds` (word-count-based), but that estimate is a planning input for visual slot counts - it is not accurate enough for subtitle sync, where even small drift is visibly wrong. `CaptionService` instead transcribes the actual generated narration MP3 (`VoiceResult.audio_file_path`) with real timestamps, so every caption's start/end reflects what was actually spoken and when, not a word-count projection.

## Caption Service is a new standalone service, consuming VoiceResult/VideoAssemblyResult - not folded into Voice or Video Assembly

Captioning is a distinct responsibility from narration synthesis (`VoiceService`) or visual assembly (`VideoAssemblyService`): it only needs their already-produced outputs (the narration audio file and the assembled MP4), not their internals. `CaptionService` (`src/services/caption_service.py`) was added as its own service consuming `VoiceResult`/`VideoAssemblyResult`, mirroring how `VisualQCService` consumes earlier stages' outputs without those stages gaining new logic. `VoiceService` and `VideoAssemblyService` were not modified.

## faster-whisper chosen for local transcription: no PyTorch, no paid API

The reference `openai-whisper` package pulls in PyTorch - a multi-GB install for what is otherwise a small local inference task. `faster-whisper` (CTranslate2-backed) does the same Whisper-model transcription with no PyTorch dependency (~90MB installed), supports CPU inference with int8 quantization out of the box, is actively maintained, and provides word-level timestamps. Model weights download once from Hugging Face and are cached locally - fully free, no per-request cost, consistent with "no paid API dependency for this MVP if local transcription works." The `base` model size is the default (configurable via `WHISPER_MODEL_SIZE`): narration audio is clean, single-speaker TTS output, not noisy real-world audio, so a larger model was judged unnecessary for this MVP.

## TranscriptionProvider is a new abstraction, synchronous like LLMProvider - not folded into VoiceProvider

Transcription is local CPU-bound inference, not network I/O, so `TranscriptionProvider.transcribe(audio_path) -> List[TranscribedSegment]` (`src/tools/transcription_provider.py`) is a plain synchronous method - mirroring `LLMProvider.generate_text`, not the async `VoiceProvider`/`MediaProvider` interfaces that wrap real async I/O. `CaptionService.generate_captions` itself stays async (matching every other service in this project, for eventual pipeline consistency) and simply calls the synchronous `transcribe()` as one blocking step, the same way `VisualContextPlanner.plan_visuals` (also synchronous) is already called from async pipeline code.

## Caption readability segmentation is deterministic and separate from SRT formatting

`src/services/caption_segmentation.py` turns raw transcribed segments into screen-ready captions using fixed rules only - no LLM call: split at sentence boundaries (or by word-level timestamps when available) when a segment exceeds `MAX_CHARS_PER_SEGMENT`, greedy word-wrap without ever breaking a word, minimum/maximum/reading-speed-driven duration bounds, and monotonic non-overlapping timestamps clamped to the real narration/video duration within `DURATION_TOLERANCE_SECONDS`. All thresholds are named module-level constants, not scattered magic numbers. SRT formatting itself (`src/services/srt_writer.py`) is a separate, even simpler module - it only knows how to render already-finalized `CaptionSegment`s as valid SRT text (always renumbering sequentially from list order, never trusting a stored index), so a future alternate subtitle format could be added without touching segmentation logic.

## Word-level timestamps are preserved in the data model but not used for word-by-word captions yet

`TranscribedWord`/`TranscribedSegment.words` carry word-level timing from Whisper when available, and caption segmentation uses them internally (to split a long segment at accurate word boundaries rather than only proportionally by character count). Karaoke/word-by-word caption rendering was explicitly out of scope for this milestone - the data is preserved for a possible future enhancement, but `CaptionSegment` (the rendered/burned unit) is always phrase/sentence-level.

## Line-length limit takes priority over the 1-2 line-count target in a rare edge case

Caption text is wrapped to ~2 lines of ≤42 characters via greedy word-wrapping. For the overwhelming majority of real narration (which has normal sentence punctuation), this reliably produces at most 2 lines. In the rare case of a very long run of words with no punctuation and unlucky word-length alignment, wrapping can need a third line; rather than forcibly merging the overflow into one over-long line to preserve a hard 2-line cap, the implementation allows the extra line - never exceeding the per-line character limit (the harder readability requirement: "avoid excessively long lines") takes priority over the softer "approximately 1-2 lines" target.

## Subtitle burning extends the existing VideoAssembler interface, not a second FFmpeg wrapper

Burning (hardcoding) subtitles into a video is FFmpeg work, and the project already has one thin FFmpeg wrapper (`VideoAssembler`/`FFmpegVideoAssembler`) with binary discovery, subprocess handling, and error types - reusing it (a new `burn_subtitles` method, following the same precedent as Visual QC's earlier `extract_frames` addition) avoids a second video-processing stack. `FFmpegVideoAssembler` stays styling-agnostic - it just runs the `subtitles` filter with whatever `force_style` string it's given; `CaptionService` owns the actual default styling (font, size, colors, margins, alignment) as centralized constants, keeping semantic/presentation decisions in the service layer and mechanical FFmpeg invocation in the tool layer.

## Subtitles are always burned into a copy; the original assembled MP4 is never overwritten

`CaptionService` derives the captioned output path as a sibling file (`<name>-captioned.mp4`) next to the original, and `FFmpegVideoAssembler.burn_subtitles` always writes to a new `output_path`, never in place. Every failure path (missing audio/video, transcription failure, empty segments, SRT-write failure, FFmpeg failure) returns a structured `CaptionResult(success=False, error=...)` before any write to the captioned path is attempted, so the original video is provably untouched on both success and failure - verified in tests and in the real validation run (original file's modification time and byte content unchanged after captioning).

## Standalone-first: Caption Service is not yet wired into the main LangGraph pipeline

Consistent with how Visual QC was validated standalone before integration, this milestone stops short of touching `src/workflows/pipeline_graph.py`. `src/caption_demo.py` is a separate standalone runner that auto-discovers the most recently generated narration MP3 and assembled MP4 under `output/audio/`/`output/video/` (skipping already-captioned copies) rather than requiring Research/Script/Voice/Visual Media/Visual QC to run again just to validate captioning. Wiring `CaptionService` in as a stage after Video Assembly is the next planned milestone.

## Caption Service integration reuses CaptionService as-is; the pipeline only adds routing and state-plumbing

Consistent with how Visual QC's integration milestone worked, `build_pipeline_graph`'s new `caption_node` calls `CaptionService.generate_captions(voice_result, video_assembly_result)` exactly as `caption_demo.py` already did - same transcription/segmentation/SRT/burn logic, unchanged. The only new code is a thin node function, a new `PipelineState.caption_result` field, and conditional routing - `CaptionService` and every module beneath it (`transcription_provider`, `caption_segmentation`, `srt_writer`, `FFmpegVideoAssembler.burn_subtitles`) remain unaware of whether they're called standalone or from the graph.

## Captions run after Video Assembly, gated on VideoAssemblyResult.success rather than the status string

`route_after_video_assembly` decides whether to run `captions` by checking `state.video_assembly_result is not None and state.video_assembly_result.success` - the same result-object-based routing style used by `route_after_media`/`route_after_voice`/etc. elsewhere in the graph (as opposed to `route_after_visual_qc`, which checks the special policy-driven `status == "qc_passed"` string). This keeps the new routing consistent with the graph's established convention: check the actual result, not a derived status string, except where a dedicated pipeline-level policy status already exists.

## Pipeline "completed" status moved from Video Assembly to Captions

With captions now the final stage, `video_assembly_node`'s own success status was renamed from `"completed"` to `"assembled"` (an intermediate status, matching the pattern of `"qc_passed"` before it), and only `caption_node`'s success path sets `status="completed"`. This keeps the same principle used when Video Assembly was added as the final stage in an earlier milestone: "completed" should only ever mean the pipeline's actual final deliverable exists - now the captioned MP4, not the intermediate assembled one.

## Caption failure marks the pipeline failed but preserves every earlier successful result, including the original MP4

If transcription or subtitle burning fails, `caption_node` sets `status="failed"` and records the error on a `CaptionResult(success=False, ...)`, but does not clear or overwrite `research_result`/`script_result`/`voice_result`/`visual_result`/`visual_qc_result`/`video_assembly_result` - all remain inspectable in the final `PipelineState`, and critically the original non-captioned MP4 on disk is untouched (captions are always burned into a copy, a property already guaranteed by the standalone `CaptionService`/`FFmpegVideoAssembler.burn_subtitles`, not something the pipeline layer had to re-implement).

## Two-video output (original + captioned MP4) is an intentional, temporary MVP behavior

Each successful run now leaves both `output/video/<name>.mp4` (original, non-captioned) and `output/video/<name>-captioned.mp4` (final, captioned) on disk. This is deliberate for the current MVP: keeping the original around is a useful development/debugging fallback, and removing it is a storage/cleanup concern, not a captioning concern. A future storage/cleanup milestone may delete the intermediate uncaptioned MP4 once captioning (and eventually upload) has succeeded - this integration milestone explicitly left that behavior unchanged rather than adding cleanup logic prematurely.

## Visual pipeline is frozen for the MVP; occasional single-word captions are accepted, not tuned

Manual review of this milestone's real end-to-end run found (a) some run-to-run variation in stock-footage relevance versus a previous run, still within the previously-accepted quality bar, and (b) occasional single-word caption segments caused by Whisper's own segment boundaries (not a `caption_segmentation.py` defect - that module only ever splits long segments, it never merges short adjacent ones). Both were judged acceptable for the current MVP and explicitly left untuned: the visual pipeline (planning → selection → QC → assembly) remains frozen unless a recurring, severe problem appears, and no caption-segmentation merge logic is planned unless a real readability problem is found in practice.

## Background music selection is driven by actual video/narration mood, not random or hardcoded per topic

`MusicContextPlanner` and `MusicSelectionService` exist specifically so BGM choice reflects what the video is actually about and how it feels (calm, energetic, serious, warm, etc.), the same "understand context, don't just match keywords" principle already applied to visual planning (`VisualContextPlanner`). Nothing in the selection path branches on a specific topic string - the mood/energy/genre profile is either inferred by the LLM from the real script/narration content, or a safe generic fallback profile, never a per-topic special case.

## Narration always has priority over background music

Every mixing decision in `AudioMixingService`/`FFmpegVideoAssembler.mix_background_audio` treats the narration track as fixed and the music track as the one shaped around it: a conservative default gain (`DEFAULT_BGM_GAIN_DB = -24.0`), sidechain ducking under speech, and a constructor-level guard rejecting any `bgm_gain_db > 0` (which would amplify music above its source level) so the service can't be misconfigured into overpowering narration.

## Instrumental tracks are preferred for narrated video; vocal tracks excluded by default

`MusicSelectionService.select_track` filters the catalog down to `instrumental=True` tracks before any mood/energy scoring runs - a track with vocals/lyrics would compete with narration for the listener's attention regardless of how well it otherwise matches the mood, so it's excluded by default rather than merely down-ranked.

## Only an approved, curated local BGM catalog is used - never automatic "No Copyright Music" downloads

Consistent with the project's "no paid APIs, nothing without known rights" posture, `MusicCatalogProvider`/`LocalMusicCatalogProvider` never search, scrape, or download music at runtime - they only return tracks a human has already placed in `assets/bgm/tracks/` with explicit source/license metadata in `assets/bgm/catalog.json` (see `assets/bgm/README.md`). The current MVP catalog is 5 tracks manually obtained from the YouTube Audio Library, each marked "Attribution not required".

## MusicCatalogProvider is a new abstraction, extensible to a future licensed provider

Music sourcing is defined behind a `MusicCatalogProvider` interface (`src/tools/music_catalog_provider.py`), with `LocalMusicCatalogProvider` as the free/local MVP implementation and a `MockMusicCatalogProvider` for tests - mirroring every other provider abstraction in this project (`SearchProvider`, `VoiceProvider`, `MediaProvider`, `TranscriptionProvider`). `AudioMixingService`/`MusicSelectionService` depend only on this interface, so a future approved provider (e.g. a licensed Mixkit catalog, or a managed licensed-music API) could be added later without changing either service.

## BGM mixing reuses the existing FFmpeg infrastructure via a new VideoAssembler method, not a second audio stack

`mix_background_audio` was added to `VideoAssembler`/`FFmpegVideoAssembler` (`src/tools/ffmpeg_video_assembler.py`), following the same precedent as `extract_frames` (Visual QC) and `burn_subtitles` (Captions): one thin FFmpeg wrapper, extended additively per milestone, never duplicated. The track is looped or trimmed to the video's exact duration using the same `-stream_loop -1` plus final-duration-capping strategy already used by `build_section_clip`; fades and ducking are built as one FFmpeg `filter_complex` (`volume`, `afade`, `sidechaincompress`, `amix` with `normalize=0` so mixing doesn't quietly attenuate narration).

## The standalone BGM demo reconstructs narration context from the existing .srt transcript instead of re-running Research/Script

No `ScriptResult` is persisted to disk anywhere in this project. An initial version of `src/bgm_demo.py` worked around this by re-running `run_research_workflow`/`run_script_workflow` for the given topic - real validation showed this was slow and unreliable under Gemini free-tier rate limiting, and wasteful: `AudioMixingService` only needs mood-planning-relevant text, and Video Assembly/Captions have already produced exactly that, as the real, timestamped `.srt` transcript `CaptionService` wrote for the same video. `bgm_demo.py` now locates that matching `.srt` file and reconstructs a minimal `ScriptResult` directly from its concatenated caption text (falling back to a topic-only context if no `.srt` exists) - Research/Script/Voice/Visual Media/Visual QC/Video Assembly are never re-run just to validate BGM.

## Gemini mood planning is an optional single-call enhancement, not a hard dependency

`AudioMixingService` only constructs a `MusicContextPlanner` when an `LLMProvider` is supplied, and `MusicContextPlanner.plan_music` never raises - any LLM failure (including the 429/503/timeout rate limiting hit during real validation) is caught internally and returns the same deterministic fallback `MusicPlan` used when no planner is configured at all. This mirrors `VisualContextPlanner`'s established fallback pattern: at most one LLM call per video, and its unavailability degrades mood quality (to a safe neutral/calm/subtle profile) rather than blocking track selection or mixing.

## Standalone-first: Audio Mixing Service is not yet wired into the main LangGraph pipeline

Consistent with how Visual QC and Captions were each validated standalone before integration, this milestone stops short of touching `src/workflows/pipeline_graph.py`. `src/bgm_demo.py` is a separate standalone runner. Wiring `AudioMixingService` in as a stage after Captions is the next planned milestone.

## BGM pipeline integration reuses AudioMixingService as-is; the pipeline only adds routing and a caption-to-video adapter

Consistent with how Visual QC's and Captions' integration milestones worked, `build_pipeline_graph`'s new `bgm_node` calls `AudioMixingService.generate_mix(...)` exactly as `bgm_demo.py` already did - same mood-planning/selection/mixing logic, unchanged. `AudioMixingService` is constructed once inside `build_pipeline_graph` from the same shared `llm_provider` (for `MusicContextPlanner`) and the same shared `assembler` (for `mix_background_audio`) already used elsewhere, plus one new injected `music_catalog_provider` argument - no new provider type, no duplicated business logic. The only genuinely new code is a thin node function, a new `PipelineState.audio_mix_result` field, conditional routing, and a small `_captioned_video_result` adapter (below).

## BGM mixes onto the captioned MP4, adapted from CaptionResult - not the pre-caption assembly

`AudioMixingService.generate_mix` expects a `VideoAssemblyResult`-shaped source video (the same shape the standalone demo already builds for its own captioned-MP4 target). Since the pipeline's real `VideoAssemblyResult` describes the pre-caption assembly, `_captioned_video_result(caption_result)` adapts `CaptionResult.captioned_video_path`/`captioned_duration_seconds` into that shape before calling `generate_mix` - a thin adapter, not new mixing/selection logic. This ensures BGM is always mixed onto what viewers will actually see (the captioned video), matching exactly what real standalone validation already targeted.

## BGM node reuses the pipeline's existing ScriptResult; Research/Script are never re-run for mood planning

Unlike the standalone `bgm_demo.py` (which has no `PipelineState` to read a `ScriptResult` from, and so reconstructs one from an existing `.srt` transcript), the pipeline's `bgm_node` passes `state.script_result` - the exact object Research/Script already produced earlier in the same run - directly into `AudioMixingService.generate_mix`. This was the entire reason the standalone demo's SRT-reconstruction approach was explicitly kept out of the main pipeline: inside a live pipeline run, the real `ScriptResult` is already sitting in state, so reconstructing one from a transcript would be pure redundancy, not a workaround for a real gap.

## Pipeline "completed" status moved from Captions to BGM/Audio Mixing

With BGM now the final stage, `caption_node`'s own success status was renamed from `"completed"` to `"captioned"` (an intermediate status, matching the pattern of `"assembled"`/`"qc_passed"` before it), and only `bgm_node`'s success path sets `status="completed"`. This is the same principle applied each time a new final stage was added: "completed" should only ever mean the pipeline's actual final deliverable - now the captioned-and-BGM-mixed MP4 - exists.

## A semantic mood-planning failure inside BGM does not fail the pipeline; only a real mixing/selection/catalog failure does

`AudioMixingService`/`MusicContextPlanner` already guarantee that an LLM failure (429/503/timeout/malformed response) falls back to a deterministic `MusicPlan` and returns `success=True` from `generate_mix` if selection/mixing then succeed - `bgm_node` doesn't need (and doesn't add) any special-case handling for this at the pipeline level, unlike the hard-QC-failure policy Visual QC needed. `bgm_node` fails the pipeline only when `AudioMixResult.success` is `False`, which `AudioMixingService` reserves for real failures: an empty approved catalog, no eligible/fallback track, a missing/corrupt selected track file, or an FFmpeg mixing failure - never a mood-planning outage alone. This mirrors `VisualContextPlanner`'s established distinction between "the LLM enhancement degraded" and "the stage itself failed."

## BGM failure marks the pipeline failed but preserves every earlier successful result, including the captioned MP4

If catalog/selection/mixing fails, `bgm_node` sets `status="failed"` and records the error on an `AudioMixResult(success=False, ...)`, but does not clear or overwrite `research_result`/`script_result`/`voice_result`/`visual_result`/`visual_qc_result`/`video_assembly_result`/`caption_result` - all remain inspectable in the final `PipelineState`, and the captioned MP4 on disk is untouched (BGM mixing is always written to a new copy, a property already guaranteed by the standalone `AudioMixingService`/`FFmpegVideoAssembler.mix_background_audio`, not something the pipeline layer had to re-implement).

## Real end-to-end validation was deferred, not skipped, for this integration milestone

Real Gemini free-tier instability (429/503/timeouts) observed during the standalone BGM milestone was still present at integration time. Rather than repeatedly retrying a real full pipeline run against an unreliable API - or changing models/providers mid-integration-milestone to work around it - this milestone's validation relied entirely on the automated pytest suite against mocked LLM/catalog/assembler doubles (including dedicated tests for the fallback-continues and hard-failure-fails paths). Real 8-stage end-to-end validation is deliberately the next milestone's job, once LLM/provider reliability is addressed on its own terms.

## Gemini model names must stay centralized, not duplicated per module

Auditing the LLM layer found the same default model name effectively hardcoded in three places (`src/config/settings.py`'s field defaults, `src/llm/gemini.py`'s `DEFAULT_GEMINI_MODEL`/`DEFAULT_FALLBACK_MODEL`, and `src/tools/gemini_visual_relevance_evaluator.py`'s `DEFAULT_VISION_MODEL`) - all effectively dead code in real usage (every real construction path passes an explicit `settings.gemini_model`), but a duplication risk if they ever drifted (as `settings.py`'s defaults already had, being reversed relative to the actual `.env` configuration). `src/llm/gemini.py` is now the single source of truth for the default model pair; `gemini_visual_relevance_evaluator.py` (which already depended on `src.llm.gemini` for `GEMINI_API_BASE`) now imports `DEFAULT_GEMINI_MODEL` from there instead of redeclaring it. `settings.py` was deliberately left as a literal (not an added import) to keep it a dependency-free leaf module - its literal was corrected to match, not eliminated, since introducing a new config→llm import direction wasn't worth it just to remove one duplicate string in an already-low-risk spot.

## A real, isolated health check - not a guess, not the full pipeline - drives model selection

`gemini-3.6-flash` (the prior fallback) had been silently timing out under real load, discovered only indirectly through slow/failing BGM and pipeline validation runs. Rather than swapping in another guessed model name, `src/gemini_health_check.py` was added: it calls the real Gemini `ListModels` endpoint to get the actual set of models the configured API key's project can see (never inventing a name), then sends exactly one minimal, low-token `generateContent` probe per candidate with no retries of its own (retry/backoff is `GeminiLLMProvider`'s concern, not this tool's), reporting success, latency, and an explicit failure category (`429`/`503`/`timeout`/`malformed`/`error`). This is deliberately narrower than `GeminiLLMProvider` itself - a raw availability/latency measurement tool, not a second copy of the retry/fallback logic.

## Fallback model changed only after real evidence; primary left untouched

A real run of the health check against 40 candidate models confirmed `gemini-3.6-flash` timing out (~15s) while the current primary `gemini-3.5-flash-lite` and several other lite-tier models (including `gemini-3.1-flash-lite`) succeeded in ~1.2-1.3s. Since the primary was already healthy, only the fallback was changed - to `gemini-3.1-flash-lite`, a distinct pinned model generation from the primary (not an alias or a minor variant of the same family), reducing the chance that whatever degrades one also takes down the other. No second fallback slot was added to `GeminiLLMProvider`: the brief allows "one or two" fallback models, and one confirmed-healthy fallback already satisfies that without extending the provider's constructor/retry-loop shape for a benefit the evidence didn't yet justify.

## Retry/backoff policy was left unchanged - the root cause was model choice, not retry mechanics

`GeminiLLMProvider`'s existing bounded retry policy (5 attempts/model, exponential backoff with jitter, retrying `{408,429,500,502,503,504}`/timeouts) was already sound in isolation - it was the *fallback model itself* being unhealthy that turned single transient errors into full retry-ladder exhaustions across both models, compounding badly across `ScriptAgent`'s many per-section calls. Fixing the model choice addresses the observed symptom directly; changing attempt counts or backoff timing without evidence they were themselves the problem would have been an unjustified, unrelated change.

## Existing API key reused; no new provider, no manual account/billing step

The Generative Language API key is scoped to the Google Cloud/AI Studio project, not to a specific model - any model name that project can access works through the same key and endpoint shape. Switching `GEMINI_FALLBACK_MODEL` required no key rotation, no billing change, and no new external LLM provider; `.env`/`.env.example` were updated with the new fallback value only, and the key itself was left untouched.

## Metadata generation is genuinely LLM-dependent; there is no deterministic content fallback

Every other LLM-enhanced stage in this project (Visual Context Planner, Visual QC's evaluator, BGM's `MusicContextPlanner`) has a safe deterministic *content* fallback it can fall back to when the LLM is unavailable. A YouTube title/description/tag set cannot be meaningfully synthesized without an LLM - there is no generic, non-fabricated substitute worth returning. `MetadataAgent` therefore treats a total LLM failure as a real failure (`MetadataResult(success=False, error=...)`), never a fake placeholder, while still reusing the existing `LLMProvider` abstraction and configured Gemini primary/fallback chain (no new provider, no hardcoded model) so a transient outage is still retried/failed-over exactly as it is everywhere else.

## Chapter timestamps are deterministic; only chapter labels come from the LLM

Asking an LLM to invent chapter timestamps risks silently-wrong timing with no way to verify it. `MetadataAgent` instead derives chapter start times from `section_timing.calculate_section_durations` - the exact same per-section proportional-timing logic `VideoAssemblyService` and `VisualMediaService` already trust as the real, authoritative section timeline - fed the actual final video duration. The single metadata-generation prompt asks the LLM only for chapter *labels*, explicitly told the timestamps are "ALREADY FIXED," aligned by section index; a missing/empty label falls back to that section's own heading rather than being left blank. If duration is unavailable or there are fewer than 2 real sections (`MIN_SECTIONS_FOR_CHAPTERS`), chapters are marked unavailable with an explicit reason - never fabricated to fill the gap. This was verified directly in real standalone validation: the SRT-reconstructed context (only 1 synthetic section) correctly produced `chapters_available=False`, not a fake single chapter.

## Deterministic validation is real repair logic, not just a schema check

`src/services/metadata_validation.py` doesn't only reject malformed LLM output - it actively repairs what it safely can: trims/length-limits title and description without cutting mid-word, deduplicates tags case-insensitively under a character budget, normalizes hashtags to a capped `#`-prefixed set, and (via `validate_and_clean_chapters`) drops individual chapters with empty labels, non-increasing/duplicate timestamps, or timestamps at/beyond the video's duration - only raising `ChapterValidationError` (chapters omitted, not a whole-result failure) when nothing usable remains or the first chapter isn't at 0:00. This mirrors the project's established "normalize when safe, fail explicitly when not" policy rather than either silently accepting bad output or being needlessly strict about recoverable issues.

## Script-context reconstruction was extracted into a shared module, not duplicated a second time

`src/bgm_demo.py` already had to solve "no `ScriptResult` is ever persisted to disk - reconstruct one from the existing `.srt` transcript instead of re-running Research/Script" for its own standalone validation. `src/metadata_demo.py` needs the exact same capability. Rather than copying that logic into a second demo file, it was extracted into `src/services/script_context_reconstruction.py`, and `bgm_demo.py` was refactored to import from there too - a minimal, deliberately-scoped change (`bgm_demo.py` is a standalone demo, not part of the main LangGraph pipeline, so this doesn't touch pipeline behavior). While moving it, the video-basename-stripping logic was extended to also handle the `-captioned-bgm` suffix chain (not just `-captioned`), since `metadata_demo.py` needs to resolve the `.srt` match starting from the final BGM-mixed MP4, which didn't exist yet when the original helper was written.

## Metadata JSON artifact is designed for a future Upload Agent, not just for this milestone's own inspection

`output/metadata/<video-slug>.json` includes every field a future YouTube Upload Agent would need (title, description, tags, hashtags, chapters, seo_summary, topic, duration) as plain, directly-serializable data - not a dump of internal model state. The filename is keyed to the video's own slug (the same slug the assembled/captioned/mixed MP4 and `.srt` already share) when known, rather than a freshly-slugified title, so a future consumer can trivially find "the metadata for video X" by shared basename.

## Standalone-first: Metadata Agent is not yet wired into the main LangGraph pipeline

Consistent with every other agent/service in this project (Visual QC, Captions, BGM), this milestone stops short of touching `src/workflows/pipeline_graph.py`. `src/metadata_demo.py` is a separate standalone runner. Real chapter generation against an actual multi-section `ScriptResult` (rather than the standalone demo's single reconstructed section) could not be validated in this milestone for the same reason - it requires the real pipeline state, which only exists once `MetadataAgent` is wired in as a stage. Wiring it in after BGM/Audio Mixing is the next planned milestone.

## Metadata pipeline integration reuses MetadataAgent as-is; the pipeline only adds routing and duration plumbing

Consistent with how BGM's integration milestone worked, `build_pipeline_graph`'s new `metadata_node` calls `MetadataAgent.generate_metadata(...)` exactly as `metadata_demo.py` already did - same prompt/parsing/validation/chapter-derivation logic, unchanged. `MetadataAgent` is constructed once inside `build_pipeline_graph` from the same shared `llm_provider` already used everywhere else - no new provider, no new API key path, no hardcoded model. The only genuinely new code is a thin node function, a new `PipelineState.metadata_result` field, and conditional routing.

## The integrated pipeline passes the real ScriptResult directly - the standalone demo's SRT-reconstruction path is deliberately unused here

The standalone `metadata_demo.py` reconstructs a `ScriptResult` from an existing `.srt` transcript because it has no `PipelineState` to read a real one from. Inside the actual pipeline, `state.script_result` - the exact multi-section object Research/Script already produced earlier in the same run - is passed straight through instead. This was the entire point of keeping the SRT-reconstruction helper in a separate, standalone-only module (`src/services/script_context_reconstruction.py`, not imported by `pipeline_graph.py` at all): the integrated path never needs it, and real validation confirmed the difference concretely - the standalone run's single reconstructed section produced `chapters_available=False`, while the integrated run's real 5-section script produced 5 real chapters.

## Pipeline "completed" status moved from BGM to Metadata

With Metadata now the final stage, `bgm_node`'s own success status was renamed from `"completed"` to `"mixed"` (an intermediate status, matching the pattern of `"assembled"`/`"captioned"` before it), and only `metadata_node`'s success path sets `status="completed"`. This is the same principle applied every time a new final stage was added: "completed" should only ever mean the pipeline's actual final deliverable - now the captioned-and-BGM-mixed MP4 *plus* its generated metadata - exists. Real validation confirmed this distinction matters: a metadata-only failure after a successful BGM mix correctly reports `status="failed"`, not `"completed"`, even though the final video itself is fine.

## Chapter timestamps use the final BGM-mixed video's own probed duration, not an earlier stage's estimate

`metadata_node` passes `audio_mix_result.output_duration_seconds` (the real, ffprobe-measured duration of the actual final deliverable) into `MetadataAgent.generate_metadata`, not `video_assembly_result.duration_seconds` or any earlier estimate - chapters must line up with the video the viewer will actually watch, and BGM mixing/looping/trimming can shift the final duration slightly from the pre-BGM assembly. This was validated for real: the final run's chapters landed correctly within the ~151.57s BGM-mixed duration.

## Real integration validation confirmed the standalone milestone's chapter limitation was a demo artifact, not an agent defect

The standalone Metadata milestone's real run produced `chapters_available=False` because its SRT-based reconstruction always yields exactly one synthetic section - `MetadataAgent`'s own `MIN_SECTIONS_FOR_CHAPTERS` guard correctly refused to fabricate a chapter split from that. Real pipeline integration validated the intended path directly: given the actual 5-section `ScriptResult`, the same unmodified `MetadataAgent` produced 5 real chapters, first at `0:00`, strictly increasing, all within the final duration - confirming the deterministic chapter-timing design works end-to-end without requiring any code change to `MetadataAgent` itself.

## Thumbnail generation splits cleanly into semantic planning and deterministic rendering, mirroring the project's established pattern

Just as `VisualContextPlanner`/`MusicContextPlanner`/`MetadataAgent` each keep LLM reasoning in a small, focused component while deterministic services own everything mechanical, thumbnail generation is split into `ThumbnailPlanner` (`src/agents/thumbnail_planner.py`, one LLM call, semantic/creative decisions only - hook text, visual concept, search query, mood, subject, composition, avoid concepts) and deterministic downstream work (`thumbnail_selection.py`, `thumbnail_renderer.py`, `thumbnail_validation.py`). The LLM is never asked for pixel coordinates or file operations - `composition` is a closed 3-value enum (`subject_left`/`subject_right`/`centered`) the renderer translates into fixed layout rules, consistent with "don't let the LLM generate arbitrary pixel coordinates."

## Pillow added as a new dependency; no paid image-generation API

The project had no existing image-processing library, and deterministic local cropping/resizing/text-compositing genuinely needs one - reusing FFmpeg for this (via `drawtext`/scaling filters) would be far clunkier for text measurement, wrapping, and layout than a proper imaging library. Pillow (`Pillow>=10.0`) was added as the minimal, free, purely-local dependency this requires; no paid AI image-generation service was introduced, consistent with the project's "no paid APIs" MVP constraint.

## Existing Pexels MediaProvider reused as-is for thumbnail source images; no second HTTP implementation

`ThumbnailAgent` calls the same `MediaProvider.search(query, prefer_video=False, ...)`/`.download(...)` interface `VisualMediaService` already depends on - `prefer_video=False` routes `PexelsMediaProvider` straight to its existing photo-search path (already `orientation=landscape` filtered server-side). No new Pexels request/response handling was written.

## Thumbnail planning tolerates a total LLM failure gracefully, unlike Metadata

Unlike `MetadataAgent` (where a title/description genuinely cannot be synthesized without an LLM), a safe thumbnail hook CAN be deterministically derived - the video's own topic (or already-generated metadata title) is definitionally accurate for its own video. `ThumbnailPlanner.plan_thumbnail` therefore never raises for an LLM failure; it always returns a usable `ThumbnailPlan` (deterministic fallback: topic/title's own words, uppercased, capped at 6), so a Gemini outage degrades thumbnail quality rather than blocking thumbnail generation entirely - a deliberate difference in fallback philosophy explicitly called for by this milestone's failure-behavior requirements.

## No local fallback image asset exists yet; a Pexels search/selection failure is a clean typed failure

Unlike Gemini's own retry/fallback-model chain, there is no pre-existing local "safe fallback image" asset in this project (nothing comparable to the curated BGM catalog). Rather than inventing one mid-milestone or silently reusing an unrelated image, a Pexels search failure or an empty/all-filtered candidate list returns a clean `ThumbnailResult(success=False, error=...)` - honest failure over a fabricated result, consistent with "do not silently use an unrelated random image."

## Deterministic rendering guarantees no text overflow, regardless of hook length

`thumbnail_renderer.py`'s `_fit_text` shrinks the font in fixed steps until the wrapped hook fits the composition's text box, and as an absolute last-resort safety net, clips to however many lines actually fit at the minimum font size rather than ever letting text visually overflow the 1280x720 canvas. Combined with a local/system font fallback chain (configurable path → common Windows/Linux/Mac bold sans-serif files → Pillow's bundled bitmap font as the ultimate fallback), rendering can never fail purely because of missing fonts or long text - only a genuinely corrupt/unreadable source image does.

## Real validation surfaced an ambiguous-hook failure pattern; a deterministic topic-alignment guard was added without a second LLM call

The first real standalone run produced a factually-grounded but ambiguous hook ("Two Hours Every Night" for "Why do humans dream?") - a viewer couldn't tell what the video was about from the thumbnail text alone. Rather than adding a second LLM call to judge the first one's output, `thumbnail_validation.py` gained a generic lexical-overlap guard (`is_ambiguous_supporting_fact_hook`, built from `looks_like_isolated_statistic` + `has_topical_overlap`): a hook dominated by a bare number/duration/percentage AND sharing no significant word with the video's own topic/title is deterministically replaced with a topic/title-derived hook (`resolve_hook_text`, preferring the metadata title over the raw topic). The word lists involved (English stopwords, number words, duration units) are generic linguistic resources, not topic-specific content - verified by a dedicated test applying the same guard to an unrelated volcano-topic example. `ThumbnailPlanner`'s prompt was also updated with an explicit hook-selection priority policy so the LLM produces a clear hook more often in the first place, with the deterministic guard remaining as the safety net, not the primary mechanism - confirmed by real re-validation, where the updated prompt alone produced a clear hook and the guard did not need to intervene.

## Thumbnail integration follows the exact status-renaming/routing precedent set by Captions/BGM/Metadata integration

Just as `video_assembly_node`'s success status was renamed `assembled` when Captions was wired in, and `caption_node`'s success status was renamed `captioned` when BGM was wired in, and `bgm_node`'s success status was renamed `mixed` when Metadata was wired in, integrating Thumbnail renames `metadata_node`'s success status from `completed` to `metadata_generated` - the pipeline's final `completed` status now belongs exclusively to whichever stage is actually last. A new `route_after_metadata` function gates the `thumbnail` node exactly like every prior `route_after_*` function gates its own next stage: only on the prior stage's real success, never on a partial/failed result. No routing logic was duplicated - one new routing function, mirroring the existing seven.

## Thumbnail Agent is constructed once per pipeline build via dependency injection, not per-request

`build_pipeline_graph` constructs one `ThumbnailAgent` instance (alongside the other eight agent/service instances it already builds) from the same `llm_provider`/`media_provider` arguments already passed in - never a second provider construction, never a per-invocation provider lookup inside the node closure. This mirrors exactly how `MetadataAgent`/`AudioMixingService`/`CaptionService` are each constructed once at graph-build time, keeping `thumbnail_node` itself a thin closure with no provider-selection logic of its own.

## Integrated Thumbnail consumes real in-memory pipeline state directly - no metadata reload, no script reconstruction

`thumbnail_node` passes `state.script_result` and `state.metadata_result.title`/`.seo_summary` straight into `ThumbnailAgent.generate_thumbnail` - the exact same real objects Research/Script/Metadata already produced earlier in the same run. This is the same "real state over reconstruction" precedent `metadata_node` already established for `ScriptResult`: the standalone demo's `script_context_reconstruction.py` (rebuilding a minimal `ScriptResult` from an `.srt` transcript) and `thumbnail_demo.py`'s metadata-JSON-file reader exist solely because standalone demos have no `PipelineState` to read a real result from - the integrated pipeline never needs either fallback, since both real results already exist in memory by the time the `thumbnail` node runs.

## Test media provider extended to write real, decodable images - a test-infrastructure fix, not an application-code change

`MockMediaProvider.download` (used pipeline-wide) writes small placeholder bytes, not a real image - sufficient for every prior stage, since `FakeVideoAssembler` fakes FFmpeg entirely and never inspects file content. Thumbnail rendering is the first pipeline stage to genuinely decode a downloaded file with real Pillow (`Image.open`), so the full-pipeline test fixture needed a media provider whose "downloaded" files actually open as valid images. Rather than changing the shared `MockMediaProvider` (used broadly, and its raw-bytes behavior is deliberate/correct for every non-Thumbnail consumer), a local test-only wrapper (`ThumbnailCapableMediaProvider` in `tests/test_pipeline_workflow.py`) delegates search/call-recording to `MockMediaProvider` unchanged and only overrides `download` to write a real small Pillow-generated JPEG - scoped entirely to test infrastructure, no production code touched.

## Compliance review splits into a deterministic primary layer and one bounded advisory LLM call, mirroring the project's established pattern

Just as `VisualContextPlanner`/`MusicContextPlanner`/`MetadataAgent`/`ThumbnailPlanner` each keep LLM reasoning in a small, focused component while deterministic logic owns everything that can be verified mechanically, compliance review splits into deterministic checks (`src/services/compliance_checks.py` - the PRIMARY safety layer: artifact presence, topic/content consistency, visual/thumbnail/BGM provenance, BGM catalog membership and attribution) and one bounded advisory semantic review (`ComplianceReviewer`, `src/agents/compliance_reviewer.py` - SECONDARY, at most one LLM call, reusing the existing shared `LLMProvider`/Gemini configuration). The LLM is never asked "is this copyright safe?" - only to identify specific, observable, describable risks (title/description/thumbnail vs. actual content mismatch, unsupported claims) - and is explicitly told it is not a legal authority.

## Centralized publish-decision rules are the single place PASS/REVIEW/BLOCK policy lives

`src/services/compliance_rules.py`'s `decide_publish_status` is the ONLY function that turns checks + a semantic review into a final decision - never scattered across individual check functions or the agent's orchestration. The priority order is fixed and tested: any deterministic blocker always forces `BLOCK` (even over a clean semantic review - a blocker is never overridden by a positive LLM result); an unperformed/failed semantic review (429/503/timeout/malformed response/no LLM provider configured) always forces `REVIEW`, never a fabricated `PASS`; a semantic finding alone (however severe) is advisory-only and can only produce `REVIEW`, never escalate to `BLOCK` on its own. `PASS`'s meaning is fixed as a constant (`DISCLAIMER` in `src/models/compliance.py`) reused verbatim in both the model and the printed report, so the "not a legal or copyright guarantee" wording can never drift between code paths.

## Visual/thumbnail/BGM provenance is read exclusively from a persisted ProvenanceManifest, never from raw VisualResult/AudioMixResult

`VisualResult`/`AudioMixResult` are only ever held in-memory `PipelineState` (same as `ScriptResult` - never persisted to disk anywhere in this project), so they are never actually available to a standalone caller. Rather than keep dead, always-`None` parameters on `ComplianceAgent`, its provenance-reading checks were redesigned around a single `ProvenanceManifest` input (`src/models/provenance.py`) - the real, persisted substitute. `ComplianceAgent` itself does not discover the manifest (that stays the caller's job, e.g. `src/compliance_demo.py` via `ProvenanceManifestStore.find_for_video`) - keeping collection/persistence/compliance-reading cleanly separated, exactly as the hardening milestone required.

## Provenance collection, persistence, and reading are three separate modules by design

`src/services/provenance_collection.py` (`build_manifest_from_pipeline_state`) only ever reads an already-completed `PipelineState` and builds a manifest object - it never touches the filesystem and never runs for a failed/partial run (no fabricated/partial manifests). `src/services/provenance_store.py` (`ProvenanceManifestStore`) is the ONE place that reads or writes manifest files on disk - no individual agent writes its own provenance file - with atomic writes (temp file in the same directory, then `os.replace`) so an interrupted process can never leave a partially-written/corrupt manifest behind; a reader either sees the complete previous file or the complete new one. `ComplianceAgent`/`compliance_checks.py` only ever read an already-built manifest object, never the filesystem directly. This three-way separation was an explicit architectural requirement, not an incidental refactor.

## Provenance manifests are named after the run's own video slug, not the topic or title

`output/provenance/<run_id>.json`, where `run_id = original_base_name(final_video_path)` - reusing the exact same helper (`script_context_reconstruction.original_base_name`) CaptionService/AudioMixingService/MetadataAgent already use to recover a video's base slug from any of its `-captioned`/`-captioned-bgm` variants, rather than inventing a second slug scheme. Since `VideoAssemblyService` embeds a fresh random 8-hex suffix per run, this guarantees two runs of the same topic never collide on the same manifest - verified directly by a dedicated test running two same-topic manifests through the store and confirming each resolves independently.

## Persisted provenance fields are deliberately narrow, and the schema itself is tested to exclude secrets

The manifest never serializes entire internal result objects (`VisualResult`/`AudioMixResult`/`ThumbnailResult`) - only the specific fields a compliance review actually needs (provider, asset ID, source URL, attribution, local path, section association for visuals; the same shape for the thumbnail's source asset; track ID/title/source/license/attribution for BGM). No API keys, tokens, prompts, or other internal LLM data are ever included - enforced not just by convention but by a test that scans `ProvenanceManifest`'s own field names for suspicious substrings (`key`, `token`, `secret`, `prompt`, `credential`), so a future field addition that violates this is caught structurally, not just by review discipline.

## The BGM catalog remains authoritative even when a persisted manifest disagrees with it

A `ProvenanceManifest`'s BGM record is a snapshot of the catalog at run time - it can drift from `assets/bgm/catalog.json` if the catalog is edited afterward (e.g. a track's attribution requirement changes). `ComplianceAgent` always calls `music_catalog_provider.list_tracks()` fresh and treats the CURRENT catalog as the source of truth for `attribution_required`/`attribution_text`, never the manifest's own copy - but a disagreement between the two is still surfaced as an explicit warning (pushing the decision to `REVIEW`) rather than either silently trusting the stale manifest or silently overriding it without comment. This mirrors the same "current catalog wins, but say so" pattern from the standalone Compliance milestone's original `check_bgm_provenance` design, extended to also compare against the manifest snapshot.

## A legacy artifact with no provenance manifest is REVIEW, never inferred or fabricated PASS

Runs from before this milestone (or any run whose manifest write failed) have no `ProvenanceManifest`. `ComplianceAgent` never attempts to infer provenance from filenames or any other heuristic for these - `check_visual_provenance`/`check_thumbnail_provenance`/`check_bgm_provenance` all report `unavailable` (a warning, contributing to `REVIEW`) exactly the same way they do for a genuinely missing manifest, with no special-cased "legacy" logic. This was verified for real: the first standalone compliance run this milestone (against a video that predated provenance persistence) correctly returned `REVIEW` with all three provenance checks `unavailable`; a fresh pipeline run afterward produced a real manifest, and re-running compliance against that video returned `PASS`.

## A pre-existing pipeline wiring gap (not application logic) was the real root cause of test-output pollution

`MetadataAgent` already supported an injectable `output_dir` constructor argument since its original implementation - the bug was entirely in `pipeline_graph.py`/`pipeline_demo.py` never threading a `metadata_output_dir` parameter through to it (every other stage's output directory - voice/media/video/subtitle/thumbnail - already was parameterized). This meant every `pytest` run of the full pipeline test suite wrote real, fixture-generated metadata JSON files into the actual `output/metadata/` directory. The fix was narrow: add `metadata_output_dir`/`provenance_output_dir` parameters to `build_pipeline_graph`/`run_pipeline` (both appended at the end of their signatures so no existing positional call site broke), and redirect them to `tmp_path` in the test fixture - no change to `MetadataAgent` itself, and production defaults are unchanged.

## Provenance persistence was added to the pipeline as pure post-run persistence, not a new pipeline stage

`run_pipeline`/`run_pipeline_demo` call `persist_provenance_if_completed` exactly once, after the LangGraph run finishes, only when `status == "completed"` - never a new graph node, never a routing change, and never able to retroactively fail an already-completed run (a filesystem error writing the manifest is caught and silently ignored, since it's best-effort supplementary data, not a required publishing artifact). This satisfies the hardening milestone's explicit instruction to persist provenance without doing Compliance Agent pipeline integration.

## Compliance integration follows the exact status-renaming/routing precedent set by every prior integration

Just as `metadata_node`'s success status was renamed `metadata_generated` when Thumbnail was wired in, integrating Compliance renames `thumbnail_node`'s success status from `completed` to `thumbnail_generated` - the pipeline's final `completed` status now belongs exclusively to Compliance, and only when its decision is `PASS`. A new `route_after_thumbnail` function gates the `compliance` node exactly like every prior `route_after_*` function gates its own next stage: only on the prior stage's real success, never on a partial/failed result. Unlike every earlier integration, Compliance's own outcome is itself three-valued (`PASS`/`REVIEW`/`BLOCK`), so `compliance_node` maps that directly to three distinct pipeline statuses (`completed`/`review_required`/`blocked`) rather than the usual binary success/`failed` - a deliberate, minimal extension of the existing status-string convention, not a new one.

## Provenance persistence timing was corrected to run inside the graph, not after it

The standalone Compliance + Provenance Hardening milestone had `persist_provenance_if_completed` run once, after the whole LangGraph finished, gated on the pipeline's overall `status == "completed"`. That guard became actively wrong once Compliance became the pipeline's own final stage: by the time the graph "finished," Compliance would already need to have run - but Compliance itself needs the manifest to already exist. The fix moves persistence into `thumbnail_node` itself, called immediately on its own success, and changes `build_manifest_from_pipeline_state`'s guard from the overall `status` string to checking `audio_mix_result.success` directly - the actual prerequisite for valid visual/BGM provenance, independent of whether any later stage (Thumbnail, Compliance) has run yet. This is a narrower, more correct condition than the one it replaced, not a loosening of the "never fabricate a manifest for a failed run" rule: a manifest still never gets built unless BGM mixing itself genuinely succeeded.

## Compliance discovers the current run's manifest via the store, not by holding it in memory

Even though `thumbnail_node` and `compliance_node` run in the same graph execution and could in principle pass a manifest object directly through `PipelineState`, `compliance_node` instead looks it up via `ProvenanceManifestStore.find_for_video(audio_mix_result.output_path)` - a real disk round-trip, not an in-memory shortcut. This exercises the exact same discovery path a future standalone re-check or Upload Agent would use, and guarantees Compliance can never accidentally review stale in-memory data from a struct that wasn't actually persisted (if the write silently failed, the read correctly returns `None`, and Compliance correctly reports provenance as unavailable rather than trusting an unpersisted object).

## A deterministic blocker overrides a clean semantic review even in the fully-integrated pipeline

The centralized decision rule (`decide_publish_status`) established in the standalone milestone needed no changes for integration - it was proven again for real here without any special-casing: a manifest/catalog attribution disagreement or a BGM track missing from the current catalog still forces `BLOCK` regardless of what the real Gemini semantic review finds, and every earlier artifact (final video, metadata, thumbnail) remains untouched on disk either way. No new decision logic was introduced at the pipeline-integration layer - `compliance_node` only maps the agent's existing decision to a pipeline status string.

## A mid-milestone laptop power interruption required only a recovery audit, not rework

The implementation, tests, and one real end-to-end validation attempt were already complete on disk when a laptop power loss occurred during that first real run. Recovery consisted of: confirming every modified file still parsed as valid Python (`ast.parse`) and matched the expected `git diff --stat`, re-running the full pytest suite to reconfirm 1156/1156, and inspecting output-directory file timestamps to determine exactly how far the interrupted real run had gotten (through Visual Media only - no video/thumbnail/compliance artifacts existed from it) before launching one fresh real validation run. No code was rewritten and no test was re-derived - the crash affected only in-progress process state, never anything already written to disk.

## YouTube Upload & Scheduling Agent built standalone-first, mirroring every prior integration milestone's pattern

Consistent with Visual QC, Captions, BGM, Metadata, Thumbnail, and Compliance all being built and real-validated as standalone components before any pipeline wiring was considered, the Upload/Scheduling Agent was built the same way: `src/youtube_upload_demo.py` is a separate CLI entry point, and `src/workflows/pipeline_graph.py` was not touched. This keeps the already-validated 11-stage content-generation graph unchanged while the publishing side is developed and real-tested in isolation - integration remains a deliberately separate, later decision, not assumed here.

## YouTube access follows the existing provider-abstraction pattern - a YouTubeClient ABC, not a direct googleapiclient dependency

Every external service in this project (search, LLM, voice, media, transcription, music catalog) sits behind a small abstract interface with a real and a mock implementation, so agents/services never depend on a concrete vendor SDK directly. YouTube access follows the same shape: `YouTubeClient` (`src/tools/youtube_client.py`) declares `get_authenticated_channel`/`insert_video`/`set_thumbnail`, with `GoogleYouTubeClient` (real, `googleapiclient`-backed, dependency-injectable `youtube_service` for testability) and `MockYouTubeClient` (in-memory, records calls) as the two implementations. `YouTubeUploadAgent` depends only on the interface.

## OAuth credentials: stored-token-first, refresh-before-reauth, secrets/ only, never logged

`src/tools/youtube_oauth.py`'s `get_credentials` tries, in order: a valid stored token, then a refresh via the stored refresh token if the access token expired, and only falls back to the interactive browser consent flow (`InstalledAppFlow.run_local_server`) if neither works - so a validated token is reused indefinitely across real runs without re-prompting the user. The token is persisted only under the Git-ignored `secrets/` directory (`secrets/youtube_token.json`), written atomically (temp file + `os.replace`, mirroring every other atomic store in this project). Neither the client secret nor the token is ever printed or logged anywhere in the implementation - a deliberate, verified constraint, not an incidental omission.

## The Compliance PASS gate is checked on the typed field directly, never via string/log parsing

`YouTubeUploadAgent.publish()` gates on `artifacts.compliance_result is not None and artifacts.compliance_result.publish_decision == "PASS"` - a direct comparison against the same typed `PublishDecision` enum `compliance_rules.decide_publish_status` already produces, never a parsed terminal message or a human-readable summary string. A missing `ComplianceResult` is treated identically to `REVIEW`/`BLOCK` (upload refused) rather than as a special "unknown, allow it" case. This makes the gate impossible to accidentally satisfy with anything other than a genuine `PASS` object, and keeps the check trivially unit-testable without needing to construct or parse any log output.

## Idempotency mirrors the existing atomic-write-per-run_id store pattern, not a database

`PublishingRecordStore` (`src/services/publishing_record_store.py`) follows the exact same shape as `ProvenanceManifestStore`/`ComplianceRecordStore`: one atomically-written JSON file per `run_id` under `output/publishing/`. `YouTubeUploadAgent.publish()` checks for an existing record before uploading (unless `force=True`) and returns early - reporting the *same* status the original upload would have reported (via a shared `_derive_status` helper), not a hardcoded generic value, so a rerun of an already-scheduled or partially-failed run is never misreported as a fresh, fully-completed upload.

## Scheduling validation and UTC normalization are centralized in one module, never duplicated

`src/services/youtube_upload_validation.py`'s `validate_scheduling` is the single place that enforces YouTube's real API requirement (`privacyStatus=private` while `status.publishAt` is set) plus this project's own safety rules (reject a naive datetime - never silently assume a timezone - and reject a past timestamp). `format_publish_at` is the single RFC3339-UTC formatter, reused identically for the real API request body, the persisted `PublishingRecord.scheduled_publish_at`, and the CLI's `--schedule` dry-run preview - so the value actually sent to YouTube, the value persisted to disk, and the value shown to a human before upload can never drift apart or be computed three different ways.

## The first real upload was deliberately private-only, and no code path can widen it silently

The MVP's first real validation of the whole publishing chain (OAuth → channel verification → compliance gate → resumable upload → thumbnail set → idempotency record) was performed with `privacyStatus=private` as a hard requirement of that validation step, not merely the default - `--execute`'s `--privacy` flag defaults to `private` but must be passed explicitly to choose anything else, and no part of `YouTubeUploadAgent`/`GoogleYouTubeClient` ever changes an already-uploaded video's privacy status after the fact.

## A pre-existing ComplianceResult-persistence gap was worked around honestly for the first real upload, then properly closed later

At the time of this milestone, `ComplianceResult` had never been persisted to disk anywhere in the project (a pre-existing gap, not introduced here) - `ComplianceRecordStore`'s write side existed but nothing called it yet. Rather than fabricating a compliance result or weakening the upload gate to proceed without one, the exact real, already-produced `PASS` result from a genuine prior real pipeline run was recorded to disk verbatim (every field copied from that run's actual output) so the gate could be checked honestly against real data. This was treated as a stopgap, not a fix: the underlying gap (compliance results never persisted automatically) was properly closed later, during the Compliance Remediation checkpoint, which wired `ComplianceRecordStore.write()` directly into `compliance_node` for every PASS/REVIEW/BLOCK decision going forward.

## Do not repeatedly execute expensive full pipelines solely to force a Compliance REVIEW

A real end-to-end validation run of Compliance Remediation (`python -m src.pipeline_demo "Why is the ocean salty?"`) reached a genuine `PASS` on its first evaluation, so the live-provider REVIEW → `script_revision` → re-evaluation branch was not naturally exercised. Deliberately choosing a new topic and re-running the full real 11-stage pipeline (real Gemini, Pexels, Edge TTS, Whisper, FFmpeg) again and again purely in the hope of eventually landing on a REVIEW would burn real API quota/cost/time for a branch that is already comprehensively covered by 43 dedicated mocked tests, without adding meaningfully more confidence than those tests already provide. The remediation branch is accepted based on that comprehensive mocked coverage for the current MVP; a naturally occurring REVIEW during future real operation (a real pipeline run made for an actual video, not manufactured for validation) will serve as this branch's live-path validation when it happens, rather than being artificially forced now.

## Known limitations

- The BGM catalog is a manually curated local library (`assets/bgm/`) - there is no automatic licensed-music-provider integration yet. Populating it is a manual, one-time-per-track MVP step; the final production goal remains zero human intervention, with automated/licensed catalog sourcing deferred to a later milestone.
- Gemini mood planning is a single optional call per video; if the configured primary/fallback models are both unavailable, the deterministic fallback plan (neutral/calm, low energy, ambient/cinematic, neutral/subtle) is used automatically - mood selection is correspondingly generic whenever the LLM call doesn't succeed.
- The main pipeline currently produces three MP4-related outputs per run (original assembled, captioned, and BGM-mixed) plus an `.srt` file, rather than a single final output - intentional for now; a future cleanup milestone may remove the intermediate files once the final mixed MP4 has been used/uploaded successfully.
- The Metadata Agent does not independently fact-check its output - metadata quality/accuracy is only as good as the input `ScriptResult`/narration; an upstream accuracy issue (observed in both the standalone milestone and the real integrated run - a Matthew Walker-related description detail inherited from Research/Script content) passes through rather than being caught. Flagged as a future accuracy/compliance hardening consideration, not addressed in this milestone.
- The Metadata Agent's JSON artifact (`output/metadata/<video-slug>.json`) is now produced by every real pipeline run but has no consumer yet - a future YouTube Upload Agent is expected to read it.
- Thumbnail composition has no true subject-detection - it only controls which side of the frame hosts the text panel/contrast scrim, not literal awareness of where a photo's subject actually is; an accepted MVP simplification.
- The current thumbnail style (real stock photo + text overlay) is intentionally simple for the MVP; more custom/cinematic styles via an AI image-generation provider are a possible future V2 enhancement, not required now.
- The generated thumbnail file (`output/thumbnails/<slug>.jpg`) is now produced by every real pipeline run but has no consumer yet, same as the metadata JSON - a future YouTube Upload Agent is expected to use both.
- Full provenance-aware compliance checking depends on a persisted `ProvenanceManifest` for the run being reviewed; any run from before the provenance-hardening milestone has none and correctly resolves to `REVIEW`, not `PASS` - an honest, by-design limitation, not a defect.
- The semantic compliance review is a single advisory LLM pass, not independent fact-checking - by design it can only ever push toward `REVIEW`, never `BLOCK`, so a well-worded but still-misleading claim could pass through undetected.
- Compliance is now the pipeline's own final gate (`completed`/`review_required`/`blocked`), but nothing outside the pipeline itself yet enforces that gate - a future Upload Agent must explicitly check `compliance_result.publish_decision == "PASS"` before publishing; this is the next planned milestone.
- Whisper transcription accuracy depends on the TTS narration's clarity; it has not been validated against noisy or multi-speaker audio, which this pipeline does not produce.
- The default `base` Whisper model occasionally produces minor punctuation/spacing artifacts and occasional single-word caption segments - both cosmetic, not correctness issues for caption sync or meaning, and not being tuned further at this stage.
- Stock footage semantic relevance can vary run-to-run with live Pexels results; the visual pipeline is considered feature-complete/frozen for the MVP and is not planned for further optimization without a new, recurring, concrete problem.
- The YouTube Upload & Scheduling Agent is standalone only - there is no automated hand-off from a completed pipeline run to an upload/scheduling action; a human (or a future orchestration layer) must still invoke `src/youtube_upload_demo.py` separately.
- Metadata artifact discovery for the Upload Agent is best-effort (matched by recorded `topic`, most-recent-first), not an exact run-id link, since `MetadataAgent`'s JSON filename is not run-scoped the way provenance/compliance/publishing records already are - a pre-existing gap this milestone did not change.
- Real scheduling has only been dry-run-previewed (local→UTC conversion, validation outcome, exact CLI command) - no real future `publishAt` has actually been accepted by the live YouTube API yet in this project.
- The OAuth project backing real uploads is very likely unverified (a standard state for a personal/MVP Google Cloud project), which may force uploads private regardless of the requested privacy status - a non-issue for this MVP since private was already the deliberate target, but worth knowing before ever requesting `public`/`unlisted`.
- `videos.insert` costs ~1600 quota units against a default 10,000-units/day project quota (~6 uploads/day) - a real constraint on how many real end-to-end validation runs can reasonably be performed per day.
