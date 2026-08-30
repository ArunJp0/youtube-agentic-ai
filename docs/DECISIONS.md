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

## Known limitations

- The deterministic fallback plan (used when semantic planning is unavailable) still has no contextual understanding of ambiguous wording - it is a safe degradation path, not a second semantic solution.
- The semantic filter judges a candidate only by lightweight text metadata (alt text or a URL slug); it does not inspect actual video/image frames, so misleading or missing metadata can still let a mismatched candidate through. A future Visual QC step inspecting actual candidate thumbnails/frames (e.g. via a vision model) is a candidate next milestone, not yet implemented.
- For long videos where Pexels lacks enough unique matching stock footage, controlled asset reuse (and, as a last resort, looping) remains an accepted fallback rather than a hard failure.
