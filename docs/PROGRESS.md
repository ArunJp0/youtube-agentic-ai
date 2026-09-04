# Progress

## Completed Milestones

- Project and virtual environment setup completed
- Core dependencies installed
- Modular project structure created
- LLM and search provider abstractions created
- Mock LLM and mock search implemented
- Research Agent implemented
- LangGraph research workflow implemented
- Automated tests created
- All 44 tests passing
- Mock end-to-end Research Agent successfully validated
- Real Wikipedia provider integrated successfully
- Real Gemini provider integrated successfully
- Retry/backoff handling added for transient Gemini errors
- Optional fallback Gemini model support added
- Live Research Agent demo succeeded
- Final structured ResearchResult produced from real sources
- 92/92 tests passing
- Research Agent MVP marked complete
- Script Agent implemented (ResearchResult → structured ScriptResult)
- ScriptAgent reuses the existing LLMProvider abstraction (no direct Gemini coupling)
- LangGraph script workflow implemented, mirroring the research workflow
- Script Agent unit tests added using mock LLM providers (no real Gemini calls)
- Research Agent -> Script Agent demo/runner validated end to end (mock providers)
- 129/129 tests passing
- Script Agent MVP marked complete
- Research Agent MVP complete
- Script Agent MVP complete
- Real Research → Script LangGraph pipeline implemented
- Real Wikipedia search used successfully
- Real Gemini LLM used successfully
- Gemini retry/backoff and fallback model handling validated
- Primary model currently configured as gemini-3.5-flash-lite for reliable MVP testing
- Fallback model configured as gemini-3.6-flash
- Live end-to-end Research → Script pipeline completed successfully
- Current automated test count: 139 passed
- Voice Generation Service implemented as a deterministic service (not an LLM agent)
- VoiceProvider abstraction created, with Mock and free Edge TTS implementations
- VoiceService extracts and orders narration (hook → introduction → sections → conclusion → call_to_action) while excluding metadata (sources, research notes, headings, visual notes)
- Real Edge TTS narration audio generated successfully, saved to the gitignored `output/audio/` directory
- Voice provider selection added via `VOICE_PROVIDER`/`VOICE_NAME` settings, mirroring the LLM/search provider pattern
- Voice Service unit tests added using a mock voice provider and a monkeypatched Edge TTS client (no real network calls)
- Current automated test count: 175 passed
- Voice Generation Service MVP marked complete
- Voice Service MVP completed using Edge TTS
- Real MP3 generation validated locally
- Script narration repetition issue identified and resolved
- Root cause was repetitive ScriptAgent section generation
- ScriptAgent now enforces distinct sections with retry-before-drop validation
- VoiceService retains deduplication as a safety layer
- Final regenerated narration kept 5/5 distinct sections
- Manual audio review confirmed no repeated narration
- 188/188 tests passing
- Research → Script → Voice LangGraph pipeline completed
- Real Wikipedia + Gemini + Edge TTS validated in one run
- VoiceResult stored in pipeline state
- Failure short-circuit behavior tested
- Real MP3 generated successfully
- 190/190 tests passing
- Visual Media Service implemented as a deterministic service (not an LLM agent)
- MediaProvider abstraction created, with Mock and free Pexels implementations
- Deterministic (non-LLM) keyword extraction derives a search query per section from its own heading/narration
- VisualMediaService downloads one asset per section into the gitignored `output/media/` directory, avoiding duplicate assets across sections
- Media provider selection added via `MEDIA_PROVIDER`/`PEXELS_API_KEY` settings, mirroring the LLM/search/voice provider pattern
- Visual Media Service unit tests added using a mock media provider and a mocked Pexels HTTP client (no real network calls)
- Media demo/runner validated end to end with mock media (real Pexels run pending a `PEXELS_API_KEY`)
- 248/248 tests passing
- Visual Media Service MVP marked complete
- Real Pexels API retrieval validated with a live `PEXELS_API_KEY`
- 5 section-specific videos downloaded successfully in one real run
- Deterministic visual-query generation improved for semantic relevance (concept mapping + relevance-ranked keyword extraction + specific/broader/topic/last-resort fallback chain)
- Manual review confirmed acceptable media relevance for all 5 sections, including the previously-irrelevant "prefrontal cortex" section (now returns an on-topic brain visual)
- 261/261 tests passing
- Visual Media Service integrated into the main LangGraph orchestration - it is now a pipeline stage, not just a standalone service
- Current working orchestration: Topic input → Research Agent → Script Agent → Voice Service → Visual Media Service
- Real end-to-end pipeline validated in one run via `python -m src.pipeline_demo "Why do humans dream?"`
- Real pipeline stages completed in that run: Research → Script → Voice → Visual Media
- Real Edge TTS audio generation succeeded within the full pipeline run
- Real Pexels media retrieval/download succeeded within the full pipeline run
- Output locations confirmed: `output/audio/` (narration MP3) and `output/media/` (section videos/images), both Git-ignored
- 263/263 tests passing
- Research → Script → Voice → Visual Media pipeline milestone marked complete
- Video Assembly Service implemented as a standalone deterministic service (not an LLM agent, not yet wired into the main pipeline)
- FFmpeg-backed assembly: real Edge TTS narration and real Pexels section media combined into a final MP4
- Deterministic (non-LLM) section timing: proportional to each section's narration word count, audio track as the authoritative timeline
- Final MP4 manually reviewed and found acceptable: voice is clear, all 5 visual sections appear correctly, no black screens, narration complete, no abnormal/sudden cuts (some stock footage repetition judged acceptable for current MVP)
- Final output format confirmed: MP4 container, H.264 video, AAC audio, 1920x1080, 30 fps, ~81.5 seconds
- ffprobe validation confirmed video/audio duration alignment
- Standalone video demo (`python -m src.video_demo`) validated end to end with real providers
- 304/304 tests passing
- Video Assembly Service (standalone) milestone marked complete
- VideoAssemblyService integrated into the main LangGraph orchestration - it is now the final pipeline stage, not just a standalone service
- Current working orchestration: Topic → Research Agent → Script Agent → Voice Service → Visual Media Service → Video Assembly Service → Final MP4
- The complete pipeline can now produce a final MP4 from a topic string in one run
- Real end-to-end orchestration validated in one run via `python -m src.pipeline_demo "Why do humans dream?"`: real Wikipedia research/search, real Gemini LLM, real Edge TTS, real Pexels media retrieval, real FFmpeg video assembly
- Research, Script, Voice, Visual Media, and Video Assembly all completed successfully in that run; Script produced 5 distinct sections; Visual Media used section-specific Pexels assets
- Final MP4 manually reviewed and considered acceptable: voice clear, visuals generally match narration, no black screens, narration complete, no abnormal cuts
- Final output confirmed: H.264/AAC MP4, 1920x1080, 30 fps, ~163.8 seconds in the full-orchestration run
- ffprobe validation succeeded
- Output locations confirmed: `output/audio/`, `output/media/`, and `output/video/` are all generated runtime artifacts and remain Git-ignored
- Known visual-quality limitation identified (not a pipeline failure): the current implementation uses a limited/fixed number of stock clips per script section, so longer videos may visibly repeat stock footage
- 307/307 tests passing
- Full Research → Script → Voice → Visual Media → Video Assembly orchestration milestone marked complete
- Duration-aware multi-clip visual planning and assembly implemented: visual slot count is now derived from each section's actual share of the real narration duration (via `VoiceResult.duration_seconds`), using a tiered cadence policy (~6-10s/clip under 5 min, ~8-12s/clip 5-10 min, ~10-15s/clip over 10 min) instead of a fixed clip count per section/video
- `VisualMediaService` now selects multiple ordered assets per section (`SectionMediaMapping.assets`, a list) when a section's duration calls for it, instead of exactly one
- Deterministic query-variant generation extended so different visual slots within the same section search for different (but still relevant) facets of that section's content - still no LLM call per slot, and still fully generic across topics (verified against non-dream domains: electric vehicles, space, volcanoes, ancient history)
- Global duplicate prevention added: selected Pexels asset IDs are tracked across the whole video; a section prefers a never-used asset first, then a broader-query match, then reuse of an asset not used in the last few slots, with immediate back-to-back reuse only as an absolute last resort
- Media download efficiency preserved/improved: only the asset actually selected for a slot is downloaded (candidates are inspected via search metadata only); a reused asset is never re-downloaded
- `VideoAssemblyService` updated to build and concatenate multiple ordered clips per section (each sized to an even share of that section's planned duration) instead of exactly one clip per section; no FFmpeg wrapper/interface changes were needed
- `VoiceService` was not modified - `VoiceResult.duration_seconds` is consumed as an upstream timing input only
- New shared `src/services/section_timing.py` module extracted so `VisualMediaService` and `VideoAssemblyService` derive per-section timing from one source of truth
- 343/343 tests passing (covering cadence-tier slot calculation, multi-asset sections, global duplicate prevention/reuse-after-gap/last-resort reuse, download-efficiency, and cross-topic query genericity)
- Real end-to-end pipeline validated in one run via `python -m src.pipeline_demo "Why do humans dream?"`: ~143s narration, 5 script sections, 18 total visual slots planned, 18 unique real Pexels assets selected and downloaded, 0 asset reuse needed in this run (enough unique stock footage was available), final MP4 assembled successfully with no black screens or broken cuts, narration completed correctly
- Manual review of the real run: visual repetition is substantially improved (clips now change throughout the video, no obvious looping observed) and overall visual quality is acceptable; some individual stock clips are only loosely related to the narration - a semantic-relevance limitation, not a pipeline failure (see Known Limitations below)
- Duration-Aware Multi-Clip Visual Planning and Assembly milestone marked **COMPLETE**
- Context-Aware Visual Planning and Semantic Media Filtering implemented: a new `VisualContextPlanner` reads the whole `ScriptResult` (topic, title, every section's heading/narration, in order) and produces a structured per-section `VisualPlan` (semantic summary, visual intents, search queries, avoid concepts, neutral fallback queries) in a single Gemini call per script - not per section, not per visual slot
- The planner reuses the existing `LLMProvider` abstraction already shared by Research/Script - no new provider or setting was added
- `VisualMediaService` now consumes the plan's search queries/avoid concepts/neutral fallbacks per slot instead of interpreting isolated keywords out of context, and tags each selected asset with a diagnostic `relevance_tier` (`high`/`neutral`/`reused`) and `relevance_score`
- A new deterministic semantic filtering layer (`src/services/semantic_visual_filter.py`) rejects any candidate whose available metadata (a Pexels photo's "alt" text, or a descriptive page-URL slug) clearly matches one of the plan's avoid concepts - simple keyword-overlap logic, not a classifier
- If the semantic-planning LLM call fails or returns an unusable response, `VisualContextPlanner` falls back internally to the exact same deterministic query-generation logic used when no planner is configured at all (now centralized in `src/services/query_generation.py`), and the result records that a fallback occurred (`VisualResult.semantic_planning_used`/`semantic_planning_fallback_reason`) - a planner outage degrades gracefully rather than breaking the pipeline
- Duration-aware multi-clip slot planning, global duplicate prevention, and download-only-what's-selected efficiency (all from the prior milestone) are unchanged by this work
- `VideoAssemblyService`/FFmpeg were not modified - they remain responsible only for timing, trimming, cropping, ordering, concatenation, and audio sync, never semantic decisions
- 408/408 tests passing (mocked/faked `VisualContextPlanner` and LLM providers throughout - no real Gemini calls in the automated suite), covering structured-plan parsing/validation, one-call-per-script enforcement, contextual-ambiguity plumbing (including a non-dream example), avoid-concept rejection, neutral-fallback selection, safe LLM-failure fallback, and all prior duration-aware/duplicate-prevention behavior continuing to pass unchanged
- Real end-to-end pipeline validated in one run via `python -m src.pipeline_demo "Why do humans dream?"`: real Gemini semantic visual planning succeeded (no fallback triggered) for all 5 script sections, producing 21 total visual slots and 21 unique real Pexels assets with 0 reuse; final MP4 assembled successfully
- Manual review confirmed the previously-reported real failure is fixed: the narration phrase "philosophical discourse features much debate regarding constructs within which humans exist" produced avoid concepts including "building construction sites" and "physical Lego blocks or scaffolding", and no construction-site footage was selected for that section
- Context-Aware Visual Planning and Semantic Media Filtering milestone marked **COMPLETE**
- Standalone Visual QC capability implemented: `VisualQCService` inspects actual representative frames from already-selected/downloaded media (not just search metadata) and judges whether each asset fits its script section's meaning, the overall topic, and the `VisualContextPlanner`'s semantic summary/avoid concepts
- Deterministic frame sampling (`src/services/frame_sampling.py`) picks 1-3 representative timestamps per clip based only on its own duration (never every frame): a 50% midpoint for short clips, 25%/75% for medium ones, 25%/50%/75% for longer ones
- A new `VisualRelevanceEvaluator` abstraction (`src/tools/visual_relevance_evaluator.py`) keeps Visual QC provider-agnostic, with a `MockVisualRelevanceEvaluator` for tests and a real `GeminiVisualRelevanceEvaluator` that reuses Gemini's existing `generateContent` endpoint with image parts added - no new/unsupported API surface, and `LLMProvider`/`GeminiLLMProvider` (used by Research/Script) were left untouched
- One vision request per script section, batching every asset selected for that section's slots - never one call per frame or per slot - keeping vision-model usage cheap regardless of section size
- Centralized relevance thresholds (`APPROVE_SCORE_THRESHOLD=0.70`, `NEUTRAL_SCORE_THRESHOLD=0.50`) turn each raw vision verdict into a decision: approved, neutral/acceptable, weak (replacement recommended), or rejected (misleading, overrides score) - a neutral relevant clip is treated as acceptable, not penalized for being non-literal
- Bounded replacement implemented: a weak/rejected asset triggers up to `max_replacement_attempts` (default 2) requests back through `VisualMediaService.acquire_replacement_asset` (a new public entry point reusing the exact same selection/global-duplicate-prevention rules, extended to never reselect an explicitly QC-rejected id), never an unbounded retry loop
- Lightweight sequence-level repetition checking (`src/services/repetition_check.py`) flags back-to-back or short-interval reuse of the same asset ID by position - not frame-level computer-vision duplicate detection
- Explicit fallback states implemented: `evaluation_source` distinguishes `vision` (real verdict), `metadata_fallback` (vision evaluator failed for that section - kept on the upstream metadata filter's prior approval, never silently marked vision-approved), and `error` (no usable verdict returned)
- `VideoAssemblyService` was not modified - Visual QC logic lives entirely in the new `VisualQCService`, never inside video assembly
- 503/503 tests passing (mocked evaluator/assembler/replacement-provider throughout - no real Gemini or FFmpeg calls in the automated suite), covering QC model validation, frame-sampling timestamp selection, all four relevance-decision tiers, bounded replacement (including exhaustion and rejected-id exclusion), vision-vs-metadata-fallback distinction, repetition detection, section/asset ordering, and non-mutation of the original `VisualResult`
- Real standalone run validated via `python -m src.visual_qc_demo "Why do humans dream?"`: real visual planning + real Pexels media (11 assets across 5 sections), then real Gemini vision QC - 5 vision calls (one per section), 10 assets highly relevant, 1 neutral/acceptable, 0 warnings, 0 rejected, 0 replacements needed, real Gemini vision succeeded for every section (metadata fallback never triggered)
- Standalone Visual QC milestone marked **COMPLETE**
- Visual QC Pipeline Integration implemented: `VisualQCService` is now wired into the main LangGraph pipeline as a `visual_qc` node, running after Visual Media succeeds and before Video Assembly - the current working orchestration is Topic → Research Agent → Script Agent → Voice Service → Visual Context Planner → Duration-Aware Visual Media Service → Visual QC → Video Assembly Service → Final MP4
- `PipelineState` extended with `visual_plan` (the plan Visual Media built, reused by QC so it never triggers a second LLM planning call), `visual_qc_result` (the structured `VisualQCResult`), and `qc_approved_visual_result` (the post-QC media mapping) - the original `visual_result` field is preserved unchanged, never silently overwritten, so both the pre-QC and post-QC media are inspectable in the final state
- Video Assembly now consumes `qc_approved_visual_result` exclusively - the raw, pre-QC `visual_result` never reaches `VideoAssemblyService`
- Hard QC failure policy added at the pipeline level (not inside `VisualQCService`, which is reused unchanged): if any asset is still flagged misleading/conflicting after Visual QC's own bounded replacement is exhausted (`rejected_count > 0`), the pipeline stops before Video Assembly with status `failed`, preserving every earlier stage's results for inspection - Video Assembly never runs on rejected media
- A vision-evaluator outage (metadata-fallback approval) does not fail the pipeline - QC falls back to the upstream metadata filter's prior approval and the pipeline continues to completion, with the fallback explicitly recorded on `VisualQCResult.fallback_used`/`fallback_reason`
- Pipeline demo progress output updated to 6 top-level stages (`[1/6] Research` … `[6/6] Video Assembly`); the Visual Context Planner remains an internal part of the Visual Media stage rather than its own top-level stage, since it isn't separately observable from outside `VisualMediaService`
- 512/512 tests passing (mocked evaluator/assembler throughout - no real Gemini, Pexels, or FFmpeg calls in the automated suite), covering the full 6-stage success path, `VisualQCResult`/`qc_approved_visual_result` stored in final state, Video Assembly receiving only QC-approved media, QC-driven replacement changing what reaches Video Assembly, Visual QC not running after any earlier-stage failure, hard-QC-failure blocking Video Assembly, metadata-fallback approval continuing the pipeline safely, and earlier-stage results preserved after a QC failure
- Real end-to-end run validated in one command (`python -m src.pipeline_demo "Why do humans dream?"`): all 6 stages completed successfully; Visual QC checked 20 assets across 5 sections using real Gemini vision - 18 vision-approved, 2 neutral/acceptable, 0 warnings, 0 final rejected, 3 weak/unverifiable assets successfully replaced via bounded replacement, 8 total vision calls, metadata fallback not needed; Video Assembly consumed the post-QC approved media and produced a final MP4 (167.0s, 1920x1080 @ 30fps)
- Final MP4 manually reviewed: approximately 80-90% of visuals judged semantically relevant, a small amount of visual repetition and 1-2 slightly weak visuals remained (judged acceptable for the current MVP), no black-screen, narration, or assembly issues observed
- (Operational note, not a pipeline issue: a transient Claude Code/API DNS "ENOTFOUND" message appeared in the assistant's own tooling after the run had already completed successfully - it did not affect the completed pipeline run or the generated MP4)
- Visual QC Pipeline Integration milestone marked **COMPLETE**
- The visual pipeline (Visual Context Planner → Duration-Aware Visual Media → Visual QC → Video Assembly) is considered feature-complete for the current MVP; further optimization of visual selection/QC is not planned unless a new concrete problem is identified
- Standalone Subtitle/Caption Service implemented: `CaptionService` transcribes the real narration audio (never estimates timing from script section durations) and burns synchronized, readable subtitles into a copy of the already-assembled MP4
- A new `TranscriptionProvider` abstraction (`src/tools/transcription_provider.py`) keeps captioning provider-agnostic, with a `MockTranscriptionProvider` for tests and a real `WhisperTranscriptionProvider` backed by `faster-whisper` running fully locally (CTranslate2-based, no PyTorch dependency, no paid API, model weights downloaded once and cached) - default model size `base`, configurable via `TRANSCRIPTION_PROVIDER`/`WHISPER_MODEL_SIZE`
- Deterministic (non-LLM) caption segmentation (`src/services/caption_segmentation.py`) turns raw transcribed segments into professional, YouTube-style captions: split at sentence/word boundaries to roughly 1-2 lines of ≤42 characters, timestamps normalized to be monotonic and non-overlapping, minimum/maximum/reading-speed-driven display duration enforced, clamped to the real narration/video duration within a small tolerance
- SRT generation (`src/services/srt_writer.py`) and subtitle burning are kept as separate concerns from transcription/segmentation; subtitle rendering reuses the existing FFmpeg infrastructure via a new `VideoAssembler.burn_subtitles` method (added to the same interface `VideoAssemblyService` and Visual QC already use) - no second video-processing stack
- Captions are burned into a **copy** of the assembled MP4 (`output/video/<name>-captioned.mp4`); the original non-captioned MP4 is never overwritten or modified
- Professional basic subtitle styling applied (readable sans-serif, white text with a black outline/shadow for contrast, bottom-center with a safe margin, no flashy animations) via a centralized default ASS style string, not scattered per-call styling
- Generated `.srt` files are stored under `output/subtitles/`; all caption output artifacts remain Git-ignored under the project's existing `output/` policy
- 618/618 tests passing (mocked transcription provider/assembler throughout - no real Whisper model, network, or FFmpeg process in the automated suite), covering caption model validation, timestamp ordering/overlap resolution, empty-transcription handling, sentence/word-level readable segmentation, punctuation cleaning, SRT formatting/special characters, missing-audio/missing-video/transcription-failure/rendering-failure handling, original-video-preservation, and deterministic segmentation behavior
- Real standalone run validated via `python -m src.caption_demo` against the existing narration MP3 and assembled MP4 from a prior real pipeline run (Research/Script/Voice/Visual Media/Visual QC were **not** re-run): real Whisper (`base`) transcription produced 44 caption segments; captioned MP4 duration (166.968s) matched the original exactly; resolution/fps/codecs (1920x1080 @30fps, h264/aac) unchanged; the original MP4's file modification time confirmed it was untouched
- Manual review confirmed subtitle synchronization with the spoken narration and overall readability were satisfactory
- Standalone Subtitle/Caption Service milestone marked **COMPLETE** - deliberately **not yet wired into the main LangGraph pipeline** (`src/workflows/pipeline_graph.py` is unchanged); `src/caption_demo.py` is a separate standalone runner for this milestone
- Subtitle/Caption Service integrated into the main LangGraph orchestration as a `captions` node running after Video Assembly - the pipeline is now 7 stages: Research → Script → Voice → Visual Media → Visual QC → Video Assembly → Subtitles/Captions
- Only 3 files were changed to wire it in: `src/workflows/pipeline_graph.py`, `src/pipeline_demo.py`, `tests/test_pipeline_workflow.py` - the existing standalone `CaptionService` and every caption sub-module (transcription, segmentation, SRT writing, FFmpeg subtitle burning) were reused exactly as validated in the standalone milestone, with no duplicated logic introduced
- `PipelineState.caption_result` added; pipeline `status` only becomes `completed` once captioning itself succeeds (`video_assembly_node`'s own success status was renamed to `assembled`); a caption failure marks the pipeline `failed` while preserving every earlier stage's successful results, including the original non-captioned MP4
- Captions run only after Video Assembly succeeds; subtitle timing is taken from the real narration transcription (via the reused `CaptionService`/Whisper path), never from estimated script-section durations
- The captioned MP4 is produced as a separate file alongside the original assembled MP4 (two-video output) - this is intentional for the current MVP; a future storage/cleanup milestone may delete the intermediate uncaptioned MP4 once captioning/upload has succeeded, but that change is deliberately out of scope for now
- 630/630 tests passing
- Real end-to-end run validated via `python -m src.pipeline_demo "Why do humans dream?"`: all 7 stages completed with final status `completed` - narration ~154s, Visual Media produced 19 slots, Visual QC approved the full set with 2 weak/unverifiable visuals replaced and 0 final rejected (fallback not needed), Video Assembly produced the original MP4, and Captions produced an SRT file plus a captioned MP4 with matching duration; the original non-captioned MP4 remained untouched
- Manual review of the real run confirmed subtitle accuracy, timing sync, readability, and end-of-video timing were all satisfactory
- Manual review also found some run-to-run variation in stock footage relevance (a few mildly irrelevant visuals compared to a previous run) - judged acceptable for the current MVP; this is expected variation in live Pexels results, not a regression, and the visual pipeline is not being further tuned because of it (see Known Limitations)
- Subtitle / Caption Pipeline Integration milestone marked **COMPLETE**
- Standalone BGM / Audio Mixing Service implemented: given a video topic, its narration context, an existing captioned MP4, and a curated local BGM catalog, the service plans the video's mood, deterministically selects one approved track, and mixes it under the narration to produce a new BGM-mixed MP4
- New typed models `BGMTrack`, `MusicPlan`, `AudioMixResult` (`src/models/music.py`)
- `MusicCatalogProvider` abstraction (`src/tools/music_catalog_provider.py`) with `LocalMusicCatalogProvider` (reads a curated JSON catalog under `assets/bgm/`) and a `MockMusicCatalogProvider` for tests - the system only ever selects from tracks explicitly present in this catalog; nothing is downloaded, scraped, or guessed at runtime
- `MusicContextPlanner` (`src/agents/music_context_planner.py`): at most one Gemini call for the whole video, producing music CHARACTERISTICS (mood, energy, genre, instrumentation, avoid-styles) - never a specific song; on any LLM failure it falls back to the same deterministic `MusicPlan` used when no planner is configured at all (neutral/calm, low energy, ambient/cinematic, neutral/subtle required)
- `MusicSelectionService` (`src/services/music_selection_service.py`): deterministic ranking of only the approved catalog against the music plan - instrumental tracks strongly preferred (vocal tracks excluded by default), `avoid_styles` filtering, mood/genre/energy scoring, deterministic tie-breaking by `track_id`, and an optional configured neutral fallback track
- `AudioMixingService` (`src/services/audio_mixing_service.py`): orchestrates plan → select → mix; narration always stays primary - conservative default BGM gain (-24 dB, with a constructor guard rejecting any positive/amplifying gain) plus sidechain ducking under narration, smooth fade-in/out, and looping/trimming the track to match the video's exact duration; never overwrites the source video, always writes a new `<name>-bgm.mp4` copy
- `VideoAssembler`/`FFmpegVideoAssembler` extended with `mix_background_audio` (same reuse pattern as `extract_frames`/`burn_subtitles`) - no second audio-processing stack
- Local approved BGM catalog scaffolded under `assets/bgm/` (`catalog.json` + `tracks/` + `README.md` documenting the required format and copyright/source policy)
- Standalone demo `src/bgm_demo.py`: reuses the most recently generated captioned MP4, and - since no `ScriptResult` is ever persisted to disk - reconstructs narration context from that same video's own already-generated `.srt` transcript (the real spoken narration text) instead of re-running Research/Script/Voice/Visual Media/Visual QC/Video Assembly just to get context for mood planning
- 709/709 tests passing
- Real standalone validation: 5 real instrumental tracks added to the catalog from the YouTube Audio Library (all marked "Attribution not required"); the demo reused the existing captioned MP4 (`why-do-humans-dream-7c674e5c-captioned.mp4`) and its matching `.srt` transcript with no Research/Script rerun; the single Gemini mood-planning call hit real 429/503/timeout rate limiting, so the deterministic fallback `MusicPlan` activated automatically and the demo completed successfully instead of failing; `MusicSelectionService` selected "Calm Music" (YouTube Audio Library, no attribution required); mixed at -24 dB gain with ducking enabled; source video duration ~154.152s, output duration ~154.133s; the original captioned MP4 was confirmed untouched (modification time and byte content unchanged)
- Manual review confirmed the mix sounds subtle and professional for the current MVP: narration stayed clearly dominant and clear, BGM was mild/subtle under speech and rose slightly during narration gaps (intentional ducking behavior), judged acceptable overall
- Standalone BGM / Audio Mixing Service milestone marked **COMPLETE** - deliberately **not yet wired into the main LangGraph pipeline** (`src/workflows/pipeline_graph.py` is unchanged); `src/bgm_demo.py` is a separate standalone runner for this milestone

## Known Limitations

- Semantic relevance now depends on the quality of the LLM's contextual understanding (when the planner succeeds) or on deterministic keyword/concept mapping (when it falls back) combined with what stock footage Pexels actually has for a given query. The deterministic fallback path still has no contextual understanding of ambiguous wording - it exists only as a safe, previously-validated degradation path, not a semantic solution in its own right.
- Visual QC inspects actual frame content (not just text metadata) and now runs inside the main pipeline, but it samples a small, fixed number of representative frames rather than every frame - a clip that changes content between sampled frames could still be judged on an unrepresentative moment.
- A QC-driven replacement re-evaluation is an additional vision call per attempt (bounded by `max_replacement_attempts`, not free).
- Repetition checking remains asset-ID/position based only, not frame-level computer-vision duplicate detection - controlled visual reuse may still occur.
- Perfect semantic stock-footage matching is not guaranteed: Pexels' inventory for a given query is finite, so even with duration-aware planning, semantic filtering, and vision QC, an occasional visual can still be only loosely related to its section - real review found this acceptable (~80-90% relevance) for the current MVP, not eliminated.
- For long videos where Pexels lacks enough unique matching stock footage, controlled asset reuse (and, as a last resort, looping) remains an accepted fallback rather than a hard failure.
- The main pipeline currently produces **two** MP4 outputs per run (the original assembled MP4 and a separate captioned MP4) rather than one final file - intentional for now; a future storage/cleanup milestone may remove the intermediate uncaptioned MP4 once captioning/upload has succeeded.
- Occasional very short, single-word caption segments can occur, driven by Whisper's own segment boundaries rather than `caption_segmentation.py`'s (split-only, never-merge) logic - manual review of the real integrated run found current readability acceptable, so no further caption segmentation tuning is planned unless a real problem is found.
- Stock footage semantic relevance remains good-enough-but-not-perfect and can vary run-to-run with live Pexels results - the visual pipeline (planning → selection → QC → assembly) is considered frozen for the current MVP and will not be further tuned unless a recurring, severe relevance problem appears.
- The BGM catalog is a manually curated local library (`assets/bgm/`) - there is no automatic licensed-music-provider integration yet. Populating it is a manual, one-time-per-track MVP step; the final production goal remains zero human intervention, with automated/licensed catalog sourcing deferred to a later milestone.
- Gemini mood planning is a single optional call per video; real validation showed it can be unavailable under Gemini free-tier rate limiting (429/503/timeouts), in which case the deterministic fallback `MusicPlan` (neutral/calm, low energy, ambient/cinematic, neutral/subtle required) is used automatically - mood selection is correspondingly generic whenever the LLM call doesn't succeed.
- The Standalone BGM / Audio Mixing Service is implemented and validated but is **standalone only** - the main pipeline's final output (`python -m src.pipeline_demo`) does not currently include background music.

## Current Next Milestone

**BGM / Audio Mixing Main Pipeline Integration** - wire the already-validated standalone `AudioMixingService` into the main orchestration as a stage after Captions:

```
Topic → Research → Script → Voice → Visual Media → Visual QC → Video Assembly
      → Subtitles / Captions → BGM / Audio Mixing → Final Mixed MP4
```

Planned sequence after that, in order:

1. BGM / Audio Mixing Main Pipeline Integration
2. Metadata Agent
3. Thumbnail Agent
4. Copyright / Compliance checks
5. YouTube Upload + Scheduling
6. Monitoring / Post-publish
7. Topic Planner
8. Final storage/cleanup hardening as appropriate (including the two-video-output cleanup noted above, and replacing manual local BGM catalog curation with an automated/licensed provider or managed catalog workflow)
