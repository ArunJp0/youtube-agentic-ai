# YouTube Agentic AI

A modular monolith that automates the creation of YouTube videos using Agentic AI techniques.

## Mission

Build an automated system that can generate YouTube videos with minimal human intervention, focusing on educational and interesting knowledge content (YouTube Shorts and long-form).

## Architecture

- **Backend**: FastAPI (async HTTP API)
- **Orchestration**: LangGraph (stateful workflows)
- **Database**: PostgreSQL via SQLAlchemy
- **LLM Abstraction**: Provider-agnostic interface supporting multiple LLM backends

## Initial Visual Strategy

- Stock footage and images
- AI-generated graphics where useful
- No dependency on expensive AI video generation for MVP

## Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting/formatting
ruff check .
ruff format .
```

## MVP Scope

The complete orchestration - Research → Script → Voice → Visual Context Planner → Duration-Aware Visual Media → Visual QC → Video Assembly → Subtitles/Captions → BGM/Audio Mixing → Metadata → Thumbnail → Copyright/Compliance - is implemented end-to-end as one LangGraph pipeline. One command turns a topic into a final captioned-and-mixed, playable MP4 plus generated YouTube upload metadata, a matching thumbnail, a persisted provenance manifest, and a pre-publishing compliance decision:

```
Topic Input → Research Agent → Script Agent → Voice Service → Visual Context Planner
            → Duration-Aware Visual Media Service → Visual QC
            → Video Assembly Service → Subtitle/Caption Service
            → BGM / Audio Mixing Service → Metadata Agent → Thumbnail Agent
            → Copyright / Compliance Agent
            → Final Captioned + BGM Mixed MP4 + Metadata JSON + Thumbnail JPEG + Provenance Manifest + Compliance Decision
```

**Validation status**: all 11 stages have been validated end-to-end together with real providers (real Gemini, real Wikipedia, real Edge TTS, real Pexels, real Whisper), including after the Gemini fallback model was changed (`gemini-3.6-flash` → `gemini-3.1-flash-lite`) following a real health check (`python -m src.gemini_health_check`) that found the old fallback timing out under real load - see `docs/DECISIONS.md`.

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Context Planner**: one Gemini call for the entire script, producing a structured visual plan per section - what it's actually about (`semantic_summary`), concrete visual concepts (`visual_intents`), search queries, `avoid_concepts` (literal-but-wrong interpretations to steer away from, e.g. "constructs a narrative" ≠ construction), and neutral fallback queries. If the call fails or returns something unusable, it falls back cleanly to the same deterministic query generation used when no planner runs at all - the pipeline never breaks because of it. Runs as an internal part of the Visual Media stage, not a separate top-level pipeline stage.
- **Visual Media Service**: duration-aware - it derives how many distinct stock clips each section needs from that section's real share of the narration duration (not a fixed count), and consumes the visual plan (rather than isolated keywords) to search, filter, and select multiple ordered assets per section. A deterministic semantic filter rejects candidates whose available metadata matches an `avoid_concept`. Only the assets actually selected are downloaded (real Pexels, with a mock provider for offline dev), saved under `output/media/`. Selected assets are tracked globally across the whole video to avoid duplicates, with exact-asset reuse/looping used only as a controlled fallback when unique stock footage runs out.
- **Visual QC**: inspects real representative frames (not just text metadata) from every selected asset with a vision-capable evaluator (real Gemini vision, with a mock evaluator for offline dev) - one batched request per script section. Assets judged weak or misleading trigger a bounded replacement request back through the Visual Media Service (never an unbounded search); an asset still flagged misleading after replacement is exhausted stops the pipeline before Video Assembly rather than risk a misleading clip in the final video. A vision-provider outage falls back to the upstream metadata filter's prior approval instead of failing the pipeline.
- **Video Assembly Service**: combines the narration audio with each section's one or more QC-approved ordered clips into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`. Makes no semantic decisions - only timing, trimming, cropping, ordering, concatenation, and audio sync.
- **Subtitle/Caption Service**: runs after Video Assembly succeeds. Transcribes the real narration audio locally (Whisper via `faster-whisper` - free, no paid API) to get actual spoken-word timestamps (never estimated from script section durations), builds readable YouTube-style captions, writes an `.srt` file under `output/subtitles/`, and burns captions into a **copy** of the assembled MP4 (`output/video/<name>-captioned.mp4`) - the original MP4 is never overwritten.
- **BGM / Audio Mixing Service**: runs after Captions succeeds, reusing the same `ScriptResult` already produced earlier in the run (Research/Script are never re-run). Plans the video's mood via a single optional Gemini call (falling back automatically to a safe deterministic mood profile if that call is unavailable), deterministically selects one instrumental track from a curated local approved catalog (`assets/bgm/` - YouTube Audio Library tracks marked "Attribution not required" for the current MVP), and mixes it under the narration on the captioned MP4 at a conservative gain with sidechain ducking, so narration stays clearly dominant and the music stays subtle. Writes a new copy (`output/video/<name>-captioned-bgm.mp4`) - the captioned MP4 is never overwritten.
- **Metadata Agent**: runs after BGM/Audio Mixing succeeds, reusing the same `ScriptResult` already produced earlier in the run and the final BGM-mixed MP4's own probed duration (Research/Script are never re-run). One Gemini call generates a professional YouTube title, description, tags, hashtags, and an SEO summary, grounded only in the actual script content - never inventing facts, clickbait, or spam keywords. Chapter timestamps are always derived deterministically from real per-section timing (never asked of the LLM); the LLM supplies only chapter labels, with a missing label falling back to the section's own heading. Writes a structured JSON artifact (`output/metadata/<video-slug>.json`) for a future YouTube Upload Agent to consume.
- **Thumbnail Agent**: runs after Metadata succeeds, reusing the same `ScriptResult` and the real `MetadataResult`'s title/SEO summary already produced earlier in the run (Research/Script/Metadata are never re-run, and nothing is reloaded from the metadata JSON file). One Gemini call (`ThumbnailPlanner`) produces a small semantic plan - a short hook, a visual concept, a Pexels search query, a mood, and a closed-vocabulary `composition` (`subject_left`/`subject_right`/`centered`) - never raw pixel coordinates. The existing Pexels `MediaProvider` finds a source photo; a deterministic Pillow-based renderer crops it to exactly 1280x720 without stretching and composites the wrapped, auto-shrunk hook text with a translucent contrast panel. A deterministic lexical guard (`resolve_hook_text`) catches a hook that reads as an ambiguous isolated statistic disconnected from the video's own topic and replaces it with a topic/title-derived hook, without a second LLM call. If Gemini planning fails outright, a safe deterministic plan is used instead of failing thumbnail generation. Writes the final JPEG to `output/thumbnails/<slug>.jpg`. On its own success, this stage also persists a machine-readable provenance manifest (`output/provenance/<run_id>.json` - exact visual/thumbnail/BGM asset provenance, no secrets or prompts) for the Compliance stage to read next.
- **Copyright / Compliance Agent**: runs after Thumbnail succeeds, reusing the exact `ScriptResult`/`MetadataResult`/`ThumbnailResult` already produced earlier in the run and the real final video path (Research/Script/Metadata/Thumbnail are never re-run). A pre-publishing safety gate, not a legal authority: deterministic checks are the primary safety layer (final video/metadata/thumbnail presence, topic/content consistency, visual/thumbnail/BGM provenance read from the current run's own provenance manifest, and BGM catalog membership/attribution always cross-checked against the current `assets/bgm/catalog.json`); one bounded, advisory Gemini call flags only specific observable risks (e.g. a title/thumbnail that doesn't match the actual content) and is never asked to make a legal or copyright determination. A deterministic blocker always wins over a clean LLM result, and a failed/unavailable semantic review always falls back to `REVIEW`, never a fabricated `PASS`. Returns a `PASS`/`REVIEW`/`BLOCK` decision, stored as `PipelineState.compliance_result` for a future Upload Agent to consume directly (no terminal-log parsing required).

All stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage - including a hard Visual QC failure, a caption/transcription failure, a real BGM catalog/selection/mixing failure, a metadata generation failure, a thumbnail generation failure, or a total compliance-review failure - stops the pipeline before the next one runs (or before completion), preserving every earlier stage's successful results. A BGM mood-planning outage alone does not stop the pipeline - the deterministic fallback plan lets mixing continue. The pipeline's final status is driven entirely by the Compliance decision: `completed` only when Compliance itself returns `PASS`; `review_required` when it returns `REVIEW` (a non-blocking concern needing human/another-system review); `blocked` when it returns `BLOCK` (a known deterministic compliance failure). Neither `review_required` nor `blocked` is ever reported as `completed`, and both preserve every earlier artifact (the final captioned-and-BGM-mixed MP4, metadata JSON, and thumbnail JPEG) untouched. `output/audio/`, `output/media/`, `output/video/`, `output/subtitles/`, `output/metadata/`, `output/thumbnails/`, and `output/provenance/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation (visual)**: perfect semantic stock-footage matching is not guaranteed - Pexels' inventory for a given query is finite, so even with duration-aware planning, semantic filtering, and vision QC, an occasional visual can still be only loosely related to its section, and controlled visual reuse may still occur in longer videos; relevance can also vary somewhat run-to-run with live Pexels results. Manual review of real runs has found this acceptable for the current MVP. The visual pipeline (planning → selection → QC → assembly) is considered feature-complete/frozen for now and is not planned for further optimization without a new, recurring, concrete problem.

**Known limitation (captions)**: occasional very short, single-word captions can occur due to Whisper's own segmentation, since caption segmentation only ever splits long segments and never merges short adjacent ones. Manual review found current readability acceptable, so no further segmentation tuning is planned at this stage.

**Multi-file output**: each run currently produces the original assembled MP4, a captioned MP4, and a final BGM-mixed MP4 (plus the `.srt` transcript) on disk, rather than a single final file. This is intentional for the current MVP - the intermediate files are useful development/debug fallbacks; a future storage/cleanup milestone may delete them once the final mixed MP4 has been used/uploaded successfully.

**Known limitation (BGM)**: the approved BGM catalog (`assets/bgm/`) is a manually curated local library - there is no automatic licensed-music-provider integration yet, so adding tracks is a manual, one-time-per-track MVP step. Gemini mood planning is a single optional call per video; it has been observed to be unavailable under real Gemini free-tier rate limiting (429/503/timeouts), in which case mood selection falls back to a safe, generic deterministic profile rather than failing.

**Known limitation (metadata)**: the Metadata Agent does not independently fact-check its output - quality/accuracy depends entirely on the input `ScriptResult`/narration, and an upstream accuracy issue will pass through into the generated title/description. Its JSON artifact is now read by both the standalone and pipeline-integrated YouTube publishing paths, though only via a best-effort topic-matched lookup (no exact on-disk run-id link exists for metadata, unlike provenance/compliance).

**Known limitation (thumbnail)**: composition has no true subject-detection - it only controls which side of the frame hosts the text panel, not literal awareness of where a photo's subject actually is. The ambiguous-hook guard is a generic lexical heuristic (statistic-shaped wording with no shared topic word), not independent fact-checking, so a suitably-worded but still-misleading hook could pass through. The current visual style (a real stock photo plus text overlay) is intentionally simple for the MVP; more custom/cinematic styles via an AI image-generation provider are a possible future enhancement, not required now. The generated thumbnail JPEG is now applied by both the standalone and pipeline-integrated YouTube publishing paths via `videos.thumbnails.set`.

**Known limitation (compliance)**: full provenance-aware checking depends on a persisted manifest existing for the run being reviewed - a run from before the provenance-hardening milestone (or any run whose manifest write failed) has none and correctly resolves to `REVIEW`, never a fabricated `PASS`. The semantic review is advisory only and can never escalate to `BLOCK` on its own, so a well-worded but still-misleading claim could pass through undetected. Compliance is now the pipeline's own final gate, but nothing outside the pipeline itself yet enforces that gate - a future Upload Agent must explicitly check `compliance_result.publish_decision == "PASS"` before publishing.

Downloaded stock media and generated narration audio are treated as temporary working assets for the MVP (safe to clean up once consumed downstream); final videos should be retained per a future retention policy. No automated cleanup/retention is implemented yet.

A standalone YouTube Upload & Scheduling Agent has been implemented and real-validated (real OAuth authentication, real channel verification, a real private upload with its thumbnail applied, idempotent publishing records, and validated scheduling logic), and is also now reachable directly from the main pipeline above via an explicit opt-in `publishing` node added after Compliance: a normal pipeline run still never uploads anything, even on a genuine PASS, unless a caller explicitly supplies a `PublishingIntent` (`mode="private"`/`"scheduled"`, default `"disabled"`) and a `YouTubeClient`. Either path - standalone CLI (`src/youtube_upload_demo.py`) or the pipeline's own `publishing` node - reuses the identical `YouTubeUploadAgent`, OAuth handling, idempotency (`PublishingRecordStore`), and compliance gate: publishing proceeds only when `compliance_result.publish_decision == "PASS"`, and a `REVIEW` or `BLOCK` decision always refuses to upload, with no bypass. A bounded Compliance Remediation loop (REVIEW → targeted script correction → fresh re-evaluation, capped at 2 attempts) runs directly inside the main pipeline's Compliance stage. One controlled real private integrated publishing validation has succeeded end to end (real OAuth reuse, real upload, real thumbnail, persisted publishing record, idempotency verified, manually confirmed Private in YouTube Studio); real scheduled publishing remains validated only via dry-run/mocked tests, not a real scheduled upload yet. Automated cleanup/retention is not implemented yet. The next planned milestone is Monitoring/Post-publish - see `docs/PROGRESS.md`.

## Project Structure

```
src/
├── agents/       # LangGraph agents (reasoning tasks)
├── workflows/    # LangGraph workflow definitions  
├── services/     # Deterministic services
├── tools/        # External API wrappers
├── llm/          # LLM abstraction layer
├── models/       # SQLAlchemy models & migrations
└── config/       # Typed settings via Pydantic
```