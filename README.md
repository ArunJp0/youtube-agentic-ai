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

The complete orchestration - Research → Script → Voice → Visual Context Planner → Duration-Aware Visual Media → Visual QC → Video Assembly → Subtitles/Captions → BGM/Audio Mixing - is implemented end-to-end as one LangGraph pipeline. One command is intended to turn a topic into a final captioned-and-mixed, playable MP4:

```
Topic Input → Research Agent → Script Agent → Voice Service → Visual Context Planner
            → Duration-Aware Visual Media Service → Visual QC
            → Video Assembly Service → Subtitle/Caption Service
            → BGM / Audio Mixing Service → Final Captioned + BGM Mixed MP4
```

**Validation status**: all 8 stages have been validated end-to-end together with real providers (real Gemini, real Wikipedia, real Edge TTS, real Pexels, real Whisper), including after the Gemini fallback model was changed (`gemini-3.6-flash` → `gemini-3.1-flash-lite`) following a real health check (`python -m src.gemini_health_check`) that found the old fallback timing out under real load - see `docs/DECISIONS.md`.

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Context Planner**: one Gemini call for the entire script, producing a structured visual plan per section - what it's actually about (`semantic_summary`), concrete visual concepts (`visual_intents`), search queries, `avoid_concepts` (literal-but-wrong interpretations to steer away from, e.g. "constructs a narrative" ≠ construction), and neutral fallback queries. If the call fails or returns something unusable, it falls back cleanly to the same deterministic query generation used when no planner runs at all - the pipeline never breaks because of it. Runs as an internal part of the Visual Media stage, not a separate top-level pipeline stage.
- **Visual Media Service**: duration-aware - it derives how many distinct stock clips each section needs from that section's real share of the narration duration (not a fixed count), and consumes the visual plan (rather than isolated keywords) to search, filter, and select multiple ordered assets per section. A deterministic semantic filter rejects candidates whose available metadata matches an `avoid_concept`. Only the assets actually selected are downloaded (real Pexels, with a mock provider for offline dev), saved under `output/media/`. Selected assets are tracked globally across the whole video to avoid duplicates, with exact-asset reuse/looping used only as a controlled fallback when unique stock footage runs out.
- **Visual QC**: inspects real representative frames (not just text metadata) from every selected asset with a vision-capable evaluator (real Gemini vision, with a mock evaluator for offline dev) - one batched request per script section. Assets judged weak or misleading trigger a bounded replacement request back through the Visual Media Service (never an unbounded search); an asset still flagged misleading after replacement is exhausted stops the pipeline before Video Assembly rather than risk a misleading clip in the final video. A vision-provider outage falls back to the upstream metadata filter's prior approval instead of failing the pipeline.
- **Video Assembly Service**: combines the narration audio with each section's one or more QC-approved ordered clips into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`. Makes no semantic decisions - only timing, trimming, cropping, ordering, concatenation, and audio sync.
- **Subtitle/Caption Service**: runs after Video Assembly succeeds. Transcribes the real narration audio locally (Whisper via `faster-whisper` - free, no paid API) to get actual spoken-word timestamps (never estimated from script section durations), builds readable YouTube-style captions, writes an `.srt` file under `output/subtitles/`, and burns captions into a **copy** of the assembled MP4 (`output/video/<name>-captioned.mp4`) - the original MP4 is never overwritten.
- **BGM / Audio Mixing Service**: runs after Captions succeeds, reusing the same `ScriptResult` already produced earlier in the run (Research/Script are never re-run). Plans the video's mood via a single optional Gemini call (falling back automatically to a safe deterministic mood profile if that call is unavailable), deterministically selects one instrumental track from a curated local approved catalog (`assets/bgm/` - YouTube Audio Library tracks marked "Attribution not required" for the current MVP), and mixes it under the narration on the captioned MP4 at a conservative gain with sidechain ducking, so narration stays clearly dominant and the music stays subtle. Writes a new copy (`output/video/<name>-captioned-bgm.mp4`) - the captioned MP4 is never overwritten.

All stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage - including a hard Visual QC failure, a caption/transcription failure, or a real BGM catalog/selection/mixing failure - stops the pipeline before the next one runs (or before completion), preserving every earlier stage's successful results. A BGM mood-planning outage alone does not stop the pipeline - the deterministic fallback plan lets mixing continue. The pipeline only reports `completed` once the final captioned-and-BGM-mixed MP4 exists. `output/audio/`, `output/media/`, `output/video/`, and `output/subtitles/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation (visual)**: perfect semantic stock-footage matching is not guaranteed - Pexels' inventory for a given query is finite, so even with duration-aware planning, semantic filtering, and vision QC, an occasional visual can still be only loosely related to its section, and controlled visual reuse may still occur in longer videos; relevance can also vary somewhat run-to-run with live Pexels results. Manual review of real runs has found this acceptable for the current MVP. The visual pipeline (planning → selection → QC → assembly) is considered feature-complete/frozen for now and is not planned for further optimization without a new, recurring, concrete problem.

**Known limitation (captions)**: occasional very short, single-word captions can occur due to Whisper's own segmentation, since caption segmentation only ever splits long segments and never merges short adjacent ones. Manual review found current readability acceptable, so no further segmentation tuning is planned at this stage.

**Multi-file output**: each run currently produces the original assembled MP4, a captioned MP4, and a final BGM-mixed MP4 (plus the `.srt` transcript) on disk, rather than a single final file. This is intentional for the current MVP - the intermediate files are useful development/debug fallbacks; a future storage/cleanup milestone may delete them once the final mixed MP4 has been used/uploaded successfully.

**Known limitation (BGM)**: the approved BGM catalog (`assets/bgm/`) is a manually curated local library - there is no automatic licensed-music-provider integration yet, so adding tracks is a manual, one-time-per-track MVP step. Gemini mood planning is a single optional call per video; it has been observed to be unavailable under real Gemini free-tier rate limiting (429/503/timeouts), in which case mood selection falls back to a safe, generic deterministic profile rather than failing.

A **Metadata Agent** has been implemented and validated standalone (`MetadataAgent`, `src/metadata_demo.py`): given a topic, the video's `ScriptResult`, and its final duration, it generates a professional YouTube title, description, tags, hashtags, an SEO summary, and chapters, using exactly one Gemini call reusing the same provider/model chain as the rest of the pipeline - never inventing facts, clickbait, or spam keywords beyond what the script actually supports. Chapter timestamps are always derived deterministically from real section-timing data (never asked of the LLM); if that timing data is insufficient, chapters are explicitly marked unavailable rather than fabricated. Output is written as a structured JSON artifact (`output/metadata/<video-slug>.json`) designed for reuse by a future YouTube Upload Agent. It is **not yet wired into the main pipeline** above; integrating it as a stage after BGM/Audio Mixing is the next planned milestone.

Downloaded stock media and generated narration audio are treated as temporary working assets for the MVP (safe to clean up once consumed downstream); final videos should be retained per a future retention policy. No automated cleanup/retention is implemented yet.

Thumbnail generation, copyright/compliance checking, YouTube upload, scheduling, and automated cleanup/retention are not implemented yet. The next planned milestone is Metadata Agent Main Pipeline Integration, followed by a Thumbnail Agent.

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