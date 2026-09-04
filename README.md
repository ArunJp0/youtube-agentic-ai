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

The complete orchestration - Research → Script → Voice → Visual Context Planner → Duration-Aware Visual Media → Visual QC → Video Assembly → Subtitles/Captions - is implemented and validated end-to-end with real providers, including real Gemini vision QC and real local Whisper transcription. One command turns a topic into a final captioned, playable MP4:

```
Topic Input → Research Agent → Script Agent → Voice Service → Visual Context Planner
            → Duration-Aware Visual Media Service → Visual QC
            → Video Assembly Service → Subtitle/Caption Service → Captioned Final MP4
```

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Context Planner**: one Gemini call for the entire script, producing a structured visual plan per section - what it's actually about (`semantic_summary`), concrete visual concepts (`visual_intents`), search queries, `avoid_concepts` (literal-but-wrong interpretations to steer away from, e.g. "constructs a narrative" ≠ construction), and neutral fallback queries. If the call fails or returns something unusable, it falls back cleanly to the same deterministic query generation used when no planner runs at all - the pipeline never breaks because of it. Runs as an internal part of the Visual Media stage, not a separate top-level pipeline stage.
- **Visual Media Service**: duration-aware - it derives how many distinct stock clips each section needs from that section's real share of the narration duration (not a fixed count), and consumes the visual plan (rather than isolated keywords) to search, filter, and select multiple ordered assets per section. A deterministic semantic filter rejects candidates whose available metadata matches an `avoid_concept`. Only the assets actually selected are downloaded (real Pexels, with a mock provider for offline dev), saved under `output/media/`. Selected assets are tracked globally across the whole video to avoid duplicates, with exact-asset reuse/looping used only as a controlled fallback when unique stock footage runs out.
- **Visual QC**: inspects real representative frames (not just text metadata) from every selected asset with a vision-capable evaluator (real Gemini vision, with a mock evaluator for offline dev) - one batched request per script section. Assets judged weak or misleading trigger a bounded replacement request back through the Visual Media Service (never an unbounded search); an asset still flagged misleading after replacement is exhausted stops the pipeline before Video Assembly rather than risk a misleading clip in the final video. A vision-provider outage falls back to the upstream metadata filter's prior approval instead of failing the pipeline.
- **Video Assembly Service**: combines the narration audio with each section's one or more QC-approved ordered clips into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`. Makes no semantic decisions - only timing, trimming, cropping, ordering, concatenation, and audio sync.
- **Subtitle/Caption Service**: runs after Video Assembly succeeds. Transcribes the real narration audio locally (Whisper via `faster-whisper` - free, no paid API) to get actual spoken-word timestamps (never estimated from script section durations), builds readable YouTube-style captions, writes an `.srt` file under `output/subtitles/`, and burns captions into a **copy** of the assembled MP4 (`output/video/<name>-captioned.mp4`) - the original MP4 is never overwritten.

All stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage - including a hard Visual QC failure or a caption/transcription failure - stops the pipeline before the next one runs (or before completion), preserving every earlier stage's successful results. The pipeline only reports `completed` once the final captioned MP4 exists. `output/audio/`, `output/media/`, `output/video/`, and `output/subtitles/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation (visual)**: perfect semantic stock-footage matching is not guaranteed - Pexels' inventory for a given query is finite, so even with duration-aware planning, semantic filtering, and vision QC, an occasional visual can still be only loosely related to its section, and controlled visual reuse may still occur in longer videos; relevance can also vary somewhat run-to-run with live Pexels results. Manual review of real runs has found this acceptable for the current MVP. The visual pipeline (planning → selection → QC → assembly) is considered feature-complete/frozen for now and is not planned for further optimization without a new, recurring, concrete problem.

**Known limitation (captions)**: occasional very short, single-word captions can occur due to Whisper's own segmentation, since caption segmentation only ever splits long segments and never merges short adjacent ones. Manual review found current readability acceptable, so no further segmentation tuning is planned at this stage.

**Two-video output**: each run currently produces both the original assembled MP4 and a separate captioned MP4, rather than one final file. This is intentional for the current MVP (the original is a useful development/debug fallback); a future storage/cleanup milestone may delete the intermediate uncaptioned MP4 once captioning/upload has succeeded.

Downloaded stock media and generated narration audio are treated as temporary working assets for the MVP (safe to clean up once consumed downstream); final videos should be retained per a future retention policy. No automated cleanup/retention is implemented yet.

Background music/audio mixing, thumbnail generation, video metadata generation, copyright/compliance checking, YouTube upload, scheduling, and automated cleanup/retention are not implemented yet. The next planned milestone is a standalone BGM/audio mixing service.

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