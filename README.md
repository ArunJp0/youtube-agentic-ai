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

The complete orchestration - Research → Script → Voice → Visual Context Planner → Duration-Aware Visual Media → Semantic Media Filtering → Video Assembly - is implemented and validated end-to-end with real providers. One command turns a topic into a final playable MP4:

```
Topic Input → Research Agent → Script Agent → Voice Service → Visual Context Planner
            → Duration-Aware Visual Media Service → Semantic Media Filtering
            → Video Assembly Service → Final MP4
```

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Context Planner**: one Gemini call for the entire script, producing a structured visual plan per section - what it's actually about (`semantic_summary`), concrete visual concepts (`visual_intents`), search queries, `avoid_concepts` (literal-but-wrong interpretations to steer away from, e.g. "constructs a narrative" ≠ construction), and neutral fallback queries. If the call fails or returns something unusable, it falls back cleanly to the same deterministic query generation used when no planner runs at all - the pipeline never breaks because of it.
- **Visual Media Service**: duration-aware - it derives how many distinct stock clips each section needs from that section's real share of the narration duration (not a fixed count), and consumes the visual plan (rather than isolated keywords) to search, filter, and select multiple ordered assets per section. A deterministic semantic filter rejects candidates whose available metadata matches an `avoid_concept`. Only the assets actually selected are downloaded (real Pexels, with a mock provider for offline dev), saved under `output/media/`. Selected assets are tracked globally across the whole video to avoid duplicates, with exact-asset reuse/looping used only as a controlled fallback when unique stock footage runs out.
- **Video Assembly Service**: combines the narration audio with each section's one or more ordered clips into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`. Makes no semantic decisions - only timing, trimming, cropping, ordering, concatenation, and audio sync.

All stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage stops the pipeline before the next one runs, and the pipeline only reports `completed` once a real final MP4 exists. `output/audio/`, `output/media/`, and `output/video/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation**: the semantic filter only judges a candidate by lightweight text metadata (a Pexels photo's alt text or a descriptive URL slug), not actual frame content - so a candidate with misleading or missing metadata can still pass through unfiltered. This is a content-quality limitation, not a pipeline failure - videos are still produced complete and correctly timed. A future Visual QC step that inspects actual candidate thumbnails/frames (e.g. via a vision model) is a candidate next milestone.

Downloaded stock media and generated narration audio are treated as temporary working assets for the MVP (safe to clean up once consumed downstream); final videos should be retained per a future retention policy. No automated cleanup/retention is implemented yet.

Subtitles, thumbnail, metadata generation, QC, copyright checking, YouTube upload, scheduling, and automated cleanup/retention are not implemented yet.

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