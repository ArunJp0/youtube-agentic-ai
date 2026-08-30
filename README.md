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

The complete orchestration - Research → Script → Voice → Duration-Aware Visual Media → Multi-Clip Video Assembly - is implemented and validated end-to-end with real providers. One command turns a topic into a final playable MP4:

```
Topic Input → Research Agent → Script Agent → Voice Service → Duration-Aware Visual Media Service → Multi-Clip Video Assembly Service → Final MP4
```

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Media Service**: duration-aware - it derives how many distinct stock clips each section needs from that section's real share of the narration duration (not a fixed count), selects multiple ordered assets per section when needed, and downloads only the assets actually selected (real Pexels, with a mock provider for offline dev), saved under `output/media/`. Selected assets are tracked globally across the whole video to avoid duplicates, with exact-asset reuse/looping used only as a controlled fallback when unique stock footage runs out.
- **Video Assembly Service**: combines the narration audio with each section's one or more ordered clips into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`.

All five stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage stops the pipeline before the next one runs, and the pipeline only reports `completed` once a real final MP4 exists. `output/audio/`, `output/media/`, and `output/video/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation**: visual semantic relevance is still bounded by deterministic (non-LLM) query generation and by what stock footage is actually available for a given query - some individual selected clips are only loosely related to their section's narration, even though visible clip repetition is now substantially reduced. This is a content-quality limitation, not a pipeline failure - videos are still produced complete and correctly timed. A future QC/visual-relevance-validation step (to reject weak matches and request alternatives) is a candidate future improvement.

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