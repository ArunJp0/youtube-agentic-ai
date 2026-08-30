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

The complete orchestration - Research → Script → Voice → Visual Media → Video Assembly - is implemented and validated end-to-end with real providers. One command turns a topic into a final playable MP4:

```
Topic Input → Research Agent → Script Agent → Voice Service → Visual Media Service → Video Assembly Service → Final MP4
```

- **Research Agent**: researches a topic (real Wikipedia search + Gemini LLM, with mock providers for offline dev) and produces a structured `ResearchResult` (summary, key points, sourced facts, source URLs).
- **Script Agent**: converts a `ResearchResult` into a structured `ScriptResult` (title, hook, introduction, narrated sections, conclusion, call to action, estimated duration, source references) - natural spoken-style narration for YouTube, grounded only in the research.
- **Voice Service**: converts a `ScriptResult` into narration audio (real Edge TTS, with a mock provider for offline dev), saved under `output/audio/`.
- **Visual Media Service**: finds and downloads a stock image/video per script section (real Pexels, with a mock provider for offline dev), saved under `output/media/`.
- **Video Assembly Service**: combines the narration audio with the section media into a final MP4 (FFmpeg: H.264 video, AAC audio, 1920x1080, 30 fps), saved under `output/video/`.

All five stages run together as one LangGraph pipeline (`python -m src.pipeline_demo "<topic>"`); a failure at any stage stops the pipeline before the next one runs, and the pipeline only reports `completed` once a real final MP4 exists. `output/audio/`, `output/media/`, and `output/video/` are all generated runtime artifacts and are Git-ignored - nothing under them is source.

**Known limitation**: the current implementation fetches a limited/fixed number of stock clips per script section regardless of that section's duration, so longer videos may visibly repeat stock footage. This is a visual-quality limitation, not a pipeline failure - videos are still produced complete and correctly timed. Duration-aware multi-clip planning (fetching only as many distinct clips as a section's actual length requires, with looping as a last resort) is the next planned improvement.

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