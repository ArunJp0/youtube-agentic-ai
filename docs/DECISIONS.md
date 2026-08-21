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
