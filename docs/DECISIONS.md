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

## Gemini 3.6 Flash as the primary LLM

Gemini 3.6 Flash is the primary LLM provider for the Research Agent, chosen as a free/low-cost real provider behind the existing `LLMProvider` interface.

## Retry with exponential backoff for transient Gemini errors

429, 408, and 5xx responses (plus timeouts/network errors) are retried with exponential backoff and jitter, since these are transient and typically self-resolve on a subsequent attempt. Non-retryable errors (400/401/403, malformed/blocked responses) fail immediately.

## Fallback model support enabled

If the primary model (Gemini 3.6 Flash) remains unavailable after exhausting retries, the provider automatically retries against a configured fallback model (Gemini 3.5 Flash Lite) before failing, improving reliability during upstream capacity issues.

## Research Agent remains provider-agnostic

All Wikipedia- and Gemini-specific logic (HTTP details, retry/backoff, fallback model selection) is contained within their respective provider implementations. `ResearchAgent` and the LangGraph workflow only depend on the abstract `LLMProvider`/`SearchProvider` interfaces and were not modified to add real-provider support.
