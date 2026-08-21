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
