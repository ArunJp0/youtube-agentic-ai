# Claude Code Project

## Goal

A **modular monolith** that builds an automated Agentic AI system for generating YouTube videos with minimal human intervention. MVP uses only free/open-source or free-tier tools — no paid APIs, no unnecessary microservices.

See [docs/DECISIONS.md](docs/DECISIONS.md) for architectural decisions and rationale, and [docs/PROGRESS.md](docs/PROGRESS.md) for completed milestones and current status.

---

## Architecture

- **LangGraph** orchestrates multi-step agent reasoning (e.g. the research workflow).
- **Interchangeable providers**: LLM and search backends sit behind abstract interfaces (`src/llm/provider.py`, `src/tools/search_provider.py`) with mock implementations for offline development, swappable for real providers without touching agent/workflow code.
- **Database persistence** (SQLAlchemy + Alembic) is scaffolded but not yet wired in — postponed until the Research Agent is validated against real providers.

```
src/
├── agents/          # LangGraph agents (reasoning tasks)
├── workflows/       # LangGraph workflow definitions
├── services/        # Deterministic services (e.g., FFmpeg wrapper)
├── tools/           # External API wrappers (search, etc.)
├── llm/             # LLM abstraction layer
├── models/          # SQLAlchemy models & Alembic migrations
└── config/          # Typed settings via Pydantic
```

---

## Coding Rules

- Keep the codebase simple: no unnecessary microservices, Celery, Redis, or Kubernetes for the MVP.
- No paid APIs during initial development — prefer free/open-source or free-tier providers.
- New LLM or search backends must implement the existing abstract provider interfaces, not bypass them.
- Favor small, testable units; every new agent/workflow should ship with tests.

---

## Commands

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .[dev]

# Run tests
pytest
```

---

## Contributing

Feel free to open issues or pull requests. Please keep the codebase simple and avoid unnecessary micro-services or paid services for the MVP.
