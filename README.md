# Enterprise Agentic RAG System

**Production-grade, agentic Retrieval-Augmented Generation platform** built for deep reasoning over technical guidebooks and enterprise knowledge bases.

**Current Focus:** "AI Agents Guidebook" (117-page illustrated reference) — a comprehensive guide to building, evaluating, and operating AI agents.

## Project Philosophy

- **Production-first**: Type-safe config, structured logging, observability hooks, reproducible pipelines, and clean separation of concerns.
- **Agentic by design**: Foundations support future multi-step reasoning, tool use, self-critique, query planning, and iterative retrieval.
- **Modular & Extensible**: Every major stage (ingestion → chunking → indexing → retrieval → generation → evaluation) is replaceable.
- **Evaluation-driven**: Metrics and golden datasets are first-class from day one.

## Current Phase: Phase 0 — Setup & Foundations

**Goals accomplished in this phase:**
- Professional project layout (src layout, packaging, Docker)
- Strongly-typed configuration (YAML + Pydantic + .env)
- Production logging setup
- PDF ingestion foundation (page-aware extraction)
- Configurable chunking strategies (extensible to semantic/agentic)
- Vector indexing abstractions (ready for Chroma / FAISS / PGVector)
- Retrieval and generation interfaces
- Evaluation scaffolding
- Reproducible notebook for Phase 0 validation

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Recommended: [uv](https://docs.astral.sh/uv/) (fastest) or pip + venv

### 2. Setup Environment

```bash
# Clone or open the folder
cd C:\Users\jains\OneDrive\Desktop\RAG-SYSTEM   # or your local path

# Create virtual environment (uv recommended)
uv venv
.venv\Scripts\activate     # Windows PowerShell

# Install the project in editable mode with dev dependencies
uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"
```

### 3. Configure Secrets

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and others as needed)
```

### 4. Place the Source Document

Copy the guidebook into the raw data folder:

```bash
# From your Downloads (example)
copy "$env:USERPROFILE\Downloads\AI Agents guidebook.pdf" "data\raw\ai_agents_guidebook.pdf"
```

### 5. Run Phase 0 Validation Notebook

Open and run:

```bash
jupyter lab notebooks/00_phase0_setup.ipynb
```

Or execute the setup validation script (to be added in later phases).

## Project Structure

```
enterprise-rag-ai-agents/          # (this repo root)
├── .env.example
├── .gitignore
├── config.yaml                    # All tunable parameters
├── pyproject.toml
├── README.md
├── src/
│   ├── config.py                  # Pydantic settings + YAML loader
│   ├── logging_config.py          # Structured logging (loguru)
│   ├── ingestion/                 # Document loading & preprocessing
│   ├── chunking/                  # Text splitting strategies
│   ├── indexing/                  # Embedding + vector store
│   ├── retrieval/                 # Retrievers + rerankers
│   ├── generation/                # LLM calls + prompt engineering
│   ├── evaluation/                # RAGAS-style + custom metrics
│   └── utils/
├── notebooks/
│   └── 00_phase0_setup.ipynb
├── docker/
│   └── Dockerfile
├── tests/
├── data/
│   ├── raw/                       # Original PDFs, docs
│   └── processed/                 # Chunked JSONL, metadata, etc.
└── app/                           # Future: FastAPI / Streamlit / Agent UI
```

## Roadmap (High-level)

- **Phase 0**: Foundations & PDF ingestion (current)
- **Phase 1**: Indexing + basic retrieval + simple generation
- **Phase 2**: Evaluation harness + golden dataset creation
- **Phase 3**: Agentic patterns (query rewriting, multi-hop, tool use, critique loops)
- **Phase 4**: Advanced retrieval (hybrid, graph, agentic routing)
- **Phase 5**: Production serving, observability, cost tracking, CI/CD

## Key Design Decisions (Phase 0)

- **Config-driven everything**: No magic numbers in code.
- **Page-aware ingestion**: Preserve page numbers and section structure for citations.
- **Token-aware chunking**: Target chunk size in tokens (not chars) for better embedding behavior.
- **Pluggable components**: Interfaces + factory functions so you can swap splitters, embedders, vector stores, and LLMs without touching pipelines.
- **Observability hooks**: Every major stage emits structured logs + (future) traces.

## Contributing / Development

```bash
# Linting & formatting
ruff check .
ruff format .

# Tests
pytest

# Type checking
mypy src
```

## License

Internal / proprietary for now. Adjust as needed.

---

**Next milestone**: Complete a working end-to-end ingest → chunk → index → retrieve → answer loop with the AI Agents Guidebook and baseline metrics.
