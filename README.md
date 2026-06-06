# Enterprise Agentic RAG System

**Production-grade Agentic Retrieval-Augmented Generation** over the *AI Agents Guidebook* (117-page illustrated reference).

**Current Tech Stack (Phase 0):**
- **Framework**: LlamaIndex (primary)
- **LLM**: Gemma (latest) via Ollama (fully local)
- **Embeddings & Reranker**: Local models served by Ollama
- **Evaluation**: Ragas
- **Vector Store**: Chroma (via LlamaIndex) — easily swappable

## Goals

- Build a **truly production-ready** RAG system with strong foundations.
- Start fully local (Ollama) for cost control, privacy, and iteration speed.
- Design for future **agentic** capabilities (planning, tool use, multi-step reasoning, self-critique).
- Evaluation-first: Ragas + custom metrics from day one.

## Phase 0: Setup & Foundations (Done)

## Phase 1: Ingestion & Vector Store (Complete)

## Phase 2: Evaluation with Ragas (Current)

- Full `RAGASEvaluator` class wrapping Ragas
- Uses **Gemma 4 8B** (via LlamaIndex) as the judge LLM
- Uses **nomic-embed-text** for Ragas embeddings
- Curated dataset of 18 high-quality questions with ground truth + reference contexts
- Metrics: faithfulness, answer_relevancy, context_precision, context_recall
- Timestamped result saving to `artifacts/evaluation_results/`
- Fully driven by `config.yaml`
- `notebooks/02_evaluation.ipynb` for running and analyzing results

Run evaluation with:
```python
from src.evaluation import run_evaluation
from src.retrieval import get_query_engine

results = run_evaluation(query_engine=get_query_engine())
```

- `UnstructuredReader` for high-quality PDF parsing of the illustrated guidebook
- `IngestionPipeline` + `SentenceSplitter` driven by `config.yaml`
- Rich metadata per node: `page_number`, `section`, `has_code`, `has_diagram`, `document_title`
- Persistent **Chroma** vector store at `data/processed/chroma_db/`
- `QueryEngine` factory using **Gemma 4 8B**
- Idempotent ingestion + excellent logging
- `notebooks/01_ingestion.ipynb` with test queries and node inspection

Run the notebook after `pip install -e ".[dev]"` (it will pull `unstructured[pdf]`).

This phase delivers:

- Clean professional project structure
- Strongly-typed configuration (`config.yaml` + Pydantic)
- Production logging (loguru)
- Seamless LlamaIndex + Ollama integration
- PDF ingestion using LlamaIndex `SimpleDirectoryReader`
- Configurable node parsing / chunking
- Setup notebook that validates the entire foundation
- Docker scaffolding
- Ready for persistent vector stores and full pipelines in Phase 1

## Project Structure

```
enterprise-rag-ai-agents/
├── .env.example
├── config.yaml                  # All models, chunk sizes, retrieval params, etc.
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings + LlamaIndex configuration helper
│   ├── logging_config.py
│   └── utils/
├── notebooks/
│   └── 00_setup.ipynb           # Phase 0 validation notebook
├── docker/
├── data/
│   ├── raw/                     # ai_agents_guidebook.pdf lives here
│   └── processed/
├── app/                         # Future CLI / API / UI
└── tests/
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Recommended: `uv` for fast dependency management

### 2. Pull Required Models

```bash
ollama pull gemma2:27b          # or gemma2:9b / whatever "Gemma 4 Latest" maps to
ollama pull nomic-embed-text    # or mxbai-embed-large, etc.
```

### 3. Setup Python Environment

```powershell
cd "C:\Users\jains\OneDrive\Desktop\RAG-SYSTEM"

# Using uv (recommended)
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"

# Or with pip
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 4. Configure Environment

```powershell
copy .env.example .env
# Edit .env if your Ollama runs on a different host/port
```

### 5. Run the Phase 0 Setup Notebook

```powershell
jupyter lab notebooks/00_setup.ipynb
```

The notebook will:
- Load and validate configuration
- Configure LlamaIndex global `Settings` with your Ollama models
- Verify model availability
- Load the AI Agents Guidebook
- Create nodes using LlamaIndex `SentenceSplitter`
- (Optional) Run a small vector index query

## Configuration Highlights

All important knobs live in `config.yaml`:

- `ollama.llm_model` → Gemma via Ollama
- `ollama.embed_model` → Local embedding model
- `llama_index.chunk_size` / `chunk_overlap`
- `llama_index.similarity_top_k`
- `retrieval.top_k`, reranker toggle, hybrid search flags
- Ragas metrics and judge model

## Development Commands

```bash
# Linting & formatting
ruff check .
ruff format .

# Tests
pytest

# Type checking
mypy src
```

## Roadmap

- **Phase 0** — Foundations + LlamaIndex + Ollama setup (current)
- **Phase 1** — Full ingestion pipeline, persistent Chroma index, basic query engine
- **Phase 2** — Advanced retrieval (hybrid, reranking), citation handling
- **Phase 3** — Ragas evaluation harness + golden dataset
- **Phase 4+** — Agentic patterns on top of LlamaIndex (query engines, agents, tools, reflection loops)

## Notes

- The project is designed to stay **local-first**. Cloud LLM fallbacks can be added later via LlamaIndex's multi-LLM support.
- Large PDFs are stored in `data/raw/` (consider Git LFS if you push frequently).
- All LlamaIndex global settings are driven from `src/config.py` → `settings.configure_llama_index()`.

---

**Next milestone**: Build a complete `IngestionPipeline` + `VectorStoreIndex` that can answer questions from the full AI Agents Guidebook using only local models.
