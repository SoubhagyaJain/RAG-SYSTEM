# =============================================================================
# Enterprise Agentic RAG System
# LlamaIndex + Ollama (local models) - Phase 0 base image
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

COPY . .

# Default entrypoint (will evolve)
CMD ["python", "-c", "from src.config import settings; print('Config loaded. Project:', settings.project.name)"]
