# =============================================================================
# Enterprise Agentic RAG System - Production Dockerfile (Phase 0 base)
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (add tesseract, poppler, etc. later for OCR / PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution (optional but recommended)
# RUN pip install uv

COPY pyproject.toml ./
# COPY uv.lock ./   # if using uv

RUN pip install --upgrade pip && \
    pip install -e ".[dev]"   # In production use a locked set of deps

COPY . .

# Default command (will be replaced by proper entrypoints later)
CMD ["python", "-m", "app.main", "hello"]
