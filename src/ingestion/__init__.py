"""Ingestion package for the Enterprise Agentic RAG System (Phase 1+)."""

from src.ingestion.ingestion_pipeline import GuidebookIngestionPipeline, run_ingestion

__all__ = ["GuidebookIngestionPipeline", "run_ingestion"]
