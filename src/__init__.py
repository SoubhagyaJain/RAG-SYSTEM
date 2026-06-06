"""Enterprise Agentic RAG System (LlamaIndex + Ollama)."""

__version__ = "0.1.0"

from src.config import get_settings, settings
from src.logging_config import logger, setup_logging

# Phase 1+ public API
from src.ingestion import GuidebookIngestionPipeline, run_ingestion
from src.retrieval import get_query_engine, get_vector_store

__all__ = [
    "get_settings",
    "settings",
    "logger",
    "setup_logging",
    "GuidebookIngestionPipeline",
    "run_ingestion",
    "get_query_engine",
    "get_vector_store",
]
