"""Enterprise Agentic RAG System (LlamaIndex + Ollama)."""

__version__ = "0.1.0"

from src.config import get_settings, settings
from src.logging_config import logger, setup_logging

# Phase 1+ public API (guarded so optional ingestion dependencies do not
# break the entire package import)
try:
    from src.ingestion import GuidebookIngestionPipeline, run_ingestion
except Exception as e:  # pragma: no cover
    GuidebookIngestionPipeline = None
    run_ingestion = None
    _ingestion_import_error = e

try:
    from src.retrieval import (
        get_query_engine,
        get_vector_store,
        get_retriever,
        SmallToBigRetriever,
        HybridRetriever,
    )
except Exception as e:  # pragma: no cover
    get_query_engine = None
    get_vector_store = None
    get_retriever = None
    SmallToBigRetriever = None
    HybridRetriever = None
    _retrieval_import_error = e

try:
    from src.evaluation import RAGASEvaluator, run_evaluation
except Exception as e:  # pragma: no cover
    RAGASEvaluator = None
    run_evaluation = None
    _evaluation_import_error = e

__all__ = [
    "get_settings",
    "settings",
    "logger",
    "setup_logging",
    "GuidebookIngestionPipeline",
    "run_ingestion",
    "get_query_engine",
    "get_vector_store",
    "RAGASEvaluator",
    "run_evaluation",
]
