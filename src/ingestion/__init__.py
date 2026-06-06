"""Ingestion package for the Enterprise Agentic RAG System (Phase 1+).

Note: PDF loading currently uses LlamaIndex `SimpleDirectoryReader` inside
`GuidebookIngestionPipeline.load_documents()`. Keeping the import surface small
means `from src.ingestion import ...` stays cheap and predictable.
"""

from src.ingestion.ingestion_pipeline import GuidebookIngestionPipeline, run_ingestion

__all__ = ["GuidebookIngestionPipeline", "run_ingestion"]
