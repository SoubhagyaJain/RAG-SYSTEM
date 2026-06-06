"""Retrieval components for the Educational RAG System.

Provides modular retrievers (vector, small-to-big, hybrid) and the main
QueryEngine factory with the educational system prompt.

Usage example:
    from src.retrieval import get_query_engine, get_retriever

    # The mode is read from config.yaml (retrieval.mode)
    qe = get_query_engine()

    # Or get a retriever directly for advanced use
    retriever = get_retriever(index, mode="hybrid")
"""

from src.retrieval.query_engine import (
    get_query_engine,
    get_vector_store,
    EDUCATIONAL_TEXT_QA_PROMPT,
)

from src.retrieval.retriever import (
    get_retriever,
    SmallToBigRetriever,
    HybridRetriever,
    VectorRetriever,  # exported for advanced / testing use
)

from src.retrieval.postprocessor import MetadataBoosterPostprocessor

__all__ = [
    "get_query_engine",
    "get_vector_store",
    "get_retriever",
    "SmallToBigRetriever",
    "HybridRetriever",
    "VectorRetriever",
    "MetadataBoosterPostprocessor",
    "EDUCATIONAL_TEXT_QA_PROMPT",
]
