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
