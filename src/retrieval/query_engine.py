"""
Retriever and QueryEngine factory.

All retrieval/generation configuration comes from config.yaml.
Uses the Gemma 4 8B model (via LlamaIndex Ollama integration) for generation.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import get_settings
from src.logging_config import logger


def get_vector_store() -> ChromaVectorStore:
    """Connect to the existing persistent Chroma collection."""
    settings = get_settings()

    # Prefer top-level vector_store config
    if hasattr(settings, "vector_store") and settings.vector_store:
        persist_dir = Path(settings.vector_store.persist_dir)
        collection_name = settings.vector_store.collection_name
    else:
        persist_dir = Path(settings.llama_index.vector_store.persist_dir)
        collection_name = settings.llama_index.vector_store.collection_name

    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    logger.debug(
        "Connected to Chroma vector store",
        extra={"collection": collection_name, "persist_dir": str(persist_dir)},
    )
    return vector_store


def get_query_engine(
    vector_store: BasePydanticVectorStore | None = None,
    similarity_top_k: int | None = None,
) -> BaseQueryEngine:
    """
    Create a configured QueryEngine using the persisted vector store + Gemma 4 8B.

    All important parameters (top_k, response_mode, etc.) are driven by config.
    """
    settings = get_settings()
    vector_store = vector_store or get_vector_store()

    top_k = similarity_top_k or settings.llama_index.similarity_top_k

    logger.info(
        "Building QueryEngine",
        extra={
            "llm_model": settings.ollama.llm_model,
            "similarity_top_k": top_k,
            "response_mode": settings.llama_index.response_mode,
        },
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=top_k,
        response_mode=settings.llama_index.response_mode,
        # We can add more advanced node_postprocessors here in later phases
    )

    return query_engine
