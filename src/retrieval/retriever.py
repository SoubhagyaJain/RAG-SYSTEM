"""
Retriever abstraction + basic semantic retriever.

Phase 0: Simple top-k semantic search.
Phase 1+: Add MMR, hybrid search (BM25 + vector), query expansion, HyDE, etc.
Phase 3+: Agentic retrieval (multi-step, tool calling, routing).
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import get_settings
from src.indexing.vectorstore import ChromaVectorStore, RetrievedChunk, VectorStore
from src.logging_config import logger


class Retriever:
    """High-level retrieval interface."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore()
        self.client = OpenAI()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval.top_k
        logger.info("Retrieving", extra={"query": query[:80], "top_k": top_k})

        # Embed the query using same model as documents
        q_emb = (
            self.client.embeddings.create(
                model=self.settings.indexing.embedding.model,
                input=[query],
            )
            .data[0]
            .embedding
        )

        results = self.vector_store.search(q_emb, top_k=top_k)
        logger.success("Retrieval complete", extra={"returned": len(results)})
        return results
