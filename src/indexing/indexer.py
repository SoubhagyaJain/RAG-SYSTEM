"""
Embedding + indexing orchestration.

Phase 0 provides a clean interface:
    indexer = Indexer()
    indexer.index_chunks(chunks)

This will:
1. Generate embeddings (currently OpenAI)
2. Store in the configured vector backend

Later: batching, cost tracking, incremental updates, different embedders.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chunking.splitter import Chunk
from src.config import get_settings
from src.indexing.vectorstore import ChromaVectorStore, VectorStore
from src.logging_config import logger


class Embedder:
    """OpenAI embedding client with retry and batching."""

    def __init__(self) -> None:
        self.settings = get_settings()
        emb_cfg = self.settings.indexing.embedding
        self.client = OpenAI()
        self.model = emb_cfg.model
        self.batch_size = emb_cfg.batch_size

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            logger.debug("Embedded batch", extra={"batch_start": i, "size": len(batch)})
        return all_embeddings


class Indexer:
    """Coordinates embedding generation and storage."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.settings = get_settings()
        self.embedder = Embedder()
        self.vector_store = vector_store or ChromaVectorStore()

    def index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info("Starting indexing", extra={"num_chunks": len(chunks)})

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        # Prepare payload for vector store
        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "doc_id": c.doc_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "metadata": {
                    "token_count": c.token_count,
                    **c.metadata,
                },
            }
            for c in chunks
        ]

        self.vector_store.add(payload, embeddings)
        logger.success("Indexing complete", extra={"indexed": len(chunks)})
