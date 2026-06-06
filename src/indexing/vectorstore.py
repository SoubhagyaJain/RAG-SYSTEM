"""
Vector store abstraction layer.

Phase 0 provides:
- Abstract VectorStore protocol
- ChromaDB implementation (local, persistent, great for iteration)

Future backends (easy to add):
- FAISS (in-memory / local index)
- pgvector
- Pinecone / Weaviate / Qdrant (managed)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.config import get_settings
from src.logging_config import logger


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    doc_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class VectorStore(ABC):
    """Abstract interface all vector stores must implement."""

    @abstractmethod
    def add(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """Store chunks + their embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int, **kwargs: Any) -> list[RetrievedChunk]:
        """Semantic similarity search."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Dangerous: wipe the collection (useful in dev)."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store (recommended for Phase 0/1)."""

    def __init__(self, collection_name: str | None = None) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        self.settings = get_settings()
        vs_cfg = self.settings.indexing.vector_store

        self.persist_dir = self.settings.paths.resolve()["vector_store"] / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name or vs_cfg.collection_name

        # We compute embeddings ourselves (via OpenAI or other) so we use "default" embedding function
        # or a no-op. For now we pass our own vectors.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": vs_cfg.distance_metric},
        )

        logger.info(
            "Chroma vector store initialized",
            extra={"collection": self.collection_name, "persist_dir": str(self.persist_dir)},
        )

    def add(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": c.get("doc_id"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                **c.get("metadata", {}),
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,
        )
        logger.success("Added chunks to vector store", extra={"count": len(chunks)})

    def search(self, query_embedding: list[float], top_k: int, **kwargs: Any) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    score=1.0 - (results["distances"][0][i] or 0.0),  # convert distance to similarity-ish
                    metadata=meta,
                    doc_id=meta.get("doc_id"),
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                )
            )
        return retrieved

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        logger.warning("Deleted collection", extra={"collection": self.collection_name})
