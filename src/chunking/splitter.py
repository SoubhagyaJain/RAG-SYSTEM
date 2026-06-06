"""
Configurable text chunking with token awareness.

Phase 0 implements a high-quality recursive splitter (inspired by LangChain best practices)
with tiktoken length function. This gives good semantic boundaries for embedding models.

Planned future strategies:
- Semantic chunking (embedding-based breakpoints)
- Agentic / hierarchical chunking
- Structure-aware chunking (respecting headings, lists, code blocks, tables)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig, get_settings
from src.logging_config import logger


@dataclass
class Chunk:
    """A single chunk with rich provenance for citations and evaluation."""

    chunk_id: str
    text: str
    doc_id: str
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Main chunking orchestrator. Strategy selected via config.yaml."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or get_settings().chunking
        self._encoder = None

        if self.config.length_function == "tiktoken":
            self._encoder = tiktoken.get_encoding(self.config.encoding_name)

    def _token_length(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(text.split())  # crude fallback

    def chunk(self, text: str, doc_id: str, page_map: list[tuple[int, str]] | None = None) -> list[Chunk]:
        """
        Split text into chunks according to current strategy.

        page_map: optional list of (page_number, page_text) to compute page ranges per chunk.
        """
        logger.info(
            "Chunking document",
            extra={
                "doc_id": doc_id,
                "strategy": self.config.strategy,
                "chunk_size": self.config.chunk_size_tokens,
                "overlap": self.config.chunk_overlap_tokens,
            },
        )

        if self.config.strategy == "recursive":
            chunks = self._recursive_chunk(text, doc_id)
        else:
            # Placeholder for future strategies
            logger.warning(f"Strategy '{self.config.strategy}' not fully implemented. Falling back to recursive.")
            chunks = self._recursive_chunk(text, doc_id)

        # Attach rough page ranges if page_map provided (best-effort)
        if page_map:
            self._attach_page_ranges(chunks, page_map)

        logger.success("Chunking complete", extra={"doc_id": doc_id, "num_chunks": len(chunks)})
        return chunks

    def _recursive_chunk(self, text: str, doc_id: str) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            separators=self.config.separators,
            chunk_size=self.config.chunk_size_tokens,
            chunk_overlap=self.config.chunk_overlap_tokens,
            length_function=self._token_length,
            keep_separator=self.config.keep_separator,
            strip_whitespace=True,
        )

        raw_chunks = splitter.split_text(text)

        result: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            token_count = self._token_length(chunk_text)
            result.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{idx:04d}",
                    text=chunk_text,
                    doc_id=doc_id,
                    token_count=token_count,
                    metadata={
                        "strategy": "recursive",
                        "chunk_index": idx,
                    },
                )
            )
        return result

    def _attach_page_ranges(self, chunks: list[Chunk], page_map: list[tuple[int, str]]) -> None:
        """Best-effort page number assignment using cumulative text matching."""
        # Simple heuristic: walk pages and assign based on chunk text appearing in page text.
        # For production, consider more robust methods (e.g. storing char offsets during load).
        current_page = 1
        for chunk in chunks:
            # Find first page where a significant prefix of the chunk appears
            prefix = chunk.text[: min(120, len(chunk.text))].strip()
            for page_num, page_text in page_map:
                if prefix and prefix in page_text:
                    chunk.page_start = page_num
                    chunk.page_end = page_num
                    current_page = page_num
                    break
            else:
                # fallback: monotonically increasing
                chunk.page_start = current_page
                chunk.page_end = current_page
