"""
Production-grade Ingestion Pipeline for the AI Agents Guidebook.

Uses:
- UnstructuredReader (excellent for complex/illustrated PDFs with layout)
- LlamaIndex IngestionPipeline + SentenceSplitter
- Chroma as persistent vector store
- Rich per-node metadata (page, section, has_code, has_diagram, etc.)

Key production features:
- Idempotent (skips if collection already contains sufficient nodes)
- Fully driven by config.yaml
- Detailed structured logging
- Clean metadata for downstream retrieval & evaluation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import Document, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.readers.unstructured import UnstructuredReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import get_settings
from src.logging_config import logger


class GuidebookIngestionPipeline:
    """
    Orchestrates loading the AI Agents Guidebook, chunking it, enriching metadata,
    and persisting to a Chroma vector store.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.paths = self.settings.paths.resolve()
        self.ingestion_cfg = self.settings.ingestion
        self.vector_store_cfg = self.settings.llama_index.vector_store

        # Resolve final Chroma location (prefer top-level vector_store if present)
        if hasattr(self.settings, "vector_store") and self.settings.vector_store:
            self.persist_dir = Path(self.settings.vector_store.persist_dir)
            self.collection_name = self.settings.vector_store.collection_name
        else:
            self.persist_dir = Path(self.vector_store_cfg.persist_dir)
            self.collection_name = self.vector_store_cfg.collection_name

        self.pdf_path = self.paths["data_raw"] / self.settings.document.primary_pdf

        # Ensure directories exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.paths["data_processed"].mkdir(parents=True, exist_ok=True)

    def _get_chroma_vector_store(self) -> ChromaVectorStore:
        """Create or connect to the persistent Chroma collection."""
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return ChromaVectorStore(chroma_collection=chroma_collection)

    def _check_already_ingested(self) -> tuple[bool, int]:
        """
        Idempotency check.
        Returns (already_done, current_node_count)
        """
        try:
            chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            threshold = self.ingestion_cfg.min_chunks_threshold
            if count >= threshold and not self.ingestion_cfg.force_reingest:
                return True, count
            return False, count
        except Exception:
            # Collection doesn't exist yet or other error → not ingested
            return False, 0

    def load_documents(self) -> list[Document]:
        """
        Load the PDF using UnstructuredReader.

        UnstructuredReader is chosen because it does a much better job than pypdf
        at preserving layout, headings, tables, and code blocks in illustrated technical PDFs.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        logger.info(
            "Loading PDF with UnstructuredReader",
            extra={
                "pdf_path": str(self.pdf_path),
                "loader": "unstructured",
                "extract_tables": self.ingestion_cfg.extract_tables,
            },
        )

        reader = UnstructuredReader()
        documents = reader.load_data(
            file=str(self.pdf_path),
            # split_documents=True would give element-level docs; we prefer full pages then split
        )

        # Attach high-level document metadata
        doc_title = self.settings.document.source_name
        for doc in documents:
            doc.metadata.setdefault("document_title", doc_title)
            doc.metadata.setdefault("source", str(self.pdf_path.name))

        logger.success(
            "Documents loaded",
            extra={"num_documents": len(documents)},
        )
        return documents

    def _infer_section(self, node: BaseNode, last_heading: str | None = None) -> str | None:
        """
        Try to extract a section/heading from the node metadata or text.

        UnstructuredReader often includes 'element_type' or heading information
        in metadata when the PDF has structure.
        """
        meta = node.metadata or {}

        # Common keys from Unstructured + LlamaIndex
        for key in ["section", "heading", "title", "Header", "element_type"]:
            if key in meta and meta[key]:
                val = str(meta[key]).strip()
                if val and len(val) < 120:
                    return val

        # Fallback: look for markdown-style or title-like first line in text
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if text:
            first_line = text.split("\n", 1)[0].strip()
            # Heuristic: short lines that look like headings
            if 5 < len(first_line) < 90 and not first_line.endswith("."):
                # Very rough heading detection
                if re.match(r"^(Chapter|Section|\d+\.|#|•|\*\*)", first_line, re.IGNORECASE):
                    return first_line[:100]

        return last_heading

    def _detect_has_code(self, text: str) -> bool:
        """Detect presence of code blocks or code-like content."""
        if not text:
            return False
        code_patterns = [
            r"```",                     # markdown code fence
            r"^\s{4,}",                 # indented block (common in PDFs)
            r"\b(def |class |import |from .* import|function |const |let |var )",
            r"```(python|bash|yaml|json|javascript)",
        ]
        return any(re.search(p, text, re.MULTILINE | re.IGNORECASE) for p in code_patterns)

    def _detect_has_diagram(self, text: str, metadata: dict[str, Any]) -> bool:
        """
        Best-effort detection of diagrams/figures/illustrations.

        Unstructured can surface image elements. We also look for common terms
        in the AI Agents Guidebook.
        """
        if not text:
            return False

        # Direct metadata signals from Unstructured
        element_type = str(metadata.get("element_type", "")).lower()
        if any(x in element_type for x in ["image", "figure", "diagram", "picture"]):
            return True

        # Textual signals
        diagram_keywords = [
            "diagram", "figure", "architecture", "flow", "workflow",
            "illustration", "schematic", "overview diagram", "component diagram",
            "agent architecture", "system diagram"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in diagram_keywords)

    def enrich_metadata(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """
        Add rich, consistent metadata to every node.

        This metadata is extremely valuable for:
        - Citation quality
        - Filtering / routing in agentic systems
        - Evaluation (Ragas + custom)
        """
        logger.info("Enriching node metadata", extra={"num_nodes": len(nodes)})

        doc_title = self.settings.document.source_name
        source_file = self.settings.document.primary_pdf
        last_heading: str | None = None

        for node in nodes:
            meta = node.metadata or {}

            # Document level
            meta["document_title"] = doc_title
            meta["source"] = source_file

            # Page number (Unstructured + LlamaIndex often put it in page_label or page_number)
            page_num = meta.get("page_number") or meta.get("page_label")
            if page_num is not None:
                try:
                    meta["page_number"] = int(str(page_num).replace("p.", "").strip())
                except (ValueError, TypeError):
                    meta["page_number"] = page_num
            else:
                meta["page_number"] = None

            # Section / heading inference
            section = self._infer_section(node, last_heading)
            if section:
                last_heading = section
            meta["section"] = section or "Unknown"

            # Code & Diagram flags
            text_content = node.get_content(metadata_mode=MetadataMode.NONE)
            meta["has_code"] = self._detect_has_code(text_content)
            meta["has_diagram"] = self._detect_has_diagram(text_content, meta)

            # Clean up some noisy unstructured keys if present
            for noisy_key in ["element_id", "filename", "file_directory"]:
                meta.pop(noisy_key, None)

            node.metadata = meta

        logger.success(
            "Metadata enrichment complete",
            extra={
                "sections_captured": len({n.metadata.get("section") for n in nodes if n.metadata.get("section")}),
            },
        )
        return nodes

    def run(self, force: bool | None = None) -> int:
        """
        Execute the full ingestion pipeline.

        Returns the number of nodes inserted into the vector store.
        """
        if force is None:
            force = self.ingestion_cfg.force_reingest

        already_done, existing_count = self._check_already_ingested()
        if already_done and not force:
            logger.info(
                "Ingestion skipped (idempotent)",
                extra={
                    "collection": self.collection_name,
                    "existing_nodes": existing_count,
                    "threshold": self.ingestion_cfg.min_chunks_threshold,
                },
            )
            return existing_count

        if force:
            logger.warning("Force re-ingestion requested. Existing data will be replaced.")

        # 1. Load
        documents = self.load_documents()

        # 2. Build transformations (chunking)
        splitter = SentenceSplitter(
            chunk_size=self.settings.llama_index.chunk_size,
            chunk_overlap=self.settings.llama_index.chunk_overlap,
        )

        pipeline = IngestionPipeline(
            transformations=[splitter],
            # We will add nodes manually after metadata enrichment for more control
        )

        logger.info(
            "Running chunking pipeline",
            extra={
                "chunk_size": self.settings.llama_index.chunk_size,
                "chunk_overlap": self.settings.llama_index.chunk_overlap,
            },
        )

        # Run splitter to get nodes
        nodes = pipeline.run(documents=documents, show_progress=True)
        logger.success("Chunking complete", extra={"num_nodes": len(nodes)})

        # 3. Enrich with rich metadata (critical for this guidebook)
        nodes = self.enrich_metadata(nodes)

        # 4. Persist to Chroma
        vector_store = self._get_chroma_vector_store()

        # Clear collection if force reingest
        if force:
            try:
                vector_store.client.delete_collection(self.collection_name)
                vector_store = self._get_chroma_vector_store()
            except Exception:
                pass

        logger.info(
            "Inserting nodes into Chroma",
            extra={
                "collection": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "num_nodes": len(nodes),
            },
        )

        # Use StorageContext + VectorStoreIndex pattern or direct add
        # Direct insert via vector_store is efficient
        vector_store.add(nodes)

        final_count = len(nodes)
        logger.success(
            "Ingestion complete",
            extra={
                "nodes_inserted": final_count,
                "collection": self.collection_name,
                "persist_dir": str(self.persist_dir),
            },
        )
        return final_count


def run_ingestion(force: bool = False) -> int:
    """Convenience function to run ingestion from scripts/notebooks."""
    pipeline = GuidebookIngestionPipeline()
    return pipeline.run(force=force)
