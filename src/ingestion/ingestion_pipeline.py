"""
Production-grade Ingestion Pipeline for the AI Agents Guidebook.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import get_settings
from src.logging_config import logger


class SectionAwareHierarchicalChunking:
    """Clean PDF page text, preserve page metadata, and create hierarchical chunks."""

    CHUNK_SIZES = [2048, 512, 128]
    CHUNK_OVERLAP = 100

    HEADER_FOOTER_PATTERNS = [
        r"(?im)^\s*DailyDoseofDS\.com\s*$",
        r"(?im)^\s*FREE AI AGENTS 2025 EDITION\s*$",
        r"(?im)^\s*THE ILLUSTRATED GUIDEBOOK\s*$",
    ]

    def __init__(self, document_title: str, source: str) -> None:
        self.document_title = document_title
        self.source = source
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.CHUNK_SIZES,
            chunk_overlap=self.CHUNK_OVERLAP,
            include_metadata=True,
        )

    def clean_text(self, text: str) -> str:
        """Remove PDF layout noise while keeping headings and local paragraph structure."""
        if not text:
            return ""

        cleaned = text.replace("\x00", " ")
        for pattern in self.HEADER_FOOTER_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned)

        cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
        return cleaned.strip()

    def _coerce_page_number(self, metadata: dict[str, Any]) -> int | None:
        page = metadata.get("page_number") or metadata.get("page_label")
        if page is None:
            return None
        match = re.search(r"\d+", str(page))
        return int(match.group(0)) if match else None

    def prepare_documents(self, documents: list[Document]) -> list[Document]:
        prepared: list[Document] = []

        for fallback_page, doc in enumerate(documents, 1):
            raw_meta = doc.metadata or {}
            page_number = self._coerce_page_number(raw_meta) or fallback_page
            text = self.clean_text(doc.get_content(metadata_mode=MetadataMode.NONE))

            if not text:
                logger.warning(f"Skipping empty page during ingestion: page {page_number}")
                continue

            prepared.append(
                Document(
                    text=text,
                    id_=f"{self.source}_page_{page_number}",
                    metadata={
                        "page_number": page_number,
                        "page_label": str(raw_meta.get("page_label") or page_number),
                        "document_title": self.document_title,
                        "source": self.source,
                        "file_name": self.source,
                    },
                )
            )

        return prepared

    def build_nodes(self, documents: list[Document]) -> tuple[list[BaseNode], list[BaseNode]]:
        prepared = self.prepare_documents(documents)
        all_nodes = self.parser.get_nodes_from_documents(prepared, show_progress=True)
        leaf_nodes = get_leaf_nodes(all_nodes)
        return all_nodes, leaf_nodes


class GuidebookIngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()

        from src.config import _find_project_root
        project_root = _find_project_root()

        vs = self.settings.llama_index.vector_store
        self.persist_dir = project_root / vs.persist_dir
        self.collection_name = vs.collection_name
        self.pdf_path = project_root / self.settings.paths.data_raw / self.settings.document.primary_pdf

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDF path resolved to: {self.pdf_path}")

    def _get_chroma_vector_store(self):
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return ChromaVectorStore(chroma_collection=chroma_collection)

    def _check_already_ingested(self) -> tuple[bool, int]:
        try:
            chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            threshold = getattr(self.settings.ingestion, "min_chunks_threshold", 100)
            force = getattr(self.settings.ingestion, "force_reingest", False)

            if count >= threshold and not force:
                return True, count
            return False, count
        except Exception:
            return False, 0

    def load_documents(self) -> list[Document]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        logger.info(f"Loading PDF from: {self.pdf_path}")

        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(self.pdf_path)],
            filename_as_id=True,
        )
        documents = reader.load_data()

        for doc in documents:
            doc.metadata.setdefault("document_title", self.settings.document.source_name)
            doc.metadata.setdefault("source", self.settings.document.primary_pdf)
            doc.metadata.setdefault("file_name", self.pdf_path.name)

        logger.success(f"Loaded {len(documents)} pages successfully")
        return documents

    # ==================== IMPROVED SECTION INFERENCE ====================
    def _infer_section(self, node: BaseNode, last_heading: str | None = None) -> str | None:
        """Robust section inference. Prevents '6. Memory' leakage into other sections."""
        meta = node.metadata or {}

        # Prefer parser-provided headings
        for key in ["section", "heading", "title", "Header", "Header_1", "Header_2"]:
            if key in meta and meta[key]:
                val = str(meta[key]).strip()
                if val and len(val) < 130:
                    return val

        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if not text:
            return last_heading

        first_lines = [l.strip() for l in text.split("\n")[:4] if l.strip()]

        # Known major sections from the PDF TOC
        MAJOR_SECTIONS = [
            "Building blocks of AI Agents",
            "5 Agentic AI Design Patterns",
            "5 Levels of Agentic AI Systems",
            "AI Agents Projects",
            "What is an AI Agent?",
            "Agent vs LLM vs RAG",
        ]

        for line in first_lines:
            clean_line = line[:130]

            if re.match(r"^(Chapter|Section|Part|#?\d+[\.\)]|\d+\.\d+)", clean_line, re.IGNORECASE):
                return clean_line

            for major in MAJOR_SECTIONS:
                if major.lower() in clean_line.lower():
                    return major

            if (
                clean_line.istitle()
                and 3 < len(clean_line.split()) < 12
                and not clean_line.endswith((".", ":", ";", ","))
            ):
                return clean_line

        # Conservative carry-forward — do not keep propagating "Memory"
        if last_heading and "Memory" in last_heading:
            return None

        return last_heading

    def _detect_has_code(self, text: str) -> bool:
        if not text:
            return False
        patterns = [r"```", r"^\s{4,}", r"\b(def |class |import |from .* import)"]
        return any(re.search(p, text, re.MULTILINE | re.IGNORECASE) for p in patterns)

    def _detect_has_diagram(self, text: str, metadata: dict[str, Any]) -> bool:
        if not text:
            return False
        element = str(metadata.get("element_type", "")).lower()
        if any(x in element for x in ["image", "figure", "diagram"]):
            return True
        keywords = ["diagram", "figure", "architecture", "flow", "workflow"]
        return any(kw in text.lower() for kw in keywords)

    def _detect_content_type(self, text: str, meta: dict[str, Any]) -> str:
        if not text:
            return "general"
        t = text.lower()
        if meta.get("has_code") or "```" in text or re.search(r"\b(def |class |import |from .* import)", text):
            return "code"
        if meta.get("has_diagram") or any(k in t for k in ["diagram", "figure", "architecture diagram", "flowchart", "schematic"]):
            return "diagram"
        if any(k in t for k in ["definition", "is defined as", "refers to the", "a [a-z]+ is a"]):
            return "definition"
        if any(k in t for k in ["for example", "e.g.", "example:", "consider the following", "suppose"]):
            return "example"
        if any(k in t for k in ["architecture", "the following components", "system consists of", "layers of"]):
            return "architecture"
        if any(k in t for k in ["workflow", "the process", "steps:", "pipeline", "sequence of"]):
            return "workflow"
        return "general"

    def enrich_metadata(self, nodes: list[BaseNode]) -> list[BaseNode]:
        logger.info(f"Enriching metadata for {len(nodes)} nodes")
        last_heading = None

        for node in nodes:
            meta = node.metadata or {}
            meta["document_title"] = self.settings.document.source_name
            meta["source"] = self.settings.document.primary_pdf

            page = meta.get("page_number") or meta.get("page_label")
            meta["page_number"] = int(str(page).replace("p.", "").strip()) if page else None

            # === IMPROVED SECTION ASSIGNMENT ===
            section = self._infer_section(node, last_heading)
            if section:
                last_heading = section
            else:
                # Try recovering from parent node
                rels = getattr(node, "relationships", {}) or {}
                parent = rels.get(NodeRelationship.PARENT)
                if parent and hasattr(parent, "metadata"):
                    section = parent.metadata.get("section")

            meta["section"] = section or last_heading or "Unknown"

            text = node.get_content(metadata_mode=MetadataMode.NONE)
            meta["has_code"] = self._detect_has_code(text)
            meta["has_diagram"] = self._detect_has_diagram(text, meta)
            meta["content_type"] = self._detect_content_type(text, meta)

            rels = getattr(node, "relationships", {}) or {}
            parent = rels.get(NodeRelationship.PARENT)
            meta["parent_id"] = getattr(parent, "node_id", None) if parent else None

            node.metadata = meta

        logger.success("Metadata enrichment completed")
        return nodes

    def run(self, force: bool | None = None) -> int:
        if force is None:
            force = getattr(self.settings.ingestion, "force_reingest", False)

        self.settings.configure_llama_index()

        already_done, count = self._check_already_ingested()
        if already_done and not force:
            logger.info(f"Ingestion skipped. Already have {count} nodes.")
            return count

        if force:
            logger.warning("Force re-ingestion enabled.")

        documents = self.load_documents()

        chunker = SectionAwareHierarchicalChunking(
            document_title=self.settings.document.source_name,
            source=self.settings.document.primary_pdf,
        )
        all_nodes, leaf_nodes = chunker.build_nodes(documents)
        logger.info(f"Hierarchical chunking produced {len(leaf_nodes)} leaf nodes")

        nodes = self.enrich_metadata(leaf_nodes)

        for i, node in enumerate(nodes):
            node.metadata = node.metadata or {}
            node.metadata["chunk_index"] = i

        # ==================== FIXED: Removed aggressive truncation ====================
        # We now use a much higher safe limit (or none) so we don't destroy context before embedding.
        MAX_EMBED_CHARS = 8000
        for node in nodes:
            text = node.get_content()
            if len(text) > MAX_EMBED_CHARS:
                node.text = text[:MAX_EMBED_CHARS]
                logger.warning(f"Truncated very long node to {MAX_EMBED_CHARS} chars")

        vector_store = self._get_chroma_vector_store()

        if force:
            try:
                vector_store.client.delete_collection(self.collection_name)
                vector_store = self._get_chroma_vector_store()
            except Exception:
                pass

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(all_nodes)

        VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        try:
            hierarchy_path = self.persist_dir / "hierarchy_nodes.json"
            node_dicts = [n.to_dict() for n in all_nodes]
            import json
            with open(hierarchy_path, "w", encoding="utf-8") as f:
                json.dump(node_dicts, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved hierarchy sidecar ({len(node_dicts)} nodes)")
        except Exception as e:
            logger.warning(f"Failed to save hierarchy sidecar: {e}")

        logger.success(f"Ingestion completed. Total leaf nodes: {len(nodes)}")
        return len(nodes)


def run_ingestion(force: bool = False) -> int:
    pipeline = GuidebookIngestionPipeline()
    return pipeline.run(force=force)