"""
PDF and document loaders with rich metadata extraction.

Phase 0 focuses on clean, page-aware text extraction from the AI Agents Guidebook.
Future phases will add:
- Image / table extraction
- OCR fallback
- Multi-format support (Markdown, HTML, DOCX, web pages, Notion, Confluence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from src.config import get_settings
from src.logging_config import logger


@dataclass
class Document:
    """Canonical document representation after loading."""

    doc_id: str
    source: str
    pages: list[dict[str, Any]] = field(default_factory=list)  # {page_num, text, metadata}
    total_pages: int = 0
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def get_full_text(self) -> str:
        return "\n\n".join(p["text"] for p in self.pages if p.get("text"))


class DocumentLoader(Protocol):
    """Interface for all future document loaders."""

    def load(self, path: Path) -> Document: ...


class PDFLoader:
    """Production PDF loader using pypdf with page-level metadata."""

    def __init__(self, extract_images: bool = False, extract_tables: bool = False) -> None:
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.settings = get_settings()

    def load(self, path: Path | str) -> Document:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Loading PDF", extra={"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 2)})

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        pages: list[dict[str, Any]] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1}", extra={"error": str(e)})
                text = ""

            # Clean common PDF artifacts
            text = self._clean_text(text)

            page_meta = {
                "page_num": i + 1,
                "text": text,
                "char_count": len(text),
                "metadata": {
                    "producer": reader.metadata.producer if reader.metadata else None,
                },
            }
            pages.append(page_meta)

        doc = Document(
            doc_id=path.stem,
            source=str(path),
            pages=pages,
            total_pages=total_pages,
            raw_metadata={
                "title": reader.metadata.title if reader.metadata else None,
                "author": reader.metadata.author if reader.metadata else None,
                "subject": reader.metadata.subject if reader.metadata else None,
                "creator": reader.metadata.creator if reader.metadata else None,
            },
        )

        logger.info(
            "PDF loaded successfully",
            extra={
                "doc_id": doc.doc_id,
                "total_pages": total_pages,
                "total_chars": sum(p["char_count"] for p in pages),
            },
        )
        return doc

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic PDF text cleanup. Extend as needed."""
        # Collapse excessive whitespace while preserving paragraph breaks
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        return text
