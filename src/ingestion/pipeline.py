"""
High-level ingestion orchestration.

Phase 0: Simple "load PDF → emit Document" pipeline.
Later phases will add:
- Parallel loading
- Preprocessing (dedup, boilerplate removal, PII redaction)
- Multi-document merging + versioning
- Change detection / incremental ingestion
"""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.ingestion.loaders import Document, PDFLoader
from src.logging_config import logger


class IngestionPipeline:
    """Orchestrates document loading and basic preprocessing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.loader = PDFLoader(
            extract_images=self.settings.ingestion.extract_images,
            extract_tables=self.settings.ingestion.extract_tables,
        )

    def run(self, pdf_path: Path | str | None = None) -> Document:
        """Execute the ingestion stage."""
        if pdf_path is None:
            raw_dir = self.settings.paths.resolve()["data_raw"]
            pdf_path = raw_dir / self.settings.document.primary_pdf

        logger.info("Starting ingestion pipeline", extra={"pdf_path": str(pdf_path)})

        doc = self.loader.load(pdf_path)

        # Attach project-level metadata from config
        for page in doc.pages:
            page["metadata"].update(self.settings.ingestion.metadata)

        logger.success(
            "Ingestion complete",
            extra={"doc_id": doc.doc_id, "pages": doc.total_pages},
        )
        return doc
