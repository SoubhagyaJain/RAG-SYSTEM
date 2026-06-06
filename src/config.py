"""
Production-grade configuration loader for Enterprise Agentic RAG (LlamaIndex + Ollama).

Loads config.yaml + .env and exposes a validated Settings object.
LlamaIndex settings are also configured from this module.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Sub-config models
# -----------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "enterprise-rag-ai-agents"
    version: str = "0.1.0"
    description: str = ""


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    artifacts: str = "artifacts"
    logs: str = "logs"

    def resolve(self, base_dir: Path | None = None) -> dict[str, Path]:
        base = base_dir or Path.cwd()
        return {k: (base / v).resolve() for k, v in self.model_dump().items()}


class DocumentConfig(BaseModel):
    primary_pdf: str = "ai_agents_guidebook.pdf"
    source_name: str = "AI Agents Guidebook"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 300

    # LLM - Gemma 4 Latest 8B via Ollama
    llm_model: str = "gemma4:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_request_timeout: int = 300
    llm_num_ctx: int = 8192

    # Embeddings
    embed_model: str = "nomic-embed-text"
    embed_batch_size: int = 64

    # Reranker (future)
    reranker_model: str = "bge-reranker-base"


class VectorStoreConfig(BaseModel):
    """Dedicated config for vector store (Chroma in Phase 1)."""
    backend: Literal["chroma", "faiss", "simple"] = "chroma"
    collection_name: str = "ai_agents_guidebook"
    persist_dir: str = "data/processed/chroma_db"


class LlamaIndexConfig(BaseModel):
    chunk_size: int = 768
    chunk_overlap: int = 120
    similarity_top_k: int = 10
    response_mode: str = "compact"

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    include_metadata: bool = True
    metadata_keys_to_include: list[str] = Field(
        default_factory=lambda: [
            "page_number", "page_label", "section", "source",
            "has_code", "has_diagram", "document_title"
        ]
    )


class IngestionConfig(BaseModel):
    loader: Literal["llama_index", "pypdf", "unstructured"] = "unstructured"
    use_llama_parse: bool = False
    extract_images: bool = False
    extract_tables: bool = True
    force_reingest: bool = False
    min_chunks_threshold: int = 50


class RetrievalConfig(BaseModel):
    top_k: int = 8
    fetch_k: int = 25
    use_hybrid_search: bool = False
    use_reranker: bool = False
    mmr_lambda: float = 0.5


class GenerationConfig(BaseModel):
    system_prompt_version: str = "v1_baseline"
    include_citations: bool = True
    citation_format: Literal["page", "section", "both"] = "page"


class EvaluationConfig(BaseModel):
    enabled: bool = True
    ragas_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    )
    llm_for_judge: str = "gemma4:8b"
    embed_model_for_ragas: str = "nomic-embed-text"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False
    log_to_file: bool = True
    log_file: str = "logs/rag_system.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class Settings(BaseSettings):
    """Root validated settings object."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    llama_index: LlamaIndexConfig = Field(default_factory=LlamaIndexConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    rag_env: Literal["development", "staging", "production"] = "development"

    @field_validator("rag_env", mode="before")
    @classmethod
    def _read_env(cls, v: Any) -> Any:
        return v or os.getenv("RAG_ENV", "development")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config.yaml") -> Settings:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            yaml_path = Path(__file__).parent.parent / "config.yaml"

        yaml_data: dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        return cls(**yaml_data)

    def configure_llama_index(self) -> None:
        """
        Apply settings to global LlamaIndex Settings.
        Call this early in your application / notebook.
        """
        from llama_index.core import Settings as LlamaSettings
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        # LLM (Gemma 4 8B)
        llm_kwargs: dict[str, Any] = {
            "num_predict": self.ollama.llm_max_tokens,
        }
        if hasattr(self.ollama, "llm_num_ctx"):
            llm_kwargs["num_ctx"] = getattr(self.ollama, "llm_num_ctx", 8192)

        LlamaSettings.llm = Ollama(
            model=self.ollama.llm_model,
            base_url=self.ollama.base_url,
            temperature=self.ollama.llm_temperature,
            request_timeout=self.ollama.llm_request_timeout,
            additional_kwargs=llm_kwargs,
        )

        # Embeddings (local via Ollama)
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=self.ollama.embed_model,
            base_url=self.ollama.base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )

        # Chunking / node parsing defaults (used by SentenceSplitter etc.)
        LlamaSettings.chunk_size = self.llama_index.chunk_size
        LlamaSettings.chunk_overlap = self.llama_index.chunk_overlap

        # Retrieval defaults
        LlamaSettings.similarity_top_k = self.llama_index.similarity_top_k

        # Make rich metadata keys available globally if needed
        LlamaSettings.metadata_keys_to_include = self.llama_index.metadata_keys_to_include


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings accessor."""
    return Settings.from_yaml()


# Global instance for convenience (notebooks, scripts)
settings = get_settings()
