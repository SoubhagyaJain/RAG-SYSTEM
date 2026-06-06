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
    vector_store: str = "data/vectorstore"
    artifacts: str = "artifacts"

    def resolve(self, base_dir: Path | None = None) -> dict[str, Path]:
        base = base_dir or Path.cwd()
        return {k: (base / v).resolve() for k, v in self.model_dump().items()}


class DocumentConfig(BaseModel):
    primary_pdf: str = "ai_agents_guidebook.pdf"
    source_name: str = "AI Agents Guidebook"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 300

    # LLM
    llm_model: str = "gemma2:27b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_request_timeout: int = 300

    # Embeddings
    embed_model: str = "nomic-embed-text"
    embed_batch_size: int = 50

    # Reranker (future)
    reranker_model: str = "bge-reranker-base"


class LlamaIndexConfig(BaseModel):
    chunk_size: int = 1024
    chunk_overlap: int = 150
    similarity_top_k: int = 8
    response_mode: str = "compact"

    vector_store: dict[str, Any] = Field(
        default_factory=lambda: {
            "backend": "chroma",
            "collection_name": "ai_agents_guidebook",
            "persist_dir": "data/vectorstore/chroma",
        }
    )
    include_metadata: bool = True
    metadata_keys_to_include: list[str] = Field(
        default_factory=lambda: ["page_label", "file_name", "source"]
    )


class IngestionConfig(BaseModel):
    loader: Literal["llama_index", "pypdf", "unstructured"] = "llama_index"
    use_llama_parse: bool = False
    extract_images: bool = False
    extract_tables: bool = False


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
    llm_for_judge: str = "gemma2:27b"
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

        # LLM
        LlamaSettings.llm = Ollama(
            model=self.ollama.llm_model,
            base_url=self.ollama.base_url,
            temperature=self.ollama.llm_temperature,
            request_timeout=self.ollama.llm_request_timeout,
            additional_kwargs={"num_predict": self.ollama.llm_max_tokens},
        )

        # Embeddings
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=self.ollama.embed_model,
            base_url=self.ollama.base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )

        # Chunking / node parsing defaults
        LlamaSettings.chunk_size = self.llama_index.chunk_size
        LlamaSettings.chunk_overlap = self.llama_index.chunk_overlap

        # Retrieval defaults
        LlamaSettings.similarity_top_k = self.llama_index.similarity_top_k


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings accessor."""
    return Settings.from_yaml()


# Global instance for convenience (notebooks, scripts)
settings = get_settings()
