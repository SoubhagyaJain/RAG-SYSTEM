"""
Central configuration loader for the Enterprise Agentic RAG System.

Loads:
- config.yaml (base configuration)
- .env (secrets and environment-specific overrides)
- Applies Pydantic validation and type coercion

Usage:
    from src.config import get_settings, Settings

    settings: Settings = get_settings()
    print(settings.chunking.chunk_size_tokens)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "enterprise-rag-ai-agents"
    version: str = "0.1.0"
    description: str = ""


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    vector_store: str = "data/vectorstore"

    def resolve(self, base_dir: Path | None = None) -> dict[str, Path]:
        """Return absolute paths relative to project root or provided base."""
        base = base_dir or Path.cwd()
        return {
            "data_raw": (base / self.data_raw).resolve(),
            "data_processed": (base / self.data_processed).resolve(),
            "vector_store": (base / self.vector_store).resolve(),
        }


class DocumentConfig(BaseModel):
    primary_pdf: str = "ai_agents_guidebook.pdf"
    source: str = "AI Agents Guidebook"


class IngestionConfig(BaseModel):
    loader: Literal["pypdf", "pdfplumber", "unstructured"] = "pypdf"
    extract_images: bool = False
    extract_tables: bool = False
    ocr: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "semantic", "agentic", "hierarchical"] = "recursive"
    chunk_size_tokens: int = Field(800, ge=128, le=8192)
    chunk_overlap_tokens: int = Field(120, ge=0, le=2048)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    keep_separator: bool = True
    length_function: Literal["tiktoken", "char", "word"] = "tiktoken"
    encoding_name: str = "cl100k_base"

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size_tokens", 800)
        if v >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be strictly less than chunk_size_tokens")
        return v


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "cohere", "huggingface", "voyage", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int | None = 1536
    batch_size: int = Field(100, ge=1, le=2048)
    max_retries: int = 5


class VectorStoreConfig(BaseModel):
    backend: Literal["chromadb", "faiss", "pgvector", "pinecone", "weaviate"] = "chromadb"
    collection_name: str = "ai_agents_guidebook"
    persist_directory: str = "data/vectorstore/chroma"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class IndexingConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


class RerankerConfig(BaseModel):
    enabled: bool = False
    provider: str = "cohere"
    model: str = "rerank-english-v3.0"
    top_n: int = 6


class RetrievalConfig(BaseModel):
    top_k: int = Field(8, ge=1, le=50)
    fetch_k: int = Field(30, ge=1, le=100)
    search_type: Literal["similarity", "mmr", "hybrid"] = "similarity"
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "groq", "together", "azure", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1200, ge=64, le=32768)
    top_p: float = 0.95
    timeout: int = 60


class GenerationConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt_version: str = "v1_baseline"
    system_prompt_name: str = "rag_system_v1"
    include_citations: bool = True
    citation_style: Literal["page", "section", "both"] = "page"


class MetricConfig(BaseModel):
    name: str
    enabled: bool = True


class EvaluationConfig(BaseModel):
    enabled: bool = True
    frameworks: list[str] = Field(default_factory=lambda: ["custom"])
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(name="context_relevance"),
            MetricConfig(name="faithfulness"),
            MetricConfig(name="answer_relevance"),
            MetricConfig(name="citation_accuracy"),
        ]
    )
    golden_dataset: str = "data/eval/golden_dataset.jsonl"
    track_cost: bool = True
    track_latency: bool = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False
    log_to_file: bool = True
    log_file: str = "logs/rag_system.log"
    rotation: str = "10 MB"
    retention: str = "14 days"


class Settings(BaseSettings):
    """Root settings object. Combines YAML config + environment variables."""

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
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime
    rag_env: Literal["development", "staging", "production"] = "development"

    @field_validator("rag_env", mode="before")
    @classmethod
    def _read_rag_env(cls, v: Any) -> Any:
        return v or os.getenv("RAG_ENV", "development")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config.yaml") -> Settings:
        """Load base configuration from YAML then overlay with Pydantic settings (.env)."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            # Fallback to default in-package config if running from src
            yaml_path = Path(__file__).parent.parent / "config.yaml"

        yaml_config: dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}

        # Instantiate with yaml values; Pydantic will also read .env on top
        return cls(**yaml_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call this everywhere instead of instantiating directly."""
    return Settings.from_yaml()


# Convenience for notebooks / scripts
settings = get_settings()
