"""Enterprise Agentic RAG System (LlamaIndex + Ollama)."""

__version__ = "0.1.0"

from src.config import get_settings, settings
from src.logging_config import logger, setup_logging

__all__ = ["get_settings", "settings", "logger", "setup_logging"]
