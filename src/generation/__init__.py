"""Generation (LLM) package foundations."""

from src.generation.generator import RAGGenerator
from src.generation.prompts import get_system_prompt

__all__ = ["RAGGenerator", "get_system_prompt"]
