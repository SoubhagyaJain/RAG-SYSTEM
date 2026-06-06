"""
Prompt templates and system instructions.

Phase 0 contains a solid baseline prompt with citation instructions.
These will evolve significantly once we introduce agentic patterns (planning, critique, etc.).
"""

from __future__ import annotations

from src.config import get_settings


def get_system_prompt(name: str | None = None) -> str:
    """Return a named system prompt. Currently only one baseline version."""
    settings = get_settings()
    prompt_name = name or settings.generation.system_prompt_name

    if prompt_name == "rag_system_v1":
        return (
            "You are an expert AI systems researcher and practitioner.\n"
            "You answer questions about building, evaluating, and operating AI agents using only the provided context.\n\n"
            "Rules:\n"
            "1. Use ONLY the information present in the retrieved context. Do not hallucinate.\n"
            "2. When you use information from the context, cite the source using page numbers when available (e.g. [p. 42]).\n"
            "3. If the context does not contain the answer, say so clearly instead of guessing.\n"
            "4. Be precise, structured, and cite specific concepts or frameworks mentioned in the guidebook.\n"
            "5. When relevant, mention trade-offs, common pitfalls, or best practices from the material.\n\n"
            "Format your final answer clearly. Use bullet points or numbered lists when helpful."
        )

    # Default / fallback
    return "You are a helpful assistant. Answer using only the provided context. Cite sources."
