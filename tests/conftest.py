"""
Pytest configuration and shared fixtures for the RAG system.

Add common fixtures here (settings overrides, temp vector stores, sample docs, etc.).
"""

import pytest

from src.config import Settings, get_settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide test-specific settings (override paths, use smaller models, etc.)."""
    # You can load a test config or patch here
    return get_settings()


@pytest.fixture
def sample_text() -> str:
    return (
        "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models. "
        "It allows language models to access external knowledge at inference time. "
        "Agentic RAG adds planning, tool use, and iterative refinement on top of classic RAG."
    )
