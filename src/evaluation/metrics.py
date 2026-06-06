"""
Evaluation metrics scaffolding.

Phase 0 defines the interface and a few simple heuristic metrics.
Real production evaluation (especially for agentic RAG) will be added in Phase 2:
- RAGAS (faithfulness, answer_relevancy, context_precision, etc.)
- Custom citation accuracy
- Human preference / LLM-as-judge
- Cost, latency, and token usage tracking
- Regression testing against golden dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.indexing.vectorstore import RetrievedChunk


@dataclass
class EvaluationResult:
    query: str
    answer: str
    retrieved: list[RetrievedChunk]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_retrieval(
    query: str,
    retrieved: list[RetrievedChunk],
    reference_chunks: list[str] | None = None,
) -> dict[str, float]:
    """
    Very basic retrieval quality signals (placeholder).

    In real usage you would compare against a labeled golden set.
    """
    metrics: dict[str, float] = {
        "num_retrieved": float(len(retrieved)),
        "avg_chunk_length": (
            sum(len(r.text) for r in retrieved) / max(1, len(retrieved))
        ),
    }

    # Example: simple lexical overlap if reference chunks are provided
    if reference_chunks:
        # Placeholder for real overlap / embedding similarity
        metrics["has_reference_overlap"] = 0.0

    return metrics
