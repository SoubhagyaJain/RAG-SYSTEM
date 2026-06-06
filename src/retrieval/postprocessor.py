"""
Metadata-based Node Postprocessor for bias correction in retrieval.

This postprocessor adjusts node scores after initial retrieval (vector, BM25, Small-to-Big, etc.)
but before the response synthesizer / LLM sees the context.

It is designed to counteract section bias (e.g., over-retrieval of the "6. Memory" section
in the AI Agents Guidebook) by applying multiplicative penalties or boosts based on
node metadata (primarily the "section" field).

Usage in get_query_engine():
    if settings.retrieval.metadata_boosting.enabled:
        postproc = MetadataBoosterPostprocessor(...)
        query_engine = RetrieverQueryEngine(
            ...,
            node_postprocessors=[postproc]
        )
"""

from __future__ import annotations

from typing import List, Optional

# Robust import for BaseNodePostprocessor across LlamaIndex versions.
# In LlamaIndex >= 0.10 the class moved to the .types submodule.
try:
    from llama_index.core.postprocessor import BaseNodePostprocessor
except ImportError:
    try:
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
    except ImportError:
        from llama_index.core.base.postprocessor.base import BaseNodePostprocessor

from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from pydantic import Field

from src.logging_config import logger


class MetadataBoosterPostprocessor(BaseNodePostprocessor):
    """
    A NodePostprocessor that boosts or penalizes retrieved nodes based on their
    metadata (especially the 'section' field).

    Primary use case: Reduce over-representation of the "Memory" section while
    optionally boosting nodes from other pedagogically important sections.
    """

    # Declare as Pydantic fields so the class (which inherits from a Pydantic model)
    # accepts them in __init__ without raising "object has no field".
    memory_section_penalty: float = Field(default=0.6, ge=0.0, le=2.0)
    boost_important_sections: bool = Field(default=True)
    important_section_boost: float = Field(default=1.25, ge=0.5, le=3.0)
    important_sections: List[str] = Field(default_factory=list)

    def __init__(
        self,
        memory_section_penalty: float = 0.6,
        boost_important_sections: bool = True,
        important_sections: Optional[List[str]] = None,
        important_section_boost: float = 1.25,
    ):
        """
        Args:
            memory_section_penalty: Multiplicative factor applied to scores of nodes
                whose section contains "memory" (case-insensitive). Values < 1.0
                penalize (recommended: 0.4 - 0.7).
            boost_important_sections: Whether to apply boosts to important sections.
            important_sections: List of keywords to look for in the section name.
                If None, a sensible default list for the AI Agents Guidebook is used.
            important_section_boost: Multiplicative factor for nodes matching
                important sections (e.g. 1.15 - 1.4).
        """
        if important_sections is None:
            # Default important sections based on the "AI Agents Guidebook" content
            important_sections = [
                "react",
                "tool use",
                "function calling",
                "reflection",
                "self-critique",
                "critique",
                "agentic rag",
                "plan-and-execute",
                "rewoo",
                "hierarchical",
                "multi-agent",
                "guardrails",
                "safety",
                "evaluation",
                "failure modes",
                "prompt",
                "architecture",
            ]
        else:
            important_sections = [s.lower() for s in important_sections]

        super().__init__(
            memory_section_penalty=memory_section_penalty,
            boost_important_sections=boost_important_sections,
            important_section_boost=important_section_boost,
            important_sections=important_sections,
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Adjust scores of nodes in-place based on their 'section' metadata.

        This runs after the retriever (custom SmallToBigRetriever, HybridRetriever, etc.)
        has returned its NodeWithScore list, but before they are passed to the
        response synthesizer.
        """
        if not nodes:
            return nodes

        modified_count = 0
        memory_penalized = 0
        boosted = 0

        for node_with_score in nodes:
            node = node_with_score.node
            meta = getattr(node, "metadata", {}) or {}
            section = str(meta.get("section", "") or "").lower().strip()

            if not section:
                continue

            original_score = node_with_score.score if node_with_score.score is not None else 0.0
            new_score = original_score
            changed = False

            # 1. Penalize the over-represented "Memory" section
            if "memory" in section:
                new_score = original_score * self.memory_section_penalty
                memory_penalized += 1
                changed = True
            # 2. Optionally boost other important sections
            elif self.boost_important_sections:
                for keyword in self.important_sections:
                    if keyword in section:
                        new_score = original_score * self.important_section_boost
                        boosted += 1
                        changed = True
                        break

            if changed:
                # Clamp score to reasonable range [0, 10]
                node_with_score.score = max(0.0, min(new_score, 10.0))
                modified_count += 1

        if modified_count > 0:
            # Re-sort by the adjusted scores (highest first) so the most relevant
            # (after bias correction) nodes come first for the LLM.
            nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

            logger.info(
                "MetadataBoosterPostprocessor applied",
                extra={
                    "total_nodes": len(nodes),
                    "modified": modified_count,
                    "memory_penalized": memory_penalized,
                    "important_boosted": boosted,
                    "memory_penalty": self.memory_section_penalty,
                },
            )

        return nodes

    # Required for some LlamaIndex versions / serialization
    @classmethod
    def class_name(cls) -> str:
        return "MetadataBoosterPostprocessor"
