"""
Modular Retrieval Strategies for the Educational RAG System.

This module provides different retrieval modes that can be selected via config.yaml
under `retrieval.mode`:

- "vector": Standard vector similarity search on leaf nodes.
- "small_to_big": Retrieve small leaves, then expand to the actual parent node
  from the docstore (real parent context from HierarchicalNodeParser).
- "hybrid": Combine vector search with BM25 keyword search using QueryFusionRetriever.

All retrievers are designed to work with the rich metadata produced by
SectionAwareHierarchicalChunking (page_number, section, content_type, parent_id, etc.).

The SmallToBigRetriever here uses *real parent nodes* (fetched via docstore)
instead of merging sibling leaves, providing cleaner, more coherent context
to the LLM for educational answers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode

from src.logging_config import logger

# BM25Retriever is optional (requires llama-index-retrievers-bm25).
# We import it lazily inside get_retriever when mode="hybrid".
BM25Retriever = None
try:
    from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
    BM25Retriever = _BM25Retriever
except ImportError:
    pass  # Will be handled with a clear error in get_retriever if hybrid is requested without the package.


class VectorRetriever(BaseRetriever):
    """Simple wrapper around the vector retriever for consistency."""

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 10):
        super().__init__()
        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)


class SmallToBigRetriever(BaseRetriever):
    """
    Real Small-to-Big retrieval using actual parent nodes from the hierarchy.

    Process:
    1. Retrieve `small_top_k` leaf nodes using vector similarity (precise matches).
    2. For each leaf, look up its `parent_id` from metadata.
    3. Fetch the *actual parent node* from the docstore (the 512-token section
       created by HierarchicalNodeParser during ingestion).
    4. Return the parent nodes as the "big" contexts for generation.

    This is superior to merging siblings because the parent node is a
    coherent, self-contained section written by the author (or the hierarchical
    parser at that level).

    Requires that ingestion stored the parent nodes in the docstore
    (see updated ingestion_pipeline.py).
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        docstore: Any,  # BaseDocumentStore or compatible
        small_top_k: int = 15,
        max_big_nodes: int = 6,
    ):
        super().__init__()
        self._base_retriever = base_retriever
        self._docstore = docstore
        self._small_top_k = small_top_k
        self._max_big_nodes = max_big_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 1. Retrieve more small leaves for good coverage
        small_nodes = self._base_retriever.retrieve(query_bundle)[: self._small_top_k]

        if not small_nodes:
            return []

        # 2. Group by parent to avoid duplicate big contexts
        parent_groups: dict[str, list[NodeWithScore]] = defaultdict(list)
        for ns in small_nodes:
            pid = ns.node.metadata.get("parent_id") or "root"
            parent_groups[pid].append(ns)

        big_nodes: list[NodeWithScore] = []

        for pid, group in parent_groups.items():
            if pid == "root":
                # No parent - use the best small node as-is
                best = max(group, key=lambda x: x.score or 0.0)
                big_nodes.append(best)
                continue

            # 3. Fetch the real parent node from docstore
            try:
                parent_node: BaseNode = self._docstore.get_document(pid)
                parent_text = parent_node.get_content()

                # Use the highest-scoring leaf's metadata for citations
                best = max(group, key=lambda x: x.score or 0.0)
                big_metadata = parent_node.metadata.copy()
                big_metadata.update({
                    "retrieval_mode": "small-to-big",
                    "parent_id": pid,
                    "num_supporting_leaves": len(group),
                    "supporting_leaf_pages": sorted({
                        m.get("page_number")
                        for m in [n.node.metadata for n in group]
                        if m.get("page_number") is not None
                    }),
                })

                big_node = TextNode(
                    text=parent_text,
                    metadata=big_metadata,
                    id_=f"parent-{pid}",
                )
                big_nodes.append(NodeWithScore(node=big_node, score=best.score))
            except Exception as e:
                logger.warning(f"Could not fetch parent {pid} from docstore: {e}. Falling back to merged leaves.")
                # Fallback: merge siblings (previous behavior)
                group.sort(key=lambda x: x.node.metadata.get("chunk_index") or 0)
                merged = "\n\n".join(n.node.get_content() for n in group)
                best = max(group, key=lambda x: x.score or 0.0)
                big_metadata = best.node.metadata.copy()
                big_metadata["retrieval_mode"] = "small-to-big-fallback"
                big_node = TextNode(text=merged, metadata=big_metadata)
                big_nodes.append(NodeWithScore(node=big_node, score=best.score))

        # Return top big contexts
        big_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return big_nodes[: self._max_big_nodes]


class HybridRetriever(BaseRetriever):
    """
    Hybrid Search = Vector + BM25 using QueryFusionRetriever.

    This combines semantic (vector) and lexical (BM25) signals.
    Very effective for technical documents with specific terminology
    (e.g. "ReAct", "agentic workflow", "tool use").
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: "BM25Retriever",
        similarity_top_k: int = 8,
        fusion_mode: str = "reciprocal_rerank",
    ):
        super().__init__()
        self._fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1,
            mode=fusion_mode,
            use_async=False,
            verbose=False,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._fusion_retriever.retrieve(query_bundle)


def get_retriever(
    index: VectorStoreIndex,
    mode: str = "small_to_big",
    similarity_top_k: int = 10,
    small_to_big_top_k: int = 15,
    hybrid_fusion_mode: str = "reciprocal_rerank",
    bm25_top_k: int = 10,
    **kwargs,
) -> BaseRetriever:
    """
    Factory to get the desired retriever based on configuration.

    This makes retrieval fully modular and switchable via config.yaml
    without changing query_engine.py logic.
    """
    if mode == "vector":
        logger.info("Using pure vector retrieval")
        return index.as_retriever(similarity_top_k=similarity_top_k)

    elif mode == "small_to_big":
        logger.info("Using Small-to-Big retrieval (real parent nodes)")
        base_retriever = index.as_retriever(similarity_top_k=small_to_big_top_k)
        return SmallToBigRetriever(
            base_retriever=base_retriever,
            docstore=index.docstore,
            small_top_k=small_to_big_top_k,
            max_big_nodes=similarity_top_k,
        )

    elif mode == "hybrid":
        logger.info(f"Using Hybrid (Vector + BM25) with fusion_mode={hybrid_fusion_mode}")

        if BM25Retriever is None:
            raise ImportError(
                "Hybrid search requires 'llama-index-retrievers-bm25'. "
                "Install with: pip install llama-index-retrievers-bm25"
            )

        vector_retriever = index.as_retriever(similarity_top_k=bm25_top_k)

        # Build BM25 on the leaf nodes for lexical search on fine-grained chunks
        leaf_nodes = [
            node for node in index.docstore.docs.values()
            if node.metadata.get("chunk_index") is not None
        ] or list(index.docstore.docs.values())

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=leaf_nodes,
            similarity_top_k=bm25_top_k,
        )

        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            similarity_top_k=similarity_top_k,
            fusion_mode=hybrid_fusion_mode,
        )

    else:
        logger.warning(f"Unknown retrieval mode '{mode}', falling back to vector")
        return index.as_retriever(similarity_top_k=similarity_top_k)
