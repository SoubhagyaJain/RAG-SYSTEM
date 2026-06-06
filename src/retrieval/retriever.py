"""
Modular Retrieval Strategies for the Educational RAG System.

Supported modes (via config.yaml → retrieval.mode):
- "vector"       : Standard vector similarity
- "small_to_big" : Retrieve leaves → expand to real parent nodes from docstore
- "hybrid"       : Vector + BM25 (QueryFusionRetriever)

This version has improved robustness in Hybrid mode.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode

from src.logging_config import logger

# BM25Retriever (optional)
BM25Retriever = None
try:
    from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
    BM25Retriever = _BM25Retriever
except ImportError:
    pass


class VectorRetriever(BaseRetriever):
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 10):
        super().__init__()
        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)


class SmallToBigRetriever(BaseRetriever):
    """
    Retrieves small leaf nodes, then expands to the actual parent node
    stored in the docstore during ingestion.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        docstore: Any,
        small_top_k: int = 15,
        max_big_nodes: int = 6,
    ):
        super().__init__()
        self._base_retriever = base_retriever
        self._docstore = docstore
        self._small_top_k = small_top_k
        self._max_big_nodes = max_big_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        small_nodes = self._base_retriever.retrieve(query_bundle)[: self._small_top_k]

        if not small_nodes:
            return []

        parent_groups: dict[str, list[NodeWithScore]] = defaultdict(list)
        for ns in small_nodes:
            pid = ns.node.metadata.get("parent_id") or "root"
            parent_groups[pid].append(ns)

        big_nodes: list[NodeWithScore] = []

        for pid, group in parent_groups.items():
            if pid == "root":
                best = max(group, key=lambda x: x.score or 0.0)
                big_nodes.append(best)
                continue

            try:
                parent_node: BaseNode = self._docstore.get_document(pid)
                parent_text = parent_node.get_content()

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
                logger.warning(f"Parent {pid} not found in docstore: {e}. Using fallback merge.")
                group.sort(key=lambda x: x.node.metadata.get("chunk_index") or 0)
                merged_text = "\n\n".join(n.node.get_content() for n in group)
                best = max(group, key=lambda x: x.score or 0.0)
                meta = best.node.metadata.copy()
                meta["retrieval_mode"] = "small-to-big-fallback"
                big_nodes.append(NodeWithScore(node=TextNode(text=merged_text, metadata=meta), score=best.score))

        big_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return big_nodes[: self._max_big_nodes]


class HybridRetriever(BaseRetriever):
    """Vector + BM25 using QueryFusionRetriever."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: Any,
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


def _get_leaf_nodes_for_bm25(index: VectorStoreIndex) -> list[TextNode]:
    """
    Robustly extract leaf nodes for BM25.
    Tries multiple sources in order:
    1. In-memory docstore (best case)
    2. Raw documents from Chroma collection
    3. Last resort: whatever is in docstore
    """
    leaf_nodes: list[TextNode] = []

    # 1. Try in-memory docstore first
    try:
        docs = getattr(index.docstore, "docs", {}) or {}
        leaf_nodes = [
            node for node in docs.values()
            if isinstance(node, TextNode) and node.metadata.get("chunk_index") is not None
        ]
        if leaf_nodes:
            return leaf_nodes
    except Exception:
        pass

    # 2. Fallback: Read directly from Chroma
    try:
        vs = getattr(index, "vector_store", None)
        collection = None

        if vs is not None:
            if hasattr(vs, "_collection") and vs._collection:
                collection = vs._collection
            elif hasattr(vs, "client"):
                client = vs.client
                cname = getattr(vs, "collection_name", None)
                if cname and hasattr(client, "get_collection"):
                    collection = client.get_collection(cname)

        if collection is not None and hasattr(collection, "get"):
            data = collection.get(include=["documents", "metadatas"])
            for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
                if text:
                    leaf_nodes.append(TextNode(text=text, metadata=meta or {}))
            if leaf_nodes:
                return leaf_nodes
    except Exception as e:
        logger.warning(f"Could not extract nodes from Chroma: {e}")

    # 3. Last resort
    try:
        return list(getattr(index.docstore, "docs", {}).values())
    except Exception:
        return []


def get_retriever(
    index: VectorStoreIndex,
    mode: str = "small_to_big",
    similarity_top_k: int = 10,
    small_to_big_top_k: int = 15,
    hybrid_fusion_mode: str = "reciprocal_rerank",
    bm25_top_k: int = 10,
    **kwargs,
) -> BaseRetriever:

    if mode == "vector":
        logger.info("Using pure vector retrieval")
        return index.as_retriever(similarity_top_k=similarity_top_k)

    elif mode == "small_to_big":
        logger.info("Using Small-to-Big retrieval (real parents)")
        base_retriever = index.as_retriever(similarity_top_k=small_to_big_top_k)
        return SmallToBigRetriever(
            base_retriever=base_retriever,
            docstore=index.docstore,
            small_top_k=small_to_big_top_k,
            max_big_nodes=similarity_top_k,
        )

    elif mode == "hybrid":
        logger.info(f"Using Hybrid retrieval (Vector + BM25)")

        if BM25Retriever is None:
            raise ImportError(
                "Hybrid mode requires 'llama-index-retrievers-bm25'. "
                "Install it with: pip install llama-index-retrievers-bm25"
            )

        vector_retriever = index.as_retriever(similarity_top_k=bm25_top_k)

        leaf_nodes = _get_leaf_nodes_for_bm25(index)

        if not leaf_nodes:
            raise ValueError(
                "Hybrid retrieval requires leaf nodes for BM25. "
                "Please re-run ingestion with the latest code, or switch to 'small_to_big' mode."
            )

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
        logger.warning(f"Unknown mode '{mode}', falling back to vector")
        return index.as_retriever(similarity_top_k=similarity_top_k)