"""
Retriever and QueryEngine factory.

Uses:
- Strong educational system prompt for detailed, structured Markdown answers
- Small-to-Big retrieval on top of HierarchicalNodeParser (parent_id metadata)
"""

from __future__ import annotations

import chromadb
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import _find_project_root, get_settings
from src.logging_config import logger
from src.retrieval.retriever import get_retriever  # New modular retrievers
from src.retrieval.postprocessor import MetadataBoosterPostprocessor  # Metadata bias correction


# =============================================================================
# Strong Educational System Prompt
# =============================================================================
EDUCATIONAL_TEXT_QA_PROMPT = PromptTemplate(
    "You are an expert educator and technical writer specializing in AI Agents and Agentic Systems.\n\n"
    "Your goal is to provide **clear, comprehensive, and educational answers** based strictly on the provided context from the \"AI Agents Guidebook\".\n\n"
    "Instructions:\n"
    "1. Explain concepts clearly as if teaching a student. Start with simple language and intuition, then gradually add technical depth.\n"
    "2. Structure every response in clean Markdown:\n"
    "   - Use `##` and `###` for main sections.\n"
    "   - Use bullet points and numbered lists.\n"
    "   - Use **bold** for key terms.\n"
    "   - Use fenced code blocks with language tags for code/examples.\n"
    "   - Use blockquotes for important takeaways.\n"
    "3. Be comprehensive but focused. Synthesize information when multiple pieces of context are relevant.\n"
    "4. Use citations from node metadata whenever possible: (Page X), (Section: Y), or (Content Type: Z, Page X).\n"
    "5. If the provided context is insufficient, clearly state what is missing. Do not hallucinate or use external knowledge.\n\n"
    "Context information (retrieved via Small-to-Big):\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Educational Answer:\n"
)


# The SmallToBigRetriever (and other strategies) now live in src/retrieval/retriever.py
# for better modularity. We re-export the main one here for backward compatibility
# if any code imports it directly from query_engine.
from src.retrieval.retriever import SmallToBigRetriever  # noqa: F401


def get_vector_store() -> ChromaVectorStore:
    """Connect to persistent Chroma collection."""
    settings = get_settings()
    project_root = _find_project_root()

    if hasattr(settings, "vector_store") and settings.vector_store:
        persist_dir = project_root / settings.vector_store.persist_dir
        collection_name = settings.vector_store.collection_name
    else:
        persist_dir = project_root / settings.llama_index.vector_store.persist_dir
        collection_name = settings.llama_index.vector_store.collection_name

    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    if chroma_collection.count() == 0:
        raise RuntimeError(
            f"Chroma collection '{collection_name}' is empty. "
            "Please run ingestion first: from src.ingestion import run_ingestion; run_ingestion(force=True)"
        )

    return ChromaVectorStore(chroma_collection=chroma_collection)


def get_query_engine(
    vector_store: BasePydanticVectorStore | None = None,
    similarity_top_k: int | None = None,
) -> BaseQueryEngine:
    """
    Creates a QueryEngine using the configured retrieval strategy.

    The actual retriever is created in src/retrieval/retriever.py based on
    `retrieval.mode` in config.yaml. This keeps query_engine.py focused on
    wiring the educational prompt + response synthesizer.
    """
    settings = get_settings()
    settings.configure_llama_index()

    vector_store = vector_store or get_vector_store()
    top_k = similarity_top_k or settings.llama_index.similarity_top_k

    logger.info(
        "Creating QueryEngine",
        extra={
            "retrieval_mode": getattr(settings.retrieval, "mode", "small_to_big"),
            "similarity_top_k": top_k,
        },
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    # === Repopulate docstore from sidecar (High priority fix for real parent / hybrid nodes) ===
    # Ingestion now writes data/processed/hierarchy_nodes.json with all hierarchical nodes.
    # This lets SmallToBigRetriever see real parents and Hybrid get leaf texts even after
    # from_vector_store() (which otherwise leaves an empty docstore).
    try:
        from src.config import _find_project_root
        from pathlib import Path
        import json
        from llama_index.core.schema import TextNode

        project_root = _find_project_root()
        # Mirror the vector_store persist_dir used in get_vector_store
        vs_cfg = getattr(settings, "vector_store", None) or settings.llama_index.vector_store
        persist_dir = project_root / vs_cfg.persist_dir
        hierarchy_path = persist_dir / "hierarchy_nodes.json"

        if hierarchy_path.exists():
            with open(hierarchy_path, "r", encoding="utf-8") as f:
                node_dicts = json.load(f)
            loaded_nodes = [TextNode.from_dict(d) for d in node_dicts if d]
            if loaded_nodes:
                index.docstore.add_documents(loaded_nodes)
                logger.info(f"Repopulated docstore with {len(loaded_nodes)} hierarchy nodes from sidecar for Small-to-Big / Hybrid")
    except Exception as e:
        logger.warning(f"Could not load hierarchy sidecar (S2B/Hybrid will use fallbacks): {e}")

    # Get the appropriate retriever (vector / small_to_big / hybrid)
    # This is fully configurable via config.yaml
    retriever = get_retriever(
        index=index,
        mode=getattr(settings.retrieval, "mode", "small_to_big"),
        similarity_top_k=top_k,
        small_to_big_top_k=getattr(settings.retrieval, "small_to_big_top_k", top_k * 2),
        hybrid_fusion_mode=getattr(settings.retrieval, "hybrid_fusion_mode", "reciprocal_rerank"),
        bm25_top_k=getattr(settings.retrieval, "bm25_top_k", top_k),
    )

    # =====================================================================
    # Metadata-based score boosting (bias correction)
    # Applied as a NodePostprocessor *after* the custom retriever (SmallToBig,
    # Hybrid, etc.) but *before* the response synthesizer / LLM.
    # This is the recommended LlamaIndex pattern for post-retrieval adjustments.
    # =====================================================================
    node_postprocessors = []

    retrieval_cfg = settings.retrieval

    # Read from the metadata_boosting dict (supports nested yaml under retrieval:)
    mb = getattr(retrieval_cfg, "metadata_boosting", {}) or {}
    enabled = mb.get("enabled", False)
    penalty = mb.get("memory_section_penalty", 0.6)
    boost = mb.get("boost_important_sections", True)

    if enabled:
        postproc = MetadataBoosterPostprocessor(
            memory_section_penalty=penalty,
            boost_important_sections=boost,
        )
        node_postprocessors.append(postproc)
        logger.info(
            "MetadataBoosterPostprocessor enabled for section bias correction",
            extra={
                "memory_section_penalty": penalty,
                "boost_important_sections": boost,
                "retrieval_mode": getattr(retrieval_cfg, "mode", "unknown"),
            },
        )

    # Always use the strong educational prompt + configured response mode
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=EDUCATIONAL_TEXT_QA_PROMPT,
        response_mode=settings.llama_index.response_mode,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )

    return query_engine