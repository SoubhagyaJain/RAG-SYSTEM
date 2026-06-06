"""Evaluation package using Ragas for the Enterprise Agentic RAG System.

Ragas (and its transitive dependencies like various langchain_* providers) are
imported lazily inside the evaluator using direct submodule imports (to avoid
ragas' broad __init__.py that pulls in all providers like VertexAI).

This means you can do:

    from src.evaluation import RAGASEvaluator

even if you haven't installed the full set of optional ragas extras yet.
The actual Ragas code is only loaded when you call .evaluate() or run_evaluation().
"""

# Lazy re-exports so that top-level import of the package does not pull in ragas
# (which has heavy optional dependencies on many LLM providers).

def __getattr__(name):
    if name in ("RAGASEvaluator", "run_evaluation"):
        from src.evaluation.evaluator import RAGASEvaluator as _R, run_evaluation as _run
        globals()[name] = _R if name == "RAGASEvaluator" else _run
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RAGASEvaluator", "run_evaluation"]
