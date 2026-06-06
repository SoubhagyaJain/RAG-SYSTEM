"""
Comprehensive Evaluation Framework for the Enterprise Agentic RAG System.

This module provides production-ready evaluation for the RAG over the
"AI Agents Guidebook".

Key features:
- Golden dataset driven (40-50 questions recommended)
- Retrieval Evaluation: Recall@K, Context Precision, Page Hit Rate
- Generation Evaluation: Ragas (faithfulness, answer_relevancy) + custom
  Educational Quality (LLM-as-judge for structure, depth, clarity, citations)
- Fully config-driven via config.yaml
- Timestamped result saving to artifacts/
- Lazy Ragas imports (avoids heavy optional LLM provider deps until needed)
- Works with the current get_query_engine() (including Small-to-Big + educational prompt)

Usage (notebook or script):
    from src.retrieval import get_query_engine
    from src.evaluation import RAGASEvaluator

    query_engine = get_query_engine()
    evaluator = RAGASEvaluator(query_engine=query_engine)
    results = evaluator.evaluate()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from llama_index.core import Settings as LlamaSettings

from src.config import get_settings
from src.logging_config import logger as app_logger  # project logger
from src.utils.io import ensure_dir


def _get_ragas_modules():
    """Lazily import ragas and its wrappers.

    Ragas has very eager imports in its __init__ and llms/ package that pull in
    many optional LLM backends (VertexAI, Haystack, etc.) even when you only
    want the LlamaIndex wrappers + a couple of metrics.

    We use importlib + direct submodule access + a broad except to give a clear
    actionable message. The root cause is almost always a missing
    langchain-google-vertexai (or an older langchain-community that no longer
    re-exports the vertexai chat model under the old path).
    """
    try:
        import importlib

        # We deliberately import the smallest pieces we actually use.
        # Even this can still trigger ragas/__init__.py in some ragas versions,
        # which is why we have the very explicit error message below.
        ragas_evaluate = importlib.import_module("ragas.evaluation").evaluate

        embeddings_mod = importlib.import_module("ragas.embeddings")
        LlamaIndexEmbeddingsWrapper = embeddings_mod.LlamaIndexEmbeddingsWrapper

        # This one is the usual culprit (ragas.llms.base does a hard import
        # of langchain_community.chat_models.vertexai).
        llms_base = importlib.import_module("ragas.llms.base")
        LlamaIndexLLMWrapper = llms_base.LlamaIndexLLMWrapper

        metrics_mod = importlib.import_module("ragas.metrics")

        return {
            "evaluate": ragas_evaluate,
            "LlamaIndexEmbeddingsWrapper": LlamaIndexEmbeddingsWrapper,
            "LlamaIndexLLMWrapper": LlamaIndexLLMWrapper,
            "answer_relevancy": metrics_mod.answer_relevancy,
            "context_precision": metrics_mod.context_precision,
            "context_recall": metrics_mod.context_recall,
            "faithfulness": metrics_mod.faithfulness,
        }
    except Exception as e:  # broad: ImportError, ModuleNotFoundError, etc.
        raise ImportError(
            "Ragas generation evaluation requires extra LLM provider packages "
            "(especially langchain-google-vertexai).\n\n"
            "Ragas eagerly imports VertexAI wrappers from langchain at import time, "
            "even when we only want the LlamaIndex path.\n\n"
            "Fix (run in your project root):\n"
            "    .\\.venv\\Scripts\\python.exe -m pip install -e \".[dev]\" langchain-google-vertexai\n\n"
            "Or explicitly:\n"
            "    .\\.venv\\Scripts\\python.exe -m pip install langchain-google-vertexai>=1.0 langchain-community>=0.2\n\n"
            "Then **restart the Jupyter kernel** completely.\n\n"
            "If you are using uv instead of pip, do the equivalent uv pip command.\n\n"
            f"Original error: {e}"
        ) from e


@dataclass
class EvaluationResult:
    """Container for evaluation results with metadata."""
    timestamp: str
    metrics: dict[str, float]
    num_questions: int
    model_used: str
    judge_model: str
    duration_seconds: float
    dataset_path: str
    details: list[dict[str, Any]] = field(default_factory=list)  # per-question scores if available


class RAGASEvaluator:
    """
    Comprehensive evaluator for both Retrieval and Generation quality.

    Supports:
    - Retrieval metrics (Recall@K, Context Precision, Page Hit Rate) using node metadata
    - Generation metrics via Ragas + custom Educational Quality LLM judge

    Designed to be run after every major change to ingestion or retrieval.
    """

    def __init__(self, query_engine: Any, config: Any | None = None):
        self.query_engine = query_engine
        self.settings = config or get_settings()
        self.eval_config = self.settings.evaluation
        self.paths = self.settings.paths.resolve()

        self.dataset_path = self.paths["evaluation"] / Path(self.eval_config.dataset_path).name
        self.output_dir = Path(self.eval_config.output_dir)
        ensure_dir(self.output_dir)

        self._ragas_llm = None
        self._ragas_embeddings = None

        app_logger.info("RAGAS Evaluator initialized", extra={"dataset": str(self.dataset_path)})

    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            ds = json.loads(text)
        except json.JSONDecodeError as e:
            app_logger.warning(f"Malformed JSON in {self.dataset_path}: {e}. Attempting auto-repair...")
            ds = self._repair_and_load_dataset(text)

        if not isinstance(ds, list):
            raise ValueError(
                f"Eval dataset at {self.dataset_path} must be a JSON array of question objects. "
                f"Got {type(ds).__name__} instead."
            )

        # Basic validation + filter
        clean = []
        for item in ds:
            if isinstance(item, dict) and item.get("question"):
                clean.append(item)

        if not clean:
            raise ValueError(f"No valid questions found in {self.dataset_path}")

        n = getattr(self.eval_config, "num_questions", len(clean))
        return clean[:n]

    def _repair_and_load_dataset(self, text: str) -> list[dict[str, Any]]:
        """Attempt to recover from common JSON corruption (extra data / concatenated arrays).
        This happens when the file was appended to instead of overwritten, or had manual edits.
        On success it rewrites the file cleanly and returns the items.
        """
        decoder = json.JSONDecoder()
        pos = 0
        items: list[dict] = []

        while pos < len(text):
            # skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, consumed = decoder.raw_decode(text, pos)
                if isinstance(obj, list):
                    items.extend([x for x in obj if isinstance(x, dict) and x.get("question")])
                elif isinstance(obj, dict) and obj.get("question"):
                    items.append(obj)
                pos += consumed
            except json.JSONDecodeError:
                pos += 1

        # Deduplicate by question text (defensive)
        seen = set()
        unique = []
        for it in items:
            q = it.get("question", "").strip()
            if q and q not in seen:
                seen.add(q)
                unique.append(it)

        if unique:
            # Rewrite the file as clean JSON so the corruption never happens again
            try:
                with open(self.dataset_path, "w", encoding="utf-8") as f:
                    json.dump(unique, f, indent=2, ensure_ascii=False)
                app_logger.warning(
                    f"Auto-repaired {self.dataset_path} (recovered {len(unique)} questions) "
                    "and wrote a clean single JSON array. A .bak of the old file may exist."
                )
            except Exception as write_err:
                app_logger.error(f"Could not rewrite repaired dataset: {write_err}")

        return unique

    # ---------------- Retrieval Evaluation ----------------
    def evaluate_retrieval(self, k: int | None = None) -> dict[str, float]:
        """Compute retrieval metrics using rich node metadata from hierarchical chunking."""
        k = k or getattr(self.eval_config, "retrieval_k", 10)
        dataset = self._load_dataset()

        # Try to get the actual retriever used by the QueryEngine.
        # This is critical so that evaluate_retrieval() measures the configured strategy
        # (Small-to-Big real-parent, Hybrid, etc.) instead of falling back to plain vector.
        retriever = getattr(self.query_engine, "retriever", None)
        if retriever is None:
            retriever = getattr(self.query_engine, "_retriever", None)

        if retriever is None:
            # Last-resort fallback: plain vector retriever (loses Small-to-Big / Hybrid behavior)
            from src.retrieval.query_engine import get_vector_store
            from llama_index.core import VectorStoreIndex
            vs = get_vector_store()
            idx = VectorStoreIndex.from_vector_store(vs, embed_model=LlamaSettings.embed_model)
            retriever = idx.as_retriever(similarity_top_k=k)

        total_recall = total_precision = total_hits = evaluated = 0.0
        details: list[dict] = []

        app_logger.info(f"Starting retrieval evaluation over {len(dataset)} questions (k={k})")

        for idx, item in enumerate(dataset, 1):
            q = item.get("question")
            exp_pages = set(item.get("expected_pages", []))
            exp_sections = set(item.get("expected_sections", []))
            must = [t.lower() for t in item.get("must_retrieve_terms", [])]
            if not q:
                continue

            app_logger.info(f"  [{idx}/{len(dataset)}] Retrieving for: {q[:60]}...")

            try:
                retrieved = retriever.retrieve(q)[:k]
            except Exception as e:
                app_logger.warning(f"    Retrieval failed for question {idx}: {e}")
                continue

            pages = set()
            relevant = 0
            retrieved_meta = []
            for ns in retrieved:
                m = ns.node.metadata or {}
                p = m.get("page_number")
                sec = m.get("section", "")
                txt = ns.node.get_content().lower()
                if p is not None:
                    pages.add(int(p))
                if (p in exp_pages) or (sec in exp_sections) or any(mt in txt for mt in must):
                    relevant += 1
                retrieved_meta.append({
                    "page": p,
                    "section": sec,
                    "content_type": m.get("content_type"),
                    "preview": ns.node.get_content()[:120].replace("\n", " ")
                })

            if exp_pages:
                total_recall += len(pages & exp_pages) / len(exp_pages)
            total_precision += relevant / max(1, len(retrieved))
            hit = bool(pages & exp_pages)
            if hit:
                total_hits += 1
            evaluated += 1

            detail = {
                "idx": idx,
                "question": q[:80],
                "expected_pages": sorted(exp_pages),
                "retrieved_pages": sorted(pages),
                "hit": hit,
                "precision_contrib": round(relevant / max(1, len(retrieved)), 3),
                "retrieved": retrieved_meta[:2],  # top 2 for brevity
            }
            details.append(detail)
            app_logger.info(f"    pages_retrieved={sorted(pages)} expected={sorted(exp_pages)} hit={hit} precision={detail['precision_contrib']}")

        app_logger.info(f"Retrieval evaluation complete. Evaluated {evaluated} questions.")

        if evaluated == 0:
            return {"recall_at_k": 0.0, "context_precision": 0.0, "page_hit_rate": 0.0, "details": []}

        agg = {
            "recall_at_k": round(total_recall / evaluated, 4),
            "context_precision": round(total_precision / evaluated, 4),
            "page_hit_rate": round(total_hits / evaluated, 4),
            "evaluated": int(evaluated),
            "details": details,
        }

        # Print a compact per-question summary for immediate diagnostics (plan Step 1)
        print("\n=== Per-question retrieval diagnostics (smoke set) ===")
        for d in details:
            print(f"Q{d['idx']}: hit={d['hit']} pages={d['retrieved_pages']} vs expected={d['expected_pages']} prec={d['precision_contrib']}")
            for r in d.get("retrieved", []):
                print(f"   -> page {r['page']} | {r['section'][:40]}... | {r['preview'][:80]}")

        print(f"\nAGGREGATE (k={k}): recall_at_k={agg['recall_at_k']} context_precision={agg['context_precision']} page_hit_rate={agg['page_hit_rate']}")
        return agg

    def _get_ragas_llm(self):
        """Wrap the project's LlamaIndex LLM (Gemma 4 8B) for Ragas."""
        if self._ragas_llm is None:
            ragas_mods = _get_ragas_modules()
            # We use the already-configured LlamaSettings.llm (Gemma 4 8B)
            # Ragas wrapper works directly with LlamaIndex LLM objects
            self._ragas_llm = ragas_mods["LlamaIndexLLMWrapper"](LlamaSettings.llm)
        return self._ragas_llm

    def _get_ragas_embeddings(self):
        """Wrap the project's LlamaIndex embeddings for Ragas."""
        if self._ragas_embeddings is None:
            ragas_mods = _get_ragas_modules()
            self._ragas_embeddings = ragas_mods["LlamaIndexEmbeddingsWrapper"](LlamaSettings.embed_model)
        return self._ragas_embeddings

    # --- Generation Evaluation (Ragas + Educational Quality) ---
    def _select_ragas_metrics(self):
        ragas_mods = _get_ragas_modules()
        mapping = {
            "faithfulness": ragas_mods["faithfulness"],
            "answer_relevancy": ragas_mods["answer_relevancy"],
        }
        return [mapping[m] for m in getattr(self.eval_config, "ragas_metrics", []) if m in mapping]

    def _evaluate_educational_quality(self, question: str, answer: str, contexts: list[str]) -> float:
        try:
            ragas_mods = _get_ragas_modules()
            judge = ragas_mods["LlamaIndexLLMWrapper"](LlamaSettings.llm)
            prompt = (
                f"You are an expert evaluator of educational RAG answers.\n"
                f"Question: {question}\n\nContext (excerpt): {' '.join(contexts)[:800]}\n\nAnswer: {answer}\n\n"
                "Score 0.0-1.0 on average of: Clarity+Structure (Markdown), Depth (simple→technical), "
                "Citations from context, Faithfulness, Educational value for students. Return only the float."
            )
            import re
            m = re.search(r"([0-9]*\.?[0-9]+)", str(judge.complete(prompt)))
            return max(0.0, min(1.0, float(m.group(1)))) if m else 0.5
        except Exception:
            return 0.5

    def evaluate_generation(self) -> dict[str, float]:
        """Run Ragas + custom educational quality metrics.

        If the full ragas + langchain provider stack is not installed (very common
        because ragas eagerly imports VertexAI etc.), we gracefully degrade:
        we still return retrieval scores + a clear note so the user doesn't lose
        their retrieval evaluation work.
        """
        dataset = self._load_dataset()

        try:
            ragas_mets = self._select_ragas_metrics()
        except ImportError as e:
            app_logger.warning(
                "Generation evaluation (Ragas) skipped because of missing provider packages.\n"
                f"{e}\n"
                "You still have the retrieval scores. Install the packages listed above and "
                "restart the kernel to get faithfulness / answer_relevancy / educational_quality."
            )
            return {
                "generation_skipped": "Missing langchain-google-vertexai / ragas LLM backends. "
                                      "See previous warning for install command."
            }

        qs, ans, ctxs, gts = [], [], [], []

        for item in dataset:
            q = item.get("question")
            if not q: continue
            try:
                resp = self.query_engine.query(q)
                a = str(resp).strip()
                c = [n.node.get_content() for n in getattr(resp, "source_nodes", [])[:5]]
            except Exception:
                a, c = "", []
            qs.append(q); ans.append(a); ctxs.append(c or [""]); gts.append(item.get("ground_truth", ""))

        scores = {}
        if ragas_mets and qs:
            from datasets import Dataset
            ds = Dataset.from_dict({"question": qs, "answer": ans, "contexts": ctxs, "ground_truth": gts})
            res = _get_ragas_modules()["evaluate"](ds, metrics=ragas_mets,
                                                   llm=self._get_ragas_llm(), embeddings=self._get_ragas_embeddings())
            if hasattr(res, "to_pandas"):
                df = res.to_pandas()
                for col in df.columns:
                    if col != "question":
                        scores[col] = float(df[col].mean())

        if "educational_quality" in getattr(self.eval_config, "custom_metrics", []):
            edu = [self._evaluate_educational_quality(q, a, c) for q, a, c in zip(qs, ans, ctxs)]
            scores["educational_quality"] = round(sum(edu) / max(1, len(edu)), 4) if edu else 0.0

        return scores

    # --- Main Entry Point ---
    def evaluate(self, save: bool | None = None) -> dict:
        start = time.time()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        retrieval_scores = self.evaluate_retrieval()
        generation_scores = self.evaluate_generation()

        # Graceful handling when generation was skipped due to missing ragas backends
        if "generation_skipped" in generation_scores:
            combined = {**retrieval_scores, **generation_scores}
            result = {
                "timestamp": ts,
                "num_questions": len(self._load_dataset()),
                "retrieval": retrieval_scores,
                "generation": generation_scores,
                "combined": combined,
                "model": str(self.settings.ollama.llm_model),
                "duration_seconds": round(time.time() - start, 2),
                "note": "Generation metrics (Ragas + educational_quality) were skipped. "
                        "See logs for the exact pip command. Retrieval results are still valid and saved.",
            }
        else:
            combined = {**retrieval_scores, **generation_scores}
            result = {
                "timestamp": ts,
                "num_questions": len(self._load_dataset()),
                "retrieval": retrieval_scores,
                "generation": generation_scores,
                "combined": combined,
                "model": str(self.settings.ollama.llm_model),
                "duration_seconds": round(time.time() - start, 2),
            }

        if save if save is not None else getattr(self.eval_config, "save_results", True):
            self._save_results(result)

        if "generation_skipped" in generation_scores:
            app_logger.warning("Evaluation finished with retrieval only (generation skipped due to missing packages).")
        else:
            app_logger.success(f"Full evaluation complete: {combined}")
        return result

    def _save_results(self, result: dict):
        base = f"evaluation_{result['timestamp']}"
        path = self.output_dir / f"{base}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        app_logger.success(f"Saved evaluation results to {path}")


def run_evaluation(query_engine: Any | None = None, save: bool = True):
    """Convenience entrypoint."""
    from src.retrieval import get_query_engine
    qe = query_engine or get_query_engine()
    return RAGASEvaluator(qe).evaluate(save=save)
