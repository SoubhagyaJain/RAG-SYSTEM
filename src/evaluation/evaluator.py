"""
Ragas-based Evaluation Framework for the Enterprise Agentic RAG System.

This module provides a production-ready wrapper around Ragas for evaluating
RAG pipelines (specifically QueryEngine from LlamaIndex).

Key features:
- Fully config-driven (metrics, judge LLM, embeddings)
- Uses Gemma 4 8B as the judge model via Ollama (wrapped for Ragas)
- Uses nomic-embed-text for embeddings in Ragas
- Supports faithfulness, answer_relevancy, context_precision, context_recall
- Automatic extraction of answer + retrieved contexts from QueryEngine
- Timestamped result saving
- Structured logging and timing
- Repeatable and scriptable

Usage:
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

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from llama_index.core import Settings as LlamaSettings

from src.config import get_settings
from src.logging_config import logger as app_logger  # project logger
from src.utils.io import ensure_dir


# Map string metric names (from config) to actual Ragas metric objects
METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


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
    Production wrapper for evaluating a LlamaIndex QueryEngine using Ragas.

    The evaluator:
    1. Loads a JSON evaluation dataset
    2. Runs the provided QueryEngine on each question
    3. Extracts generated answers + retrieved contexts
    4. Wraps the project's LLM and embeddings for Ragas
    5. Runs selected Ragas metrics using Gemma 4 8B as judge
    6. Returns and optionally persists results
    """

    def __init__(
        self,
        query_engine: Any,
        config: Any | None = None,
    ):
        """
        Args:
            query_engine: A LlamaIndex QueryEngine (from get_query_engine())
            config: Optional settings object. If None, uses the global get_settings()
        """
        self.query_engine = query_engine
        self.settings = config or get_settings()
        self.eval_config = self.settings.evaluation
        self.paths = self.settings.paths.resolve()

        # Resolve paths
        self.dataset_path = self.paths["evaluation"] / Path(self.eval_config.dataset_path).name
        self.output_dir = Path(self.eval_config.output_dir)

        ensure_dir(self.output_dir)

        # Prepare Ragas LLM and Embeddings wrappers (using project's configured models)
        self._ragas_llm = None
        self._ragas_embeddings = None

        app_logger.info(
            "RAGAS Evaluator initialized",
            extra={
                "judge_model": self.eval_config.llm_for_judge,
                "embed_model": self.eval_config.embed_model_for_ragas,
                "metrics": self.eval_config.ragas_metrics,
                "dataset": str(self.dataset_path),
            },
        )

    def _get_ragas_llm(self):
        """Wrap the project's LlamaIndex LLM (Gemma 4 8B) for Ragas."""
        if self._ragas_llm is None:
            # We use the already-configured LlamaSettings.llm (Gemma 4 8B)
            # Ragas wrapper works directly with LlamaIndex LLM objects
            self._ragas_llm = LlamaIndexLLMWrapper(LlamaSettings.llm)
        return self._ragas_llm

    def _get_ragas_embeddings(self):
        """Wrap the project's LlamaIndex embeddings for Ragas."""
        if self._ragas_embeddings is None:
            self._ragas_embeddings = LlamaIndexEmbeddingsWrapper(LlamaSettings.embed_model)
        return self._ragas_embeddings

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load the evaluation dataset from JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at: {self.dataset_path}. "
                "Please create data/evaluation/eval_dataset.json with questions and ground_truth."
            )

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Evaluation dataset must be a list of question objects.")

        app_logger.info(f"Loaded evaluation dataset with {len(dataset)} questions")
        return dataset

    def _run_query_engine(self, question: str) -> tuple[str, list[str]]:
        """
        Run the QueryEngine on a question and extract answer + contexts.

        Returns:
            (answer, list_of_context_strings)
        """
        response = self.query_engine.query(question)

        answer = str(response).strip()

        # Extract retrieved contexts (source nodes)
        contexts: list[str] = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                try:
                    text = node.get_content() if hasattr(node, "get_content") else str(node.node.text)
                    if text:
                        contexts.append(text.strip())
                except Exception:
                    continue

        # Fallback: some engines put context in response.metadata or similar
        if not contexts and hasattr(response, "metadata"):
            ctx = response.metadata.get("context") or response.metadata.get("contexts")
            if isinstance(ctx, list):
                contexts = [str(c).strip() for c in ctx if c]
            elif isinstance(ctx, str):
                contexts = [ctx.strip()]

        return answer, contexts

    def _select_metrics(self) -> list:
        """Convert metric names from config into actual Ragas metric instances."""
        selected = []
        for name in self.eval_config.ragas_metrics:
            metric = METRIC_MAP.get(name.lower())
            if metric is None:
                app_logger.warning(f"Unknown Ragas metric '{name}' - skipping")
                continue
            selected.append(metric)
        return selected

    def evaluate(self, save: bool | None = None) -> EvaluationResult:
        """
        Run the full Ragas evaluation on the configured dataset.

        Args:
            save: Override the config setting for saving results.

        Returns:
            EvaluationResult with aggregated scores and metadata.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dataset = self._load_dataset()
        metrics = self._select_metrics()

        if not metrics:
            raise ValueError("No valid Ragas metrics selected. Check evaluation.ragas_metrics in config.yaml.")

        app_logger.info(
            "Starting Ragas evaluation",
            extra={
                "num_questions": len(dataset),
                "metrics": [m.name for m in metrics],
                "judge": self.eval_config.llm_for_judge,
            },
        )

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        # Run the RAG system on every question in the dataset
        for i, item in enumerate(dataset, 1):
            question = item.get("question") or item.get("query")
            ground_truth = item.get("ground_truth") or item.get("reference_answer")

            if not question or not ground_truth:
                app_logger.warning(f"Skipping item {i}: missing question or ground_truth")
                continue

            try:
                answer, retrieved_contexts = self._run_query_engine(question)
            except Exception as e:
                app_logger.error(f"Query failed for question {i}: {e}")
                answer, retrieved_contexts = "", []

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts or [""])
            ground_truths.append(ground_truth)

            if i % 5 == 0 or i == len(dataset):
                app_logger.info(f"Processed {i}/{len(dataset)} questions...")

        # Prepare Ragas dataset
        ragas_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        # Get wrapped LLM and embeddings for Ragas (Gemma 4 8B + nomic-embed-text)
        ragas_llm = self._get_ragas_llm()
        ragas_embeddings = self._get_ragas_embeddings()

        # Run evaluation
        app_logger.info("Running Ragas evaluate() with local models...")
        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            # raise_exceptions=False,  # Uncomment for more robustness with local models
        )

        # Extract scores (Ragas returns a dict-like or pandas-like object)
        scores: dict[str, float] = {}
        try:
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                for col in df.columns:
                    if col != "question":
                        scores[col] = float(df[col].mean())
            else:
                # Fallback for different Ragas versions
                scores = {k: float(v) for k, v in dict(result).items() if k != "question"}
        except Exception as e:
            app_logger.warning(f"Could not parse detailed scores: {e}")
            scores = {"overall": 0.0}

        duration = time.time() - start_time

        eval_result = EvaluationResult(
            timestamp=timestamp,
            metrics=scores,
            num_questions=len(questions),
            model_used=str(self.settings.ollama.llm_model),
            judge_model=self.eval_config.llm_for_judge,
            duration_seconds=round(duration, 2),
            dataset_path=str(self.dataset_path),
        )

        app_logger.success(
            "Ragas evaluation complete",
            extra={
                "duration_sec": eval_result.duration_seconds,
                "scores": scores,
                "num_questions": eval_result.num_questions,
            },
        )

        # Save results if requested
        do_save = save if save is not None else self.eval_config.save_results
        if do_save:
            self._save_results(eval_result, ragas_dataset, result)

        return eval_result

    def _save_results(
        self,
        eval_result: EvaluationResult,
        ragas_dataset: Dataset,
        ragas_result: Any,
    ) -> None:
        """Save evaluation results with timestamp."""
        timestamp = eval_result.timestamp
        base_name = f"evaluation_{timestamp}"

        # Save main summary as JSON
        summary_path = self.output_dir / f"{base_name}.json"
        summary = {
            "timestamp": timestamp,
            "model": eval_result.model_used,
            "judge_model": eval_result.judge_model,
            "dataset": eval_result.dataset_path,
            "num_questions": eval_result.num_questions,
            "duration_seconds": eval_result.duration_seconds,
            "metrics": eval_result.metrics,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save detailed per-question results if possible
        try:
            if hasattr(ragas_result, "to_pandas"):
                df = ragas_result.to_pandas()
                detail_path = self.output_dir / f"{base_name}_details.csv"
                df.to_csv(detail_path, index=False)
                app_logger.info(f"Saved detailed results to {detail_path}")
        except Exception as e:
            app_logger.warning(f"Could not save detailed CSV: {e}")

        app_logger.success(f"Evaluation results saved to {self.output_dir} (prefix: {base_name})")


def run_evaluation(
    query_engine: Any | None = None,
    save: bool = True,
) -> EvaluationResult:
    """
    Convenience function to run evaluation end-to-end.

    If query_engine is None, it will be created using get_query_engine().
    """
    from src.retrieval import get_query_engine

    if query_engine is None:
        query_engine = get_query_engine()

    evaluator = RAGASEvaluator(query_engine=query_engine)
    return evaluator.evaluate(save=save)
