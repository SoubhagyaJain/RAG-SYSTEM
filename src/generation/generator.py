"""
RAG Generator — combines retrieved context with an LLM call.

Phase 0: Simple "stuff context into prompt" baseline.
Future phases will add:
- Map-reduce / refine summarization for very long contexts
- Agentic generation (planning, tool use, self-critique, reflection)
- Structured output (Pydantic / JSON mode)
- Streaming + token usage tracking
"""

from __future__ import annotations

from openai import OpenAI

from src.config import get_settings
from src.generation.prompts import get_system_prompt
from src.indexing.vectorstore import RetrievedChunk
from src.logging_config import logger


class RAGGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = OpenAI()
        self.llm_cfg = self.settings.generation.llm

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        context = self._build_context(retrieved)
        system = get_system_prompt(self.settings.generation.system_prompt_name)

        logger.info(
            "Calling LLM",
            extra={
                "model": self.llm_cfg.model,
                "num_context_chunks": len(retrieved),
                "temperature": self.llm_cfg.temperature,
            },
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved Context:\n{context}\n\nAnswer:",
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.llm_cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
            top_p=self.llm_cfg.top_p,
            timeout=self.llm_cfg.timeout,
        )

        answer = resp.choices[0].message.content or ""
        logger.success("Generation complete", extra={"answer_length": len(answer)})
        return answer

    def _build_context(self, retrieved: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(retrieved, 1):
            citation = ""
            if chunk.page_start:
                citation = f" [p. {chunk.page_start}]"
            parts.append(f"---\nSource {i}{citation}\n{chunk.text}\n")
        return "\n".join(parts)
