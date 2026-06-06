"""
CLI entrypoint (Phase 0 placeholder).

Future commands will include:
- rag ingest
- rag index
- rag query "question here"
- rag evaluate
"""

import typer

app = typer.Typer(help="Enterprise Agentic RAG System (LlamaIndex + Ollama)")


@app.command()
def hello():
    """Sanity check."""
    typer.echo("Enterprise RAG System - Phase 0 (LlamaIndex + Ollama) foundations ready.")


@app.command()
def config():
    """Print current resolved configuration (safe subset)."""
    from src.config import get_settings
    s = get_settings()
    typer.echo(f"Project: {s.project.name} v{s.project.version}")
    typer.echo(f"Ollama LLM: {s.ollama.llm_model}")
    typer.echo(f"Ollama Embed: {s.ollama.embed_model}")
    typer.echo(f"Chunk size: {s.llama_index.chunk_size}")


def cli():
    app()


if __name__ == "__main__":
    cli()
