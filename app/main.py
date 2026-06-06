"""
Future CLI / application entrypoint.

For now this is a placeholder. Phase 1+ will add:
- `rag ingest`
- `rag index`
- `rag query "your question"`
- `rag evaluate`
"""

import typer

app = typer.Typer(help="Enterprise Agentic RAG System")


@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("Enterprise RAG system - Phase 0 foundations ready.")


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
