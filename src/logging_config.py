"""
Production-grade structured logging configuration.

Uses loguru for beautiful console output + optional JSON/file logging.
All application code should import `logger` from here.

Example:
    from src.logging_config import logger, setup_logging

    setup_logging()
    logger.info("Pipeline started", extra={"phase": "ingestion", "doc": "guidebook"})
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings


def setup_logging() -> None:
    """
    Configure loguru sinks based on config.yaml + environment.

    - Rich colored console (always)
    - Rotating file logs (optional)
    - JSON structured logs for production (optional)
    """
    settings = get_settings()
    log_cfg = settings.logging

    # Remove default handler
    logger.remove()

    # Console handler (beautiful for humans)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=log_cfg.level,
        format=console_format,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # File logging
    if log_cfg.log_to_file:
        log_path = Path(log_cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message} | {extra}"
        )

        logger.add(
            log_path,
            level=log_cfg.level,
            format=file_format,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    # JSON logging (great for containers / centralized logging)
    if log_cfg.json_logs:
        logger.add(
            sys.stderr,
            level=log_cfg.level,
            format="{message}",
            serialize=True,  # produces JSON
            filter=lambda record: record["level"].no >= 20,  # INFO and above
        )

    logger.info(
        "Logging initialized",
        extra={
            "level": log_cfg.level,
            "json": log_cfg.json_logs,
            "file": str(log_cfg.log_file) if log_cfg.log_to_file else None,
            "env": settings.rag_env,
        },
    )


# Default logger instance for direct import
# Call setup_logging() once at application entrypoint (CLI, notebook, FastAPI lifespan, etc.)
setup_logging()
