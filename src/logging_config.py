"""
Production logging configuration using loguru.

Import `logger` after calling `setup_logging()` (done automatically on import).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings


def setup_logging() -> None:
    """Configure loguru sinks based on config.yaml."""
    cfg = get_settings().logging

    logger.remove()

    # Human-friendly console
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=cfg.level,
        format=console_fmt,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # File logging
    if cfg.log_to_file:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message} | {extra}"
        logger.add(
            log_path,
            level=cfg.level,
            format=file_fmt,
            rotation=cfg.rotation,
            retention=cfg.retention,
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    # JSON mode (useful for containers / centralized logging)
    if cfg.json_logs:
        logger.add(
            sys.stderr,
            level=cfg.level,
            format="{message}",
            serialize=True,
        )

    logger.info(
        "Logging initialized",
        extra={
            "level": cfg.level,
            "json": cfg.json_logs,
            "env": get_settings().rag_env,
        },
    )


# Auto-initialize on import
setup_logging()
