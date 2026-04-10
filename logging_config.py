"""
logging_config.py — Centralized logging configuration for the autotrade system.

Usage:
    from logging_config import setup_logging
    logger = setup_logging(level="INFO")  # or "DEBUG", "WARNING", "ERROR", "CRITICAL"
"""

import logging
import sys


LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d %(message)s"


def setup_logging(level="INFO"):
    """Configure root logger with standard format and level.

    Clears any existing handlers to avoid duplicate output, then adds a single
    StreamHandler writing to stderr.

    Args:
        level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
               or as a logging.* integer constant.

    Returns:
        The root logger instance, now configured.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Clear any pre-existing handlers to prevent duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    return root
