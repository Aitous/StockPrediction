import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Environment variables:
        LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FILE: Path to log file (if set, logs will be written to this file)
        LOG_TO_CONSOLE: Set to 'false' to disable console logging (default: true)

    Args:
        name: The name of the logger (usually __name__)
        level: Optional logging level override (defaults to INFO or LOG_LEVEL env var)

    Returns:
        Configured logger instance

    Example:
        export LOG_FILE=ranker_debug.log
        export LOG_LEVEL=DEBUG
        python cli/main.py
    """
    logger = logging.getLogger(name)

    # If logger is already configured, return it
    if logger.hasHandlers():
        return logger

    # Get level from environment variable or use provided/default
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add file handler if LOG_FILE is set
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler (unless explicitly disabled)
    log_to_console = os.getenv("LOG_TO_CONSOLE", "true").lower() != "false"
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
