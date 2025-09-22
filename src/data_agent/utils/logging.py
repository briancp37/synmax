"""Structured logging utilities for data agent."""

import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any, Optional

# Global run ID for correlating logs across a single execution
RUN_ID = str(uuid.uuid4())[:8]


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "run_id": RUN_ID,
            "module": record.name,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add timing information if present
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        # Add operation context if present
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation

        return json.dumps(log_data, default=str)


def setup_logging(level: str = "INFO", structured: bool = True) -> logging.Logger:
    """
    Set up structured logging for the data agent.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON format

    Returns:
        Configured logger instance
    """
    # Get root logger for data_agent
    logger = logging.getLogger("data_agent")

    # Clear any existing handlers
    logger.handlers.clear()

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Set formatter
    formatter: logging.Formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"data_agent.{name}")


def log_operation(
    logger: logging.Logger, operation: str, duration_ms: Optional[float] = None, **extra_fields: Any
) -> None:
    """
    Log an operation with timing and extra context.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **extra_fields: Additional fields to include in log
    """
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"Operation: {operation}",
        args=(),
        exc_info=None,
    )

    record.operation = operation
    if duration_ms is not None:
        record.duration_ms = duration_ms
    if extra_fields:
        record.extra_fields = extra_fields

    logger.handle(record)


def log_timing(logger: logging.Logger, stage: str, duration_ms: float) -> None:
    """Log timing information for a stage."""
    log_operation(logger, f"timing.{stage}", duration_ms=duration_ms, stage=stage)
