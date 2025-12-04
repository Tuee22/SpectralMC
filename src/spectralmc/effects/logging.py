"""
Logging Effect ADTs for structured, interpreter-driven logging.

This module defines immutable dataclasses for logging requests, ensuring
all log side effects are routed through the Effect Interpreter rather
than being executed directly inside business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LogMessage:
    """Request to emit a log message.

    Attributes:
        kind: Discriminator for pattern matching. Always "LogMessage".
        level: Log level to emit ("debug", "info", "warning", "error", "critical").
        message: Log message payload.
        logger_name: Logger name to use; defaults to module-level logger when empty.
        exc_info: Whether to include exception info in the log record.
    """

    kind: Literal["LogMessage"] = "LogMessage"
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    message: str = ""
    logger_name: str = ""
    exc_info: bool = False


# Logging Effect Union
LoggingEffect = LogMessage


__all__ = [
    "LogMessage",
    "LoggingEffect",
]
