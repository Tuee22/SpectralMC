"""
Logging Effect ADTs for structured, interpreter-driven logging.

This module defines immutable dataclasses for logging requests and a simple
interpreter that routes logs through the standard logging module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from spectralmc.effects.errors import LoggingError
from spectralmc.result import Failure, Result, Success

LogLevel = Literal["debug", "info", "warning", "error", "critical"]


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
    level: LogLevel = "info"
    message: str = ""
    logger_name: str = ""
    exc_info: bool = False


# Logging Effect Union
LoggingEffect = LogMessage


class LoggingInterpreter:
    """Interpreter for logging effects."""

    def __init__(self, default_logger_name: str = "spectralmc") -> None:
        self._default_logger_name = default_logger_name

    async def interpret(self, effect: LoggingEffect) -> Result[object, LoggingError]:
        """Execute logging effect."""
        return self._log_message(effect)

    def _log_message(self, effect: LogMessage) -> Result[object, LoggingError]:
        """Emit a log message at the requested level."""
        logger_name = effect.logger_name or self._default_logger_name
        logger = logging.getLogger(logger_name)

        level_map: dict[LogLevel, int] = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        level_no = level_map.get(effect.level)
        if level_no is None:
            return Failure(LoggingError(message="invalid_log_level", logger_name=logger_name))

        try:
            logger.log(level_no, effect.message, exc_info=effect.exc_info)
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            return Failure(LoggingError(message=str(exc), logger_name=logger_name))


__all__ = [
    "LogMessage",
    "LoggingEffect",
    "LoggingInterpreter",
]
