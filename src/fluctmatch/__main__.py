"""Command-line interface."""
from __future__ import annotations

import logging
import sys

from loguru import logger

from fluctmatch.cli import main

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class InterceptHandler(logging.Handler):
    """Intercept standard logging."""

    def emit(self: InterceptHandler, record: logging.LogRecord) -> None:
        """Emit standard logging to loguru.

        Parameters
        ----------
        record : logging.LogRecord
            logging record
        """
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

with logger.catch(message="An unexpected error occurred while running the program."):
    main()
