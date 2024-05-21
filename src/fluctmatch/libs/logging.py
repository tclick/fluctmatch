# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#  DAMAGE.
# ------------------------------------------------------------------------------
"""Logging module."""

from __future__ import annotations

import getpass
import logging
import sys
from pathlib import Path

from loguru import logger
from loguru_logging_intercept import setup_loguru_logging_intercept


def config_logger(name: str, logfile: str | Path | None = None, level: str | int = "INFO") -> None:
    """Configure logger.

    Parameters
    ----------
    name : str
        name associated with the logger
    logfile: str or Path, optional
        name of log file
    level : str
        minimum level for logging
    """
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": "{time:YYYY-MM-DD HH:mm} | <level>{level.name}</level> | {message}",
                "colorize": True,
                "level": level,
                "backtrace": True,
                "diagnose": True,
            },
        ],
        "extra": {"name": name, "user": getpass.getuser()},
    }
    if logfile is not None:
        config["handlers"].append(
            {"sink": logfile, "format": "{time:YYYY-MM-DD HH:mm} | {level} | {message}", "level": level},
        )

    logger.remove()
    logger.configure(**config)
    setup_loguru_logging_intercept(level=logging.DEBUG, modules=f"root {name}".split())
