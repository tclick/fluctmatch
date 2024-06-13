# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This file is part of fluctmatch.
#
# Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
#
# Reference:
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
"""Logging module."""

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
