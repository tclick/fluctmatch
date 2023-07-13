"""Fluctuation matching."""
import getpass
import sys
from typing import ParamSpec, TypeVar

from loguru import logger

T = TypeVar("T")
P = ParamSpec("P")
__version__: str = "4.0.0a0"
__copyright__: str = """Copyright (C) 2013-2023 Timothy H. Click <timothy.click@briarcliff.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

logger.remove()


def config_logger(logfile: str = "mdsetup.log", level: str = "INFO") -> None:
    """Configure logger.

    Parameters
    ----------
    logfile: str
        name of log file
    level : str
        minimum level for logging
    """
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                "colorize": True,
                "level": level,
                "backtrace": True,
                "diagnose": True,
            },
            {"sink": logfile, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", "level": level},
        ],
        "extra": {"user": f"{getpass.getuser()}"},
    }
    logger.configure(**config)
