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
"""Fluctuation Matching."""

import getpass
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from click_loguru import ClickLoguru
from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger
else:
    from loguru import logger as Logger  # noqa: N812

NAME = "fluctmatch"
T = TypeVar("T")
P = ParamSpec("P")
__version__: str = "4.0.0a0"
__copyright__: str = """Copyright (C) 2013-2024 Timothy H. Click <thclick@umary.edu>

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

click_loguru = ClickLoguru(
    NAME,
    __version__,
    log_dir_parent=Path.cwd() / "logs",
    timer_log_level="info",
)


def config_logger(name: str, logfile: str = "fluctmatch.log", level: str = "INFO") -> Logger:
    """Configure logger.

    Parameters
    ----------
    name : str
        name associated with the logger
    logfile: str
        name of log file
    level : str
        minimum level for logging

    Returns
    -------
    Logger
        logging object
    """
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                "colorize": True,
                "level": level,
                "backtrace": True,
                "diagnose": True,
            },
            {"sink": logfile, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", "level": level},
        ],
        "extra": "",
        "patcher": "",
    }
    logger.configure(**config)
    return logger.bind(name=name, user=getpass.getuser())
