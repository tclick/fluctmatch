# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2023 Timothy H. Click, Ph.D.
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
"""Fluctuation matching."""

import getpass
import importlib
import pkgutil
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import MDAnalysis as mda
from loguru import logger

import fluctmatch.core.models
import fluctmatch.parsers.parsers
import fluctmatch.parsers.readers
import fluctmatch.parsers.writers
from fluctmatch.core.base import ModelBase

if TYPE_CHECKING:
    from loguru import Logger
else:
    from loguru import logger as Logger  # noqa: N812

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


def config_logger(logfile: str = "fluctmatch.log", level: str = "INFO") -> Logger:
    """Configure logger.

    Parameters
    ----------
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
        "extra": {"user": f"{getpass.getuser()}"},
    }
    logger.configure(**config)
    return logger


def iter_namespace(ns_pkg) -> Iterator[pkgutil.ModuleInfo]:  # noqa: ANN001
    """Iterate over a namespace package. [1]_.

    Parameters
    ----------
    ns_pkg : namespace

    References
    ----------
    .. [1] https://packaging.python.org/guides/creating-and-discovering-plugins/
    """
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


# Update the parsers in MDAnalysis
mda._PARSERS.update(
    {
        name.split(".")[-1].upper(): importlib.import_module(name).Reader
        for _, name, _ in iter_namespace(fluctmatch.parsers.parsers)
    }
)

mda._PARSERS["COR"] = mda._PARSERS["CRD"]

# Update the readers in MDAnalysis
mda._READERS.update(
    {
        name.split(".")[-1].upper(): importlib.import_module(name).Reader
        for _, name, _ in iter_namespace(fluctmatch.parsers.readers)
    }
)

# Update the writers in MDAnalysis
mda._SINGLEFRAME_WRITERS.update(
    {
        name.split(".")[-1].upper(): importlib.import_module(name).Writer
        for _, name, _ in iter_namespace(fluctmatch.parsers.writers)
    }
)

_MODELS: dict[str, ModelBase] = {
    name.split(".")[-1]: importlib.import_module(name).Model for _, name, _ in iter_namespace(fluctmatch.core.models)
}
